"""Persistent, concurrency-safe store of named enzyme kinetic profiles.

Profiles are kept in a JSON file on disk so they survive process restarts.
Concurrency safety works on two levels:

* an in-process :class:`threading.RLock` serializes threads in one worker;
* a POSIX ``fcntl`` lock on a sidecar lock file additionally serializes the
  multiple worker processes gunicorn forks.

Writes replace the data file atomically via temp-file + ``os.replace`` and
persist with ``fsync``, so a reader either sees the previous version or the
new complete one — never a torn write.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Iterator

DEFAULT_HEXOKINASE = {
    # Hexokinase, glucose phosphorylation (illustrative hand-checkable values).
    "vmax": 100.0,          # μmol product / min / mg enzyme
    "km": 0.1,              # mM glucose (~blood glucose order of magnitude)
    "description": (
        "Hexokinase demonstration profile: Km(glucose)=0.1 mM, "
        "Vmax=100 umol/min/mg. At [S]=Km the rate is exactly 50; "
        "competitive inhibition by a 0.2 mM ligand with Ki=0.2 mM "
        "gives Km_app=0.2 mM."
    ),
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class EnzymeProfile:
    """A named set of Michaelis constants for one enzyme."""

    vmax: float
    km: float
    description: str = ""
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    def to_json(self) -> dict[str, object]:
        return asdict(self)


class EnzymeNotFoundError(KeyError):
    """Raised when a named enzyme profile is not registered."""


class EnzymeStore:
    """JSON-file backed store safe for threaded and multi-process use."""

    def __init__(self, path: str, *, seed_default: bool = True) -> None:
        self.path = os.path.abspath(path)
        self._lock_path = self.path + ".lock"
        self._thread_lock = threading.RLock()
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with self._process_lock(exclusive=True):
            if not os.path.exists(self.path):
                data = (
                    {"hexokinase": {**DEFAULT_HEXOKINASE, "created_at": _utc_now_iso(),
                                    "updated_at": _utc_now_iso()}}
                    if seed_default
                    else {}
                )
                self._write_unlocked(data)

    # ------------------------------------------------------------------ locking
    @contextlib.contextmanager
    def _process_lock(self, *, exclusive: bool) -> Iterator[None]:
        lock_file = open(self._lock_path, "a+")
        try:
            fcntl.flock(
                lock_file.fileno(),
                fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH,
            )
            with self._thread_lock:
                yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()

    # ------------------------------------------------------------------- io
    def _read_unlocked(self) -> dict[str, dict[str, object]]:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"enzyme store at {self.path} is unreadable: {exc}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"enzyme store at {self.path} is corrupt")
        return data

    def _write_unlocked(self, data: dict[str, dict[str, object]]) -> None:
        directory = os.path.dirname(self.path) or "."
        fd, tmp_path = tempfile.mkstemp(prefix=".enzymes-", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, self.path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.remove(tmp_path)
            raise

    @staticmethod
    def _from_record(record: dict[str, object]) -> EnzymeProfile:
        return EnzymeProfile(
            vmax=float(record["vmax"]),  # type: ignore[arg-type]
            km=float(record["km"]),      # type: ignore[arg-type]
            description=str(record.get("description", "")),
            created_at=str(record.get("created_at", "")),
            updated_at=str(record.get("updated_at", "")),
        )

    # ------------------------------------------------------------------ public
    def list_names(self) -> list[str]:
        with self._process_lock(exclusive=False):
            return sorted(self._read_unlocked().keys())

    def get(self, name: str) -> EnzymeProfile:
        with self._process_lock(exclusive=False):
            data = self._read_unlocked()
        if name not in data:
            raise EnzymeNotFoundError(name)
        return self._from_record(data[name])

    def exists(self, name: str) -> bool:
        with self._process_lock(exclusive=False):
            return name in self._read_unlocked()

    def upsert(
        self, name: str, vmax: float, km: float, description: str = ""
    ) -> tuple[EnzymeProfile, bool]:
        """Create or replace a profile. Returns (profile, created)."""
        with self._process_lock(exclusive=True):
            data = self._read_unlocked()
            created = name not in data
            now = _utc_now_iso()
            created_at = (
                str(data[name].get("created_at", now)) if not created else now
            )
            profile = EnzymeProfile(
                vmax=float(vmax),
                km=float(km),
                description=description,
                created_at=created_at,
                updated_at=now,
            )
            data[name] = profile.to_json()
            self._write_unlocked(data)
        return profile, created

    def delete(self, name: str) -> None:
        with self._process_lock(exclusive=True):
            data = self._read_unlocked()
            if name not in data:
                raise EnzymeNotFoundError(name)
            del data[name]
            self._write_unlocked(data)
