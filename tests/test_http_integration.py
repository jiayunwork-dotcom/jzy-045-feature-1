"""Integration tests against a live HTTP server.

These exercise actual sockets + threads (rather than the in-process test
client) to prove concurrent requests stay independent, and prove profiles
survive a full server process-object restart against the same data file.
"""

import json
import threading
import urllib.error
import urllib.request
from wsgiref.simple_server import WSGIRequestHandler, make_server

import pytest

from app import create_app


class _QuietHandler(WSGIRequestHandler):
    def log_message(self, *args):  # silence stderr access log
        pass


@pytest.fixture()
def live_server(tmp_path):
    store_path = str(tmp_path / "enzymes.json")
    application = create_app(store_path=store_path)
    server = make_server("127.0.0.1", 0, application, handler_class=_QuietHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", store_path
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _request(base, path, payload):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode())


def test_concurrent_calculations_are_independent(live_server):
    base, _ = live_server
    errors = []

    def worker(worker_id):
        try:
            for i in range(50):
                vmax = 10.0 + worker_id
                km = 0.1 * (i + 1)
                substrate = km  # always the half-saturation point
                status, body = _request(base, "/v1/rate", {
                    "vmax": vmax, "km": km, "substrate": substrate,
                })
                assert status == 200
                assert body["rate"] == pytest.approx(vmax / 2.0, rel=1e-12)
                assert body["km"] == km
                assert body["vmax"] == vmax
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


def test_concurrent_profile_registrations_do_not_cross_talk(live_server):
    base, _ = live_server
    errors = []

    def register_worker(worker_id):
        try:
            for i in range(10):
                name = f"enzyme-w{worker_id}-{i}"
                data = json.dumps({"vmax": float(worker_id + 1),
                                   "km": float(i + 1)}).encode()
                req = urllib.request.Request(
                    f"{base}/v1/enzymes/{name}", data=data,
                    headers={"Content-Type": "application/json"}, method="PUT",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    assert resp.status in (200, 201)
                _, body = _request(base, "/v1/rate",
                                   {"enzyme": name, "substrate": float(i + 1)})
                # Reading back each own profile: rate must be its Vmax/2
                # (substrate == its own Km) — no profile may overwrite another.
                assert body["rate"] == pytest.approx(
                    (worker_id + 1) / 2.0, rel=1e-12
                ), body
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=register_worker, args=(w,))
               for w in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []

    with urllib.request.urlopen(base + "/v1/enzymes", timeout=10) as resp:
        names = json.loads(resp.read().decode())["enzymes"]
    for w in range(8):
        for i in range(10):
            assert f"enzyme-w{w}-{i}" in names
    assert "hexokinase" in names


def test_profile_available_after_server_restart(live_server):
    base, store_path = live_server

    data = json.dumps({"vmax": 33.3, "km": 0.77}).encode()
    req = urllib.request.Request(
        f"{base}/v1/enzymes/restart_check", data=data,
        headers={"Content-Type": "application/json"}, method="PUT",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 201

    # Simulate a restart: build a brand-new app/store over the same file.
    application2 = create_app(store_path=store_path)
    server2 = make_server("127.0.0.1", 0, application2,
                          handler_class=_QuietHandler)
    thread2 = threading.Thread(target=server2.serve_forever, daemon=True)
    thread2.start()
    try:
        base2 = f"http://127.0.0.1:{server2.server_address[1]}"
        _, body = _request(base2, "/v1/rate",
                           {"enzyme": "restart_check", "substrate": 0.77})
        assert body["rate"] == pytest.approx(33.3 / 2.0, rel=1e-12)
        # Pre-seeded demo enzyme also remains.
        _, body = _request(base2, "/v1/rate",
                           {"enzyme": "hexokinase", "substrate": 0.1})
        assert body["rate"] == pytest.approx(50.0)
    finally:
        server2.shutdown()
        server2.server_close()
        thread2.join(timeout=5)


def test_http_400_includes_reason(live_server):
    base, _ = live_server
    req = urllib.request.Request(
        base + "/v1/rate",
        data=json.dumps({"vmax": -1, "km": 1, "substrate": 1}).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=10)
    err = exc_info.value
    assert err.code == 400
    body = json.loads(err.read().decode())
    assert body["error"] == "validation_error"
    assert "vmax" in body["reason"]
