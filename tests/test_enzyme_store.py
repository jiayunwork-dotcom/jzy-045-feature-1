"""Tests for the on-disk enzyme profile store."""

import json

import pytest

from app.enzyme_store import EnzymeStore, EnzymeNotFoundError


def test_hexokinase_seeded_and_hand_checkable(store):
    profile = store.get("hexokinase")
    assert profile.vmax == 100.0
    assert profile.km == 0.1
    assert profile.description  # documents the hand-checkable demo values


def test_seed_only_when_file_missing(store_path):
    EnzymeStore(store_path)
    store1 = EnzymeStore(store_path)
    store1.upsert("custom", vmax=5.0, km=1.0, description="mine")
    # Re-opening must not overwrite existing data with the seed.
    store2 = EnzymeStore(store_path)
    assert store2.exists("custom")
    assert store2.exists("hexokinase")


def test_profile_persists_across_store_restart(store_path):
    store1 = EnzymeStore(store_path)
    store1.upsert("lactate_dehydrogenase", vmax=42.5, km=0.25,
                  description="example")
    del store1

    store2 = EnzymeStore(store_path)
    profile = store2.get("lactate_dehydrogenase")
    assert profile.vmax == 42.5
    assert profile.km == 0.25
    assert profile.description == "example"
    assert profile.created_at
    assert profile.updated_at


def test_update_preserves_created_at(store):
    profile, created = store.upsert("enz", 1.0, 1.0)
    assert created
    updated, created2 = store.upsert("enz", 2.0, 3.0)
    assert not created2
    assert updated.created_at == profile.created_at
    assert updated.updated_at >= profile.updated_at
    assert store.get("enz").vmax == 2.0


def test_get_missing_raises(store):
    with pytest.raises(EnzymeNotFoundError):
        store.get("ghost")


def test_delete_roundtrip(store):
    store.upsert("temp", 1.0, 1.0)
    store.delete("temp")
    assert not store.exists("temp")
    with pytest.raises(EnzymeNotFoundError):
        store.delete("temp")


def test_list_names_sorted(store):
    store.upsert("zzz", 1.0, 1.0)
    store.upsert("aaa", 1.0, 1.0)
    names = store.list_names()
    assert names == sorted(names)
    assert {"hexokinase", "zzz", "aaa"}.issubset(names)


def test_file_is_valid_json_and_tempfiles_gone(store_path):
    store = EnzymeStore(store_path)
    store.upsert("persisted", 9.0, 8.0)
    with open(store_path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["persisted"]["vmax"] == 9.0
    import os
    leftovers = [f for f in os.listdir(os.path.dirname(store_path))
                 if f.startswith(".enzymes-")]
    assert leftovers == []


def test_concurrent_upserts_do_not_lose_profiles(store):
    import threading

    def worker(prefix):
        for i in range(25):
            store.upsert(f"{prefix}-{i}", vmax=1.0 + i, km=1.0)

    threads = [threading.Thread(target=worker, args=(f"t{t}",)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    names = set(store.list_names())
    for t in range(8):
        for i in range(25):
            assert f"t{t}-{i}" in names
    # Every profile must be independently readable with its own constants.
    for t in range(8):
        for i in range(25):
            p = store.get(f"t{t}-{i}")
            assert p.vmax == 1.0 + i


def test_empty_store_seeds_nothing_when_disabled(tmp_path):
    path = str(tmp_path / "empty.json")
    store = EnzymeStore(path, seed_default=False)
    assert store.list_names() == []
