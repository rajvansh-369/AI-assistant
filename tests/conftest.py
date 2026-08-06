"""Keep tests away from the real memory store.

`memory/store.py` opens a process-wide connection to `memory/memory.db` and
migrates `memory/long_term.json` into it on first use. Without this fixture a
test run would read, write and migrate the actual user's facts.
"""
from __future__ import annotations

import pytest

from memory import store


@pytest.fixture(autouse=True)
def isolated_memory(tmp_path, monkeypatch):
    """Point the store at a fresh database for every test."""
    store.close()
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "memory.db")
    monkeypatch.setattr(store, "JSON_PATH", tmp_path / "long_term.json")
    yield
    store.close()
