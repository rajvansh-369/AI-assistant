"""The once-a-day gate in background_monitor, and the store bug that broke it.

`check_all` gates on `last_check == today`, but after the SQLite migration the
monitors table had no `last_check` column — `replace_monitors` silently dropped
the field on every write.  The gate therefore never held, and every monitored
topic hit the DDG news API on every 30-minute cycle instead of once a day.

Schema v2 adds the column; these tests pin the round-trip and the gate.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from actions import background_monitor as bm
from memory import memory_manager as mm
from memory import store


TODAY     = datetime.now().strftime("%Y-%m-%d")
YESTERDAY = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")


@pytest.fixture
def ddg(monkeypatch):
    """Replace the news fetch with a counter the tests control."""
    state = {"calls": 0, "headline": "First headline"}

    def fake_news(query, max_results=5, allow_text_fallback=True):
        state["calls"] += 1
        return [{"title": state["headline"], "snippet": "s", "url": "u", "source": "src"}]

    import actions.web_search
    monkeypatch.setattr(actions.web_search, "_ddg_news", fake_news)
    return state


def _set_last_check(topic: str, day: str) -> None:
    slug = bm._slug(topic)
    with mm.memory_txn() as memory:
        memory["monitors"][slug]["last_check"] = day


# ── the store round-trip ──────────────────────────────────────────────────────

def test_last_check_round_trips_through_the_store():
    store.replace_monitors({
        "ai": {"topic": "AI news", "last_hash": "abc", "last_check": TODAY},
    })
    assert store.get_monitors()["ai"]["last_check"] == TODAY


def test_memory_txn_preserves_last_check():
    """The exact write path check_all uses."""
    bm.add_monitor("space exploration")
    _set_last_check("space exploration", TODAY)

    slug = bm._slug("space exploration")
    assert store.get_monitors()[slug]["last_check"] == TODAY


# ── the daily gate ────────────────────────────────────────────────────────────

def test_first_check_fires_and_alerts(ddg):
    bm.add_monitor("space exploration")
    alerts = bm.check_all()
    assert len(alerts) == 1
    assert "First headline" in alerts[0]
    assert ddg["calls"] == 1


def test_second_check_same_day_is_skipped(ddg):
    """The regression: this used to hit the network every 30-minute cycle."""
    bm.add_monitor("space exploration")
    bm.check_all()
    assert bm.check_all() == []
    assert ddg["calls"] == 1, "check_all queried DDG again on the same day"


def test_next_day_checks_again_but_same_headline_stays_silent(ddg):
    bm.add_monitor("space exploration")
    bm.check_all()

    _set_last_check("space exploration", YESTERDAY)
    assert bm.check_all() == [], "an unchanged headline must not re-alert"
    assert ddg["calls"] == 2


def test_next_day_alerts_on_a_new_headline(ddg):
    bm.add_monitor("space exploration")
    bm.check_all()

    _set_last_check("space exploration", YESTERDAY)
    ddg["headline"] = "Second headline"
    alerts = bm.check_all()
    assert len(alerts) == 1
    assert "Second headline" in alerts[0]


# ── schema upgrade ────────────────────────────────────────────────────────────

def test_v1_database_gains_the_column():
    """An existing install upgrades in place, keeping its monitors."""
    import sqlite3

    store.close()
    store.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(store.DB_PATH)
    conn.executescript("""
        CREATE TABLE monitors (
            slug      TEXT PRIMARY KEY,
            topic     TEXT NOT NULL,
            last_hash TEXT,
            updated   TEXT
        );
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO meta VALUES ('schema_version', '1');
        INSERT INTO monitors VALUES ('ai', 'AI news', 'abc', '2026-01-01');
    """)
    conn.commit()
    conn.close()

    monitors = store.get_monitors()          # connect() runs the upgrade
    assert monitors["ai"]["topic"] == "AI news"
    assert monitors["ai"]["last_check"] == ""
    assert store.get_meta("schema_version") == "2"
