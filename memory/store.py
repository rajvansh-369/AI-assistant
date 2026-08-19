"""SQLite behind the memory API.

`long_term.json` had a hard cap of 2200 characters — about a page — because
everything in it was pasted into the system prompt on every connect. Past that,
`_trim_to_limit` deleted the oldest entries, and since entries were sorted by
`updated` and identity facts are usually the oldest thing in the store, **the
user's own name was the first thing forgotten**. Reproduced: forty note writes
on a fresh store left `identity` completely empty.

The cap belonged to the prompt, not to storage. So: store everything here, and
let `memory_manager.format_memory_for_prompt` decide what fits in the prompt —
identity always, then whatever is most relevant to what is being discussed.

Why SQLite rather than a bigger JSON file:

* writes are atomic and crash-safe, which is what `memory_txn` was working
  around by hand
* embeddings are a natural BLOB column, kept next to the fact they describe
* `last_used` / `access_count` make it possible to forget by disuse rather than
  by age, which is what the old trim got backwards

Schema is deliberately small. There is no ORM and no migration framework — one
`schema_version` in `meta`, and an `_upgrade` ladder if it ever moves.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from core.log import get_logger
from core.paths import BASE_DIR

log = get_logger("memory.store")

DB_PATH   = BASE_DIR / "memory" / "memory.db"
JSON_PATH = BASE_DIR / "memory" / "long_term.json"

SCHEMA_VERSION = 2

#: The categories the model is allowed to write to. `identity` is special: it is
#: never dropped from the prompt, so it stays small by convention.
CATEGORIES = ("identity", "preferences", "projects", "relationships", "wishes", "notes")

#: Still enforced per value — a rambling "fact" is a summarisation failure, and
#: it would crowd out real ones in the prompt budget.
MAX_VALUE_LENGTH = 380

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id           INTEGER PRIMARY KEY,
    category     TEXT    NOT NULL,
    key          TEXT    NOT NULL,
    value        TEXT    NOT NULL,
    created      TEXT    NOT NULL,
    updated      TEXT    NOT NULL,
    last_used    TEXT,
    access_count INTEGER NOT NULL DEFAULT 0,
    embedding    BLOB,
    embed_model  TEXT,
    UNIQUE (category, key)
);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts (category);
CREATE INDEX IF NOT EXISTS idx_facts_updated  ON facts (updated);

CREATE TABLE IF NOT EXISTS sessions (
    id       INTEGER PRIMARY KEY,
    date     TEXT NOT NULL,
    summary  TEXT NOT NULL,
    language TEXT,
    created  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS monitors (
    slug       TEXT PRIMARY KEY,
    topic      TEXT NOT NULL,
    last_hash  TEXT,
    last_check TEXT,
    updated    TEXT
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

# One connection, one lock. Writes are small and rare (a fact per few turns);
# contention is not the problem this needs to solve, correctness is. RLock so a
# transaction body may call back into a reading helper.
_lock = threading.RLock()
_conn: sqlite3.Connection | None = None


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    """The process-wide connection, opening and migrating on first use."""
    global _conn
    with _lock:
        if _conn is not None:
            return _conn

        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # WAL survives a hard kill mid-write, which matters because the app is
        # normally closed by os._exit(0) from shutdown_jarvis.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        conn.commit()

        _conn = conn
        _migrate_from_json(conn)
        _upgrade(conn)
        return conn


def close() -> None:
    """Release the connection — tests point DB_PATH somewhere else between runs."""
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None


@contextmanager
def txn() -> Iterator[sqlite3.Connection]:
    """Read-modify-write under the lock, committed on clean exit."""
    conn = connect()
    with _lock:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


# ── meta ──────────────────────────────────────────────────────────────────────

def get_meta(key: str, default: str | None = None) -> str | None:
    row = connect().execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_meta(key: str, value: str) -> None:
    with txn() as conn:
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )


# ── migration ─────────────────────────────────────────────────────────────────

def _migrate_from_json(conn: sqlite3.Connection) -> None:
    """Import long_term.json once, then never look at it again.

    The JSON file is deliberately left on disk. It is the user's only copy of
    facts about themselves, it is already gitignored, and a rename buys tidiness
    at the cost of being the one irreversible step in this whole change.
    """
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    if row is not None:
        return                                  # already initialised

    facts = sessions = monitors = 0
    if JSON_PATH.exists():
        try:
            data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            log.error(f"Could not read {JSON_PATH.name} for migration: {e}")
            data = {}

        if isinstance(data, dict):
            for category, items in data.items():
                if category == "sessions" and isinstance(items, list):
                    for entry in items:
                        if isinstance(entry, dict) and entry.get("summary"):
                            conn.execute(
                                "INSERT INTO sessions (date, summary, language, created) "
                                "VALUES (?, ?, ?, ?)",
                                (entry.get("date") or _today(), entry["summary"],
                                 entry.get("language"), _now()),
                            )
                            sessions += 1
                    continue

                if category == "monitors" and isinstance(items, dict):
                    for slug, fields in items.items():
                        if isinstance(fields, dict):
                            conn.execute(
                                "INSERT OR REPLACE INTO monitors (slug, topic, last_hash, updated) "
                                "VALUES (?, ?, ?, ?)",
                                (slug, fields.get("topic", slug),
                                 fields.get("last_hash"), fields.get("updated") or _today()),
                            )
                            monitors += 1
                    continue

                if not isinstance(items, dict):
                    continue
                for key, entry in items.items():
                    value, updated = _unwrap(entry)
                    if not value:
                        continue
                    conn.execute(
                        "INSERT OR IGNORE INTO facts "
                        "(category, key, value, created, updated) VALUES (?, ?, ?, ?, ?)",
                        (category, key, value[:MAX_VALUE_LENGTH], updated, updated),
                    )
                    facts += 1

    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
    )
    conn.commit()

    if facts or sessions or monitors:
        log.info(
            f"Migrated {facts} facts, {sessions} sessions, {monitors} monitors "
            f"from {JSON_PATH.name} (left in place as a backup)"
        )


def _unwrap(entry: Any) -> tuple[str, str]:
    """A stored entry is either {'value':…, 'updated':…} or a bare scalar."""
    if isinstance(entry, dict):
        return str(entry.get("value", "")).strip(), str(entry.get("updated") or _today())
    return str(entry).strip(), _today()


def _upgrade(conn: sqlite3.Connection) -> None:
    """Bring an existing database up to SCHEMA_VERSION.

    v1 → v2: the monitors table gains `last_check`.  Without it, the daily gate
    in background_monitor never persisted — `replace_monitors` silently dropped
    the field — so every monitored topic hit the news API on every 30-minute
    cycle instead of once a day.
    """
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    try:
        version = int(row["value"]) if row else SCHEMA_VERSION
    except (TypeError, ValueError):
        version = SCHEMA_VERSION

    if version < 2:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(monitors)")}
        if "last_check" not in cols:
            conn.execute("ALTER TABLE monitors ADD COLUMN last_check TEXT")
        conn.execute(
            "UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(SCHEMA_VERSION),)
        )
        conn.commit()
        log.info(f"Schema upgraded v{version} → v{SCHEMA_VERSION}")


# ── facts ─────────────────────────────────────────────────────────────────────

def put_fact(category: str, key: str, value: str) -> bool:
    """Insert or update one fact. Returns True if anything actually changed.

    An unchanged rewrite is a no-op on purpose: `updated` is used for ranking,
    and the model re-saves the same fact often enough that touching the
    timestamp would make every old fact look new.
    """
    category = category if category in CATEGORIES else "notes"
    key      = (key or "").strip()
    value    = (value or "").strip()[:MAX_VALUE_LENGTH]
    if not key or not value:
        return False

    with txn() as conn:
        row = conn.execute(
            "SELECT value FROM facts WHERE category = ? AND key = ?", (category, key)
        ).fetchone()
        if row and row["value"] == value:
            return False

        if row:
            conn.execute(
                "UPDATE facts SET value = ?, updated = ?, embedding = NULL, embed_model = NULL "
                "WHERE category = ? AND key = ?",
                (value, _today(), category, key),
            )
        else:
            conn.execute(
                "INSERT INTO facts (category, key, value, created, updated) "
                "VALUES (?, ?, ?, ?, ?)",
                (category, key, value, _today(), _today()),
            )
    return True


def delete_fact(category: str, key: str) -> bool:
    with txn() as conn:
        cur = conn.execute(
            "DELETE FROM facts WHERE category = ? AND key = ?", (category, key)
        )
    return cur.rowcount > 0


def all_facts(category: str | None = None) -> list[sqlite3.Row]:
    sql    = "SELECT * FROM facts"
    params: tuple = ()
    if category:
        sql   += " WHERE category = ?"
        params = (category,)
    sql += " ORDER BY category, key"
    return connect().execute(sql, params).fetchall()


def facts_by_ids(ids: list[int]) -> list[sqlite3.Row]:
    if not ids:
        return []
    marks = ",".join("?" * len(ids))
    return connect().execute(
        f"SELECT * FROM facts WHERE id IN ({marks})", tuple(ids)
    ).fetchall()


def count_facts() -> int:
    return connect().execute("SELECT COUNT(*) AS n FROM facts").fetchone()["n"]


def mark_used(ids: list[int]) -> None:
    """Record that these facts were surfaced — the input to forgetting by disuse."""
    if not ids:
        return
    with txn() as conn:
        conn.executemany(
            "UPDATE facts SET access_count = access_count + 1, last_used = ? WHERE id = ?",
            [(_now(), i) for i in ids],
        )


def set_embedding(fact_id: int, blob: bytes, model: str) -> None:
    with txn() as conn:
        conn.execute(
            "UPDATE facts SET embedding = ?, embed_model = ? WHERE id = ?",
            (blob, model, fact_id),
        )


def facts_missing_embeddings(model: str, limit: int = 64) -> list[sqlite3.Row]:
    return connect().execute(
        "SELECT * FROM facts WHERE embedding IS NULL OR embed_model IS NOT ? LIMIT ?",
        (model, limit),
    ).fetchall()


# ── sessions ──────────────────────────────────────────────────────────────────

def add_session(summary: str, language: str = "") -> None:
    summary = (summary or "").strip()
    if not summary:
        return
    with txn() as conn:
        conn.execute(
            "INSERT INTO sessions (date, summary, language, created) VALUES (?, ?, ?, ?)",
            (_today(), summary[:280], language or None, _now()),
        )


def pop_session() -> dict | None:
    """Return and delete the newest session summary.

    Consuming it is what stops the morning briefing repeating "yesterday you
    were…" forever.
    """
    with txn() as conn:
        row = conn.execute("SELECT * FROM sessions ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            return None
        conn.execute("DELETE FROM sessions WHERE id = ?", (row["id"],))
    entry = {"date": row["date"], "summary": row["summary"]}
    if row["language"]:
        entry["language"] = row["language"]
    return entry


def recent_sessions(limit: int = 3) -> list[dict]:
    rows = connect().execute(
        "SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [{"date": r["date"], "summary": r["summary"], "language": r["language"] or ""}
            for r in rows]


# ── monitors ──────────────────────────────────────────────────────────────────

def get_monitors() -> dict[str, dict]:
    rows = connect().execute("SELECT * FROM monitors").fetchall()
    return {
        r["slug"]: {"topic": r["topic"], "last_hash": r["last_hash"],
                    "last_check": r["last_check"] or "", "updated": r["updated"]}
        for r in rows
    }


def put_monitor(slug: str, topic: str, last_hash: str | None = None,
                last_check: str | None = None) -> None:
    with txn() as conn:
        conn.execute(
            "INSERT INTO monitors (slug, topic, last_hash, last_check, updated) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(slug) DO UPDATE SET topic = excluded.topic, "
            "last_hash = excluded.last_hash, last_check = excluded.last_check, "
            "updated = excluded.updated",
            (slug, topic, last_hash, last_check, _today()),
        )


def delete_monitor(slug: str) -> bool:
    with txn() as conn:
        cur = conn.execute("DELETE FROM monitors WHERE slug = ?", (slug,))
    return cur.rowcount > 0


def replace_monitors(monitors: dict[str, dict]) -> None:
    """Overwrite the whole monitor set — the shape `memory_txn` callers expect.

    `last_check` must round-trip: background_monitor's once-a-day gate reads it,
    and dropping it here is what used to reduce "daily" to "every cycle".
    """
    with txn() as conn:
        conn.execute("DELETE FROM monitors")
        conn.executemany(
            "INSERT INTO monitors (slug, topic, last_hash, last_check, updated) "
            "VALUES (?, ?, ?, ?, ?)",
            [(slug, f.get("topic", slug), f.get("last_hash"),
              f.get("last_check") or None, f.get("updated") or _today())
             for slug, f in monitors.items() if isinstance(f, dict)],
        )
