"""What JARVIS knows about you.

The public functions here are unchanged from the JSON era on purpose — eight
call sites across main.py, proactive.py and background_monitor.py keep working
without edits. What changed is underneath (`memory/store.py`, SQLite) and at the
edges:

* **Storage is no longer capped.** The old 2200-character limit existed because
  the whole store went into the system prompt. Now the store is unbounded and
  `format_memory_for_prompt` does the budgeting.
* **Identity is pinned.** The old `_trim_to_limit` sorted by `updated` and
  deleted the oldest, and identity facts are the oldest thing most stores have —
  so the user's own name was the first thing forgotten. Identity now always goes
  in the prompt, and nothing is deleted to make room.
* **Memory is no longer frozen at connect.** `recall(query)` retrieves mid
  session, which is what the `recall_memory` tool calls.

`load_memory()` still returns the whole store as the old nested dict, because
that is what callers expect and what `format_memory_for_prompt` consumes. It is
a *view* now, rebuilt per call, not the file itself.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime

from core.log import get_logger

from memory import store
from memory.store import CATEGORIES, MAX_VALUE_LENGTH  # noqa: F401 — re-exported

log = get_logger("memory")

#: How much of the system prompt memory is allowed to take. Identity is exempt:
#: it is small, and being wrong about who the user is defeats the point.
PROMPT_CHAR_BUDGET = 2000

#: How many non-identity facts to retrieve for a prompt or a recall.
TOP_K = 12


# ── the dict view ─────────────────────────────────────────────────────────────

def load_memory() -> dict:
    """The whole store, shaped like the old long_term.json."""
    memory: dict = {c: {} for c in CATEGORIES}
    for row in store.all_facts():
        memory.setdefault(row["category"], {})[row["key"]] = {
            "value":   row["value"],
            "updated": row["updated"],
        }
    memory["sessions"] = store.recent_sessions(limit=3)[::-1]   # oldest first
    memory["monitors"] = store.get_monitors()
    return memory


def save_memory(memory: dict) -> None:
    """Write back a whole dict view. Kept for callers that still think in files."""
    if not isinstance(memory, dict):
        return
    for category, items in memory.items():
        if category in ("sessions", "monitors") or not isinstance(items, dict):
            continue
        for key, entry in items.items():
            value = entry.get("value") if isinstance(entry, dict) else entry
            if value:
                store.put_fact(category, key, str(value))
    if isinstance(memory.get("monitors"), dict):
        store.replace_monitors(memory["monitors"])


def update_memory(memory_update: dict) -> dict:
    """Merge `{category: {key: {'value': …}}}` into the store.

    Accepts the bare-scalar form too (`{category: {key: 'value'}}`), which the
    old recursive merge tolerated and some callers rely on.
    """
    if not isinstance(memory_update, dict) or not memory_update:
        return load_memory()

    changed = []
    for category, items in memory_update.items():
        if not isinstance(items, dict):
            continue
        for key, entry in items.items():
            value = entry.get("value") if isinstance(entry, dict) else entry
            if value is None or not str(value).strip():
                continue
            if store.put_fact(category, key, str(value)):
                changed.append(f"{category}/{key}")

    if changed:
        log.info(f"Saved: {', '.join(changed)}")
    return load_memory()


@contextmanager
def memory_txn():
    """Read-modify-write the dict view atomically.

    Only `background_monitor` still uses this, and only for `memory['monitors']`.
    Kept so that module did not have to change alongside the storage swap; new
    code should call the `store` functions directly.
    """
    with store._lock:
        memory = load_memory()
        yield memory
        save_memory(memory)


# ── retrieval ─────────────────────────────────────────────────────────────────

def recall(query: str, k: int = TOP_K, include_identity: bool = False) -> list[dict]:
    """The facts most relevant to `query`, best first.

    Ranking is relevance-led with a small nudge for facts that have proved
    useful before, so a fact JARVIS has actually needed beats an equally similar
    one he never has.
    """
    from memory import embeddings

    rows = [r for r in store.all_facts()
            if include_identity or r["category"] != "identity"]
    if not rows or not (query or "").strip():
        return [_as_dict(r) for r in rows[:k]]

    scores  = embeddings.score(query, rows)
    ranked  = sorted(
        zip(rows, scores),
        key=lambda pair: pair[1] + _usage_bonus(pair[0]),
        reverse=True,
    )
    hits = [r for r, s in ranked if s > 0][:k]

    store.mark_used([r["id"] for r in hits])
    return [_as_dict(r) for r in hits]


def _usage_bonus(row) -> float:
    """At most a small tiebreak — never enough to outrank a better match.

    Without the cap a fact recalled fifty times would pin itself to the top of
    every query regardless of what was asked.
    """
    from math import log1p
    return min(0.05, 0.01 * log1p(row["access_count"]))


def _as_dict(row) -> dict:
    return {
        "category": row["category"],
        "key":      row["key"],
        "value":    row["value"],
        "updated":  row["updated"],
    }


# ── prompt rendering ──────────────────────────────────────────────────────────

_ID_ORDER = ["name", "age", "birthday", "city", "job", "language", "school", "nationality"]

_HEADINGS = {
    "preferences":   "Preferences",
    "projects":      "Active Projects / Goals",
    "relationships": "People in their life",
    "wishes":        "Wishes / Plans / Wants",
    "notes":         "Other notes",
}


def format_memory_for_prompt(memory: dict | None = None, query: str = "") -> str:
    """Render memory for the system instruction, within a character budget.

    `memory` is accepted and ignored when a `query` is given — callers pass the
    dict view out of habit, but retrieval reads the store directly so it can
    rank. With no query it falls back to the whole store, newest first, which is
    what the old function did.

    Identity always goes in first and is never budgeted away.
    """
    lines: list[str] = []

    identity = {r["key"]: r["value"] for r in store.all_facts("identity")}
    for field in _ID_ORDER:
        if identity.get(field):
            lines.append(f"{field.title()}: {identity.pop(field)}")
    for key, value in identity.items():
        lines.append(f"{key.replace('_', ' ').title()}: {value}")

    others = (recall(query, k=TOP_K) if (query or "").strip()
              else [_as_dict(r) for r in _by_recency()])

    used = sum(len(line) + 1 for line in lines)
    grouped: dict[str, list[str]] = {}
    for fact in others:
        line = f"  - {fact['key'].replace('_', ' ').title()}: {fact['value']}"
        if used + len(line) + 1 > PROMPT_CHAR_BUDGET:
            break
        used += len(line) + 1
        grouped.setdefault(fact["category"], []).append(line)

    for category, heading in _HEADINGS.items():
        if grouped.get(category):
            lines.append("")
            lines.append(f"{heading}:")
            lines.extend(grouped[category])

    if not lines:
        return ""

    header = "[WHAT YOU KNOW ABOUT THIS PERSON — use naturally, never recite like a list]\n"
    return header + "\n".join(lines) + "\n"


def _by_recency() -> list:
    rows = [r for r in store.all_facts() if r["category"] != "identity"]
    return sorted(rows, key=lambda r: r["updated"], reverse=True)


# ── small helpers kept from the old module ────────────────────────────────────

def remember(key: str, value: str, category: str = "notes") -> str:
    if category not in CATEGORIES:
        category = "notes"
    store.put_fact(category, key, value)
    return f"Remembered: {category}/{key} = {value}"


def forget(key: str, category: str = "notes") -> str:
    if store.delete_fact(category, key):
        return f"Forgotten: {category}/{key}"
    return f"Not found: {category}/{key}"


forget_memory = forget


# ── session memory ────────────────────────────────────────────────────────────

def save_session_summary(summary: str, language: str = "") -> None:
    """Append a 1-2 sentence summary of the session that just ended."""
    summary = (summary or "").strip()
    if not summary:
        return
    store.add_session(summary, language)
    log.info(f"Session saved ({datetime.now():%Y-%m-%d}): {summary[:60]}…")


def last_session_topic() -> str:
    """The newest session summary, without consuming it.

    Used to seed retrieval at connect: with no conversation yet there is nothing
    to rank facts against, and "what we were talking about last time" is a much
    better guess than "whatever was edited most recently".
    """
    recent = store.recent_sessions(limit=1)
    return recent[0]["summary"] if recent else ""


def pop_last_session() -> dict | None:
    """Return AND remove the most recent session summary.

    Consuming it is what stops the briefing repeating the same "yesterday you
    were…" every morning.
    """
    try:
        return store.pop_session()
    except Exception as e:
        log.error(f"pop_last_session error: {e}")
        return None
