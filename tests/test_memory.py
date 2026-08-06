"""Memory v2 — the store, the retrieval, and the bug that motivated both.

The headline case is `test_identity_survives_a_flood_of_notes`. Under the old
JSON store, `_trim_to_limit` sorted entries by `updated` and deleted the oldest
once the file passed 2200 characters — and identity facts are the oldest thing
in most stores. Forty note writes left `identity` empty: JARVIS forgot the
user's name to make room for trivia. Identity is now pinned by construction,
and nothing is deleted to make room for anything.

Every test gets a fresh database via the autouse fixture in conftest.py.
"""
from __future__ import annotations

import json
import threading

import pytest

from memory import memory_manager as mm
from memory import store


@pytest.fixture(autouse=True)
def lexical_only(monkeypatch):
    """Pin scoring to the local backend.

    Otherwise a machine with a paid key would make live embedding calls during
    the test run, and ranking assertions would depend on the network.
    """
    from memory import embeddings
    monkeypatch.setattr(embeddings, "available", lambda: False)


# ── migration ─────────────────────────────────────────────────────────────────

def write_json(payload: dict) -> None:
    store.close()
    store.JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    store.JSON_PATH.write_text(json.dumps(payload), encoding="utf-8")


def test_migration_imports_facts():
    write_json({"identity": {"name": {"value": "Snehal", "updated": "2026-01-02"}},
                "preferences": {"editor": {"value": "VS Code", "updated": "2026-01-03"}}})

    memory = mm.load_memory()
    assert memory["identity"]["name"]["value"] == "Snehal"
    assert memory["preferences"]["editor"]["value"] == "VS Code"
    assert memory["identity"]["name"]["updated"] == "2026-01-02", "lost the original date"


def test_migration_imports_sessions_and_monitors():
    write_json({
        "sessions": [{"date": "2026-01-01", "summary": "built a drone", "language": "Hindi"}],
        "monitors": {"ai-news": {"topic": "AI news", "last_hash": "abc"}},
    })

    assert store.get_monitors()["ai-news"]["topic"] == "AI news"
    assert mm.pop_last_session()["summary"] == "built a drone"


def test_migration_tolerates_bare_scalars():
    """The old store sometimes held a plain string instead of {'value': …}."""
    write_json({"identity": {"city": "Pune"}})
    assert mm.load_memory()["identity"]["city"]["value"] == "Pune"


def test_migration_leaves_the_json_alone():
    """It is the user's only copy — importing must not be destructive."""
    write_json({"identity": {"name": {"value": "Snehal"}}})
    mm.load_memory()
    assert store.JSON_PATH.exists()


def test_migration_runs_once():
    write_json({"identity": {"name": {"value": "Snehal"}}})
    mm.load_memory()

    mm.remember("name", "Someone Else", "identity")
    store.close()                                   # reopen: must not re-import

    assert mm.load_memory()["identity"]["name"]["value"] == "Someone Else"


def test_migration_survives_a_corrupt_file():
    store.close()
    store.JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    store.JSON_PATH.write_text("{not json", encoding="utf-8")

    assert mm.load_memory()["identity"] == {}       # empty, not a crash


def test_empty_start_is_fine():
    assert store.count_facts() == 0
    assert mm.format_memory_for_prompt() == ""


# ── the eviction bug ──────────────────────────────────────────────────────────

def test_identity_survives_a_flood_of_notes():
    """The regression that motivated memory v2: identity used to go first."""
    mm.remember("name", "Snehal", "identity")
    mm.remember("language", "Hindi", "identity")

    for i in range(40):
        mm.remember(f"note_{i}", f"some passing remark number {i} " * 6, "notes")

    memory = mm.load_memory()
    assert memory["identity"]["name"]["value"] == "Snehal"
    assert memory["identity"]["language"]["value"] == "Hindi"

    prompt = mm.format_memory_for_prompt()
    assert "Snehal" in prompt, "the user's own name was evicted from the prompt"
    assert "Hindi" in prompt


def test_nothing_is_deleted_to_make_room():
    for i in range(60):
        mm.remember(f"note_{i}", f"fact number {i}", "notes")
    assert store.count_facts() == 60, "storage is no longer capped"


def test_prompt_stays_within_budget():
    mm.remember("name", "Snehal", "identity")
    for i in range(60):
        mm.remember(f"note_{i}", "x" * 300, "notes")

    prompt = mm.format_memory_for_prompt()
    assert len(prompt) < mm.PROMPT_CHAR_BUDGET + 600, f"prompt was {len(prompt)} chars"


# ── writing ───────────────────────────────────────────────────────────────────

def test_update_memory_accepts_the_nested_form():
    mm.update_memory({"identity": {"name": {"value": "Snehal"}}})
    assert mm.load_memory()["identity"]["name"]["value"] == "Snehal"


def test_update_memory_accepts_bare_values():
    mm.update_memory({"notes": {"habit": "codes at night"}})
    assert mm.load_memory()["notes"]["habit"]["value"] == "codes at night"


def test_blank_values_are_ignored():
    mm.update_memory({"notes": {"empty": "", "none": None, "spaces": "   "}})
    assert store.count_facts() == 0


def test_unknown_category_lands_in_notes():
    mm.update_memory({"nonsense": {"thing": "value"}})
    assert mm.load_memory()["notes"]["thing"]["value"] == "value"


def test_rewriting_the_same_value_is_a_no_op():
    """`updated` drives ranking; re-saving must not make an old fact look new."""
    mm.remember("name", "Snehal", "identity")
    before = store.all_facts("identity")[0]["updated"]

    assert store.put_fact("identity", "name", "Snehal") is False
    assert store.all_facts("identity")[0]["updated"] == before


def test_changing_a_value_clears_its_embedding():
    """A vector must never outlive the text it described."""
    mm.remember("job", "backend developer", "identity")
    row = store.all_facts("identity")[0]
    store.set_embedding(row["id"], b"\x00" * 16, "text-embedding-004")

    mm.remember("job", "engineering manager", "identity")
    assert store.all_facts("identity")[0]["embedding"] is None


def test_long_values_are_truncated():
    mm.remember("rambling", "y" * 5000, "notes")
    assert len(mm.load_memory()["notes"]["rambling"]["value"]) <= store.MAX_VALUE_LENGTH


def test_forget_removes_a_fact():
    mm.remember("secret", "hidden", "notes")
    assert "Forgotten" in mm.forget("secret", "notes")
    assert "secret" not in mm.load_memory()["notes"]


def test_forget_reports_a_miss():
    assert "Not found" in mm.forget("never_existed", "notes")


def test_concurrent_writes_do_not_lose_updates():
    """The lost-update race the old load/save pair had."""
    def writer(n):
        for i in range(20):
            mm.remember(f"t{n}_{i}", f"value {i}", "notes")

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert store.count_facts() == 80


# ── retrieval ─────────────────────────────────────────────────────────────────

def seed_facts():
    mm.remember("name", "Snehal", "identity")
    mm.remember("sister_name", "Priya", "relationships")
    mm.remember("favorite_food", "pizza", "preferences")
    mm.remember("drone_project", "building a quadcopter flight controller", "projects")
    mm.remember("travel_plan", "wants to visit Iceland", "wishes")


def test_recall_finds_the_matching_fact():
    seed_facts()
    hits = mm.recall("what is my sister called")
    assert hits and hits[0]["key"] == "sister_name"


def test_recall_matches_on_the_value_too():
    seed_facts()
    hits = mm.recall("quadcopter")
    assert any(h["key"] == "drone_project" for h in hits)


def test_recall_returns_nothing_for_an_unrelated_query():
    seed_facts()
    assert mm.recall("photosynthesis in ferns") == []


def test_recall_excludes_identity_by_default():
    """Identity is already in every prompt; repeating it wastes the top-k."""
    seed_facts()
    assert all(h["category"] != "identity" for h in mm.recall("name"))


def test_recall_can_include_identity():
    seed_facts()
    hits = mm.recall("name", include_identity=True)
    assert any(h["category"] == "identity" for h in hits)


def test_recall_respects_k():
    for i in range(30):
        mm.remember(f"note_{i}", "shared keyword here", "notes")
    assert len(mm.recall("shared keyword", k=5)) == 5


def test_recall_records_usage():
    """`access_count` is the input to forgetting by disuse rather than by age."""
    seed_facts()
    mm.recall("sister")
    row = next(r for r in store.all_facts() if r["key"] == "sister_name")
    assert row["access_count"] == 1
    assert row["last_used"] is not None


def test_usage_never_outranks_a_better_match():
    seed_facts()
    for _ in range(50):
        mm.recall("pizza")                      # inflate favorite_food

    hits = mm.recall("what is my sister called")
    assert hits[0]["key"] == "sister_name", "a popular fact hijacked an unrelated query"


def test_query_shapes_the_prompt():
    seed_facts()
    for i in range(40):
        mm.remember(f"filler_{i}", f"unrelated trivia {i}", "notes")

    prompt = mm.format_memory_for_prompt(query="tell me about the drone")
    assert "quadcopter" in prompt
    assert "Snehal" in prompt, "identity must be present regardless of the query"


def test_stopwords_do_not_match_everything():
    seed_facts()
    assert mm.recall("what is the of and to") == []


# ── synonym expansion (the local backend's substitute for embeddings) ─────────

@pytest.mark.parametrize("query, expected", [
    ("what do I do for work",     "current_job"),
    ("where do I work",           "current_job"),
    ("what am I building",        "drone_project"),
    ("my side project",           "drone_project"),
    ("who is my sibling",         "sister_name"),
    ("what do I like to eat",     "favorite_food"),
    ("where do I want to travel", "travel_plan"),
])
def test_synonyms_bridge_the_wording_gap(query, expected):
    """Word matching alone connects none of these to the stored key."""
    seed_facts()
    mm.remember("current_job", "Senior Backend Developer", "identity")

    hits = mm.recall(query, include_identity=True)
    assert hits, f"{query!r} found nothing"
    assert hits[0]["key"] == expected, f"{query!r} -> {[h['key'] for h in hits]}"


def test_an_exact_word_beats_a_synonym():
    """A synonym hit must never outrank the word the user actually said."""
    mm.remember("job_title", "engineer", "identity")
    mm.remember("work_history", "five previous employers", "notes")

    hits = mm.recall("what is my job title", include_identity=True)
    assert hits[0]["key"] == "job_title"


def test_expansion_does_not_match_everything():
    seed_facts()
    assert mm.recall("quantum chromodynamics") == []


# ── sessions ──────────────────────────────────────────────────────────────────

def test_session_summary_round_trip():
    mm.save_session_summary("worked on the drone controller", "English")
    entry = mm.pop_last_session()
    assert entry["summary"] == "worked on the drone controller"
    assert entry["language"] == "English"


def test_popping_a_session_consumes_it():
    """Otherwise the briefing repeats 'yesterday you were…' every morning."""
    mm.save_session_summary("built something")
    mm.pop_last_session()
    assert mm.pop_last_session() is None


def test_pop_returns_the_newest():
    mm.save_session_summary("older")
    mm.save_session_summary("newer")
    assert mm.pop_last_session()["summary"] == "newer"


def test_blank_summaries_are_dropped():
    mm.save_session_summary("   ")
    assert mm.pop_last_session() is None


def test_last_session_topic_does_not_consume():
    """_build_config seeds retrieval with it; the briefing still needs it after."""
    mm.save_session_summary("worked on the drone")
    assert "drone" in mm.last_session_topic()
    assert mm.pop_last_session() is not None


def test_last_session_topic_is_empty_on_a_fresh_store():
    assert mm.last_session_topic() == ""


# ── monitors (background_monitor still uses memory_txn) ───────────────────────

def test_memory_txn_round_trips_monitors():
    with mm.memory_txn() as memory:
        memory["monitors"]["ai-news"] = {"topic": "AI news", "last_hash": "abc"}

    assert store.get_monitors()["ai-news"]["last_hash"] == "abc"


def test_background_monitor_still_works():
    from actions import background_monitor as bm

    bm.add_monitor("space exploration")
    assert "space exploration" in bm.list_monitors()

    bm.remove_monitor("space exploration")
    assert bm.list_monitors() == []
