"""Where the briefing thinks you are, and the ladder it walks to find news.

Two things matter here beyond correctness:

* **Privacy.** IP geolocation sends the user's address to a third party. It must
  not happen when JARVIS already knows where they live, and it must be possible
  to switch off entirely.
* **Honesty.** A small town falls through to its region or country. JARVIS says
  the place name out loud, so it has to name what was actually fetched — not the
  town it started from.

No test here touches the network; providers are stubbed.
"""
from __future__ import annotations

import json
import time

import pytest

from actions import location as loc
from actions.location import Location

#: Captured before the autouse fixture stubs it out, for the two tests that
#: need the real lookup rather than the guard.
_REAL_FROM_IP = loc._from_ip


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail loudly if a test reaches for a geolocation provider unexpectedly."""
    def boom(*a, **kw):
        raise AssertionError("unexpected network call")
    monkeypatch.setattr(loc, "_from_ip", boom)


@pytest.fixture
def settings(monkeypatch):
    """Control what actions.location sees in config."""
    def apply(**extra):
        class S:
            pass
        s = S()
        s.extra = extra
        monkeypatch.setattr(loc, "get_settings", lambda: s)
        return s
    apply()
    return apply


# ── resolution order ──────────────────────────────────────────────────────────

def test_config_wins(settings, monkeypatch):
    settings(location="Berlin")
    from memory import memory_manager as mm
    mm.remember("city", "Pune", "identity")

    got = loc.resolve()
    assert got.city == "Berlin"
    assert got.source == "config"


def test_config_parses_city_and_region(settings):
    settings(location="Pune, Maharashtra")
    got = loc.resolve()
    assert (got.city, got.region) == ("Pune", "Maharashtra")


def test_memory_is_used_when_config_is_empty(settings):
    settings(location="")
    from memory import memory_manager as mm
    mm.remember("city", "Pune", "identity")

    got = loc.resolve()
    assert got.city == "Pune"
    assert got.source == "memory"


def test_a_known_city_makes_no_network_call(settings):
    """The privacy guarantee: nothing leaves the machine if we already know."""
    settings(location="")
    from memory import memory_manager as mm
    mm.remember("city", "Pune", "identity")

    loc.resolve()          # the autouse fixture asserts _from_ip is never called


def test_cache_is_used_before_the_network(settings):
    settings(location="")
    from memory import store
    store.set_meta("geo_cache", json.dumps(
        {"city": "Pune", "region": "MH", "country": "India", "at": time.time()}
    ))

    got = loc.resolve()
    assert got.city == "Pune"
    assert got.source == "ip-cache"


def test_a_stale_cache_is_ignored(settings, monkeypatch):
    settings(location="")
    from memory import store
    store.set_meta("geo_cache", json.dumps(
        {"city": "Pune", "at": time.time() - loc.CACHE_TTL - 1}
    ))

    monkeypatch.setattr(loc, "_from_ip", lambda: Location(city="Berlin", source="ip"))
    assert loc.resolve().city == "Berlin"


def test_a_corrupt_cache_is_ignored(settings, monkeypatch):
    settings(location="")
    from memory import store
    store.set_meta("geo_cache", "{not json")

    monkeypatch.setattr(loc, "_from_ip", lambda: None)
    assert loc.resolve() is None


def test_resolve_returns_none_when_nothing_is_known(settings, monkeypatch):
    """Callers must handle this — the briefing falls back to world news."""
    settings(location="")
    monkeypatch.setattr(loc, "_from_ip", lambda: None)
    assert loc.resolve() is None


def test_allow_network_false_skips_the_lookup(settings):
    settings(location="")
    assert loc.resolve(allow_network=False) is None


def test_ip_geolocation_can_be_disabled(settings, monkeypatch):
    """The opt-out has to bail before any request is built, not just ignore it."""
    settings(ip_geolocation=False)

    def boom(*a, **kw):
        raise AssertionError("a provider was contacted despite the opt-out")
    monkeypatch.setattr("requests.get", boom)

    assert _REAL_FROM_IP() is None


def test_ip_lookup_survives_every_provider_failing(settings, monkeypatch):
    settings(ip_geolocation=True)

    def boom(*a, **kw):
        raise OSError("no route to host")
    monkeypatch.setattr("requests.get", boom)

    assert _REAL_FROM_IP() is None


# ── the query ladder ──────────────────────────────────────────────────────────

def test_ladder_runs_narrow_to_wide():
    got = Location(city="Naugachhia", region="Bihar", country="India").news_queries()
    assert got == ["Naugachhia Bihar news", "Bihar news", "India news"]


def test_ladder_omits_the_word_today():
    """Measured: adding 'today' returns nothing for a small town."""
    assert all("today" not in q
               for q in Location(city="Pune", region="MH").news_queries())


def test_ladder_handles_a_city_with_no_region():
    assert Location(city="Berlin", country="Germany").news_queries() == [
        "Berlin news", "Germany news"
    ]


def test_ladder_handles_country_only():
    assert Location(city="", country="India").news_queries() == ["India news"]


@pytest.mark.parametrize("query, expected", [
    ("Naugachhia Bihar news", "Naugachhia"),
    ("Bihar news",            "Bihar"),
    ("India news",            "India"),
])
def test_label_names_what_was_actually_fetched(query, expected):
    """Announcing your town then reading national headlines is worse than
    naming the wider place."""
    place = Location(city="Naugachhia", region="Bihar", country="India")
    assert place.label_for(query) == expected


# ── the briefing ladder ───────────────────────────────────────────────────────

HEADLINES = "Latest news: X\n\n1. Something genuinely happened in the region today\n   https://e.x"


def test_ladder_stops_at_the_first_hit(monkeypatch):
    import main
    seen = []

    def fake(query, strict=False):
        seen.append(query)
        return HEADLINES

    monkeypatch.setattr(main, "_fetch_news_sync", fake)
    text, used = main._fetch_news_ladder(["Pune MH news", "MH news"])

    assert used == "Pune MH news"
    assert seen == ["Pune MH news"], "kept searching after a hit"
    assert text == HEADLINES


def test_ladder_widens_past_an_empty_place(monkeypatch):
    import main

    def fake(query, strict=False):
        return HEADLINES if query == "Bihar news" else "No news found for: " + query

    monkeypatch.setattr(main, "_fetch_news_sync", fake)
    _, used = main._fetch_news_ladder(["Nowhere Bihar news", "Bihar news"])
    assert used == "Bihar news"


def test_local_rungs_are_strict(monkeypatch):
    """Without strict, DDG answers a town query with its air-quality page and
    the ladder never widens."""
    import main
    modes = []

    def fake(query, strict=False):
        modes.append(strict)
        return HEADLINES

    monkeypatch.setattr(main, "_fetch_news_sync", fake)
    main._fetch_news_ladder(["Pune MH news"])
    assert modes == [True]


def test_world_is_the_last_resort_and_not_strict(monkeypatch):
    """An empty briefing is worse than a slightly off-topic one."""
    import main
    calls = []

    def fake(query, strict=False):
        calls.append((query, strict))
        return "" if query != main.WORLD_NEWS_QUERY else HEADLINES

    monkeypatch.setattr(main, "_fetch_news_sync", fake)
    text, used = main._fetch_news_ladder(["Pune MH news"])

    assert used == main.WORLD_NEWS_QUERY
    assert calls[-1] == (main.WORLD_NEWS_QUERY, False)
    assert text == HEADLINES


def test_no_location_goes_straight_to_world(monkeypatch):
    import main
    monkeypatch.setattr(main, "_fetch_news_sync", lambda q, strict=False: HEADLINES)
    _, used = main._fetch_news_ladder([])
    assert used == main.WORLD_NEWS_QUERY


def test_a_raising_backend_does_not_break_the_briefing(monkeypatch):
    import main

    def fake(query, strict=False):
        if strict:
            raise RuntimeError("network down")
        return HEADLINES

    monkeypatch.setattr(main, "_fetch_news_sync", fake)
    text, used = main._fetch_news_ladder(["Pune MH news"])
    assert used == main.WORLD_NEWS_QUERY
    assert text == HEADLINES


def test_total_failure_returns_empty_not_an_exception(monkeypatch):
    import main

    def fake(query, strict=False):
        raise RuntimeError("everything is down")

    monkeypatch.setattr(main, "_fetch_news_sync", fake)
    text, used = main._fetch_news_ladder(["Pune MH news"])
    assert text == ""
    assert used == main.WORLD_NEWS_QUERY
