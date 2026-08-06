"""Where the user is, for local news and anything else that needs a place.

Resolution order, most authoritative first:

1. `location` in `config/api_keys.json` — an explicit override always wins
2. `identity.city` in long-term memory — what the user actually told JARVIS
3. IP geolocation — a guess, cached, only reached when nothing better exists

The ordering is the privacy design as much as the correctness one: if JARVIS
already knows where you live, **no request leaves the machine**. The lookup only
happens when it would otherwise have nothing.

Geolocation sends the user's IP to a third-party service, so:

* it is skipped entirely when a city is already known
* `ip_geolocation: false` in `config/api_keys.json` disables it outright
* the inferred city is cached in the store's `meta` table, deliberately *not*
  written into `identity` — memory is what the user said about themselves, and
  an IP guess is wrong often enough (VPN, mobile, corporate egress) that it has
  no business overwriting that
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from core.log import get_logger
from core.settings import get_settings

log = get_logger("location")

#: Both are free and keyless. ipapi.co is HTTPS; ip-api.com is the fallback
#: because its free tier is HTTP-only.
PROVIDERS = (
    ("https://ipapi.co/json/",  "city", "region",     "country_name"),
    ("http://ip-api.com/json/", "city", "regionName", "country"),
)

#: Short — this sits on the startup path, in front of the first spoken word.
TIMEOUT = 3.0

#: Cities do not move. A week means a trip abroad is picked up eventually
#: without a lookup on every launch.
CACHE_TTL = 7 * 86400

_META_KEY = "geo_cache"


@dataclass(frozen=True)
class Location:
    city:    str
    region:  str = ""
    country: str = ""
    source:  str = "unknown"     #: config | memory | ip | ip-cache

    def label(self) -> str:
        """Short human name — what JARVIS would say out loud."""
        return self.city

    def news_queries(self) -> list[str]:
        """Search queries from most local to least, to try in order.

        A city name alone returns nothing for anywhere small — measured:
        "Naugachhia Bihar news today" gets no results while "Bihar news today"
        is fine. Rather than guess whether somewhere is big enough, ask for the
        narrowest thing first and widen until something answers.
        """
        # No "today": measured, "Naugachhia Bihar news today" returns nothing
        # while "Naugachhia Bihar news" finds real local articles. A small
        # town's coverage is thin enough that the extra word excludes it.
        queries = []
        if self.city and self.region:
            queries.append(f"{self.city} {self.region} news")
        elif self.city:
            queries.append(f"{self.city} news")
        if self.region:
            queries.append(f"{self.region} news")
        if self.country:
            queries.append(f"{self.country} news")
        return queries

    def label_for(self, query: str) -> str:
        """Which place a winning query actually covered.

        JARVIS says this out loud, so it has to match what was really fetched —
        announcing local news for your town and then reading national headlines
        is worse than just saying "India".
        """
        for place in (self.city, self.region, self.country):
            if place and query.startswith(place):
                return place
        return self.city or self.region or self.country


# ── sources ───────────────────────────────────────────────────────────────────

def _from_config() -> Location | None:
    raw = str(get_settings().extra.get("location", "")).strip()
    if not raw:
        return None
    # "Pune, Maharashtra" or just "Pune"
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return Location(city=parts[0], region=parts[1] if len(parts) > 1 else "",
                    source="config")


def _from_memory() -> Location | None:
    try:
        from memory import store
        rows = {r["key"]: r["value"] for r in store.all_facts("identity")}
    except Exception as e:
        log.debug(f"Could not read memory for location: {e}")
        return None

    city = (rows.get("city") or "").strip()
    if not city:
        return None
    return Location(city=city, country=(rows.get("nationality") or "").strip(),
                    source="memory")


def _cached() -> Location | None:
    try:
        from memory import store
        raw = store.get_meta(_META_KEY)
        if not raw:
            return None
        data = json.loads(raw)
        if time.time() - data.get("at", 0) > CACHE_TTL:
            return None
        return Location(city=data["city"], region=data.get("region", ""),
                        country=data.get("country", ""), source="ip-cache")
    except Exception:
        return None


def _cache(loc: Location) -> None:
    try:
        from memory import store
        store.set_meta(_META_KEY, json.dumps({
            "city": loc.city, "region": loc.region,
            "country": loc.country, "at": time.time(),
        }))
    except Exception as e:
        log.debug(f"Could not cache location: {e}")


def _from_ip() -> Location | None:
    """Ask a geolocation service where this IP is. Returns None on any failure."""
    if not bool(get_settings().extra.get("ip_geolocation", True)):
        log.info("IP geolocation disabled in config")
        return None

    import requests

    for url, city_key, region_key, country_key in PROVIDERS:
        try:
            resp = requests.get(url, timeout=TIMEOUT,
                                headers={"User-Agent": "MarkL/1.0"})
            resp.raise_for_status()
            data = resp.json()
            city = str(data.get(city_key) or "").strip()
            if not city:
                continue
            loc = Location(
                city    = city,
                region  = str(data.get(region_key) or "").strip(),
                country = str(data.get(country_key) or "").strip(),
                source  = "ip",
            )
            log.info(f"Located via {url.split('/')[2]}: {loc.city}, {loc.country}")
            _cache(loc)
            return loc
        except Exception as e:
            log.debug(f"{url.split('/')[2]} failed: {e}")

    log.info("IP geolocation failed — no provider answered")
    return None


# ── public ────────────────────────────────────────────────────────────────────

def resolve(allow_network: bool = True) -> Location | None:
    """Best known location, or None if there is genuinely nothing to go on.

    Callers must handle None — a briefing that fails because a geolocation
    service was down is worse than one that reads world news.
    """
    for source in (_from_config, _from_memory, _cached):
        loc = source()
        if loc:
            return loc
    return _from_ip() if allow_network else None
