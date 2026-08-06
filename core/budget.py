"""Free-tier / paid-tier policy for every Gemini call.

The 429 this exists to stop:

    RESOURCE_EXHAUSTED — You exceeded your current quota

It came from `web_search`, which asks Gemini to answer with the grounded
`google_search` tool. That tool has the tightest free-tier allowance in the
whole app, and the assistant calls it for every search, news check and morning
brief — so a free key runs out of it long before anything else.

Two modes, switchable from the UI:

    paid  — everything on: grounded search, the fast model, LLM classification
    free  — only what a free key can sustain:
              * no grounded search        → DuckDuckGo instead (no quota at all)
              * lite model everywhere     → the biggest free per-day allowance
              * no optional LLM calls     → the ones that already have a
                                            non-LLM fallback, e.g. intent
                                            detection in code_helper
              * a local rate limit        → FREE_RPM requests/minute, so a
                                            burst cannot trip a per-minute quota

Free mode is not the only protection: a 429 from *any* call (paid mode
included) trips a cooldown via `report()`, during which the app degrades itself
exactly as if free mode were on. Quota comes back on its own; hammering the API
while it is exhausted only produces more errors.

Callers use two functions:

    budget.reserve("grounded_search")   # raises QuotaExhausted → use fallback
    budget.model("fast")                # the model id to actually send
"""
from __future__ import annotations

import threading
import time
from collections import deque

from core.log import get_logger
from core.settings import get_settings, save_settings

log = get_logger("budget")

FREE = "free"
PAID = "paid"

#: Requests per minute allowed while degraded. Free-tier per-minute limits sit
#: at 10-15 depending on the model; staying under the lowest one is the point.
FREE_RPM = 8

#: Longer than this and waiting for a slot is worse than falling back.
MAX_WAIT = 12.0

#: How long a 429 keeps the app degraded.
QUOTA_COOLDOWN = 900.0

#: Features a free key cannot sustain. One-shot vision and normal generation are
#: not here on purpose — they are the assistant's core, and the lite model covers
#: them within the free allowance.
#:
#: `video_stream` is: a frame a second for as long as it is on, on top of the
#: audio already flowing. It is the single most expensive thing the app can do,
#: and unlike a search it has no cheaper fallback to degrade to — so it is
#: refused up front rather than discovered as a 429 mid-conversation.
FREE_BLOCKED = {"grounded_search", "optional_llm", "video_stream"}


class QuotaExhausted(RuntimeError):
    """Raised instead of making a call that is expected to fail with 429."""


_lock         = threading.Lock()
_calls: deque = deque()      #: monotonic timestamps of recent Gemini calls
_quota_until  = 0.0          #: monotonic deadline of the current cooldown
_notified     = False        #: cooldown already logged


# ── mode ──────────────────────────────────────────────────────────────────────

def get_mode() -> str:
    return FREE if (get_settings().mode or FREE).strip().lower() != PAID else PAID


def set_mode(mode: str) -> str:
    mode = PAID if str(mode).strip().lower() == PAID else FREE
    save_settings(mode=mode)
    log.info(f"Mode → {mode.upper()}")
    return mode


def is_free() -> bool:
    return get_mode() == FREE


def cooldown_remaining() -> float:
    with _lock:
        return max(0.0, _quota_until - time.monotonic())


def degraded() -> bool:
    """Free mode, or paid mode inside a quota cooldown."""
    return is_free() or cooldown_remaining() > 0


# ── policy ────────────────────────────────────────────────────────────────────

def model(tier: str = "fast") -> str:
    """The model id to send for `tier` — "fast", "lite" or "summary".

    Config overrides in api_keys.json still win; degraded mode drops every tier
    to the lite model.
    """
    s = get_settings()
    if degraded():
        return s.lite_model
    return {"fast": s.fast_model,
            "lite": s.lite_model,
            "summary": s.summary_model}.get(tier, s.fast_model)


def allows(feature: str) -> bool:
    return not (degraded() and feature in FREE_BLOCKED)


def reserve(feature: str = "generate") -> None:
    """Clear a Gemini call, or raise QuotaExhausted so the caller falls back.

    Blocks for up to MAX_WAIT seconds when the local rate limit is the only
    thing in the way — a short wait is better than a failed search.
    """
    if not allows(feature):
        raise QuotaExhausted(
            f"{feature} is off in free mode — using the free fallback instead."
        )

    if not degraded():
        return

    while True:
        now = time.monotonic()
        with _lock:
            while _calls and now - _calls[0] >= 60.0:
                _calls.popleft()
            if len(_calls) < FREE_RPM:
                _calls.append(now)
                return
            wait = 60.0 - (now - _calls[0])

        if wait > MAX_WAIT:
            raise QuotaExhausted(
                f"free-mode rate limit reached — {wait:.0f}s until the next slot."
            )
        time.sleep(min(wait, MAX_WAIT) + 0.05)


def is_quota_error(exc: object) -> bool:
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "429" in text


def report(exc: object) -> bool:
    """Record a failed call. Returns True if it was a quota error.

    Every `except` around a Gemini call should pass the exception here — that is
    what turns one 429 into a cooldown instead of a repeating log line.
    """
    global _quota_until, _notified
    if not is_quota_error(exc):
        return False
    with _lock:
        first = time.monotonic() >= _quota_until
        _quota_until = time.monotonic() + QUOTA_COOLDOWN
        if first:
            _notified = False
        notify, _notified = not _notified, True
    if notify:
        log.warning(
            f"Quota exhausted (429) — running in reduced mode for "
            f"{QUOTA_COOLDOWN / 60:.0f} min: lite model, no grounded search."
        )
    return True


def status() -> str:
    """One line for the UI / log."""
    cd = cooldown_remaining()
    if cd > 0:
        return f"{get_mode().upper()} · quota cooldown {cd / 60:.0f} min"
    return get_mode().upper()
