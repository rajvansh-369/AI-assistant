"""Typed, cached access to config/api_keys.json.

Replaces ten hand-rolled `_get_api_key()` helpers (which disagreed on whether a
missing key should raise, return "" or return None) and the model IDs that were
hardcoded across twelve files.

    from core.settings import get_settings, get_api_key, MODELS

    s = get_settings()
    client = genai.Client(api_key=s.gemini_api_key)
    model  = s.fast_model

The parsed file is cached and invalidated automatically when its mtime changes,
so a key edited from the UI is picked up without a restart.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field, replace

from core.paths import CONFIG_PATH, CONFIG_DIR


def _log():
    """Imported lazily so settings stays usable if logging is misconfigured."""
    from core.log import get_logger
    return get_logger("settings")

# ── Model IDs ─────────────────────────────────────────────────────────────────
# Change a model here, not in twelve action modules.  Any of these can be
# overridden per-install by adding the same key to config/api_keys.json.

LIVE_MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"
# The 2.5 line is closed to new API keys — calls to gemini-2.5-flash and
# gemini-2.5-flash-lite return 404 "no longer available to new users" even
# though models.list() still reports them.  Verify a replacement with a real
# generate_content call, not the model listing.
FAST_MODEL = "gemini-3.6-flash"        # general reasoning, summaries, vision
LITE_MODEL = "gemini-3.5-flash-lite"   # cheap classification / parameter extraction

# End-of-session summarisation.  Carried over verbatim from a local edit in
# main.py; it is deliberately separate from FAST_MODEL so the choice is visible
# and revertible in one place.
SUMMARY_MODEL = "gemini-3.6-flash"


@dataclass(frozen=True)
class Settings:
    gemini_api_key:        str  = ""
    os_system:             str  = "windows"
    assistant_name:        str  = "JARVIS"
    user_name:             str  = ""
    morning_brief_enabled: bool = True
    ui_color:              str  = ""

    #: "free" keeps every Gemini call inside what a free API key can sustain;
    #: "paid" unlocks grounded search and the fast model.  See core.budget.
    mode:                  str  = "free"

    live_model:    str = LIVE_MODEL
    fast_model:    str = FAST_MODEL
    lite_model:    str = LITE_MODEL
    summary_model: str = SUMMARY_MODEL

    #: Everything else present in the file, so nothing is silently dropped on save.
    extra: dict = field(default_factory=dict, repr=False)

    @property
    def is_configured(self) -> bool:
        return bool(self.gemini_api_key) and len(self.gemini_api_key) > 15


_KNOWN = {f for f in Settings.__dataclass_fields__ if f != "extra"}

_lock:   threading.RLock = threading.RLock()
_cache:  Settings | None = None
_cache_mtime: float      = -1.0


def _read_raw() -> dict:
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        # Never crash the assistant over a malformed config — fall back to defaults.
        _log().warning(f"Could not parse {CONFIG_PATH.name}: {e}")
        return {}


def _mtime() -> float:
    try:
        return CONFIG_PATH.stat().st_mtime
    except OSError:
        return -1.0


def get_settings(force: bool = False) -> Settings:
    """Return the current settings, re-reading the file only when it changed."""
    global _cache, _cache_mtime
    with _lock:
        mtime = _mtime()
        if _cache is not None and not force and mtime == _cache_mtime:
            return _cache

        raw    = _read_raw()
        known  = {k: v for k, v in raw.items() if k in _KNOWN}
        extra  = {k: v for k, v in raw.items() if k not in _KNOWN}
        try:
            _cache = Settings(**known, extra=extra)
        except TypeError as e:
            _log().warning(f"Ignoring bad config values: {e}")
            _cache = Settings(extra=extra)
        _cache_mtime = mtime
        return _cache


def invalidate() -> None:
    """Force the next get_settings() to re-read from disk."""
    global _cache, _cache_mtime
    with _lock:
        _cache = None
        _cache_mtime = -1.0


def get_api_key(required: bool = True) -> str:
    """The Gemini API key.

    Raises RuntimeError when missing and `required` (the behaviour every caller
    that builds a client wants); pass required=False for optional probes.
    """
    key = get_settings().gemini_api_key.strip()
    if not key and required:
        raise RuntimeError(
            "gemini_api_key not found in config/api_keys.json. "
            "Copy config/api_keys.example.json and add your key."
        )
    return key


def save_settings(**changes) -> Settings:
    """Merge `changes` into config/api_keys.json and refresh the cache.

    Unknown keys already in the file are preserved.
    """
    with _lock:
        raw = _read_raw()
        raw.update({k: v for k, v in changes.items() if v is not None})
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(
            json.dumps(raw, indent=4, ensure_ascii=False), encoding="utf-8"
        )
        invalidate()
        return get_settings()
