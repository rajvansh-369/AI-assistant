"""Single source of truth for every filesystem path MARK L uses.

Before this module, `get_base_dir()` was copy-pasted into eleven files with three
subtly different bodies.  Import from here instead of re-deriving paths locally.
"""
from __future__ import annotations

import sys
from pathlib import Path


def get_base_dir() -> Path:
    """Project root — works both from source and from a PyInstaller bundle."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR    = get_base_dir()

CONFIG_DIR  = BASE_DIR / "config"
CONFIG_PATH = CONFIG_DIR / "api_keys.json"
CERTS_DIR   = CONFIG_DIR / "certs"
CERT_PATH   = CERTS_DIR / "jarvis.crt"
KEY_PATH    = CERTS_DIR / "jarvis.key"

CORE_DIR    = BASE_DIR / "core"
PROMPT_PATH = CORE_DIR / "prompt.txt"

MEMORY_DIR  = BASE_DIR / "memory"
MEMORY_PATH = MEMORY_DIR / "long_term.json"

LOG_DIR     = BASE_DIR / "logs"
LOG_PATH    = LOG_DIR / "markl.log"


def uploads_dir() -> Path:
    """Where phone uploads land — first writable candidate wins."""
    for candidate in (
        Path.home() / "Downloads" / "JARVIS Uploads",
        Path.home() / "Documents" / "JARVIS Uploads",
        BASE_DIR / "uploads",
    ):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except Exception:
            continue
    return BASE_DIR / "uploads"
