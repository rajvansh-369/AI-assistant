"""Config accessors kept for backwards compatibility.

All reads and writes now go through core.settings, which owns parsing, caching
and cache invalidation.  This module is a thin facade so existing call sites in
ui.py and main.py keep working — prefer core.settings in new code.
"""
from core.paths import CONFIG_DIR, CONFIG_PATH as CONFIG_FILE
from core.settings import get_settings, save_settings


def ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def config_exists() -> bool:
    return CONFIG_FILE.exists()


def load_api_keys() -> dict:
    s = get_settings()
    return {**s.extra,
            "gemini_api_key":        s.gemini_api_key,
            "os_system":             s.os_system,
            "assistant_name":        s.assistant_name,
            "user_name":             s.user_name,
            "morning_brief_enabled": s.morning_brief_enabled}


def save_api_keys(gemini_api_key: str) -> None:
    save_settings(gemini_api_key=gemini_api_key.strip())


def get_gemini_key() -> str | None:
    return get_settings().gemini_api_key or None


def is_configured() -> bool:
    return get_settings().is_configured


def get_assistant_name() -> str:
    return get_settings().assistant_name or "JARVIS"


def get_user_name() -> str:
    return get_settings().user_name


def save_assistant_config(assistant_name: str, user_name: str) -> None:
    save_settings(assistant_name=assistant_name.strip() or "JARVIS",
                  user_name=user_name.strip())


def get_brief_enabled() -> bool:
    return get_settings().morning_brief_enabled


def save_brief_enabled(enabled: bool) -> None:
    save_settings(morning_brief_enabled=bool(enabled))
