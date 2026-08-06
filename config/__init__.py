# config/__init__.py — platform helpers, backed by core.settings
import platform

from core.settings import get_settings


def _platform_os() -> str:
    """Auto-detect OS when the config has no explicit setting."""
    return {"Windows": "windows", "Darwin": "mac", "Linux": "linux"}.get(
        platform.system(), "linux"
    )


def get_config() -> dict:
    """Deprecated — prefer core.settings.get_settings(). Kept for existing callers."""
    s = get_settings()
    return {**s.extra,
            "os_system":      s.os_system,
            "assistant_name": s.assistant_name,
            "user_name":      s.user_name}


def get_os() -> str:
    """Returns: 'windows' | 'mac' | 'linux'"""
    return (get_settings().os_system or _platform_os()).lower()


def is_windows() -> bool: return get_os() == "windows"
def is_mac()     -> bool: return get_os() == "mac"
def is_linux()   -> bool: return get_os() == "linux"
