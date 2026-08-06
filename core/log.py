"""Logging for MARK L.

Replaces ~250 bare `print()` calls.  Two concrete problems it solves:

1. Emoji in print statements crash on a non-UTF-8 console.  `print("[Memory] 💾 …")`
   raises UnicodeEncodeError under Windows cp1252 — inside update_memory(), so a
   logging line could take down a memory write.  Handlers here are encoding-safe:
   the stream is switched to UTF-8 where possible, and characters that still fail
   are replaced rather than raised.
2. There was no way to turn the noise down, no timestamps, and no file to read
   after a crash.  Console level is configurable; the rotating file always keeps
   full DEBUG detail.

Usage:

    from core.log import get_logger
    log = get_logger("vision")          # -> logger "markl.vision"
    log.info("Screen captured: %d bytes", len(img))
    log.exception("capture failed")     # inside an except block
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys

from core.paths import LOG_DIR, LOG_PATH

ROOT_NAME    = "markl"
MAX_BYTES    = 2 * 1024 * 1024   # 2 MB per file
BACKUP_COUNT = 3

_configured = False


class _SafeStreamHandler(logging.StreamHandler):
    """StreamHandler that degrades unencodable characters instead of raising.

    Windows consoles frequently run cp1252; the log messages inherited from the
    original code are full of emoji.  Losing a glyph is fine, losing the process
    is not.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
        except UnicodeEncodeError:
            try:
                msg      = self.format(record)
                encoding = getattr(self.stream, "encoding", None) or "ascii"
                self.stream.write(
                    msg.encode(encoding, errors="replace").decode(encoding, errors="replace")
                    + self.terminator
                )
                self.flush()
            except Exception:
                self.handleError(record)
        except Exception:
            self.handleError(record)


class _CallbackHandler(logging.Handler):
    """Forwards records to an arbitrary sink — used to mirror warnings into the HUD."""

    def __init__(self, callback, level=logging.WARNING):
        super().__init__(level)
        self._callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._callback(self.format(record))
        except Exception:
            pass   # a broken UI sink must never break logging


def _console_level() -> int:
    """Console verbosity, overridable with MARKL_LOG_LEVEL=DEBUG|INFO|WARNING|…"""
    raw = os.environ.get("MARKL_LOG_LEVEL", "INFO").upper().strip()
    return getattr(logging, raw, logging.INFO)


def setup_logging(console_level: int | None = None) -> logging.Logger:
    """Configure the `markl` logger tree.  Safe to call more than once."""
    global _configured
    root = logging.getLogger(ROOT_NAME)
    if _configured:
        return root

    root.setLevel(logging.DEBUG)
    root.propagate = False

    # Console: keeps the original "[Tag] message" look, no timestamps to stay readable.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # py3.7+
    except Exception:
        pass
    console = _SafeStreamHandler(sys.stdout)
    console.setLevel(_console_level() if console_level is None else console_level)
    console.setFormatter(logging.Formatter("[%(shortname)s] %(message)s"))
    console.addFilter(_ShortName())
    root.addHandler(console)

    # File: always full detail, always UTF-8, rotated.
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_PATH, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)-22s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root.addHandler(file_handler)
    except Exception as e:
        root.warning("File logging disabled: %s", e)

    _configured = True
    return root


class _ShortName(logging.Filter):
    """Exposes %(shortname)s — the logger name without the 'markl.' prefix."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.shortname = record.name.split(".", 1)[-1] if "." in record.name else record.name
        return True


def get_logger(name: str) -> logging.Logger:
    """Return the `markl.<name>` logger, configuring the tree on first use."""
    if not _configured:
        setup_logging()
    return logging.getLogger(f"{ROOT_NAME}.{name}" if name else ROOT_NAME)


def add_ui_sink(callback, level: int = logging.WARNING) -> None:
    """Mirror records at `level` and above into the HUD log panel.

    Lets modules report problems by logging instead of reaching for a UI object.
    """
    root    = setup_logging()
    handler = _CallbackHandler(callback, level)
    handler.setFormatter(logging.Formatter("%(shortname)s: %(message)s"))
    handler.addFilter(_ShortName())
    root.addHandler(handler)
