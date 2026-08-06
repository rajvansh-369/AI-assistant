"""Importing this module registers every tool.

`core/registry.py` fills itself from `@tool` decorators, which only run when the
module holding them is imported.  Rather than have main.py depend on the
side-effect of twenty unrelated imports, everything the registry needs is listed
here once — so a new tool is one decorated function plus one line in this file.

    import actions.tools   # noqa: F401  — registers the tool table
"""
from __future__ import annotations

from core.log import get_logger
from core.registry import names

# Import order is irrelevant; each module registers itself on import.
from actions import (          # noqa: F401
    background_monitor,
    browser_control,
    code_helper,
    computer_control,
    computer_settings,
    desktop,
    dev_agent,
    file_controller,
    file_processor,
    flight_finder,
    game_updater,
    open_app,
    reminder,
    send_message,
    session_tools,
    system_monitor,
    weather_report,
    web_builder,
    web_search,
    youtube_video,
)

log = get_logger("tools")
log.debug(f"{len(names())} tools registered: {', '.join(names())}")
