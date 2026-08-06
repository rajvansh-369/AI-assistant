"""Tool registry — one place a tool declares itself.

Before this module, adding a tool meant editing two places in a 1535-line
main.py: a 448-line TOOL_DECLARATIONS literal and a 20-branch if/elif chain in
_execute_tool.  Every tool also had a slightly different call signature, so each
branch needed its own bespoke lambda.

Now a tool owns its schema next to its implementation:

    from core.registry import tool, ToolContext

    @tool(
        name="weather_report",
        description="Gives the weather report to user",
        parameters={"type": "OBJECT",
                    "properties": {"city": {"type": "STRING", "description": "City name"}},
                    "required": ["city"]},
        timeout=20,
    )
    def weather_tool(params: dict, ctx: ToolContext) -> str:
        return weather_action(parameters=params, player=ctx.ui) or "Weather delivered."

Registration happens on import, so main.py only has to import the module.

Two properties this buys beyond tidiness:

* `declarations()` generates the Gemini function schema, so the declaration and
  the implementation can never drift apart.
* Every tool gets a timeout.  Previously a tool that hung — a Playwright
  selector that never matched, a Steam download, a scrape — held a thread
  forever while Gemini waited for a function response that would never arrive,
  and the user heard silence with no recovery path.
"""
from __future__ import annotations

import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from core.log import get_logger

log = get_logger("registry")

#: Tools run here rather than on the default executor, so a pile of hung tools
#: cannot starve the rest of the process of threads.
TOOL_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool")

DEFAULT_TIMEOUT = 30.0


@dataclass
class ToolContext:
    """Everything a tool is allowed to reach for.

    Replaces the old grab-bag of `player=`, `response=`, `speak=` and
    `session_memory=` keyword arguments, which differed per tool.
    """
    ui:      Any = None                     #: JarvisUI — write_log, show_content, state
    speak:   Callable[[str], None] | None = None   #: inject text into the live session
    jarvis:  Any = None                     #: JarvisLive, for tools touching the session
    current_file: str | None = None         #: last file dropped on the HUD

    def log_line(self, text: str) -> None:
        if self.ui is not None:
            try:
                self.ui.write_log(text)
            except Exception:
                pass


@dataclass
class Tool:
    name:        str
    description: str
    parameters:  dict
    fn:          Callable[..., Any]
    timeout:     float | None = DEFAULT_TIMEOUT   #: None disables the limit
    silent:      bool = False                      #: response marked silent to the model
    is_async:    bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.is_async = inspect.iscoroutinefunction(self.fn)

    def declaration(self) -> dict:
        return {
            "name":        self.name,
            "description": self.description,
            "parameters":  self.parameters,
        }


_REGISTRY: dict[str, Tool] = {}


def tool(
    name:        str,
    description: str,
    parameters:  dict | None = None,
    timeout:     float | None = DEFAULT_TIMEOUT,
    silent:      bool = False,
):
    """Register a function as a Gemini-callable tool.

    The function takes `(params: dict, ctx: ToolContext)` and returns a string.
    It may be sync (run on TOOL_EXECUTOR) or async (awaited on the event loop —
    use this only for tools that must touch the live session).
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in _REGISTRY:
            log.warning(f"Tool '{name}' registered twice — keeping the newer one.")
        _REGISTRY[name] = Tool(
            name        = name,
            description = description,
            parameters  = parameters or {"type": "OBJECT", "properties": {}},
            fn          = fn,
            timeout     = timeout,
            silent      = silent,
        )
        return fn
    return decorator


def get(name: str) -> Tool | None:
    return _REGISTRY.get(name)


def unregister(name: str) -> Tool | None:
    """Drop a tool from the table, returning it if it was there.

    Used by tests, and by the plugin loader to pull a plugin whose tool raised
    on load rather than let it break the whole session.
    """
    return _REGISTRY.pop(name, None)


def names() -> list[str]:
    return sorted(_REGISTRY)


def all_tools() -> list[Tool]:
    return [_REGISTRY[n] for n in sorted(_REGISTRY)]


def declarations() -> list[dict]:
    """Gemini function declarations for every registered tool."""
    return [t.declaration() for t in all_tools()]


class ToolTimeout(Exception):
    """Raised when a tool exceeds its declared timeout."""


async def run(name: str, params: dict, ctx: ToolContext) -> str:
    """Execute a registered tool and return its result string.

    Raises KeyError if the tool is not registered, ToolTimeout if it overran,
    and whatever the tool itself raised otherwise.  The caller is responsible
    for turning all three into a function response — the model must always get
    one back, or the conversation stalls waiting forever.  ToolTimeout's message
    is already phrased to be spoken as-is.
    """
    t = _REGISTRY[name]
    loop = asyncio.get_running_loop()

    if t.is_async:
        coro = t.fn(params, ctx)
    else:
        coro = loop.run_in_executor(TOOL_EXECUTOR, t.fn, params, ctx)

    try:
        if t.timeout:
            result = await asyncio.wait_for(coro, timeout=t.timeout)
        else:
            result = await coro
    except asyncio.TimeoutError:
        log.warning(f"{name} exceeded its {t.timeout}s timeout")
        # The thread may still be running; it is bounded by TOOL_EXECUTOR's size
        # and will not block the event loop.
        raise ToolTimeout(
            f"{name} took longer than {t.timeout:.0f} seconds and was stopped."
        )
    return "Done." if result is None else str(result)
