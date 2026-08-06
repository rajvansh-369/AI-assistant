"""What the registry is actually for: a tool can never hang the conversation.

Before the registry, `_execute_tool` offloaded every tool to the default
executor with no time limit.  A Playwright selector that never matched, a Steam
download, a scrape behind a dead host — any of them held a thread forever while
Gemini waited for a function response that would never arrive, and the user
heard silence with no way to recover.

These tests exercise the dispatch path with throwaway tools, so nothing here
opens a browser or touches the OS.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from core import registry
from core.registry import ToolContext, ToolTimeout, tool


@pytest.fixture
def temp_tool():
    """Register tools for one test, then remove them again.

    The registry is process-global, so leaking a fake tool would break
    test_registry.py's "no unexpected tools" check.
    """
    created: list[str] = []

    def make(name: str, fn, *, timeout=5.0, silent=False):
        tool(name=name, description=f"test tool {name}",
             parameters={"type": "OBJECT", "properties": {}},
             timeout=timeout, silent=silent)(fn)
        created.append(name)
        return name

    yield make

    for name in created:
        registry.unregister(name)


@pytest.fixture
def ctx():
    return ToolContext()


# ── happy path ────────────────────────────────────────────────────────────────

async def test_sync_tool_returns_its_string(temp_tool, ctx):
    name = temp_tool("t_sync", lambda p, c: "all good")
    assert await registry.run(name, {}, ctx) == "all good"


async def test_async_tool_is_awaited(temp_tool, ctx):
    async def fn(p, c):
        await asyncio.sleep(0)
        return "from a coroutine"

    name = temp_tool("t_async", fn)
    assert await registry.run(name, {}, ctx) == "from a coroutine"


async def test_params_reach_the_tool(temp_tool, ctx):
    name = temp_tool("t_params", lambda p, c: f"got {p['city']}")
    assert await registry.run(name, {"city": "Istanbul"}, ctx) == "got Istanbul"


async def test_none_becomes_a_speakable_default(temp_tool, ctx):
    """A tool returning None must not send `null` back to the model."""
    name = temp_tool("t_none", lambda p, c: None)
    assert await registry.run(name, {}, ctx) == "Done."


async def test_non_string_is_stringified(temp_tool, ctx):
    name = temp_tool("t_dict", lambda p, c: {"cpu": 12})
    assert await registry.run(name, {}, ctx) == "{'cpu': 12}"


# ── the reason this module exists ─────────────────────────────────────────────

async def test_hung_sync_tool_times_out(temp_tool, ctx):
    name = temp_tool("t_hang", lambda p, c: time.sleep(30), timeout=0.2)

    started = time.monotonic()
    with pytest.raises(ToolTimeout) as excinfo:
        await registry.run(name, {}, ctx)
    elapsed = time.monotonic() - started

    assert elapsed < 5, "wait_for did not cut the call short"
    # The message goes straight to the model to be spoken, so it has to read
    # like a sentence, not like a stack trace.
    assert "t_hang" in str(excinfo.value)
    assert "longer than" in str(excinfo.value)


async def test_hung_async_tool_times_out(temp_tool, ctx):
    async def fn(p, c):
        await asyncio.sleep(30)

    name = temp_tool("t_hang_async", fn, timeout=0.2)
    with pytest.raises(ToolTimeout):
        await registry.run(name, {}, ctx)


async def test_a_hung_tool_does_not_block_the_event_loop(temp_tool, ctx):
    """Audio playback shares this loop — a stuck tool must not stall it."""
    # The heartbeat needs ~0.2 s; the timeout is set well beyond that so the
    # test measures loop responsiveness rather than scheduler jitter.
    hung = temp_tool("t_block", lambda p, c: time.sleep(5), timeout=1.0)

    ticks = 0

    async def heartbeat():
        nonlocal ticks
        for _ in range(10):
            await asyncio.sleep(0.02)
            ticks += 1

    with pytest.raises(ToolTimeout):
        await asyncio.gather(registry.run(hung, {}, ctx), heartbeat())

    assert ticks == 10, "the event loop stopped running while a tool hung"


async def test_tool_exception_propagates_to_the_caller(temp_tool, ctx):
    def fn(p, c):
        raise ValueError("disk on fire")

    name = temp_tool("t_boom", fn)
    with pytest.raises(ValueError, match="disk on fire"):
        await registry.run(name, {}, ctx)


async def test_unknown_tool_raises_keyerror(ctx):
    with pytest.raises(KeyError):
        await registry.run("no_such_tool", {}, ctx)


# ── context plumbing ──────────────────────────────────────────────────────────

async def test_ctx_reaches_the_tool(temp_tool):
    spoken: list[str] = []
    ctx = ToolContext(speak=spoken.append, current_file="C:/tmp/a.pdf")

    def fn(p, c):
        c.speak("working on it")
        return c.current_file

    name = temp_tool("t_ctx", fn)
    assert await registry.run(name, {}, ctx) == "C:/tmp/a.pdf"
    assert spoken == ["working on it"]


def test_log_line_survives_a_missing_ui():
    """Tools call ctx.log_line() unconditionally; headless runs have no UI."""
    ToolContext(ui=None).log_line("should not raise")


def test_log_line_survives_a_broken_ui():
    class Broken:
        def write_log(self, _):
            raise RuntimeError("Qt is gone")

    ToolContext(ui=Broken()).log_line("should not raise")


# ── real registrations ────────────────────────────────────────────────────────

def test_save_memory_is_the_only_silent_tool():
    """Silent responses skip being spoken; anything else would go unanswered."""
    import actions.tools  # noqa: F401

    silent = [t.name for t in registry.all_tools() if t.silent]
    assert silent == ["save_memory"]
