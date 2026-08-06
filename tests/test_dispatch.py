"""JarvisLive._execute_tool — the seam between the live session and the registry.

Gemini stalls forever if a function call goes unanswered, so the one invariant
here is: *every* path returns a FunctionResponse.  Success, unknown tool, tool
raised, tool hung — all four.

Uses throwaway tools and a stub UI, so nothing opens an app or touches Qt.
"""
from __future__ import annotations

import time
import types as pytypes

import pytest

import main
from core import registry
from core.registry import tool


class StubUI:
    """The surface _execute_tool actually touches."""

    def __init__(self):
        self.muted        = False
        self.current_file = None
        self.states: list[str]   = []
        self.logs:   list[str]   = []
        # JarvisLive.__init__ assigns these callbacks
        self.on_text_command = None
        self.on_remote_clicked = None
        self.on_interrupt = None

    def set_state(self, s):  self.states.append(s)
    def write_log(self, t):  self.logs.append(t)
    def show_content(self, label, body): pass


class FakeCall:
    """Stands in for google.genai's FunctionCall."""

    def __init__(self, name, args=None, id="call-1"):
        self.name = name
        self.args = args or {}
        self.id   = id


@pytest.fixture
def jarvis():
    ui = StubUI()
    j  = main.JarvisLive(ui)
    return j


@pytest.fixture
def temp_tool():
    created: list[str] = []

    def make(name, fn, *, timeout=5.0, silent=False):
        tool(name=name, description=f"test {name}",
             parameters={"type": "OBJECT", "properties": {}},
             timeout=timeout, silent=silent)(fn)
        created.append(name)
        return name

    yield make
    for name in created:
        registry.unregister(name)


async def test_successful_call_returns_the_result(jarvis, temp_tool):
    name = temp_tool("d_ok", lambda p, c: "23 degrees and clear")
    fr = await jarvis._execute_tool(FakeCall(name))

    assert fr.name == name
    assert fr.id == "call-1"
    assert fr.response == {"result": "23 degrees and clear"}


async def test_unknown_tool_still_answers(jarvis):
    fr = await jarvis._execute_tool(FakeCall("tool_that_never_existed"))
    assert "Unknown tool" in fr.response["result"]


async def test_raising_tool_still_answers(jarvis, temp_tool):
    def boom(p, c):
        raise RuntimeError("the printer is on fire")

    name = temp_tool("d_boom", boom)
    fr = await jarvis._execute_tool(FakeCall(name))

    assert "failed" in fr.response["result"]
    assert "the printer is on fire" in fr.response["result"]


async def test_hung_tool_still_answers(jarvis, temp_tool):
    """The whole point of the registry: silence is not an acceptable outcome."""
    name = temp_tool("d_hang", lambda p, c: time.sleep(30), timeout=0.2)

    started = time.monotonic()
    fr = await jarvis._execute_tool(FakeCall(name))
    elapsed = time.monotonic() - started

    assert elapsed < 5
    assert "longer than" in fr.response["result"]
    assert any("timed out" in line for line in jarvis.ui.logs)


async def test_silent_tool_is_marked_silent(jarvis, temp_tool):
    name = temp_tool("d_quiet", lambda p, c: "ok", silent=True)
    fr = await jarvis._execute_tool(FakeCall(name))
    assert fr.response.get("silent") is True


async def test_loud_tool_is_not_marked_silent(jarvis, temp_tool):
    name = temp_tool("d_loud", lambda p, c: "ok")
    fr = await jarvis._execute_tool(FakeCall(name))
    assert "silent" not in fr.response


async def test_args_are_forwarded(jarvis, temp_tool):
    name = temp_tool("d_args", lambda p, c: f"city={p.get('city')}")
    fr = await jarvis._execute_tool(FakeCall(name, {"city": "Ankara"}))
    assert fr.response["result"] == "city=Ankara"


async def test_ui_returns_to_listening(jarvis, temp_tool):
    name = temp_tool("d_state", lambda p, c: "ok")
    await jarvis._execute_tool(FakeCall(name))
    assert jarvis.ui.states == ["THINKING", "LISTENING"]


async def test_muted_ui_stays_muted(jarvis, temp_tool):
    jarvis.ui.muted = True
    name = temp_tool("d_muted", lambda p, c: "ok")
    await jarvis._execute_tool(FakeCall(name))
    assert "LISTENING" not in jarvis.ui.states


async def test_dropped_file_fills_in_an_empty_file_path(jarvis, temp_tool):
    jarvis.ui.current_file = r"C:\tmp\report.pdf"
    name = temp_tool("d_ctx", lambda p, c: c.current_file or "none")
    fr = await jarvis._execute_tool(FakeCall(name))
    assert fr.response["result"] == r"C:\tmp\report.pdf"


def test_declarations_are_what_the_session_sends():
    """_build_config wires registry.declarations() into LiveConnectConfig."""
    decls = registry.declarations()
    assert decls, "no tools registered"
    assert {d["name"] for d in decls} == set(registry.names())
    for d in decls:
        # Anything else in the dict is rejected by the Live API as an unknown
        # field on FunctionDeclaration.
        assert set(d) == {"name", "description", "parameters"}
