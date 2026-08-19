"""End-of-session summarisation — the write path and its failure path.

The failure path is the one with history: `_save_session_summary` used to bind
`log = self._session_log`, shadowing the module logger with a list.  A failed
Gemini call then raised AttributeError *inside the except block*, which killed
whatever was awaiting the summary — including the shutdown task, leaving JARVIS
running after he had said "Shutting down".
"""
from __future__ import annotations

import pytest

import main
from memory import memory_manager as mm


class StubUI:
    muted = False
    current_file = None

    def __init__(self):
        self.logs: list[str] = []
        self.on_text_command = None
        self.on_remote_clicked = None
        self.on_interrupt = None

    def set_state(self, s): pass
    def write_log(self, t): self.logs.append(t)


@pytest.fixture
def jarvis(monkeypatch):
    monkeypatch.setattr(main, "_get_api_key", lambda: "test-key")
    j = main.JarvisLive(StubUI())
    j._session_log = [
        "User: how is the drone project going",
        "JARVIS: the flight controller firmware compiled cleanly",
        "User: great, flash it tomorrow",
        "JARVIS: noted, sir",
    ]
    return j


class _StubClient:
    """Stands in for google.genai.Client."""

    class _Models:
        def generate_content(self, model=None, contents=None):
            class R:
                text = "Worked on the drone flight controller."
            return R()

    def __init__(self, api_key=None, **kw):
        self.models = self._Models()


async def test_summary_is_saved(jarvis, monkeypatch):
    monkeypatch.setattr("google.genai.Client", _StubClient)

    await jarvis._save_session_summary()

    entry = mm.pop_last_session()
    assert entry is not None
    assert "drone" in entry["summary"]
    assert jarvis._session_log == [], "log must reset for the next session"


async def test_short_sessions_are_not_saved(jarvis):
    jarvis._session_log = ["User: hello"]
    await jarvis._save_session_summary()
    assert mm.pop_last_session() is None


async def test_a_failing_api_call_does_not_raise(jarvis, monkeypatch):
    """The shadowed-logger regression: this used to raise AttributeError."""
    def boom(*a, **kw):
        raise RuntimeError("503 model overloaded")

    monkeypatch.setattr("google.genai.Client", boom)

    await jarvis._save_session_summary()          # must not raise

    assert mm.pop_last_session() is None
    assert jarvis._session_log == []
