"""JarvisLive._send_video — the loop that puts frames on the wire.

Covers the things that would be expensive or silent to get wrong: sending when
the stream is off, taking the whole TaskGroup down over one dropped frame, or
spinning at 1 Hz forever against a socket that is never coming back.

The session is a stub; nothing here opens a websocket.
"""
from __future__ import annotations

import asyncio
import time

import pytest
from google.genai import types

import main
from actions import vision_stream as vs
from actions.vision_stream import STREAM as VISION


class FakeSession:
    def __init__(self, fail: bool = False):
        self.videos: list[bytes] = []
        self.fail = fail

    async def send_realtime_input(self, **kw):
        if self.fail:
            raise ConnectionError("socket closed")
        if "video" in kw:
            self.videos.append(kw["video"].data)


class StubUI:
    def __init__(self):
        self.muted = False
        self.current_file = None
        self.logs: list[str] = []
        self.on_text_command = None
        self.on_remote_clicked = None
        self.on_interrupt = None

    def set_state(self, s): pass
    def write_log(self, t): self.logs.append(t)
    def start_camera_stream(self): pass
    def stop_camera_stream(self): pass


@pytest.fixture
def jarvis(monkeypatch):
    j = main.JarvisLive(StubUI())
    j._send_lock = asyncio.Lock()
    j.session = FakeSession()
    j._last_user_speech = time.monotonic()

    # The real stream is a module-level singleton shared with the tool; reset it
    # so a test never inherits another test's state.
    VISION.stop()
    monkeypatch.setattr(VISION, "interval", 0.01)   # bypasses start()'s 1 FPS clamp
    yield j
    VISION.stop()


async def pump(jarvis, seconds: float = 0.15):
    """Run the send loop briefly, the way the TaskGroup would."""
    task = asyncio.create_task(jarvis._send_video())
    await asyncio.sleep(seconds)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def arm(monkeypatch, source="screen", value=100):
    """Turn the stream on with a stubbed grab."""
    import numpy as np
    monkeypatch.setattr(
        VISION, "_grab_screen",
        lambda: np.full((64, 64, 3), value, dtype=np.uint8),
    )
    VISION.start(source)
    monkeypatch.setattr(VISION, "interval", 0.01)


async def test_nothing_is_sent_while_the_stream_is_off(jarvis):
    await pump(jarvis)
    assert jarvis.session.videos == []


async def test_frames_reach_the_session(jarvis, monkeypatch):
    arm(monkeypatch)
    await pump(jarvis)
    assert jarvis.session.videos, "no frame was sent"
    assert jarvis.session.videos[0][:2] == b"\xff\xd8"   # JPEG


async def test_a_static_screen_sends_exactly_one_frame(jarvis, monkeypatch):
    """The change gate is what makes leaving this on affordable."""
    arm(monkeypatch)
    await pump(jarvis, seconds=0.3)
    assert len(jarvis.session.videos) == 1


async def test_a_silent_user_stops_the_frames(jarvis, monkeypatch):
    arm(monkeypatch)
    jarvis._last_user_speech = time.monotonic() - (vs.IDLE_AFTER + 5)
    await pump(jarvis)
    assert jarvis.session.videos == []


async def test_frames_are_sent_as_video_not_media(jarvis, monkeypatch):
    """`media=` is the audio path; the Live API wants `video=` for frames."""
    seen = {}

    class Recorder(FakeSession):
        async def send_realtime_input(self, **kw):
            seen.update(kw)

    jarvis.session = Recorder()
    arm(monkeypatch)
    await pump(jarvis)

    assert "video" in seen and "media" not in seen
    assert isinstance(seen["video"], types.Blob)
    assert seen["video"].mime_type == "image/jpeg"


async def test_a_send_failure_does_not_raise(jarvis, monkeypatch):
    """Raising here would tear down the TaskGroup and drop the whole session."""
    jarvis.session = FakeSession(fail=True)
    arm(monkeypatch)
    await pump(jarvis)          # must not raise


async def test_repeated_failures_give_up(jarvis, monkeypatch):
    jarvis.session = FakeSession(fail=True)
    arm(monkeypatch)
    await pump(jarvis, seconds=0.4)

    assert VISION.active is False, "stream kept retrying a dead socket"
    assert any("Vision stream stopped" in line for line in jarvis.ui.logs)


async def test_a_capture_failure_stops_the_stream(jarvis, monkeypatch):
    def boom():
        raise RuntimeError("camera unplugged")

    monkeypatch.setattr(VISION, "_grab_screen", boom)
    VISION.start("screen")
    monkeypatch.setattr(VISION, "interval", 0.01)

    await pump(jarvis)
    assert VISION.active is False
    assert any("camera unplugged" in line for line in jarvis.ui.logs)


# ── telling the model it can see ──────────────────────────────────────────────
#
# Found live: with the feed running, the model still called screen_process for a
# view it was already being sent — because nothing in the session said the feed
# was on. That fact only ever existed in the tool result that started it, so any
# reconnect mid-stream lost it and JARVIS went back to announcing "let me look".

def test_prompt_says_nothing_when_the_feed_is_off(jarvis):
    assert "[LIVE VISION" not in jarvis._build_config().system_instruction


def test_prompt_announces_a_running_feed(jarvis, monkeypatch):
    arm(monkeypatch)
    prompt = jarvis._build_config().system_instruction

    assert "[LIVE VISION — ON]" in prompt
    assert "screen" in prompt


def test_prompt_names_the_actual_source(jarvis, monkeypatch):
    import numpy as np
    monkeypatch.setattr(vs, "_CV2", True)
    monkeypatch.setattr(
        VISION, "_grab_camera",
        lambda: np.full((64, 64, 3), 100, dtype=np.uint8),
    )
    VISION.start("camera")

    prompt = jarvis._build_config().system_instruction
    assert "camera" in prompt


def test_prompt_forbids_the_redundant_tool_call(jarvis, monkeypatch):
    arm(monkeypatch)
    prompt = jarvis._build_config().system_instruction
    assert "Never call screen_process" in prompt


# ── interaction with the one-shot tool ────────────────────────────────────────

async def test_screen_process_short_circuits_while_streaming(jarvis, monkeypatch):
    """Capturing again would send the model a second copy of what it can see."""
    from actions.session_tools import screen_process_tool
    from core.registry import ToolContext

    arm(monkeypatch)
    ctx = ToolContext(ui=jarvis.ui, jarvis=jarvis)
    result = await screen_process_tool({"angle": "screen", "text": "what is this"}, ctx)

    assert "[VISION_LIVE]" in result
    assert jarvis._pending_vision is None, "captured despite the live feed"
    assert jarvis._vision_busy is False


async def test_screen_process_still_works_for_the_other_source(jarvis, monkeypatch):
    """Streaming the screen must not block a one-off look at the camera."""
    from actions import session_tools
    from core.registry import ToolContext

    arm(monkeypatch)
    monkeypatch.setattr(session_tools, "_capture_camera", lambda: (b"\xff\xd8jpg", "image/jpeg"))

    ctx = ToolContext(ui=jarvis.ui, jarvis=jarvis)
    result = await session_tools.screen_process_tool(
        {"angle": "camera", "text": "who is this"}, ctx
    )

    assert "[VISION_ACTIVE]" in result
    assert jarvis._pending_vision is not None
