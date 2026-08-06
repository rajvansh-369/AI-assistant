"""The two gates that decide what continuous vision costs.

A frame a second, forever, is the most expensive thing this app can do. Almost
all of the value in `actions/vision_stream.py` is in the frames it *declines* to
send, so that is what these cover:

* the activity gate — nothing flows while the user is away
* the change gate  — a screen that has not changed costs nothing

Plus the failure modes that would silently blind JARVIS: a frame that never
reached the model being treated as seen, a reconnect leaving the change gate
primed against a frame the new session never got, and a source switch comparing
a webcam frame against the last screenshot.

Capture backends are stubbed, so no test opens a camera or grabs a display.
"""
from __future__ import annotations

import io
import time

import numpy as np
import pytest

from actions import vision_stream as vs
from actions.vision_stream import VisionStream


def flat(value: int = 100, size: int = 64) -> np.ndarray:
    """A flat RGB image of a single brightness."""
    return np.full((size, size, 3), value, dtype=np.uint8)


def noisy(seed: int, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)


def send(stream) -> vs.Frame | None:
    """capture + commit — what _send_video does when a send succeeds."""
    frame = stream.capture()
    if frame is not None:
        stream.commit(frame)
    return frame


@pytest.fixture
def stream(monkeypatch):
    """A stream whose screen grab returns whatever `stream.next` is set to."""
    s = VisionStream()
    s.next = flat(100)                       # type: ignore[attr-defined]
    monkeypatch.setattr(s, "_grab_screen", lambda: s.next)
    s.start("screen")
    return s


# ── change gate ───────────────────────────────────────────────────────────────

def test_first_frame_is_always_sent(stream):
    assert send(stream) is not None
    assert stream.frames_sent == 1


def test_identical_frame_is_skipped(stream):
    send(stream)
    assert stream.capture() is None
    assert stream.frames_sent == 1
    assert stream.frames_skipped == 1


def test_a_changed_frame_is_sent(stream):
    send(stream)
    stream.next = flat(200)
    assert send(stream) is not None
    assert stream.frames_sent == 2


def test_imperceptible_change_is_skipped(stream):
    """A ticking clock or blinking cursor must not count as activity."""
    send(stream)
    stream.next = flat(101)                  # 1/255 brighter
    assert stream.capture() is None


def test_scrolling_content_is_sent(stream):
    stream.next = noisy(1)
    send(stream)
    stream.next = noisy(2)
    assert stream.capture() is not None


def test_an_uncommitted_frame_is_offered_again(stream):
    """A frame that failed to send has not been seen — the gate must not move.

    Otherwise one dropped send leaves a static screen skipped forever, and
    JARVIS is blind while believing he can see.
    """
    first = stream.capture()                 # no commit: pretend the send failed
    assert first is not None
    assert stream.capture() is not None, "frame was treated as delivered"
    assert stream.frames_sent == 0


def test_keepalive_resends_a_static_screen(stream, monkeypatch):
    """A frame that never repeats would age out of context behind new audio."""
    send(stream)
    assert stream.capture() is None

    real = time.monotonic
    monkeypatch.setattr(vs.time, "monotonic", lambda: real() + vs.KEEPALIVE + 1)
    assert stream.capture() is not None, "static screen never refreshed"


def test_output_is_a_jpeg(stream):
    jpeg = stream.capture().jpeg
    assert jpeg[:2] == b"\xff\xd8"           # SOI
    assert jpeg[-2:] == b"\xff\xd9"          # EOI


def test_large_frames_are_downscaled(monkeypatch):
    s = VisionStream()
    monkeypatch.setattr(s, "_grab_screen", lambda: noisy(3, size=2000))
    s.start("screen")

    import PIL.Image
    img = PIL.Image.open(io.BytesIO(s.capture().jpeg))
    assert max(img.size) <= vs.FRAME_MAX


def test_byte_counter_tracks_committed_frames_only(stream):
    stream.capture()                         # not committed
    assert stream.bytes_sent == 0
    send(stream)
    assert stream.bytes_sent > 0


# ── activity gate ─────────────────────────────────────────────────────────────

def test_recent_speech_lets_frames_through(stream):
    assert stream.should_send(time.monotonic()) is True


def test_a_silent_user_stops_the_stream(stream):
    long_ago = time.monotonic() - (vs.IDLE_AFTER + 5)
    assert stream.should_send(long_ago) is False


def test_speaking_again_wakes_it(stream):
    assert stream.should_send(time.monotonic() - (vs.IDLE_AFTER + 5)) is False
    assert stream.should_send(time.monotonic()) is True


def test_nothing_flows_while_inactive(stream):
    stream.stop()
    assert stream.should_send(time.monotonic()) is False
    assert stream.capture() is None


# ── lifecycle ─────────────────────────────────────────────────────────────────

def test_reconnect_forces_the_next_frame(stream):
    """The new session has seen nothing; a static screen must not stay skipped."""
    send(stream)
    assert stream.capture() is None

    stream.on_session_start()
    assert stream.capture() is not None, "model would have been blind after reconnect"


def test_switching_source_clears_the_comparison(stream, monkeypatch):
    """A webcam frame must not be diffed against the last screenshot."""
    send(stream)
    monkeypatch.setattr(stream, "_grab_camera", lambda: stream.next)
    monkeypatch.setattr(vs, "_CV2", True)

    stream.start("camera")
    assert stream._last_thumb is None
    assert stream.capture() is not None


def test_restarting_does_not_reset_the_counters(stream):
    send(stream)
    stream.start("screen")                   # same source, already running
    assert stream.frames_sent == 1


def test_stop_then_start_resets_the_counters(stream):
    send(stream)
    stream.stop()
    stream.start("screen")
    assert stream.frames_sent == 0


def test_interval_is_clamped_to_the_api_ceiling(stream):
    """Gemini rejects video faster than 1 FPS."""
    stream.start("screen", interval=0.1)
    assert stream.interval >= vs.MIN_INTERVAL


def test_a_slower_interval_is_honoured(stream):
    stream.start("screen", interval=5)
    assert stream.interval == 5


def test_unknown_source_is_refused(stream):
    assert "Unknown vision source" in stream.start("telepathy")


def test_status_reads_as_a_sentence(stream):
    assert "off" in VisionStream().status().lower()
    send(stream)
    assert "screen" in stream.status()


def test_stop_is_idempotent(stream):
    stream.stop()
    assert "already off" in stream.stop()


# ── budget ────────────────────────────────────────────────────────────────────

def test_free_mode_blocks_video_streaming():
    """Unlike a search, there is no cheap fallback to degrade to."""
    from core import budget
    assert "video_stream" in budget.FREE_BLOCKED
