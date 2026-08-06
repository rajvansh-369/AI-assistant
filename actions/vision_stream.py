"""Continuous vision — JARVIS sees, instead of calling a tool to look.

The `screen_process` tool works, but it is a *request*: the model has to decide
to look, capture, stall for a turn while the image is injected, then answer.
Five booleans coordinate that handshake across two turns, and a 4-second
cooldown exists purely because JARVIS hears his own "looking at your screen now"
filler through the speakers and tries to look again.

Streaming replaces the decision with a fact.  Frames go up the same websocket as
the microphone, so by the time you finish asking "what's this error", the model
already has the screen in context.  No tool call, no stall, no cooldown.

Gemini caps video input at 1 FPS, and every frame costs tokens whether or not it
was worth sending, so this module is mostly about *not* sending:

* **Activity gate** — frames only flow while the conversation is warm.  Go quiet
  for `IDLE_AFTER` seconds and the stream goes dormant; the next thing you say
  wakes it.  Google's own guidance is to send video only during audio activity.
* **Change gate** — a frame that looks like the one before it is dropped.  A
  static screen therefore costs nothing at all, which is the common case while
  you are reading rather than doing.

Capture handles are held open across frames.  The one-shot `_capture_camera` in
screen_processor.py opens the device, burns ten warm-up frames and releases it
on every call — around a second each time, which is unusable at 1 FPS.
"""
from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from core.log import get_logger

log = get_logger("vision_stream")

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

try:
    import mss
    _MSS = True
except ImportError:
    _MSS = False

try:
    import PIL.Image
    _PIL = True
except ImportError:
    _PIL = False


#: Google recommends 768x768 for streamed frames; larger costs tokens for
#: detail the model does not use at this frame rate.
FRAME_MAX = 768
JPEG_Q    = 70

#: The API's own ceiling. Sending faster is rejected, not merely wasteful.
MIN_INTERVAL     = 1.0
DEFAULT_INTERVAL = 1.0

#: Mean absolute difference (0-255) between two 32x32 grayscale thumbnails,
#: below which the screen counts as unchanged.  A blinking cursor and a clock
#: ticking over sit well under 2.0; scrolling text sits well above it.
CHANGE_THRESHOLD = 2.0

#: Seconds of user silence after which the stream goes dormant.  Long enough to
#: cover a pause mid-thought, short enough that walking away stops the spend.
IDLE_AFTER = 60.0

#: Even on a static screen, re-send occasionally so the frame does not age out
#: of the context window behind newer audio.
KEEPALIVE = 30.0

SOURCES = ("screen", "camera")


@dataclass
class Frame:
    """A frame that is worth sending, plus what the change gate needs to
    remember about it *once it has actually been sent*."""
    jpeg:  bytes
    thumb: object = field(repr=False)


@dataclass
class VisionStream:
    """Streaming state plus the capture handles it keeps open.

    One instance, owned by main.py, read by the `vision_stream` tool.  Every
    mutating method takes `_lock` because the tool runs on the tool executor
    while `_send_video` captures on an asyncio worker thread.
    """

    active:   bool  = False
    source:   str   = "screen"
    interval: float = DEFAULT_INTERVAL

    frames_sent:    int = 0
    frames_skipped: int = 0
    bytes_sent:     int = 0
    started_at:     float = 0.0

    _lock:       threading.RLock = field(default_factory=threading.RLock, repr=False)
    _tls:        threading.local = field(default_factory=threading.local, repr=False)
    _cam:        object | None   = field(default=None, repr=False)
    _last_thumb: object | None   = field(default=None, repr=False)
    _last_sent:  float           = field(default=0.0, repr=False)

    # ── control ───────────────────────────────────────────────────────────────

    def start(self, source: str = "screen", interval: float | None = None) -> str:
        source = (source or "screen").strip().lower()
        if source not in SOURCES:
            return f"Unknown vision source '{source}'. Use screen or camera."
        if source == "camera" and not _CV2:
            return "Camera streaming needs OpenCV. Run: pip install opencv-python"
        if source == "screen" and not _MSS:
            return "Screen streaming needs mss. Run: pip install mss"

        with self._lock:
            if self.source != source:
                self._release_camera()
                self._last_thumb = None
            self.source   = source
            self.interval = max(MIN_INTERVAL, float(interval or DEFAULT_INTERVAL))
            if not self.active:
                self.active         = True
                self.started_at     = time.monotonic()
                self.frames_sent    = 0
                self.frames_skipped = 0
                self.bytes_sent     = 0
                self._last_thumb    = None
                self._last_sent     = 0.0

        log.info(f"Streaming {source} at 1 frame / {self.interval:.0f}s")
        return (
            f"Continuous vision on — I can see your {source} from now on. "
            f"You do not need to ask me to look."
        )

    def stop(self) -> str:
        with self._lock:
            if not self.active:
                return "Continuous vision is already off."
            sent, skipped = self.frames_sent, self.frames_skipped
            self.active      = False
            self._last_thumb = None
            self._release_camera()

        log.info(f"Stopped after {sent} frames sent, {skipped} skipped")
        return f"Continuous vision off. I sent {sent} frames while it was on."

    def on_session_start(self) -> None:
        """Forget what the model has seen — a new session has seen nothing.

        The change gate compares against the last frame *sent*. After a
        reconnect the model's context is empty (or rewound to a resume point),
        so a static screen would otherwise be skipped forever and JARVIS would
        stream nothing while believing he could see.
        """
        with self._lock:
            self._last_thumb = None
            self._last_sent  = 0.0

    def status(self) -> str:
        with self._lock:
            if not self.active:
                return "Continuous vision is off. I can only see when you ask me to look."
            mins = (time.monotonic() - self.started_at) / 60.0
            return (
                f"Watching your {self.source}, one frame every {self.interval:.0f}s, "
                f"for {mins:.0f} minutes. {self.frames_sent} frames sent, "
                f"{self.frames_skipped} skipped as unchanged "
                f"({self.bytes_sent / 1024:.0f} KB total)."
            )

    # ── capture ───────────────────────────────────────────────────────────────

    def should_send(self, last_user_speech: float) -> bool:
        """False while the user is away — the activity gate.

        `last_user_speech` is a `time.monotonic()` stamp.
        """
        with self._lock:
            if not self.active:
                return False
        return (time.monotonic() - last_user_speech) <= IDLE_AFTER

    def capture(self) -> Frame | None:
        """One frame worth sending, or None when the view has not changed.

        Blocking — call it with `asyncio.to_thread`, never on the event loop.

        Does not advance the change gate. Call `commit()` once the frame is
        actually on the wire: a frame that failed to send has not been seen by
        the model, and treating it as seen would leave a static screen skipped
        indefinitely — JARVIS blind while believing he could see.
        """
        with self._lock:
            if not self.active:
                return None
            source = self.source

        raw = self._grab_camera() if source == "camera" else self._grab_screen()
        if raw is None:
            return None

        thumb = _thumbnail(raw)
        with self._lock:
            stale = (time.monotonic() - self._last_sent) >= KEEPALIVE
            if not stale and _unchanged(self._last_thumb, thumb):
                self.frames_skipped += 1
                return None

        jpeg = _encode(raw)
        if jpeg is None:
            return None
        return Frame(jpeg=jpeg, thumb=thumb)

    def commit(self, frame: Frame) -> None:
        """Record that `frame` reached the model."""
        with self._lock:
            self._last_thumb  = frame.thumb
            self._last_sent   = time.monotonic()
            self.frames_sent += 1
            self.bytes_sent  += len(frame.jpeg)

    # ── backends ──────────────────────────────────────────────────────────────

    def _grab_screen(self):
        """RGB ndarray of the primary display.

        mss instances are bound to the thread that created them, and captures
        run on whatever worker asyncio.to_thread happens to pick — hence the
        thread-local rather than one shared grabber.
        """
        sct = getattr(self._tls, "sct", None)
        if sct is None:
            sct = self._tls.sct = mss.mss()

        monitors = sct.monitors
        target   = monitors[1] if len(monitors) > 1 else monitors[0]
        shot     = sct.grab(target)
        arr      = np.asarray(shot)          # BGRA
        return arr[:, :, [2, 1, 0]]          # -> RGB

    def _grab_camera(self):
        with self._lock:
            cap = self._cam
            if cap is None or not cap.isOpened():
                from actions.screen_processor import _cv2_backend, _get_camera_index
                cap = cv2.VideoCapture(_get_camera_index(), _cv2_backend())
                if not cap.isOpened():
                    cap.release()
                    raise RuntimeError("Camera could not be opened for streaming.")
                self._cam = cap

            ok, frame = cap.read()

        if not ok or frame is None:
            return None
        return frame[:, :, ::-1]             # BGR -> RGB

    def _release_camera(self) -> None:
        cap, self._cam = self._cam, None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


def _thumbnail(rgb) -> np.ndarray:
    """32x32 grayscale, for the change comparison. Strided, so no resize cost."""
    h, w = rgb.shape[:2]
    ys = np.linspace(0, h - 1, 32).astype(np.intp)
    xs = np.linspace(0, w - 1, 32).astype(np.intp)
    small = rgb[np.ix_(ys, xs)]
    return small.mean(axis=2).astype(np.int16)


def _unchanged(prev, current) -> bool:
    if prev is None:
        return False
    return float(np.abs(prev - current).mean()) < CHANGE_THRESHOLD


def _encode(rgb) -> bytes | None:
    if not _PIL:
        log.warning("Pillow is not installed — cannot encode frames.")
        return None
    img = PIL.Image.fromarray(rgb.astype(np.uint8), "RGB")
    img.thumbnail((FRAME_MAX, FRAME_MAX), PIL.Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_Q, optimize=False)
    return buf.getvalue()


#: The single instance. main.py drives it; the vision_stream tool reads it.
STREAM = VisionStream()
