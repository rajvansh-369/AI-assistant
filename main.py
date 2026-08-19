import platform as _platform
import subprocess as _subprocess

# ── Nuclear: force CREATE_NO_WINDOW on EVERY subprocess call on Windows ───────
# This patches Popen itself, so no per-file flag is needed anywhere.
if _platform.system() == "Windows":
    _OrigPopen = _subprocess.Popen

    class _Popen(_OrigPopen):
        def __init__(self, args, **kw):
            kw["creationflags"] = kw.get("creationflags", 0) | _subprocess.CREATE_NO_WINDOW
            kw.pop("startupinfo", None)   # drop any stale/shared STARTUPINFO
            super().__init__(args, **kw)

    _subprocess.Popen = _Popen
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import re
import threading
import time
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
from google import genai
from google.genai import types
from ui import JarvisUI
from memory.memory_manager import (
    load_memory, format_memory_for_prompt, last_session_topic,
    save_session_summary, pop_last_session,
)

# Every tool lives in the registry now.  Importing actions.tools runs the @tool
# decorators that fill it; the named imports below are only the pieces main.py
# calls outside the tool path — background loops and the startup briefing.
import actions.tools  # noqa: F401  — registers the tool table
from actions.location import resolve as resolve_location
from actions.vision_stream import STREAM as VISION_STREAM
from actions.background_monitor import check_all as monitor_check_all
from actions.background_monitor import list_monitors
from actions.proactive         import ProactiveEngine
from actions.system_monitor    import SystemMonitor
from actions.web_search        import _news as _fetch_news_sync
from memory.config_manager     import get_brief_enabled

from core import budget, registry
from core.barge_in import BargeInDetector
from core.log import add_ui_sink, get_logger, setup_logging
from core.paths import BASE_DIR, CONFIG_PATH as API_CONFIG_PATH, LOG_PATH, PROMPT_PATH
from core.registry import ToolContext, ToolTimeout
from core.settings import LIVE_MODEL, get_settings
from core.settings import get_api_key as _get_api_key

log = get_logger("jarvis")

CHANNELS            = 1
SEND_SAMPLE_RATE    = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE          = 1024

def _load_system_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except Exception:
        return (
            "You are JARVIS, Tony Stark's AI assistant. "
            "Be concise, direct, and always use the provided tools to complete tasks. "
            "Never simulate or guess results — always call the appropriate tool."
        )

#: Prebuilt Live API voice. "Charon" is documented as *informative* — accurate
#: for a briefing, flat for a conversation. Override with `voice` in
#: config/api_keys.json; the full set of 30 is auditionable in AI Studio.
DEFAULT_VOICE = "Charon"

WORLD_NEWS_QUERY = "top world news today"


def _fetch_news_ladder(queries: list[str]) -> tuple[str, str]:
    """First place with real headlines, as (text, query_used).

    `queries` runs narrowest to widest — city, region, country. Somewhere small
    has no news of its own and there is no way to tell that from a place name,
    so the ladder widens on an empty result rather than guessing.

    Each rung is strict: no text-search fallback. Without that, DDG answers a
    query about a town with the town's air-quality page, the ladder counts it as
    success and never widens — the user gets websites read out as headlines.
    World news is the last resort and is *not* strict, because the briefing
    coming back empty is worse than it being slightly off-topic.
    """
    for query in queries:
        try:
            text = _fetch_news_sync(query, strict=True)
        except Exception as e:
            log.warning(f"News fetch failed for {query!r}: {e}")
            continue
        if text and len(text) > 60 and not text.startswith("No news"):
            return text, query
        log.info(f"No local news for {query!r} — widening")

    try:
        return _fetch_news_sync(WORLD_NEWS_QUERY), WORLD_NEWS_QUERY
    except Exception as e:
        log.warning(f"World news fallback failed: {e}")
        return "", WORLD_NEWS_QUERY


_CTRL_RE = re.compile(r"<ctrl\d+>", re.IGNORECASE)

def _clean_transcript(text: str) -> str:
    text = _CTRL_RE.sub("", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", "", text)
    return text.strip()

def _flatten_error(exc: BaseException) -> str:
    """All messages in an exception tree, joined.

    TaskGroup wraps failures in a BaseExceptionGroup whose str() is only
    "unhandled errors in a TaskGroup (1 sub-exception)" — the actual cause
    (e.g. the 1007 close reason) lives in .exceptions, so classifying a
    failure by str(e) alone silently misses every TaskGroup error.
    """
    parts = [f"{type(exc).__name__}: {exc}"]
    for sub in getattr(exc, "exceptions", ()) or ():
        parts.append(_flatten_error(sub))
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        parts.append(f"{type(cause).__name__}: {cause}")
    return " | ".join(parts)


class JarvisLive:

    def __init__(self, ui: JarvisUI):
        self.ui             = ui
        self._asst_name     = "JARVIS"   # updated each session from config
        self.session              = None
        self.audio_in_queue       = None
        self.out_queue            = None
        self._loop                = None
        self._is_speaking         = False
        self._speaking_lock       = threading.Lock()
        # The SDK writes straight to the websocket with no lock of its own, and
        # mic audio, video frames, injected text and tool responses are all
        # produced by different coroutines. Serialise them.
        self._send_lock: asyncio.Lock | None = None
        self._idle_ticks          = 0       # consecutive empty-queue polls in _play_audio
        self._last_inject         = "none"  # last _inject_text source, for 1007 diagnosis
        self._resume_handle       = None    # server-issued session resumption token
        self._phone_active        = False   # True while phone mic is streaming; pauses PC mic
        self._pending_vision       = None    # (img_bytes, mime_type, question, angle) to inject after tool response
        self._vision_cam_active    = False   # True if camera was opened for vision → auto-close after response
        self._vision_close_pending = False   # True after vision injected; next turn_complete closes camera
        self._vision_last_time     = 0.0     # monotonic time of last screen_process call (cooldown guard)
        self._vision_busy          = False   # True while a vision capture/inject cycle is in flight
        self._interrupted          = False   # True while draining audio after user interrupt
        self._barge: BargeInDetector | None = None   # built in _listen_audio
        # Affective dialog is model- and version-dependent. Assume it works,
        # and drop it permanently for this process if a connect is rejected
        # because of it — a flatter voice beats an assistant that will not start.
        self._affective_supported  = True
        self.ui.on_text_command   = self._on_text_command
        self.ui.on_remote_clicked = self._make_remote_key
        # Called with no arguments from the HUD button; the only path that stops
        # him while barge-in is off.
        self.ui.on_interrupt      = lambda: self.interrupt(source="button")
        self._turn_done_event: asyncio.Event | None = None
        self._dashboard     = None
        self._briefing_sent    = False          # morning briefing fires once per process
        self._sys_monitor      = SystemMonitor()  # persistent cooldown state
        self._proactive        = ProactiveEngine()
        self._last_user_speech = time.monotonic()  # updated on every user utterance
        self._session_log: list[str] = []          # conversation turns for end-of-session summary

    def _log_turn(self, line: str) -> None:
        """Append to the session transcript, bounded.

        The process is designed to stay up for days; the summary uses the last
        40 turns and proactive mode the last 8, so keeping everything ever said
        is only a slow leak.
        """
        self._session_log.append(line)
        if len(self._session_log) > 120:
            del self._session_log[:-80]

    def _make_remote_key(self):
        """Called from Qt main thread when user presses Remote Control."""
        if self._dashboard is None:
            self.ui.write_log(
                "SYS: Dashboard unavailable. "
                "Run: pip install fastapi \"uvicorn[standard]\" cryptography"
            )
            return None
        key    = self._dashboard.new_key()
        url    = self._dashboard.get_url()
        manual = self._dashboard.get_manual_url()
        return url, key, f"{url}/auto-login?key={key}", manual

    async def _inject_text(self, text: str, source: str = "?") -> None:
        """Push a text prompt into the live session mid-conversation.

        Native-audio models treat send_client_content as *initial history
        seeding* only; using it once the conversation is under way makes the
        server reject the next mic frame and close the socket with
        1007 "The audio content type (CONTENT_TYPE_AUDIO) is not supported for
        this model configuration".  send_realtime_input is the supported path
        for anything sent after the session starts streaming.
        See https://ai.google.dev/gemini-api/docs/live-api/capabilities
        """
        if not self.session:
            return
        self._last_inject = source      # surfaced if the socket dies right after
        async with self._send_lock:
            await self.session.send_realtime_input(text=text)

    def _inject_text_threadsafe(self, text: str, source: str = "?") -> None:
        if not self._loop or not self.session:
            return
        asyncio.run_coroutine_threadsafe(self._inject_text(text, source), self._loop)

    def _on_text_command(self, text: str):
        self._inject_text_threadsafe(text, "text_command")

    def set_speaking(self, value: bool):
        with self._speaking_lock:
            changed = self._is_speaking != value
            self._is_speaking = value
        # _play_audio now polls every 100 ms, so skip redundant UI signal emits.
        if not changed:
            return
        if value:
            self.ui.set_state("SPEAKING")
        elif not self.ui.muted:
            self.ui.set_state("LISTENING")

    def interrupt(self, source: str = "local") -> None:
        """Stop JARVIS mid-speech: drain queued audio and open mic immediately.

        `source` is "server" when the Live API's own VAD reported the
        interruption and "local" when our RMS heuristic did. The server signal
        is authoritative — it decided from the same audio, with far more than an
        amplitude threshold to go on — so it never has to justify itself in the
        log the way a local guess does.
        """
        self._interrupted = True
        if self._barge is not None:
            self._barge.note_interrupt(time.monotonic())

        drained = 0
        q = self.audio_in_queue
        if q:
            while True:
                try:
                    q.get_nowait()
                    drained += 1
                except Exception:
                    break

        self.set_speaking(False)
        if self._turn_done_event:
            self._turn_done_event.clear()

        log.info(f"✋ Interrupted by {source} — {drained} audio chunks discarded")
        # Only announce it if he was genuinely cut off. Interrupting silence is
        # a false positive, and saying so every few seconds was the symptom that
        # made this worth fixing.
        if drained:
            self.ui.write_log("SYS: Interrupted — listening...")

    def speak(self, text: str):
        self._inject_text_threadsafe(text, "speak")

    def speak_error(self, tool_name: str, error: str):
        short = str(error)[:120]
        self.ui.write_log(f"ERR: {tool_name} — {short}")
        self.speak(f"Sir, {tool_name} encountered an error. {short}")

    def barge_in_enabled(self) -> bool:
        """One switch for the whole interruption policy.

        Read by `_build_config` (whether the server may interrupt itself) and by
        `_listen_audio` (whether the mic stays open while he talks). They must
        agree, or the server would interrupt on audio the mic never sent.
        """
        return bool(get_settings().extra.get("barge_in", self.BARGE_IN))

    def _build_config(self) -> types.LiveConnectConfig:
        from datetime import datetime

        # Load customization from config (cached; re-read only if the file changed)
        _cfg            = get_settings()
        self._asst_name = (_cfg.assistant_name or "JARVIS").strip()
        _user_name      = (_cfg.user_name or "").strip()

        memory     = load_memory()
        # Nothing has been said yet, so there is no conversation to rank facts
        # against — last session's topic is the best available stand-in.
        mem_str    = format_memory_for_prompt(memory, query=last_session_topic())
        sys_prompt = _load_system_prompt()

        now      = datetime.now()
        time_str = now.strftime("%A, %B %d, %Y — %I:%M %p")
        time_ctx = (
            f"[CURRENT DATE & TIME]\n"
            f"Right now it is: {time_str}\n"
            f"Use this to calculate exact times for reminders.\n\n"
        )

        # Identity injection — overrides any hardcoded name in prompt.txt
        _addr = (f"ADDRESS: Always call the user '{_user_name}'."
                 if _user_name
                 else "ADDRESS: When speaking Turkish → always say \"efendim\". "
                      "When speaking English → say \"sir\". Never mix languages.")
        identity_ctx = (
            f"[IDENTITY]\n"
            f"Your name is {self._asst_name}. "
            f"Always refer to yourself as {self._asst_name}.\n"
            f"{_addr}\n\n"
        )

        parts = [time_ctx, identity_ctx]

        # Whether the feed is on has so far only ever been stated in the tool
        # result that turned it on — so a reconnect mid-stream left the model
        # believing it was blind, and it went back to calling screen_process for
        # a view it was already being sent.
        if VISION_STREAM.active:
            src = VISION_STREAM.source
            parts.append(
                f"[LIVE VISION — ON]\n"
                f"You are receiving the user's {src} right now as a continuous video "
                f"feed, about one frame a second. You can see it.\n"
                f"Answer questions about the {src} directly from those frames.\n"
                f"Never call screen_process for the {src} while this is on, and never "
                f"say that you are looking, checking or analysing it — you are already "
                f"watching, so just answer.\n\n"
            )

        if mem_str:
            parts.append(mem_str)
        parts.append(sys_prompt)

        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription={},
            input_audio_transcription={},
            system_instruction="\n".join(parts),
            tools=[{"function_declarations": registry.declarations()}],
            # Passing the handle back turns a dropped socket into a warm resume:
            # the conversation survives and the reconnect skips re-sending the
            # system prompt and the whole tool schema.  Without it every blip
            # cost a cold restart and wiped the context.
            session_resumption=types.SessionResumptionConfig(
                handle=self._resume_handle
            ),
            # NO_INTERRUPTION keeps voice activity detection on — turn-taking
            # still works normally — but stops detected speech from cutting the
            # model off mid-answer. Without it the server interrupts on its own,
            # and no amount of client-side tuning can prevent that, because the
            # decision is made on the server from the audio we send.
            realtime_input_config=types.RealtimeInputConfig(
                activity_handling=(
                    types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS
                    if self.barge_in_enabled()
                    else types.ActivityHandling.NO_INTERRUPTION
                )
            ),
            # Without this an audio+video session runs out of context in about
            # two minutes — every frame is tokens that never fall off on their
            # own.  The sliding window evicts the oldest turns instead of the
            # server closing the socket, so continuous vision can stay on.
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow(),
            ),
            # Native-audio models pick the output language themselves from the
            # conversation — speech_config.language_code is not supported here
            # and setting it is an error, not a hint.
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=_cfg.extra.get("voice", DEFAULT_VOICE)
                    )
                )
            ),
            # Lets the model hear *how* something was said and answer in kind,
            # rather than reading every reply in the same register. Dropped
            # automatically if the model rejects it — see _affective_supported.
            enable_affective_dialog=self._affective_supported,
        )

    def _tool_ctx(self) -> ToolContext:
        """Everything a tool is allowed to reach for on this session."""
        return ToolContext(
            ui           = self.ui,
            speak        = self.speak,
            jarvis       = self,
            current_file = getattr(self.ui, "current_file", None),
        )

    async def _execute_tool(self, fc) -> types.FunctionResponse:
        """Dispatch one function call through the registry.

        The model must always get a response back.  A tool that raises, or one
        that hangs past its declared timeout, still returns a sentence JARVIS
        can say — otherwise Gemini waits forever for a function response and the
        user hears nothing with no way to recover.
        """
        name = fc.name
        args = dict(fc.args or {})

        log.info(f"🔧 {name}  {args}")
        self.ui.set_state("THINKING")

        t = registry.get(name)
        if t is None:
            log.warning(f"❓ Unregistered tool: {name}")
            return types.FunctionResponse(
                id=fc.id, name=name,
                response={"result": f"Unknown tool: {name}"},
            )

        started = time.monotonic()
        try:
            result = await registry.run(name, args, self._tool_ctx())
        except ToolTimeout as e:
            result = str(e)          # already phrased for the model to speak
            log.warning(f"⌛ {result}")
            self.ui.write_log(f"ERR: {name} timed out")
        except Exception as e:
            result = f"Tool '{name}' failed: {e}"
            log.error(f"❌ {name}: {e}")
            traceback.print_exc()
            self.speak_error(name, e)

        if not self.ui.muted:
            self.ui.set_state("LISTENING")

        elapsed = time.monotonic() - started
        log.info(f"📤 {name} → {str(result)[:80]}  [{elapsed:.1f}s]")

        response: dict = {"result": result}
        if t.silent:
            response["silent"] = True
        return types.FunctionResponse(id=fc.id, name=name, response=response)

    #: Whether talking over JARVIS interrupts him. **Off by default**: with no
    #: echo cancellation the microphone hears the speakers, so "someone is
    #: talking" and "JARVIS is talking" are not reliably distinguishable, and
    #: every false positive cuts him off mid-sentence. While off, the Interrupt
    #: button is the only thing that stops him — which is unambiguous.
    #:
    #: Set `barge_in: true` in config/api_keys.json to turn voice interruption
    #: back on. That switch controls the whole policy, not just the local
    #: heuristic: see `_build_config` (ActivityHandling) and `_listen_audio`
    #: (mic gating).
    BARGE_IN = False

    #: Sensitivity of the *local* fast path, used only when barge-in is on. The
    #: authoritative signal is `server_content.interrupted`; this exists so audio
    #: stops without waiting for a round trip, and it must therefore be
    #: conservative — a false positive cuts JARVIS off for no reason, a miss only
    #: costs the few hundred milliseconds until the server says so.
    #:
    #: Overridable via `barge_in_rms`, `barge_in_blocks`, `barge_in_margin`.
    BARGE_IN_RMS    = 1500

    #: Mic blocks are 1024 samples @ 16 kHz = 64 ms. Seven of them is ~450 ms of
    #: sustained sound. The old value of three (~190 ms) was inside the length of
    #: a single loud syllable of JARVIS's own voice coming back through the
    #: speakers, which is why he kept interrupting himself.
    BARGE_IN_BLOCKS = 7

    #: There is no echo cancellation, so the bleed level depends on speaker
    #: volume, mic gain and how far apart they are — a fixed threshold cannot be
    #: right on two machines. The floor is measured continuously instead, and
    #: real speech has to stand this many times above it.
    BARGE_IN_MARGIN = 2.5

    #: Ceiling on the learned floor, so a persistently loud room cannot raise
    #: the bar until barge-in stops working altogether.
    BARGE_IN_FLOOR_MAX = 4000.0

    #: Quiet after an interrupt. Without it the tail of the same utterance
    #: immediately triggers the next one.
    BARGE_IN_COOLDOWN = 2.0

    #: Mic frames are 1024 samples @ 16 kHz = 64 ms = 2048 bytes each.
    _SEND_BATCH_BYTES = 6400    # ~200 ms per websocket send
    _SEND_MAX_BACKLOG = 15      # >~1 s queued → the queue has become a delay line

    async def _send_realtime(self):
        while True:
            msg = await self.out_queue.get()

            # A single network hiccup or GC pause used to leave a permanent
            # backlog here: the mic produces exactly realtime, so the queue
            # never drains again and the model hears everything seconds late.
            # Stale speech is worth less than a responsive assistant — drop it.
            if self.out_queue.qsize() > self._SEND_MAX_BACKLOG:
                dropped = 0
                while True:
                    try:
                        self.out_queue.get_nowait()
                        dropped += 1
                    except asyncio.QueueEmpty:
                        break
                log.warning(f"🎤 Mic backlog — dropped {dropped} stale chunks")

            # Coalesce whatever is already queued into one send, so a normal
            # turn costs ~5 websocket writes/second instead of ~16.  Only
            # same-mime frames merge, so a non-PCM producer stays intact.
            mime = msg.get("mime_type", "audio/pcm")
            buf  = bytearray(msg["data"])
            while len(buf) < self._SEND_BATCH_BYTES:
                try:
                    nxt = self.out_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if nxt.get("mime_type", "audio/pcm") != mime:
                    async with self._send_lock:
                        await self.session.send_realtime_input(
                            media={"data": bytes(buf), "mime_type": mime}
                        )
                    mime = nxt.get("mime_type", "audio/pcm")
                    buf  = bytearray(nxt["data"])
                    continue
                buf.extend(nxt["data"])

            async with self._send_lock:
                await self.session.send_realtime_input(
                    media={"data": bytes(buf), "mime_type": mime}
                )

    #: Consecutive failed frame sends before the stream gives up. A dead camera
    #: or a closing socket should not spin at 1 Hz logging forever.
    _VIDEO_MAX_FAILURES = 5

    async def _send_video(self):
        """Stream frames up the same socket as the microphone.

        This is what makes JARVIS *see* rather than *look*: by the time the user
        finishes asking "what is this error", the screen is already in context —
        no tool call, no filler sentence, no two-turn image injection.

        Idle by default. `vision_stream` turns it on, and even then most ticks
        send nothing: VisionStream drops frames while the user is away and
        frames that look like the one before.
        """
        failures = 0
        while True:
            await asyncio.sleep(VISION_STREAM.interval)

            if not VISION_STREAM.should_send(self._last_user_speech) or not self.session:
                continue

            try:
                frame = await asyncio.to_thread(VISION_STREAM.capture)
            except Exception as e:
                log.warning(f"📷 Frame capture failed: {e}")
                self.ui.write_log(f"SYS: Vision stream stopped — {e}")
                VISION_STREAM.stop()
                continue

            if frame is None:          # unchanged, or stopped mid-capture
                continue

            try:
                async with self._send_lock:
                    await self.session.send_realtime_input(
                        video=types.Blob(data=frame.jpeg, mime_type="image/jpeg")
                    )
                # Only now has the model seen it — see VisionStream.commit.
                VISION_STREAM.commit(frame)
                failures = 0
            except Exception as e:
                # A send failure usually means the socket is already going down,
                # and _receive_audio will raise and trigger the reconnect. Do not
                # take the TaskGroup down from here over a dropped frame.
                failures += 1
                log.warning(f"📷 Frame send failed ({failures}): {e}")
                if failures >= self._VIDEO_MAX_FAILURES:
                    log.warning("📷 Giving up on the video stream.")
                    self.ui.write_log("SYS: Vision stream stopped — send failures.")
                    VISION_STREAM.stop()
                    failures = 0

    def _enqueue_mic(self, data: bytes) -> None:
        """Queue a mic block, dropping it if the send queue is saturated.

        Called via call_soon_threadsafe, where a raised QueueFull would surface
        as an unraisable exception rather than anything actionable.
        """
        try:
            self.out_queue.put_nowait({"data": data, "mime_type": "audio/pcm"})
        except asyncio.QueueFull:
            pass

    async def _listen_audio(self):
        log.info("🎤 Mic started")
        loop = asyncio.get_event_loop()

        # Barge-in tuning, overridable per-install from config/api_keys.json.
        _extra       = get_settings().extra
        barge_on     = self.barge_in_enabled()
        barge_rms    = float(_extra.get("barge_in_rms",    self.BARGE_IN_RMS))
        barge_blocks = int(_extra.get("barge_in_blocks",   self.BARGE_IN_BLOCKS))
        barge_margin = float(_extra.get("barge_in_margin", self.BARGE_IN_MARGIN))
        if barge_on:
            log.info(
                f"✋ Barge-in on (rms>{barge_rms:.0f}, {barge_blocks} blocks, "
                f"{barge_margin:.1f}x floor)"
            )
        else:
            log.info("✋ Barge-in off — only the Interrupt button stops him")

        self._barge = BargeInDetector(
            rms_threshold = barge_rms,
            blocks        = barge_blocks,
            margin        = barge_margin,
            floor_max     = self.BARGE_IN_FLOOR_MAX,
            cooldown      = self.BARGE_IN_COOLDOWN,
        )

        def callback(indata, frames, time_info, status):
            if self.ui.muted or self._phone_active:
                return

            with self._speaking_lock:
                jarvis_speaking = self._is_speaking

            if jarvis_speaking:
                if not barge_on:
                    # Hard-gate the mic. Not only to stop interruptions — with
                    # no echo cancellation the mic hears the speakers, so
                    # anything sent now is JARVIS's own voice, which the server
                    # would transcribe as something the *user* said and fold
                    # into the conversation.
                    return
                rms = float(np.sqrt(np.mean(np.square(indata.astype(np.float32)))))
                if self._barge.feed(rms, time.monotonic()):
                    log.info(
                        f"✋ Barge-in (rms={rms:.0f}, "
                        f"floor={self._barge.floor:.0f}, "
                        f"threshold={self._barge.threshold:.0f})"
                    )
                    loop.call_soon_threadsafe(self.interrupt)
            else:
                self._barge.reset()

            # Falls through after a barge-in so the word that triggered it is
            # not swallowed.
            loop.call_soon_threadsafe(self._enqueue_mic, indata.tobytes())

        try:
            with sd.InputStream(
                samplerate=SEND_SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=CHUNK_SIZE,
                callback=callback,
            ):
                log.info("🎤 Mic stream open")
                while True:
                    await asyncio.sleep(0.1)
        except Exception as e:
            log.error(f"❌ Mic: {e}")
            raise

    async def _receive_audio(self):
        log.info("👂 Recv started")
        out_buf, in_buf = [], []

        try:
            while True:
                async for response in self.session.receive():

                    # Refreshed periodically by the server; the newest one is the
                    # only one that can resume this conversation.
                    upd = response.session_resumption_update
                    if upd and upd.resumable and upd.new_handle:
                        self._resume_handle = upd.new_handle

                    # Sent ~before the server tears the socket down, so the
                    # reconnect below is expected rather than a failure.
                    if response.go_away is not None:
                        log.info(
                            f"👋 Server go_away (time_left="
                            f"{getattr(response.go_away, 'time_left', '?')}) — will resume"
                        )

                    if response.data:
                        if self._interrupted:
                            pass  # discard: interrupted
                        else:
                            if self._turn_done_event and self._turn_done_event.is_set():
                                self._turn_done_event.clear()
                            # Split into ~50 ms chunks so interrupt() stops audio within 50 ms
                            # (24000 Hz × 2 bytes/sample × 0.05 s = 2400 bytes per slice)
                            _audio_data = response.data
                            _SLICE = 2400
                            for _i in range(0, len(_audio_data), _SLICE):
                                self.audio_in_queue.put_nowait(_audio_data[_i : _i + _SLICE])

                    if response.server_content:
                        sc = response.server_content

                        # The server runs its own VAD over the same mic stream
                        # and tells us when it stopped generating because the
                        # user spoke. That is the real barge-in signal; the RMS
                        # check in _listen_audio is only a local fast path.
                        #
                        # Still honoured when barge-in is off, where it should
                        # never arrive: NO_INTERRUPTION plus a gated mic leaves
                        # the server nothing to trigger on. If it does arrive,
                        # the model has already stopped generating, so playing
                        # out the rest of the queue would only voice half a
                        # sentence it has abandoned.
                        if sc.interrupted:
                            self.interrupt(source="server")

                        if sc.output_transcription and sc.output_transcription.text:
                            txt = _clean_transcript(sc.output_transcription.text)
                            if txt and txt != (out_buf[-1] if out_buf else ""):
                                out_buf.append(txt)

                        if sc.input_transcription and sc.input_transcription.text:
                            txt = _clean_transcript(sc.input_transcription.text)
                            if txt:
                                in_buf.append(txt)
                                self._last_user_speech = time.monotonic()

                        if sc.turn_complete:
                            if self._turn_done_event:
                                self._turn_done_event.set()

                            # If this turn_complete ends an interrupted response, clear the
                            # flag and skip all further processing for that turn.
                            if self._interrupted:
                                self._interrupted = False
                                in_buf  = []
                                out_buf = []
                                continue

                            full_in = " ".join(in_buf).strip()
                            if full_in:
                                self.ui.write_log(f"You: {full_in}")
                                self._log_turn(f"User: {full_in}")
                                if self._dashboard:
                                    asyncio.create_task(self._dashboard.broadcast({
                                        "type": "log", "speaker": "user",
                                        "text": full_in,
                                        "ts": datetime.now().isoformat(),
                                    }))
                            in_buf = []

                            full_out = " ".join(out_buf).strip()
                            if full_out:
                                self.ui.write_log(f"{self._asst_name}: {full_out}")
                                self._log_turn(f"{self._asst_name}: {full_out}")
                                if self._dashboard:
                                    asyncio.create_task(self._dashboard.broadcast({
                                        "type": "log", "speaker": "jarvis",
                                        "text": full_out,
                                        "ts": datetime.now().isoformat(),
                                    }))
                            out_buf = []

                            # Vision injection: model finished tool-response turn → now send the image
                            if self._pending_vision and self.session:
                                import base64 as _b64
                                img_b, mime_t, question, angle = self._pending_vision
                                self._pending_vision = None
                                # Off-loop: this coroutine also drains playback
                                # audio, so encoding a screenshot inline stutters
                                # whatever JARVIS is saying.
                                b64 = await asyncio.to_thread(
                                    lambda: _b64.b64encode(img_b).decode("ascii")
                                )
                                log.info(f"[Vision] 📤 {len(img_b):,} bytes (angle={angle}) → main session")
                                async with self._send_lock:
                                    await self.session.send_client_content(
                                        turns={"parts": [
                                            {"inline_data": {"mime_type": mime_t, "data": b64}},
                                            {"text": question},
                                        ]},
                                        turn_complete=True,
                                    )
                                # Mark next turn_complete behaviour depending on angle
                                if self._vision_cam_active:
                                    # Camera: keep busy until JARVIS finishes speaking the answer
                                    self._vision_cam_active    = False
                                    self._vision_close_pending = True
                                else:
                                    # Screen-only: no camera to close; release busy flag now
                                    self._vision_busy = False
                            elif self._vision_close_pending:
                                # This turn_complete IS the vision answer — close camera + release busy flag
                                self._vision_close_pending = False
                                self._vision_busy = False
                                async def _cam_close():
                                    await asyncio.sleep(2.0)
                                    self.ui.stop_camera_stream()
                                asyncio.create_task(_cam_close())

                    if response.tool_call:
                        # Never await tool work inside this loop: while a tool ran
                        # (web_builder takes ~13 s) no server frames were drained,
                        # so playback stalled and the whole session looked frozen.
                        calls = list(response.tool_call.function_calls)
                        for fc in calls:
                            log.info(f"📞 {fc.name}")

                        async def _run_tools(calls=calls):
                            try:
                                fn_responses = await asyncio.gather(
                                    *(self._execute_tool(fc) for fc in calls)
                                )
                                if self.session:
                                    async with self._send_lock:
                                        await self.session.send_tool_response(
                                            function_responses=list(fn_responses)
                                        )
                            except Exception as e:
                                log.error(f"❌ Tool dispatch: {e}")
                                traceback.print_exc()

                        # screen_process sets _pending_vision, which the
                        # turn_complete branch above consumes — dispatching it
                        # concurrently would let turn_complete win that race and
                        # defer the image by a whole turn.  Keep it ordered.
                        if any(fc.name == "screen_process" for fc in calls):
                            await _run_tools()
                        else:
                            asyncio.create_task(_run_tools())
        except Exception as e:
            log.error(f"❌ Recv: {e}")
            traceback.print_exc()
            raise

    async def _play_audio(self):
        log.info("🔊 Play started")

        # latency defaults to 'high' on Windows WASAPI, which parks a few hundred
        # ms of buffer in front of every reply.
        stream = sd.RawOutputStream(
            samplerate=RECEIVE_SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=CHUNK_SIZE,
            latency="low",
        )
        stream.start()

        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        self.audio_in_queue.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    # Releasing the mic used to also require _turn_done_event, but
                    # line ~969 clears that event on any late response.data — so a
                    # straggler frame could leave _is_speaking stuck True and the
                    # mic gated off until the next turn.  Queue-idle is enough.
                    self._idle_ticks += 1
                    if self._idle_ticks >= 3 and self.audio_in_queue.empty():
                        self.set_speaking(False)
                        if self._turn_done_event:
                            self._turn_done_event.clear()
                    continue

                self._idle_ticks = 0
                self.set_speaking(True)

                # Batch all immediately-available chunks into one write to reduce
                # thread-pool round-trips (was one asyncio.to_thread per 50ms slice).
                # Cap at ~200 ms so interrupt() still stops audio within ~200 ms.
                batch = bytearray(chunk)
                while len(batch) < 9600:   # 9600 bytes ≈ 200 ms at 24 kHz / 16-bit mono
                    try:
                        batch.extend(self.audio_in_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                try:
                    await asyncio.to_thread(stream.write, bytes(batch))
                except (RuntimeError, asyncio.CancelledError):
                    break   # executor shutting down — exit cleanly
        except Exception as e:
            log.error(f"❌ Play: {e}")
            raise
        finally:
            self.set_speaking(False)
            stream.stop()
            stream.close()

    # ── Morning briefing ────────────────────────────────────────────────────────

    async def _send_startup_briefing(self) -> None:
        """
        Two-phase briefing optimized for speed:
          Phase 1 — instant greeting (no tools) → speech starts in <1s
          Phase 2 — news pre-fetched in a background thread while Phase 1 plays,
                    delivered as ready text (no Gemini tool-call round-trip) and
                    shown on the UI content panel. Waits for turn_complete event
                    instead of a fixed sleep so there is no unnecessary gap.
        """
        memory   = load_memory()
        identity = memory.get("identity", {})

        def _val(k: str) -> str:
            e = identity.get(k, {})
            return (e.get("value", "") if isinstance(e, dict) else str(e)).strip()

        lang = _val("language")
        name = _val("name")
        time_str = datetime.now().strftime("%H:%M")

        # Start fetching news immediately — runs in parallel while phase 1 plays.
        # Resolving the location first costs nothing when the city is already
        # known, and only reaches the network when it is not.
        loop = asyncio.get_event_loop()
        location = await asyncio.to_thread(resolve_location)
        if location:
            log.info(f"Briefing news for {location.city} (via {location.source})")
        # World news is appended by the ladder itself, as a non-strict last resort.
        queries = location.news_queries() if location else []
        news_future = loop.run_in_executor(None, _fetch_news_ladder, queries)

        await asyncio.sleep(0.3)
        if not self.session:
            return

        # ── Phase 1: instant greeting ─────────────────────────────────────────
        lang_clause = f" Respond in {lang}." if lang else ""
        name_clause = f" Address the user as {name}." if name else ""

        # Inject last session context if available — pop removes it so it's never repeated
        last = await asyncio.to_thread(pop_last_session)
        session_clause = ""
        if last:
            try:
                _delta = (datetime.now() - datetime.strptime(last["date"], "%Y-%m-%d")).days
                _when  = "earlier today" if _delta == 0 else ("yesterday" if _delta == 1 else f"{_delta} days ago")
            except Exception:
                _when = "last time"
            session_clause = (
                f" Also briefly and naturally mention that {_when}: {last['summary']}"
            )

        news_clause = (f" and say you are fetching today's local news for {location.city} now"
                       if location else " and say you are fetching today's news now")
        p1 = (
            f"Greet the user warmly, mention it is {time_str},{news_clause}.{session_clause} "
            f"Keep it to 2 short sentences max. Do not call any tools.{lang_clause}{name_clause}"
        )

        # Clear the turn-done event so we can wait for Phase 1 to finish
        if self._turn_done_event:
            self._turn_done_event.clear()

        await self._inject_text(p1, "briefing_p1")
        self.ui.write_log("SYS: Briefing phase 1 (greeting) sent.")

        # ── Phase 2: fire as soon as Phase 1 audio is done ───────────────────
        async def _deliver_news():
            try:
                lang_str = f" Respond in {lang}." if lang else ""

                # Wait for news fetch (already running) and Phase 1 turn-complete
                # in parallel — whichever takes longer determines the wait time
                news_done   = asyncio.wrap_future(news_future)
                turn_waited = False
                if self._turn_done_event:
                    try:
                        await asyncio.wait_for(self._turn_done_event.wait(), timeout=6.0)
                        turn_waited = True
                    except asyncio.TimeoutError:
                        pass

                # Extra buffer: turn_complete fires when Gemini finishes *generating*
                # Phase 1, but audio may still be playing.  Waiting a beat here
                # prevents Phase 2 audio from arriving while Phase 1 is mid-sentence
                # (which sounds like a "repeated first response" to the user).
                if turn_waited:
                    await asyncio.sleep(0.8)
                else:
                    await asyncio.sleep(1.0)

                try:
                    news_text, news_query = await asyncio.wait_for(news_done, timeout=8.0)
                except Exception:
                    news_text, news_query = "", ""

                if not self.session:
                    return

                if news_text and len(news_text) > 60:
                    # Show on UI content panel immediately
                    self.ui.show_content(f"NEWS — {news_query}", news_text)

                    # Name the place the winning query actually covered — a small
                    # town falls through to its region or country, and announcing
                    # local news before reading national headlines is worse than
                    # naming the wider place honestly.
                    place = (location.label_for(news_query)
                             if location and news_query != WORLD_NEWS_QUERY else "")
                    where = f" from {place}" if place else ""
                    p2 = (
                        f"[BRIEFING] Here are today's top news headlines{where}:\n{news_text}\n\n"
                        f"Say briefly that this is the news{where}, pick ONE headline, "
                        "summarise it in one sentence, then say the full list "
                        f"is displayed on screen. Do not call any tools.{lang_str}"
                    )
                else:
                    p2 = (
                        "News headlines could not be fetched right now. "
                        f"Let the user know briefly.{lang_str}"
                    )

                await self._inject_text(p2, "briefing_p2")
                self.ui.write_log("SYS: Briefing phase 2 (news) sent.")
            except Exception as e:
                log.error(f"[Briefing] Phase 2 error: {e}")
                self.ui.write_log(f"SYS: Briefing phase 2 failed: {e}")

        asyncio.create_task(_deliver_news())

    # ── Session memory ──────────────────────────────────────────────────────────

    async def _save_session_summary(self) -> None:
        """Summarise the current session in 1-2 sentences and save to long_term.json."""
        # Not named `log`: that shadowed the module logger, so the except branch
        # below called .error() on a list and raised AttributeError instead of
        # logging — which killed the shutdown task that was awaiting this.
        turns = self._session_log
        if len(turns) < 3:        # need at least one exchange to be worth saving
            return
        self._session_log = []    # reset immediately so the next session starts clean

        memory = load_memory()
        lang_entry = memory.get("identity", {}).get("language", {})
        lang = (lang_entry.get("value", "") if isinstance(lang_entry, dict) else str(lang_entry)).strip()
        lang = lang or "English"

        convo = "\n".join(turns[-40:])   # cap at last 40 turns to stay within token budget
        prompt = (
            f"Summarize this conversation in 1-2 sentences in {lang}. "
            "Focus on what the user accomplished or discussed. "
            "Output ONLY the summary text, nothing else:\n\n" + convo
        )
        try:
            from google import genai as _genai
            client = _genai.Client(api_key=_get_api_key())
            resp   = await asyncio.to_thread(
                client.models.generate_content,
                model=budget.model("summary"),
                contents=prompt,
            )
            summary = (resp.text or "").strip()
            if summary:
                save_session_summary(summary, lang)
        except Exception as e:
            budget.report(e)
            log.error(f"[Memory] ⚠️ Session summary failed: {e}")

    # ── System monitor ──────────────────────────────────────────────────────────

    async def _run_system_monitor(self) -> None:
        """Background task: voice alerts when metrics exceed thresholds."""
        while True:
            await asyncio.sleep(10)
            alert = await asyncio.to_thread(self._sys_monitor.check)
            if not alert or not self.session:
                continue
            # Don't interrupt an active conversation
            with self._speaking_lock:
                speaking = self._is_speaking
            if speaking or (time.monotonic() - self._last_user_speech) < 10:
                continue
            try:
                await self._inject_text(alert, "system_monitor")
            except Exception as e:
                log.error(f"[Monitor] ⚠️ Could not send alert: {e}")

    # ── Background monitor ──────────────────────────────────────────────────────

    async def _run_background_monitor(self) -> None:
        """Check user-configured topics once per day; speak alerts when new headlines appear."""
        await asyncio.sleep(300)          # wait 5 min after startup before first check
        while True:
            if self.session:
                # Don't interrupt if user spoke recently or JARVIS is mid-sentence
                with self._speaking_lock:
                    speaking = self._is_speaking
                recent_speech = (time.monotonic() - self._last_user_speech) < 30
                if not speaking and not recent_speech:
                    try:
                        alerts = await asyncio.to_thread(monitor_check_all)
                        memory = load_memory()
                        lang_e = memory.get("identity", {}).get("language", {})
                        lang   = (lang_e.get("value", "") if isinstance(lang_e, dict) else str(lang_e)).strip() or "English"
                        for alert in alerts:
                            msg = (
                                f"{alert}\n\n"
                                f"Inform the user about this development naturally in {lang}. "
                                "One brief sentence only."
                            )
                            await self._inject_text(msg, "background_monitor")
                            self.ui.write_log(f"SYS: Monitor alert sent.")
                            await asyncio.sleep(6)   # gap between consecutive alerts
                    except Exception as e:
                        log.error(f"[Monitor] ⚠️ Background check error: {e}")
            await asyncio.sleep(1800)     # check every 30 minutes

    # ── Proactive mode ──────────────────────────────────────────────────────────

    async def _run_proactive_mode(self) -> None:
        """
        Background task: periodically checks if the user has been silent long enough,
        then hands time + memory context to Gemini so it can decide what (if anything)
        to say proactively. No hardcoded rules — Gemini makes the call.
        """
        while True:
            await asyncio.sleep(60)   # evaluate once per minute

            if not self.session:
                continue

            with self._speaking_lock:
                speaking = self._is_speaking
            if speaking:
                continue

            if not self._proactive.should_trigger(self._last_user_speech):
                continue

            self._proactive.mark_triggered()

            try:
                memory       = await asyncio.to_thread(load_memory)
                monitors     = await asyncio.to_thread(list_monitors)
                recent_turns = self._session_log[-8:] if self._session_log else []
                prompt = self._proactive.build_prompt(
                    memory       = memory,
                    monitors     = monitors or None,
                    recent_turns = recent_turns or None,
                )
                await self._inject_text(prompt, "proactive")
                self.ui.write_log("SYS: Proactive check-in.")
            except Exception as e:
                log.warning(f"[Proactive] ⚠️ {e}")

    # ── Phone audio relay ────────────────────────────────────────────────────────

    async def _relay_phone_audio(self) -> None:
        """Forward phone mic PCM chunks from dashboard queue into the Gemini Live session."""
        q = self._dashboard._phone_audio_queue
        while True:
            try:
                chunk = await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # No audio for 1 s → phone mic inactive, give PC mic back
                self._phone_active = False
                continue
            self._phone_active = True   # phone is streaming — silence PC mic
            with self._speaking_lock:
                speaking = self._is_speaking
            if not speaking and not self.ui.muted:
                try:
                    self.out_queue.put_nowait(chunk)
                except asyncio.QueueFull:
                    pass

    def _on_phone_connected(self) -> None:
        self.ui.write_log("SYS: Phone connected via Remote Dashboard.")
        self.ui.notify_phone_connected()

    # ── dashboard command relay ─────────────────────────────────────────────

    async def _process_dashboard_commands(self) -> None:
        while True:
            try:
                text = await asyncio.wait_for(
                    self._dashboard._command_queue.get(), timeout=0.5
                )
                if not text:
                    continue
                # Wait up to 8s for session to become ready after a wake
                for _ in range(80):
                    if self.session:
                        break
                    await asyncio.sleep(0.1)
                if self.session:
                    await self._inject_text(text, "dashboard")
                    self.ui.write_log(f"[Web]: {text}")
                else:
                    log.warning(f"[Dashboard] Dropped command (no session): {text}")
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                log.error(f"[Dashboard] Command error: {e}")
                await asyncio.sleep(0.5)

    # ── main loop ───────────────────────────────────────────────────────────

    async def run(self):
        self._loop = asyncio.get_event_loop()

        # Start dashboard (optional — needs: pip install fastapi "uvicorn[standard]" cryptography)
        try:
            from dashboard.server import DashboardServer
            self._dashboard = DashboardServer()
            self._dashboard.set_connect_callback(self._on_phone_connected)
            asyncio.create_task(self._dashboard.serve())
            # Runs for the whole lifetime, not just inside an active session
            asyncio.create_task(self._process_dashboard_commands())
        except Exception as e:
            log.info(f"[Dashboard] Disabled: {e}")
            self._dashboard = None

        while True:
            connected = False
            try:
                log.info(
                    "Resuming..." if self._resume_handle else "Connecting..."
                )
                self.ui.set_state("THINKING")
                config = self._build_config()

                # Fresh client on every reconnect — avoids stale HTTP session state
                client = genai.Client(
                    api_key=_get_api_key(),
                    http_options={"api_version": "v1beta"}
                )

                async with (
                    client.aio.live.connect(model=LIVE_MODEL, config=config) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self.session          = session
                    self.audio_in_queue   = asyncio.Queue()
                    self.out_queue        = asyncio.Queue(maxsize=200)
                    self._turn_done_event = asyncio.Event()
                    self._send_lock       = asyncio.Lock()

                    # Reset transient state that must not carry over from a previous session
                    self._pending_vision       = None
                    self._vision_cam_active    = False
                    self._vision_close_pending = False
                    self._vision_busy          = False
                    self._vision_last_time     = 0.0
                    self._interrupted          = False
                    # The new session has seen nothing, so the next frame must
                    # go even if the screen has not changed.
                    VISION_STREAM.on_session_start()

                    connected = True
                    # A healthy connection clears the exponential backoff.
                    # Without this, one bad patch of network pinned every later
                    # reconnect at the 60s ceiling for the rest of the process.
                    self._conn_backoff = 3
                    log.info("Resumed." if self._resume_handle else "Connected.")
                    self.ui.set_state("LISTENING")
                    self.ui.write_log("SYS: JARVIS online.")

                    if self._dashboard:
                        await self._dashboard.broadcast({"type": "status", "state": "active"})

                    tg.create_task(self._send_realtime())
                    tg.create_task(self._send_video())
                    tg.create_task(self._listen_audio())
                    tg.create_task(self._receive_audio())
                    tg.create_task(self._play_audio())
                    tg.create_task(self._run_system_monitor())
                    tg.create_task(self._run_background_monitor())
                    tg.create_task(self._run_proactive_mode())
                    if self._dashboard:
                        tg.create_task(self._relay_phone_audio())

                    # Morning briefing — fires once per process launch (if enabled)
                    if not self._briefing_sent and get_brief_enabled():
                        self._briefing_sent = True
                        tg.create_task(self._send_startup_briefing())

            except KeyboardInterrupt:
                raise
            except SystemExit:
                raise
            except BaseException as e:
                # Catches both Exception and BaseExceptionGroup (Python 3.11+
                # TaskGroup raises BaseExceptionGroup when tasks are cancelled
                # externally, which `except Exception` would miss, letting the
                # exception escape the while-loop and causing asyncio.run() to
                # start shutdown — resulting in "executor after shutdown" errors).
                err_str = _flatten_error(e)
                log.error(f"Error ({type(e).__name__}): {e}")
                traceback.print_exc()

                # Failed before the session came up while holding a handle: assume
                # the handle is stale/rejected and drop it, otherwise every retry
                # replays the same bad token and never reconnects.
                if not connected and self._resume_handle:
                    log.warning("Resume handle rejected — retrying with a fresh session.")
                    self._resume_handle = None

                # Affective dialog is not available on every model or API
                # version, and the rejection arrives at connect time. Retrying
                # with the same config would loop forever, so give it up for
                # this process: a flatter voice beats no assistant.
                if (not connected and self._affective_supported
                        and "affective" in err_str.lower()):
                    log.warning(
                        "Affective dialog rejected by the server — reconnecting without it."
                    )
                    self.ui.write_log("SYS: Expressive voice unavailable on this model.")
                    self._affective_supported = False
                    continue

                # 1007 is the websocket "invalid frame payload" close code; the
                # server uses it for BOTH a bad key and a rejected payload, so it
                # can only mean "bad key" when the text actually says so.
                # Blocking on the re-enter-key overlay for a content-type 1007
                # wedged the assistant until the user typed a key it already had.
                is_auth_err = (
                    "API key not valid" in err_str
                    or "API_KEY_INVALID" in err_str
                    or "UNAUTHENTICATED" in err_str
                )
                if is_auth_err:
                    self.ui.write_log("ERR: API key invalid — please re-enter your key.")
                    self.ui.set_state("SLEEPING")
                    self.ui.prompt_reconfig()
                    while not self.ui._win._ready:
                        await asyncio.sleep(1)
                    log.info("New API key saved — reconnecting...")
                    self._conn_backoff = 3
                    continue

                # Network / timeout errors — log clearly and back off
                is_net_err = any(k in err_str for k in (
                    "TimeoutError", "timed out", "getaddrinfo", "CancelledError",
                    "ConnectionRefusedError", "OSError", "Cannot connect",
                ))

                # Payload rejected mid-session. Name the last text injection so the
                # offending call site is identifiable instead of anonymous.
                if "CONTENT_TYPE_AUDIO" in err_str or "1007" in err_str:
                    log.error(
                        f"Session rejected our payload (1007). "
                        f"Last text injection: {self._last_inject}"
                    )
                    self.ui.write_log("SYS: Session reset — reconnecting.")
                    self._conn_backoff = 1
                elif is_net_err:
                    _conn_backoff = min(getattr(self, "_conn_backoff", 3) * 2, 60)
                    self._conn_backoff = _conn_backoff
                    self.ui.write_log(
                        f"NET: Bağlantı kurulamadı — {_conn_backoff}s sonra tekrar deneniyor. "
                        "(VPN gerekiyor olabilir)"
                    )
                else:
                    self._conn_backoff = 3
            finally:
                self.session = None
                # Only save if there was a real conversation (≥3 turns)
                if len(self._session_log) >= 3:
                    asyncio.create_task(self._save_session_summary())

            self.set_speaking(False)
            self.ui.set_state("SLEEPING")

            if self._dashboard:
                await self._dashboard.broadcast({"type": "status", "state": "sleeping"})

            delay = getattr(self, "_conn_backoff", 3)
            log.info(f"Reconnecting in {delay}s...")
            await asyncio.sleep(delay)

def main():
    setup_logging()
    log.info("MARK L starting — logs at %s", LOG_PATH)

    ui = JarvisUI("face.png")
    # Warnings and errors from any module surface in the HUD log panel,
    # so modules no longer need a UI handle just to report a problem.
    add_ui_sink(lambda msg: ui.write_log(f"LOG: {msg}"))

    def runner():
        ui.wait_for_api_key()
        jarvis = JarvisLive(ui)
        try:
            asyncio.run(jarvis.run())
        except KeyboardInterrupt:
            log.info("🔴 Shutting down...")

    threading.Thread(target=runner, daemon=True).start()
    ui.root.mainloop()

if __name__ == "__main__":
    main()