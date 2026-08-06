"""Live check of the one path unit tests cannot reach: video frames on a real socket.

    python tests/live_vision_check.py

NOT a pytest test — the filename keeps it out of collection on purpose. It needs
a working API key, captures your screen, sends it to Google, and spends a few
frames plus one short exchange of quota.

Drives the real `_build_config` and `_send_video` against an actual Gemini Live
session with no Qt window, no microphone and no speaker, then asks "what am I
looking at right now?". Passing means the model answered correctly *and* did it
without calling a tool, echoing a control tag, or announcing that it was looking
— which is only possible if the frames arrived as video input.

Worth re-running after any edit to `core/prompt.txt`: the failures this catches
(a control tag spoken aloud, a redundant screen_process call) are invisible to
the unit tests, because they are model behaviour rather than code paths.

VisionStream is started directly rather than through the `vision_stream` tool, so
the free-mode budget gate is bypassed for the check without touching the
configured mode.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google import genai                       # noqa: E402
from google.genai import types                 # noqa: E402

import main                                    # noqa: E402
from actions.vision_stream import STREAM as VISION   # noqa: E402
from core.settings import LIVE_MODEL, get_api_key    # noqa: E402

# Phrased the way a user actually would: no hint that an image was sent, no
# instruction about tools. If continuous vision works, this is answerable.
QUESTION = "What am I looking at right now? One short sentence."

#: Internal control signals that must never reach the user's ears.
TAGS = ("[VISION_ACTIVE]", "[VISION_LIVE]", "[SYSTEM_ALERT]",
        "[BRIEFING]", "[PROACTIVE_CHECK]")

FRAME_SECONDS = 4.0     # let _send_video tick a few times
ANSWER_TIMEOUT = 30.0


class StubUI:
    """Only what JarvisLive.__init__ and _send_video touch."""
    muted = False
    current_file = None

    def __init__(self):
        self.on_text_command = None
        self.on_remote_clicked = None
        self.on_interrupt = None

    def set_state(self, s):  pass
    def write_log(self, t):  print(f"   [ui] {t}")
    def start_camera_stream(self): pass
    def stop_camera_stream(self):  pass


async def main_test() -> int:
    j = main.JarvisLive(StubUI())

    # Started before the config is built, the way a reconnect mid-stream would
    # find it — _build_config must then tell the model it can already see.
    VISION.start("screen")
    config = j._build_config()

    print("config:")
    print(f"   tools declared            : {len(config.tools[0].function_declarations)}")
    print(f"   LIVE VISION in prompt     : {'[LIVE VISION — ON]' in config.system_instruction}")
    print(f"   context_window_compression: {config.context_window_compression is not None}")
    print(f"   session_resumption        : {config.session_resumption is not None}")

    client = genai.Client(api_key=get_api_key(), http_options={"api_version": "v1beta"})

    print(f"\nconnecting to {LIVE_MODEL} ...")
    async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
        print("connected.")

        j.session = session
        j._send_lock = asyncio.Lock()
        j._last_user_speech = time.monotonic()      # keep the activity gate open

        VISION.on_session_start()
        video_task = asyncio.create_task(j._send_video())

        print(f"streaming frames for {FRAME_SECONDS:.0f}s ...")
        await asyncio.sleep(FRAME_SECONDS)
        print(f"   frames sent   : {VISION.frames_sent}")
        print(f"   frames skipped: {VISION.frames_skipped}")
        print(f"   bytes sent    : {VISION.bytes_sent:,}")

        if VISION.frames_sent == 0:
            print("\nFAIL: no frame was sent.")
            video_task.cancel()
            return 1

        print("\nasking what it can see ...")
        await j._inject_text(QUESTION, "livetest")

        answer: list[str] = []
        audio_bytes = 0
        tool_calls: list[str] = []

        async def read():
            nonlocal audio_bytes
            async for response in session.receive():
                if response.data:
                    audio_bytes += len(response.data)

                # The real _receive_audio answers tool calls; without this the
                # model waits forever for a function response and the turn
                # never completes.
                if response.tool_call:
                    calls = list(response.tool_call.function_calls)
                    for fc in calls:
                        tool_calls.append(fc.name)
                        print(f"   [tool] {fc.name} {dict(fc.args or {})}")
                    frs = [await j._execute_tool(fc) for fc in calls]
                    for fr in frs:
                        print(f"   [tool] -> {str(fr.response.get('result'))[:100]}")
                    async with j._send_lock:
                        await session.send_tool_response(function_responses=frs)
                    continue

                sc = response.server_content
                if not sc:
                    continue
                if sc.output_transcription and sc.output_transcription.text:
                    answer.append(sc.output_transcription.text)
                if sc.turn_complete and answer:
                    return

        try:
            await asyncio.wait_for(read(), timeout=ANSWER_TIMEOUT)
        except asyncio.TimeoutError:
            print("timed out waiting for the answer")

        video_task.cancel()
        VISION.stop()

        text = "".join(answer).strip()
        print(f"\n   audio returned: {audio_bytes:,} bytes")
        print(f"   tools called  : {tool_calls or 'none'}")
        print(f"   JARVIS said   : {text!r}")

        if not text:
            print("\nFAIL: no answer.")
            return 1

        leaked = [t for t in TAGS if t in text]
        if leaked:
            print(f"\nFAIL: spoke internal control tag(s) {leaked}")
            return 1

        if tool_calls:
            print(f"\nFAIL: called {tool_calls} — it should answer from the feed.")
            return 1

        low = text.lower()
        filler = [p for p in ("looking at your", "analyzing your", "analysing your",
                              "checking your", "let me look", "let me see")
                  if p in low]
        if filler:
            print(f"\nFAIL: announced looking ({filler}) instead of just answering.")
            return 1

        print("\nPASS: answered straight from the live feed — no tool call, "
              "no tag leak, no filler.")
        return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main_test()))
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
