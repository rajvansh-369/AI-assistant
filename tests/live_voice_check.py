"""Live check of how JARVIS speaks — transcript analysed, audio saved to WAV.

    python tests/live_voice_check.py

NOT a pytest test; the filename keeps it out of collection. Needs a working API
key and spends a short exchange of quota.

Two halves, because only one of them can be automated:

* **Delivery** is textual. Native-audio models generate speech from the words,
  so contractions, sentence-length variation and spoken-form numbers are all
  visible in `output_transcription` — and those are exactly what the HOW YOU
  SOUND section of prompt.txt is trying to produce. Those are checked here.
* **Timbre** is not. Whether the chosen voice actually sounds right is a
  listening judgement, so the audio is written to a WAV for a human to play.

Prompts are chosen to tempt the specific failure modes: a time question invites
"14:31" instead of "about half past two", and an open greeting invites
"Certainly! How may I assist you today?".
"""
from __future__ import annotations

import asyncio
import re
import statistics
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google import genai                            # noqa: E402
from google.genai import types                      # noqa: E402

import main                                         # noqa: E402
from core.settings import LIVE_MODEL, get_api_key   # noqa: E402

OUT_WAV = Path(__file__).resolve().parent.parent / "logs" / "voice_sample.wav"

TURNS = [
    "Hey, how's it going?",
    "What time is it?",
    "My CPU's been running hot all afternoon. Should I be worried?",
]

TURN_TIMEOUT = 30.0

CONTRACTIONS = re.compile(
    r"\b(i'm|it's|that's|you're|we're|there's|don't|doesn't|isn't|can't|won't|"
    r"i've|we've|you've|i'll|we'll|you'll|let's|here's|what's|he's|she's|they're|"
    r"didn't|haven't|hasn't|wouldn't|couldn't|shouldn't|it'll|that'll)\b",
    re.I,
)

ROBOT_OPENERS = (
    "certainly", "of course", "i'd be happy to", "as an ai",
    "i am happy to", "sure thing", "absolutely",
)

#: Clock times read out as digits, which the prompt asks him to speak instead.
DIGIT_TIME = re.compile(r"\b\d{1,2}[:.]\d{2}\b")


class StubUI:
    muted = False
    current_file = None

    def __init__(self):
        self.on_text_command = None
        self.on_remote_clicked = None
        self.on_interrupt = None

    def set_state(self, s): pass
    def write_log(self, t): pass
    def show_content(self, a, b): pass


def analyse(text: str) -> list[tuple[bool, str]]:
    """Each of the delivery rules the prompt actually asks for."""
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 3]
    lengths = [len(s.split()) for s in sentences]
    spread = statistics.pstdev(lengths) if len(lengths) > 1 else 0.0

    contractions = len(CONTRACTIONS.findall(text))
    opener = next((o for o in ROBOT_OPENERS if text.lower().lstrip().startswith(o)), None)
    digit_times = DIGIT_TIME.findall(text)

    return [
        (contractions >= 2,
         f"contractions: {contractions} found"),
        (spread >= 2.5,
         f"sentence-length spread: {spread:.1f} words "
         f"(lengths {lengths[:8]}{'...' if len(lengths) > 8 else ''})"),
        (opener is None,
         f"robotic opener: {opener or 'none'}"),
        (not digit_times,
         f"digit clock times: {digit_times or 'none'}"),
    ]


async def main_test() -> int:
    j = main.JarvisLive(StubUI())
    config = j._build_config()
    voice = config.speech_config.voice_config.prebuilt_voice_config.voice_name

    print("config:")
    print(f"   voice            : {voice}")
    print(f"   affective dialog : {config.enable_affective_dialog}")
    print(f"   language_code    : {config.speech_config.language_code}")
    print(f"   HOW YOU SOUND    : {'HOW YOU SOUND' in config.system_instruction}")

    client = genai.Client(api_key=get_api_key(), http_options={"api_version": "v1beta"})

    print(f"\nconnecting to {LIVE_MODEL} ...")
    audio = bytearray()
    said: list[str] = []

    async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
        print("connected — affective dialog accepted.\n")
        j.session = session
        j._send_lock = asyncio.Lock()

        for turn in TURNS:
            print(f"  you    : {turn}")
            await j._inject_text(turn, "voicecheck")

            reply: list[str] = []

            async def read():
                async for response in session.receive():
                    if response.data:
                        audio.extend(response.data)
                    sc = response.server_content
                    if not sc:
                        continue
                    if sc.output_transcription and sc.output_transcription.text:
                        reply.append(sc.output_transcription.text)
                    if sc.turn_complete and reply:
                        return

            try:
                await asyncio.wait_for(read(), timeout=TURN_TIMEOUT)
            except asyncio.TimeoutError:
                print("  (timed out)")

            text = "".join(reply).strip()
            said.append(text)
            print(f"  jarvis : {text}\n")

    if not any(said):
        print("FAIL: he said nothing.")
        return 1

    OUT_WAV.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUT_WAV), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(main.RECEIVE_SAMPLE_RATE)
        w.writeframes(bytes(audio))

    secs = len(audio) / (main.RECEIVE_SAMPLE_RATE * 2)
    print(f"audio: {len(audio):,} bytes ({secs:.1f}s) -> {OUT_WAV}")
    print("       play it to judge the voice itself; the rest is checked below.\n")

    print("delivery (what the prompt actually asks for):")
    joined = " ".join(said)
    results = analyse(joined)
    for ok, detail in results:
        print(f"   {'PASS' if ok else 'FAIL'}  {detail}")

    failed = [d for ok, d in results if not ok]
    print()
    if failed:
        print(f"{len(failed)} of {len(results)} delivery checks failed.")
        return 1
    print("All delivery checks passed. Timbre is a listening call — play the WAV.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main_test()))
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
