"""Tools that need the live session, not just the OS.

Everything in `actions/` is a pure function of its parameters — call it, get a
string back.  These four are not:

* `screen_process` stashes an image on JarvisLive for the *next* turn to inject
  (see the two-turn vision handshake in `_receive_audio`)
* `close_camera` and `save_memory` touch the UI / long-term store directly
* `shutdown_jarvis` has to outlive its own function response

They live here rather than in main.py so that the registry holds every tool, and
`_execute_tool` stays a dispatch table with no special cases.  They reach the
session through `ctx.jarvis`.
"""
from __future__ import annotations

import asyncio
import time

from core import budget
from core.log import get_logger
from core.registry import ToolContext, tool
from memory.memory_manager import recall, update_memory

from actions.screen_processor import _capture_camera, _capture_screen
from actions.vision_stream import STREAM as VISION

log = get_logger("session_tools")

#: Vision re-entry guard.  JARVIS hears his own "looking at your screen now"
#: filler sentence through the speakers, and without this he answers it by
#: calling screen_process again.
VISION_COOLDOWN = 4.0


@tool(
    name="screen_process",
    description=(
        "Captures the screen or webcam image and lets you analyze it. "
        "MUST be called when user asks what is on screen, what you see, "
        "look at camera, analyze my screen, etc. "
        "You have NO visual ability without this tool. "
        "After the image is captured it is sent directly to you — describe what you see and answer the user's question. "
        "When using camera: the live view stays open until user says close it or calls close_camera."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "angle": {"type": "STRING", "description": "'screen' to capture display, 'camera' for webcam. Default: 'screen'"},
            "text":  {"type": "STRING", "description": "The question or instruction about the captured image"},
        },
        "required": ["text"],
    },
    timeout=30,
)
async def screen_process_tool(params: dict, ctx: ToolContext) -> str:
    j = ctx.jarvis
    now = time.monotonic()

    angle = str(params.get("angle", "screen")).lower()

    # Continuous vision already puts this view in context every second, so
    # capturing again would send a second copy of what the model can see and
    # cost a turn doing it.
    if VISION.active and VISION.source == angle:
        return (
            f"[VISION_LIVE] You are already receiving the user's {angle} continuously. "
            f"Answer from the frames you have — do not call this tool for the "
            f"{angle} while continuous vision is on."
        )

    if j._vision_busy or (now - j._vision_last_time) < VISION_COOLDOWN:
        wait = max(0.0, VISION_COOLDOWN - (now - j._vision_last_time))
        log.info(f"⏳ Cooldown active ({wait:.1f}s remaining) — ignoring duplicate call")
        return "Vision is still processing the previous request. I will not call this again."

    j._vision_busy      = True
    j._vision_last_time = now

    user_text = params.get("text", "What do you see?")

    try:
        if angle == "camera":
            img_b, mime_t = await asyncio.to_thread(_capture_camera)
            ctx.ui.start_camera_stream()
            j._vision_cam_active = True
            log.info(f"📷 Camera: {len(img_b):,} bytes")
            stall = "camera"
        else:
            img_b, mime_t = await asyncio.to_thread(_capture_screen)
            log.info(f"🖥️  Screen: {len(img_b):,} bytes")
            stall = "screen"
    except BaseException:
        # Without this the flag stays True forever and vision is dead for the
        # rest of the session — the release path only runs on a successful
        # capture, via turn_complete.  Covers CancelledError from a timeout too.
        j._vision_busy = False
        raise

    j._pending_vision = (img_b, mime_t, user_text, angle)
    return (
        f"[VISION_ACTIVE] {stall.capitalize()} captured. "
        f"Immediately say ONE short natural sentence in the user's own language, "
        f"telling them you are looking at their {stall} right now. "
        f"Do NOT describe or guess content — the actual image arrives in the NEXT message."
    )


@tool(
    name="vision_stream",
    description=(
        "Turns continuous vision on or off. While it is on you receive the user's "
        "screen or webcam as a live video feed and can see what they are doing "
        "without calling screen_process. "
        "Turn it ON when the user says: watch my screen, keep an eye on this, "
        "look at what I'm doing, stay with me, follow along, watch me work. "
        "Turn it OFF when they say: stop watching, stop looking, you can look away. "
        "Use 'status' when they ask whether you are watching. "
        "This costs significantly more than a one-off look — never turn it on "
        "unless the user asked for ongoing watching."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "action": {"type": "STRING", "description": "start | stop | status"},
            "source": {"type": "STRING", "description": "screen (default) | camera"},
            "interval": {
                "type": "NUMBER",
                "description": "Seconds between frames. Minimum and default 1.",
            },
        },
        "required": ["action"],
    },
    timeout=15,
)
def vision_stream_tool(params: dict, ctx: ToolContext) -> str:
    action = str(params.get("action", "")).strip().lower()
    source = str(params.get("source", "screen")).strip().lower()

    if action == "status":
        return VISION.status()

    if action == "stop":
        if source == "camera" or VISION.source == "camera":
            ctx.ui.stop_camera_stream()
        return VISION.stop()

    if action == "start":
        # Continuous video is exactly the workload a free key cannot carry: a
        # frame a second, indefinitely. budget.py already knows whether we are
        # degraded, so ask it rather than discovering it as a 429 mid-sentence.
        if not budget.allows("video_stream"):
            return (
                "Continuous vision is off in free mode — a frame every second "
                "would exhaust the quota within minutes. Switch to paid mode to "
                "use it. I can still look on request with screen_process."
            )
        result = VISION.start(source=source, interval=params.get("interval"))
        if VISION.active and VISION.source == "camera":
            ctx.ui.start_camera_stream()
        return result

    return "Specify action: start, stop or status."


@tool(
    name="close_camera",
    description=(
        "Closes the live camera view shown on screen. "
        "Call when user says: close camera, stop camera, turn off camera, "
        "kamerayı kapat, kapat, creepy, etc."
    ),
    parameters={"type": "OBJECT", "properties": {}, "required": []},
    timeout=10,
)
def close_camera_tool(params: dict, ctx: ToolContext) -> str:
    ctx.ui.stop_camera_stream()
    return "Camera closed."


@tool(
    name="save_memory",
    description=(
        "Save an important personal fact about the user to long-term memory. "
        "Call this silently whenever the user reveals something worth remembering: "
        "name, age, city, job, preferences, hobbies, relationships, projects, or future plans. "
        "Do NOT call for: weather, reminders, searches, or one-time commands. "
        "Do NOT announce that you are saving — just call it silently. "
        "Values must be in English regardless of the conversation language."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "category": {
                "type": "STRING",
                "description": (
                    "identity — name, age, birthday, city, job, language, nationality | "
                    "preferences — favorite food/color/music/film/game/sport, hobbies | "
                    "projects — active projects, goals, things being built | "
                    "relationships — friends, family, partner, colleagues | "
                    "wishes — future plans, things to buy, travel dreams | "
                    "notes — habits, schedule, anything else worth remembering"
                ),
            },
            "key":   {"type": "STRING", "description": "Short snake_case key (e.g. name, favorite_food, sister_name)"},
            "value": {"type": "STRING", "description": "Concise value in English (e.g. Fatih, pizza, older sister)"},
        },
        "required": ["category", "key", "value"],
    },
    timeout=10,
    silent=True,
)
def save_memory_tool(params: dict, ctx: ToolContext) -> str:
    category = params.get("category", "notes")
    key      = params.get("key", "")
    value    = params.get("value", "")
    if key and value:
        update_memory({category: {key: {"value": value}}})
        log.info(f"💾 save_memory: {category}/{key} = {value}")
    return "ok"


@tool(
    name="recall_memory",
    description=(
        "Search everything you have ever been told about the user and get back "
        "the facts that match. "
        "Only a handful of the most relevant facts are loaded at the start of a "
        "conversation, so use this whenever the user refers to something personal "
        "you cannot see in front of you: 'what was my sister's name', "
        "'what did I say I wanted to build', 'which framework do I use', "
        "'remind me what I told you about my job'. "
        "Also use it before saying you do not know or do not remember something "
        "personal — the fact is very often stored, just not loaded."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "query": {
                "type": "STRING",
                "description": (
                    "What to look for, in English — a topic or question, "
                    "e.g. 'sister', 'programming languages', 'travel plans'"
                ),
            },
        },
        "required": ["query"],
    },
    timeout=20,
)
def recall_memory_tool(params: dict, ctx: ToolContext) -> str:
    query = str(params.get("query", "")).strip()
    if not query:
        return "Tell me what to look for."

    hits = recall(query, include_identity=True)
    if not hits:
        return f"I have nothing stored about '{query}'."

    lines = [f"{h['key'].replace('_', ' ')}: {h['value']}" for h in hits]
    log.info(f"recall '{query}' -> {len(hits)} hit(s)")
    return "Here is what I have stored:\n" + "\n".join(lines)


@tool(
    name="shutdown_jarvis",
    description=(
        "Shuts down the assistant completely. "
        "Call this when the user expresses intent to end the conversation, "
        "close the assistant, say goodbye, or stop Jarvis. "
        "The user can say this in ANY language."
    ),
    parameters={"type": "OBJECT", "properties": {}},
    timeout=10,
)
async def shutdown_jarvis_tool(params: dict, ctx: ToolContext) -> str:
    j = ctx.jarvis
    ctx.log_line("SYS: Shutdown requested.")

    async def _do_shutdown():
        # The exit lives in a finally: a failure anywhere in the cleanup —
        # summary API call, goodbye injection — must still end the process.
        # Before this, a raise here left JARVIS running after saying
        # "Shutting down", with no second chance at stopping him.
        try:
            try:
                await j._save_session_summary()
            except Exception as e:
                log.error(f"Session summary failed during shutdown: {e}")
            if j.session:
                try:
                    await j._inject_text("Say a brief natural goodbye to the user.", "shutdown")
                except Exception:
                    pass
            await asyncio.sleep(1.5)
        finally:
            import os as _os
            _os._exit(0)

    # Deliberately not awaited: the shutdown has to outlive this function
    # response, otherwise the goodbye never reaches the model.
    asyncio.create_task(_do_shutdown())
    return "Shutting down."
