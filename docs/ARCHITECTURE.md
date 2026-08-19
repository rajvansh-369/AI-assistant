# MARK L — Architecture Guide

A map of what this codebase actually is, how a spoken sentence turns into an action, and where every moving part lives. Written from a read of the code as of commit `b4d6ae9`.

---

## 1. What it is in one paragraph

MARK L is a **desktop voice agent**. A PyQt6 HUD runs on the main thread; a background thread runs an asyncio loop that holds an open WebSocket to the **Gemini Live API**. Your microphone streams raw PCM up that socket continuously; Gemini streams audio back down and plays it through your speakers. When Gemini decides a task needs real-world action, it emits a **function call**, and `main.py` dispatches that to one of ~20 Python modules in `actions/` that drive your OS, browser, files, or the web. Persistent facts about you live in a single JSON file. An optional FastAPI server lets your phone act as a second microphone and remote control.

There is no LangChain, no vector DB, no agent framework. It is a hand-rolled tool-calling loop around one long-lived streaming session.

---

## 2. Process and thread model

This is the single most important thing to understand, because almost every bug and every optimization lives here.

```
┌─ Process ────────────────────────────────────────────────────────┐
│                                                                  │
│  MAIN THREAD  (Qt event loop)                                    │
│    ui.py → MainWindow, HudCanvas, log panel, camera preview      │
│    Never blocks. Owns all widgets.                               │
│                                                                  │
│  THREAD "runner"  (asyncio.run → JarvisLive.run)                 │
│    ├── _send_realtime()          mic PCM → Gemini                │
│    ├── _send_video()             JPEG frames → Gemini (≤1 FPS)   │
│    ├── _listen_audio()           sounddevice InputStream         │
│    ├── _receive_audio()          Gemini → audio + transcripts +  │
│    │                             tool_call dispatch              │
│    ├── _play_audio()             queue → speaker RawOutputStream │
│    ├── _run_system_monitor()     every 10 s                      │
│    ├── _run_background_monitor() every 30 min                    │
│    ├── _run_proactive_mode()     every 60 s                      │
│    ├── _relay_phone_audio()      phone PCM → same out_queue      │
│    └── _process_dashboard_commands()                             │
│                                                                  │
│  DEFAULT ThreadPoolExecutor  (implicit)                          │
│    Every actions/* call runs here via run_in_executor(None, …)   │
│                                                                  │
│  THREAD (uvicorn)  dashboard/server.py — FastAPI on :8000        │
│  THREAD (sounddevice callback)  fills out_queue from mic         │
│  THREAD (_SysMetrics)  CPU/RAM/GPU polling for the HUD           │
└──────────────────────────────────────────────────────────────────┘
```

Key consequences:

- **The Qt thread and the asyncio thread are different threads.** UI calls from tools go through `self.ui.*` methods, which must marshal onto the Qt thread (`ui.py` does this with signals). Any new UI touch from a tool must follow the same path.
- **All tool work is synchronous Python offloaded to the default thread pool.** `run_in_executor(None, …)` means an unbounded default pool — a slow tool does not block audio, but ten slow tools will happily spawn ten threads.
- **`asyncio.TaskGroup` in `run()` is the supervisor.** If any task raises, the whole group tears down, the session drops, and the outer `while True` reconnects with backoff. That is why `_receive_audio` re-raises everything.

---

## 3. The request lifecycle

Follow one sentence end to end:

1. **Capture** — `sounddevice.InputStream` callback (`main.py:900`) fires every 1024 frames at 16 kHz. If JARVIS is not speaking, not muted, and the phone mic is idle, the raw bytes are pushed onto `out_queue`.
2. **Upload** — `_send_realtime()` drains `out_queue` into `session.send_realtime_input()`. Continuous, no VAD on our side — Gemini does endpointing.
3. **Model turn** — Gemini streams back `response.data` (24 kHz PCM), `input_transcription`, `output_transcription`, and optionally `tool_call`.
4. **Playback** — audio is sliced into 2400-byte (~50 ms) chunks into `audio_in_queue`, then `_play_audio()` re-batches up to ~200 ms per `stream.write` to cut thread-pool round-trips. The 50/200 ms split is a deliberate latency-vs-overhead tradeoff: interrupt latency is bounded by the 200 ms batch.
5. **Tool dispatch** — `_execute_tool()` (`main.py:273`) looks `fc.name` up in `core.registry` and awaits `registry.run()`, which offloads to a bounded 8-thread pool under that tool's declared timeout. Success, exception and timeout all return a `types.FunctionResponse` — the model is never left waiting on a response that does not come.
6. **Result** — responses go back via `send_tool_response()`; Gemini speaks the outcome.
7. **Logging** — on `turn_complete`, buffered transcripts are written to the HUD log, appended to `self._session_log`, and broadcast to any connected phone over WebSocket.

### Vision: two paths

**Continuous (`actions/vision_stream.py`)** — the default way JARVIS sees. `_send_video()` streams JPEG frames up the same websocket as the microphone via `send_realtime_input(video=…)`, so the screen is already in context by the time you finish asking about it. No tool call, no filler sentence, no state machine. Off until `vision_stream` turns it on.

The API caps video at 1 FPS and every frame costs tokens, so most ticks send nothing:

- **Activity gate** — dormant after `IDLE_AFTER` (60 s) of user silence, awake on the next utterance. Google's guidance is to send video only during audio activity.
- **Change gate** — a frame whose 32×32 grayscale thumbnail is within `CHANGE_THRESHOLD` of the last one *sent* is dropped. A static screen costs nothing; `KEEPALIVE` (30 s) forces a refresh so the frame does not age out behind newer audio.

`capture()` deliberately does **not** advance the change gate — `commit()` does, and only after the send succeeded. A frame that failed to send has not been seen by the model, and treating it as seen would leave a static screen skipped indefinitely: JARVIS blind while believing he can see.

Because frames never fall out of context on their own, `_build_config` sets `context_window_compression` with a sliding window. Without it an audio+video session exhausts its context in about two minutes.

Continuous vision is in `budget.FREE_BLOCKED` — a frame a second indefinitely is not something a free key survives, and unlike a search there is no cheaper fallback to degrade to.

`_build_config` injects a `[LIVE VISION — ON]` block into the system instruction while the stream is running. Without it the only thing that ever said the feed was on was the tool result that started it, so a reconnect mid-stream left the model believing it was blind — it went back to calling `screen_process` for frames it was already receiving, and announcing "let me look" before every answer.

**One-shot (`screen_process`)** — still the right thing when the stream is off, and it is the only way to look at the camera while streaming the screen. It cannot return an image through a function response, so it uses a **two-turn injection**:

- Turn 1: the tool captures, stashes bytes in `self._pending_vision`, and returns a `[VISION_ACTIVE]` instruction telling the model to say one filler sentence ("Looking at your screen now, sir").
- Turn 2: on the next `turn_complete`, `_receive_audio` base64-encodes the image (off-loop) and sends it as `inline_data` with the original question.
- A 4-second cooldown plus `_vision_busy` guards against the model re-calling the tool when it hears its own filler sentence through the speakers.

Five booleans (`_pending_vision`, `_vision_cam_active`, `_vision_close_pending`, `_vision_busy`, `_vision_last_time`) coordinate that across two turns — still the most fragile state machine in the codebase. When the stream is live for the requested source, `screen_process` short-circuits with `[VISION_LIVE]` and never enters it.

### How he sounds

Native-audio models generate speech directly rather than reading synthesised text, so three things control the voice and only one of them is a setting.

**The system prompt is the biggest lever.** The audio is generated *from the words*, so instructions about writing are instructions about delivery. `prompt.txt` previously carried three separate directions to be clipped — "No fluff", "Respond as fast as you can", "Speak immediately" — and the result sounded like a terminal. The `HOW YOU SOUND` section now directs contractions, varied sentence length, reacting before reporting, and spoken-form numbers ("about half past two", not "14:31"), with an explicit list of the tells that make an assistant sound synthetic.

**`enable_affective_dialog`** lets the model hear *how* something was said and answer in kind. It is not available on every model or API version and the rejection arrives at connect time, so `_affective_supported` drops it for the process and reconnects rather than retrying the same config forever — a flatter voice beats an assistant that will not start.

**`voice`** selects one of 30 prebuilt voices, auditionable in AI Studio. The default was `Charon`, which Google documents as *Informative* — right for a briefing, flat for a conversation.

`speech_config.language_code` is deliberately never set: native-audio models choose the output language from the conversation, and setting it explicitly is rejected rather than treated as a hint.

### Barge-in

**Off by default.** With no echo cancellation the microphone hears the speakers, so "someone is talking" and "JARVIS is talking" are not reliably distinguishable, and every false positive cuts him off mid-sentence. While off, the HUD's Interrupt button is the only thing that stops him — which is unambiguous.

`barge_in` in `config/api_keys.json` is one switch over the whole policy, because switching off any single part leaves the others running:

| `barge_in` | Server (`ActivityHandling`) | Mic while speaking | Local RMS check |
|---|---|---|---|
| `false` (default) | `NO_INTERRUPTION` | gated | disabled |
| `true` | `START_OF_ACTIVITY_INTERRUPTS` | open | enabled |

`NO_INTERRUPTION` rather than disabling activity detection: turn-taking still has to work, so the server must keep detecting when the user stops speaking — it just must not act on that mid-answer. This is the half that cannot be fixed client-side, since the server decides from audio we already sent.

The mic gate is not only about interruptions. Anything sent while he talks is his own voice, which the server transcribes as something the *user* said and folds into the conversation.

The rest of this section applies when `barge_in` is `true`. Two paths then stop him, and only one is authoritative.

**Server** — the Live API runs its own VAD over the mic stream we are already sending and sets `server_content.interrupted` when it stops generating because the user spoke. `_receive_audio` calls `interrupt(source="server")`, which drains `audio_in_queue`. This is the real barge-in.

**Local** (`core/barge_in.py`) — an RMS heuristic that stops playback without waiting for the round trip. Its error budget is lopsided: a false positive cuts JARVIS off for no reason, a miss costs a few hundred milliseconds until the server says so. It is tuned to be conservative, and it was previously the cause of JARVIS interrupting himself every ~15 seconds:

| | Was | Now |
|---|---|---|
| Threshold | fixed 1500 | `max(1500, floor × 2.5)`, floor measured continuously |
| Sustain | 3 blocks (190 ms) | 7 blocks (450 ms) |
| Cooldown | none | 2 s, shared with the server path |
| Before calibration | fired immediately | suppressed for 40 blocks |

The floor is an asymmetric estimator — `alpha_down` 0.30, `alpha_up` 0.01 — so it falls fast and rises slowly, tracking room noise while a burst of speech barely moves it. Learning only from *quiet* blocks does not work: a room whose bleed already exceeds the absolute threshold never produces one, which was exactly the failing case (measured bleed 1700–2700 against a threshold of 1500).

Warm-up exists because the floor starts as a guess. Measured, a steady bleed of 1800 fired at block 6, before the estimate had found the room; the local path is therefore suppressed until it has seen 40 blocks, counted across the session rather than per turn. Tuning: `barge_in_rms`, `barge_in_blocks`, `barge_in_margin` in config, `barge_in: false` to hard-gate the mic while he talks.

### Concurrent sends

Mic audio, video frames, injected text and tool responses are produced by four different coroutines, and the SDK writes straight to the websocket with no lock of its own. `self._send_lock` serialises all five `session.send_*` call sites.

---

## 4. Module map

| Path | Lines | Role |
|---|---:|---|
| `main.py` | 1110 | Session lifecycle, tool dispatch, background loops, briefing |
| `ui.py` | 3399 | Entire PyQt6 HUD: canvas animation, metrics, log, drop zone, overlays, camera, theming, autostart, shortcuts |
| `dashboard/server.py` | 990 | FastAPI: QR pairing, AES command channel, phone-mic WebSocket, file upload, firewall setup |
| `actions/*.py` | ~10 000 | 23 tool implementations, each carrying its own schema |
| `actions/tools.py` | 40 | Imports every tool module — importing it fills the registry |
| `actions/session_tools.py` | 245 | The five tools that need the live session (vision, streaming, camera, memory, shutdown) |
| `actions/vision_stream.py` | 285 | Continuous vision: capture handles, activity gate, change gate |
| `memory/memory_manager.py` | 295 | Load/save/trim `long_term.json`, `memory_txn()`, prompt formatting, session summaries |
| `memory/config_manager.py` | 58 | Backwards-compatible facade over `core.settings` |
| `core/registry.py` | 184 | Tool table, Gemini declarations, bounded executor, per-tool timeouts |
| `core/budget.py` | 185 | Free/paid tier policy, 429 cooldown, model tier → model id |
| `core/paths.py` | 49 | Every filesystem path in one place |
| `core/settings.py` | 156 | Typed, mtime-cached config + the model IDs |
| `core/log.py` | 147 | Logger tree: encoding-safe console, rotating file, HUD sink |
| `core/prompt.txt` | 2.9 KB | System prompt: identity, routing rules, language detection |
| `tests/` | 418 | Registry schema snapshot, timeout behaviour, dispatch integration |

The `core/` package used to also hold `llm_client.py`, `tts.py`, `stt.py` and `installer.py` — 1259 lines of pre-Live-API local-LLM code that nothing imported. Deleted in sprint 2; recover from git history if the Ollama path is ever wanted back.

### Shared foundation (added in sprint 2)

Three modules everything else now leans on. Import from these rather than re-deriving:

```python
from core.paths    import BASE_DIR, CONFIG_PATH, MEMORY_PATH, PROMPT_PATH, LOG_PATH
from core.settings import get_settings, get_api_key, FAST_MODEL, LITE_MODEL, LIVE_MODEL
from core.log      import get_logger

log = get_logger("vision")          # -> logger "markl.vision"
```

- `core/settings.py` parses `config/api_keys.json` once and re-reads only when the file's mtime changes, so a key edited in the UI is picked up without a restart. Unknown keys land in `Settings.extra` and survive a `save_settings()`.
- Model IDs live *only* here. `FAST_MODEL`, `LITE_MODEL`, `LIVE_MODEL`, `SUMMARY_MODEL`.
- `core/log.py` console output keeps the old `[Tag] message` look; the rotating file at `logs/markl.log` always keeps full DEBUG. `MARKL_LOG_LEVEL=DEBUG` raises console verbosity. `add_ui_sink()` mirrors WARNING+ into the HUD panel.

### The `actions/` contract

A tool is a decorated function that owns its schema:

```python
from core.registry import ToolContext, tool

@tool(
    name="weather_report",
    description="Gives the weather report to user",
    parameters={"type": "OBJECT",
                "properties": {"city": {"type": "STRING", "description": "City name"}},
                "required": ["city"]},
    timeout=30,
)
def weather_report_tool(params: dict, ctx: ToolContext) -> str:
    return weather_action(parameters=params, player=ctx.ui) or "Weather delivered."
```

Adding a tool is that block plus one line in `actions/tools.py`. `registry.declarations()` generates the Gemini schema from the same objects that hold the implementations, so a declaration cannot drift from its code — and `tests/test_registry.py` holds a snapshot of all 23 schemas to prove it hasn't.

`ToolContext` carries `ui` (for `write_log` / `show_content`), `speak` (injects text into the live session), `jarvis` (the session itself, for the four tools in `session_tools.py`) and `current_file` (the last file dropped on the HUD).

The underlying `actions/*` functions still take the old loose keyword signature — `(parameters, player=, response=, speak=, session_memory=)`, inconsistently across modules. The decorated wrapper absorbs that; nothing above it has to know.

**Timeouts.** Every tool declares one (`web_search` 60 s, `computer_settings` 30 s, `dev_agent` 600 s, `file_processor` 300 s for video work). On overrun `registry.run()` raises `ToolTimeout` whose message is already a speakable sentence, and `_execute_tool` returns it as the function response. Note that `asyncio.wait_for` cancels the *future*, not the thread — a genuinely wedged tool keeps a worker until it returns. `TOOL_EXECUTOR` is capped at 8 threads, so the blast radius is bounded, but eight simultaneous wedged tools will stall the ninth.

---

## 5. Memory model

SQLite at `memory/memory.db` — `facts`, `sessions`, `monitors`, `meta`. WAL mode, one process-wide connection behind an `RLock`.

`memory/store.py` is the storage; `memory/memory_manager.py` keeps the old function names (`load_memory`, `update_memory`, `memory_txn`, `format_memory_for_prompt`, …) so the eight call sites in `main.py`, `proactive.py` and `background_monitor.py` did not have to change. `load_memory()` still returns the old nested dict — it is a *view* rebuilt per call, not a file.

**The cap moved from storage to the prompt.** The old JSON store was hard-capped at 2200 characters because all of it went into the system instruction, and `_trim_to_limit` deleted the oldest entries past that. Identity facts are the oldest thing in most stores, so the user's own name was the first thing forgotten — forty note writes emptied `identity` completely. Now storage is unbounded, identity always goes into the prompt, and nothing is deleted to make room for anything.

**Retrieval.** `format_memory_for_prompt(memory, query)` renders identity in full, then the top-k most relevant other facts within `PROMPT_CHAR_BUDGET`. At connect there is no conversation to rank against, so `_build_config` seeds the query with the last session's summary — "what we were talking about last time" beats "whatever was edited most recently".

`recall_memory(query)` is the fix for memory being frozen for the whole session: the model can pull a fact mid-conversation instead of only at connect. `access_count` / `last_used` give a small ranking tiebreak, capped so a popular fact cannot hijack an unrelated query.

**Two scoring backends** (`memory/embeddings.py`):

| | Used when | Notes |
|---|---|---|
| Gemini `text-embedding-004` | paid mode, no quota cooldown | Cached per fact in the `embedding` column; cleared whenever a value changes so a vector cannot outlive its text |
| Weighted token overlap | free mode, cooldown, or any embedding failure | IDF-weighted, plus a small synonym table so "what do I do for work" reaches a fact stored as `current_job`. Entirely local |

The lexical path is not a degraded afterthought — on a free key it is the one that runs, so it carries a synonym table covering the vocabulary the six categories actually attract. Embeddings are the upgrade, not the baseline.

Migration from `memory/long_term.json` runs once, guarded by `schema_version` in `meta`. **The JSON is deliberately left on disk** — it is the user's only copy of facts about themselves, it is already gitignored, and renaming it would be the one irreversible step in the change.

### The old format, for reference

```json
{
  "identity":      {"name": {"value": "…", "updated": "2026-07-29"}},
  "preferences":   {…},
  "projects":      {…},
  "relationships": {…},
  "wishes":        {…},
  "notes":         {…},
  "sessions":      [{"date": "…", "summary": "…", "language": "…"}],
  "monitors":      {"slug": {"topic": "…", "last_hash": "…"}}
}
```

That shape is still what `load_memory()` hands back, so callers written against it keep working. The rest of the old behaviour is gone:

- The 2200-character cap and `_trim_to_limit` — replaced by prompt-side budgeting.
- Memory refreshing only on reconnect — `recall_memory` reads mid-session.
- The lost-update race in `update_memory` (load and save took the lock separately, so two concurrent saves dropped one) — writes are single SQLite statements now, and `test_concurrent_writes_do_not_lose_updates` pins it.

Still true:

- Written by the model calling `save_memory`, which routes to `update_memory()`.
- **Session memory**: at shutdown a flash call summarises the last 40 turns; at next startup `pop_last_session()` reads *and deletes* it so the briefing never repeats itself. `last_session_topic()` peeks without consuming, for seeding retrieval.
- **Monitors** live in the same store. `background_monitor.py` hashes the top DDG headline per topic and alerts only when the hash changes. Crypto/finance keywords are hard-blocked at the code level. It is the last caller of `memory_txn()`; new code should use the `store` functions directly.

---

## 6. Autonomy loops

Three background behaviours make it feel "alive". All three share the same gate: don't speak while JARVIS is speaking, and don't speak if the user spoke recently.

| Loop | Period | Gate | What it sends |
|---|---|---|---|
| `_run_system_monitor` | 10 s | not speaking, user silent 10 s | `[SYSTEM_ALERT]` when CPU/RAM/GPU/temp crosses a threshold |
| `_run_proactive_mode` | 60 s | user silent 15 min, 20 min cooldown | `[PROACTIVE_CHECK]` with a rotating focus (projects / wellbeing / interesting) |
| `_run_background_monitor` | 30 min (5 min initial delay) | user silent 30 s | Per-topic news alert when headline hash changes |

`ProactiveEngine` decides *when* and builds context; **Gemini decides what to say**. There are no hardcoded proactive lines.

---

## 7. Remote dashboard

`dashboard/server.py` runs FastAPI on port 8000 in the same process.

- Press **Remote Control** in the HUD → a 6-character one-time key is minted (10 min TTL) and rendered as a QR code pointing at `/auto-login?key=…`.
- The phone hits that URL, gets a bearer token in `sessionStorage` and a long-lived device token in `localStorage` for silent reconnect.
- Commands from the phone are **AES-256-CBC** encrypted client-side with a key derived as `SHA256(sessionKey ‖ "JARVIS-DASHBOARD-v1")` and pushed into `_command_queue`.
- `/ws/phone-audio` streams phone mic PCM into `_phone_audio_queue`; `_relay_phone_audio()` forwards it into the same `out_queue` as the PC mic and sets `_phone_active` to mute the PC mic.
- On first run it tries to open the firewall port — elevated `.bat` via `ShellExecuteW` on Windows, `osascript` on macOS, `ufw`/`firewalld`/`iptables` on Linux.

---

## 8. Configuration surface

`config/api_keys.json` (untracked, created at first run):

```json
{
  "gemini_api_key": "…",
  "os_system": "windows",
  "assistant_name": "JARVIS",
  "user_name": "",
  "morning_brief_enabled": true,
  "ui_color": "#…",
  "llm_url": "http://localhost:11434",
  "llm_model": "llama3.2",
  "llm_provider": "ollama"
}
```

The last three keys are only read by the dead `core/llm_client.py`.

Model IDs are hardcoded across 12 files: `gemini-2.5-flash` (general), `gemini-2.5-flash-lite` (cheap classification in `computer_settings`, `computer_control`, `flight_finder`), and `models/gemini-2.5-flash-native-audio-preview-12-2025` for the Live session. There is one uncommitted local edit changing the session-summary model to `gemini-3.6-flash` in `main.py:1239`.

---

## 9. Startup sequence

```
main()
 └─ JarvisUI("face.png")            → Qt app + MainWindow constructed
 └─ thread: ui.wait_for_api_key()   → blocks on SetupOverlay if unconfigured
    └─ asyncio.run(JarvisLive.run())
        ├─ DashboardServer().serve() as a task
        ├─ loop forever:
        │   ├─ _build_config()      → prompt + memory + identity + tools
        │   ├─ client.aio.live.connect(...)
        │   ├─ TaskGroup spawns the 8 coroutines
        │   └─ once per process: _send_startup_briefing()
        │        Phase 1 — instant greeting + "yesterday you were…" (no tools)
        │        Phase 2 — news, pre-fetched in parallel, delivered as text
        └─ on any error: classify → invalid key / network / other → backoff 3→60 s
 └─ ui.root.mainloop()              → Qt takes over the main thread
```

The two-phase briefing exists so the first word is spoken in under a second instead of waiting on a news round-trip.

### Local news

Phase 2 reads news for where the user actually is. `actions/location.py` resolves that in priority order — `location` in config, then `identity.city` in memory, then IP geolocation (cached 7 days in the store's `meta` table). The ordering is the privacy design as much as the correctness one: **if JARVIS already knows where you live, no request leaves the machine**, and `ip_geolocation: false` disables the lookup outright. The inferred city is deliberately *not* written into `identity` — memory is what the user said about themselves, and an IP guess is wrong often enough (VPN, mobile, corporate egress) to have no business overwriting that.

`_fetch_news_ladder` then walks progressively wider queries — `"Naugachhia Bihar news"` → `"Bihar news"` → `"India news"` → world — because somewhere small has no news of its own and a place name does not reveal whether it does. Two details that are load-bearing:

- **Local rungs are strict** (`_news(query, strict=True)`, no text-search fallback). Without it DDG answers a query about a town with the town's air-quality page, the ladder counts that as success and never widens, and the user gets websites read out as headlines.
- **The word "today" is omitted.** Measured: `"Naugachhia Bihar news today"` returns nothing while `"Naugachhia Bihar news"` finds real articles. Thin coverage cannot absorb the extra term.

World news is the last rung and is *not* strict — an empty briefing is worse than a slightly off-topic one. JARVIS names the place the winning query actually covered, so a fall-through says "Bihar" rather than claiming local news for a town.

---

## 10. Reading order for a newcomer

1. `main.py:99-262` — `JarvisLive.__init__` and `_build_config`. This is the whole contract with Gemini.
2. `main.py:441-590` — `_receive_audio`. Everything that happens to you, happens here.
3. `core/registry.py` — the tool table, then `main.py:264-317` for the dispatch that uses it.
4. `core/prompt.txt` — how the model is told to route.
5. Any one file in `actions/` — e.g. `web_search.py` (small, self-contained, shows the Gemini-grounded + DDG parallel race pattern, and ends with its own `@tool` block).
6. `memory/memory_manager.py` — small and complete.
7. `ui.py` last, and only the section you need.
