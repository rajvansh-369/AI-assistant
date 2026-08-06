# MARK L — Scaling & Optimization Plan

Concrete, ordered work to take this from a 16 k-line single-developer build to something that can absorb 50 more tools, multiple contributors, and lower latency per turn. Every item cites the file it touches. Read `ARCHITECTURE.md` first.

Ordering principle: **fix what leaks or corrupts → remove what blocks change → then make it fast → then make it bigger.**

---

## Sprint 1 outcome — Phase 0 shipped

All of Phase 0 is done and verified. Three things surfaced during the work that were not in the original plan:

- **The committed certificate was already broken.** Its SAN pinned `192.168.1.107`; this machine is now `192.168.1.2`, so HTTPS had a hostname mismatch on top of the leaked key. `_refresh_cert_if_stale()` now regenerates automatically when the LAN IP moves or the cert expires.
- **`_trim_to_limit` evicts identity first.** Entries are sorted by `updated`, and identity facts are usually the oldest — so when memory fills, *your name is the first thing forgotten*. Reproduced: 40 note writes on a fresh store left `identity` completely empty. Not fixed (it is a behaviour change, and §3.2 replaces this mechanism wholesale), but it should be a one-line pin of `identity` in the trim sort before then.
- **The emoji-in-`print` problem is real, not theoretical.** `print("[Memory] 💾 …")` raises `UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f4be'` on a cp1252 console — a hard crash inside `update_memory`, not a cosmetic issue. This raises §1.5 (real logging) from cleanup to correctness.

Verification harnesses live in the session scratchpad (`test_memory.py`, 18 checks; `test_dashboard.py`, 28 checks). They are plain scripts, not pytest — promote them to `tests/` in sprint 4.

---

## Sprint 7 outcome — JARVIS stopped interrupting himself

Reported symptom: `SYS: Interrupted — listening...` over and over. The log showed five interrupts in three minutes, each discarding 32-82 queued audio chunks — he was being cut off mid-sentence by his own voice coming back through the speakers.

Root cause was structural, not a tuning miss: **the client was doing VAD the server already does, and ignoring the server's answer.** `server_content.interrupted` was never read. The Live API runs its own VAD over the same mic stream and reports when it stopped generating because the user spoke; the local RMS check was a strictly worse duplicate whose only possible contribution over the server was false positives. That signal is now handled, and the local check is demoted to a fast path with a much higher bar. See ARCHITECTURE.md §3.

Two things the work turned up that are worth remembering:

- **The obvious adaptive fix does not work.** Learning the bleed floor from blocks *below* the threshold is the natural design, and it cannot fix this bug: a room whose bleed already exceeds the absolute threshold never produces such a block, so the floor never learns and every block still looks like speech. That is this exact case — measured bleed 1700-2700 against a threshold of 1500. It needs an asymmetric estimator that updates on every block, rising slowly enough that speech cannot drag it up.
- **A settled estimator is not enough; the cold start is where it fired.** With the floor fixed, a simulation of the user's own 1800 bleed still produced a false interrupt at block 6, before the estimate had found the room. Hence the 40-block warm-up, which is free because the server VAD covers the same audio.

The decision logic was pulled out into `core/barge_in.py` — as a closure inside the sounddevice callback it was untestable, and both bugs above were found by simulation rather than by reading. 22 tests, including a replay of the exact RMS values from the user's log.

Not verified live: the fix is proven against recorded values and simulation, not against a real session. Worth watching the log for `✋ Barge-in` lines that do not correspond to actually speaking.

---

## Sprint 6 outcome — local news briefing

The briefing reads news for where the user is rather than "top world news today". `actions/location.py` + a widening query ladder in `main.py`. See ARCHITECTURE.md §9 for the resolution order and why the local rungs are strict.

The feature was straightforward; testing it against the real machine was not, and turned up a dependency fault that had nothing to do with the request:

- **`duckduckgo-search` is dead and had been failing silently.** Every call to its news endpoint returned `403 Ratelimit`, so `_ddg_news` fell through to a plain *text* search on every single query — for months, presumably. World news still "worked" because a text search for "top world news today" happens to return news sites, which masked it completely. Local queries did not: a search for a town returned its air-quality page and a YouTube video. The package was renamed to `ddgs`; `web_search.py` already preferred it, so the fix was one line in `requirements.txt`. With `ddgs` installed, the news endpoint returns a real Times of India article about the user's actual town in 0.8 s.
- **A text-search fallback is wrong when emptiness is the answer.** The ladder needs "does this place have news" to be answerable, and the fallback made "no" indistinguishable from "here are some unrelated websites" — so it always stopped on the first rung. `_ddg_news` and `_news` grew an `allow_text_fallback` / `strict` flag; the ladder's local rungs pass it, the final world rung does not.
- **Query wording mattered more than expected.** `"Naugachhia Bihar news today"` → nothing; `"Naugachhia Bihar news"` → real articles. Dropping one word is the difference between local news working and not.

Worth noting for anyone re-running the manual checks: DDG rate-limits several searches in quick succession, so testing the ladder by hand produces false negatives. Two of the "failures" during this work were the test procedure, not the code.

---

## Sprint 5 outcome — memory v2 (§3.2)

SQLite (`memory/store.py`) + two-backend retrieval (`memory/embeddings.py`) + a `recall_memory` tool. `memory_manager.py` keeps every old function name, so none of the eight call sites changed.

What it fixes, in order of how much it mattered:

- **Identity is no longer the first thing forgotten.** The §1 note about `_trim_to_limit` sorting by `updated` was right and had never been fixed: identity facts are the oldest in most stores, so filling the 2200 characters emptied `identity` first. `test_identity_survives_a_flood_of_notes` reproduces the old failure — 40 note writes — and asserts the name survives both storage and prompt.
- **The cap moved to where it belonged.** It was a *prompt* budget being enforced as a *storage* limit. Storage is unbounded now; `format_memory_for_prompt` budgets.
- **Memory is no longer frozen at connect.** `recall_memory(query)` retrieves mid-session.
- **The lost-update race is gone** by construction — writes are single statements, not load-then-save.

Two decisions worth recording:

- **The lexical backend is not a fallback, it is the default here.** This install runs on a free key, and embeddings are in the degraded path, so token-overlap scoring is what actually executes. Testing it on the real store showed "what do I do for work" returning nothing against a fact stored as `current_job` — word matching cannot bridge that. It now carries an IDF weighting plus a small synonym table scoped to the six categories, with expanded terms weighted below terms the user actually said. Not an ontology; a stopgap sized to the problem, and documented as one.
- **`long_term.json` is left on disk after migration.** It is the user's only copy of facts about themselves. Migration is guarded by `schema_version` in `meta`, so the file is read once and then ignored — renaming or deleting it would have been the only irreversible step in the whole change.

Verified against the real store: 4 facts migrated, JSON intact, re-open idempotent, recall answering in the user's own phrasing.

Left for later: pruning by disuse. `access_count` and `last_used` are recorded and used for ranking, but nothing decays or deletes yet — with a store this size there is nothing to prune, and a forgetting policy is worth designing against real data rather than guessing.

---

## Sprint 4 outcome — continuous vision

`3.x`-adjacent, taken out of order because it is the change that most moves "assistant" toward "presence": JARVIS now *sees* rather than *looks*. `_send_video()` streams JPEG frames up the same websocket as the mic, so the screen is in context before the question finishes. `actions/vision_stream.py` + one tool (`vision_stream`) + `_send_lock`; the old `screen_process` path is untouched and still handles one-off looks and the camera-while-streaming-screen case.

Cost control is the whole design — see ARCHITECTURE.md §3. Two gates (activity, change) mean a static screen costs nothing, and `budget.FREE_BLOCKED` refuses it outright on a free key.

Three things worth recording:

- **A failed send must not advance the change gate.** The first version committed the frame inside `capture()`. A dropped send then left a static screen skipped forever — JARVIS blind while reporting he could see. Split into `capture()` / `commit()`, committing only after the send returns. `test_an_uncommitted_frame_is_offered_again` pins it; the bug was found by a test, not by reading.
- **Audio+video sessions need `context_window_compression`.** Frames never fall out of context on their own, so without a sliding window the session dies in ~2 minutes. Added to `_build_config`, which also helps long audio-only sessions.
- **The SDK does not lock the websocket.** `send_realtime_input`, `send_client_content` and `send_tool_response` all call `self._ws.send(...)` directly, and there are now four producers. `self._send_lock` wraps all five call sites. This was already latent before video; adding a second high-frequency producer made it worth fixing.

Also picked up here: 2.4's base64-off-the-event-loop fix, which was one line once the vision path was being touched anyway.

**Verified against a live session.** A headless harness (no Qt, no mic, no speaker) drives the real `_build_config` and `_send_video` against an actual socket and asks "what am I looking at right now?". Frames are accepted as video input and answered correctly. Two bugs only that run could find:

- **JARVIS spoke the internal tag.** First live answer was literally `"[VISION_ACTIVE] I am looking at your screen now. …"`. With frames arriving unprompted the model pattern-matched "I have an image" onto the one-shot handshake and read the control tag aloud. `prompt.txt` had per-tag "do not read this out" rules for `[BRIEFING]` and `[PROACTIVE_CHECK]` but no general one; it now has a `CONTROL TAGS` section covering all of them, and the VISION section distinguishes "you called screen_process" from "an image simply arrived".
- **Nothing told the model the feed was on.** It kept calling `screen_process` for a view it was already being sent. The `[VISION_LIVE]` short-circuit caught it — correctly, and confirmed live — but that still costs a wasted turn and a filler sentence. The fact only ever existed in the tool result that started the stream, so *any reconnect mid-stream lost it*. `_build_config` now injects a `[LIVE VISION — ON]` block whenever the stream is active. After that fix the same question returns `"You are currently looking at an IDE displaying a code project, sir."` with no tool call, no tag and no preamble.

Both are pinned by tests (`test_prompt_announces_a_running_feed` and friends), though the tag-leak class of bug is inherently only catchable live — worth re-running the harness after any `prompt.txt` edit.

---

## Sprint 3 outcome — the registry landed

1.3 (registry) and 2.3 (bounded executor + per-tool timeouts) are done, and 1.6 has a real start: 119 tests where there were zero.

All 23 tools were ported in one pass rather than 2–3 per sitting, and the old `if/elif` chain was deleted rather than kept as a fallback. That is a deviation from the plan below, and it was only safe because of the substitute safety net: `tests/golden_declarations.json` is the pre-refactor `TOOL_DECLARATIONS` literal extracted from the old file with `ast.literal_eval` — not retyped — and `tests/test_registry.py` asserts every registered tool's schema is byte-identical to it. A dead fallback branch nobody exercises proves less than a snapshot that fails on drift.

Numbers: `main.py` 1748 → 1110 lines (488-line declarations literal deleted, 188-line dispatch chain → 55).

Three things surfaced that were not in the plan:

- **`screen_process` could permanently disable vision.** `_vision_busy` is set True before the capture and released two turns later on `turn_complete`. If the capture itself raised — no webcam, mss failure — the flag was never cleared and *every subsequent vision request for the session* returned "Vision is still processing the previous request." The old code's `except` swallowed the error one level up, so it looked like a cooldown, not a crash. Now `try/except BaseException` clears the flag and re-raises (`session_tools.py`), and `BaseException` is deliberate — a timeout cancellation would otherwise leak the same way.
- **Timeouts are a behaviour change, not just a safety net.** Two are close enough to real work to matter: `file_processor` at 300 s can cut off a long video transcode, and `dev_agent` at 600 s can cut off a large project build. Both were unbounded before. If either bites, raise it in that module's `@tool` block — it is one number in one place now, which is the point.
- **`asyncio.wait_for` does not kill the thread.** It cancels the future; a wedged sync tool keeps its `TOOL_EXECUTOR` worker until it returns on its own. Eight of those starve the pool. Bounded blast radius rather than a fix — `test_a_hung_tool_does_not_block_the_event_loop` pins the part that actually matters, which is that audio keeps flowing.

Not done in this sprint: 1.4 (`ui/` split), and 1.6 beyond the registry — `memory_manager`, `background_monitor` and dashboard auth still have no tests.

---

## Phase 0 — Must fix before anything else — ✅ SHIPPED

### 0.1 A private TLS key is committed to git — ✅ done

`config/certs/jarvis.key` and `config/certs/jarvis.crt` are tracked (`git ls-files config/`). Anyone who has ever cloned or forked this repo holds the private key for the dashboard's HTTPS. Since the dashboard also carries phone microphone audio and AES-encrypted commands, that key is worth removing.

Fix:
```bash
git rm --cached config/certs/jarvis.key config/certs/jarvis.crt
# regenerate a fresh self-signed pair locally at first run instead of shipping one
```
Then generate the cert on demand in `DashboardServer.serve()` if `_ssl_enabled()` is false and the user wants HTTPS. Rotating means every previously-paired phone re-accepts the cert once — acceptable.

Note this only stops future exposure; the old key stays in git history. Treat it as compromised and never reuse it.

### 0.2 No `.gitignore` — ✅ done

`config/api_keys.json` (your Gemini key) and `memory/long_term.json` (your name, city, job, relationships, projects) are currently untracked only by luck. One `git add .` publishes both.

```gitignore
__pycache__/
*.py[cod]
config/api_keys.json
config/certs/
memory/long_term.json
uploads/
.venv/
```

Ship a `config/api_keys.example.json` instead.

### 0.3 Dashboard auth hardening — ✅ done

`dashboard/server.py` is reachable from every device on the LAN (and the firewall rule is opened automatically).

| Issue | Location | Fix |
|---|---|---|
| `/login` accepts unlimited attempts against a 6-char key | `server.py:481` | Per-IP counter, lock out after 5 failures for 60 s |
| `self._tokens` is a `set` that is never pruned | `server.py:373` | Store `{token: expiry}`, sweep on access, 12 h TTL |
| `_device_sessions` never expires | `server.py:382` | Same TTL treatment; `/api/revoke-devices` already exists but nothing calls it on a schedule |
| AES key = one round of `SHA256(pin ‖ salt)` | `server.py:76` | Use `hashlib.pbkdf2_hmac` or HKDF; a 6-char PIN behind one SHA-256 is brute-forceable offline if a ciphertext leaks |
| `download_file` takes the token as a query param | `server.py:711` | Acceptable tradeoff (browsers can't set headers on `<a download>`) but log it and use short-lived signed URLs if you scale this |

### 0.4 Lost-update race in memory writes — ✅ done

`update_memory()` (`memory_manager.py:111`) calls `load_memory()` (takes lock, releases) then `save_memory()` (takes lock again). Two tools saving concurrently — very possible, since tools run in a thread pool — will silently drop one write. `background_monitor._save()` has the same shape.

Fix: one re-entrant lock held across the whole read-modify-write, exposed as a context manager:

```python
@contextmanager
def memory_txn():
    with _lock:
        mem = _read_unlocked()
        yield mem
        _write_unlocked(mem)
```

### 0.5 Prompt references a tool that does not exist — ✅ done

`core/prompt.txt` routes to `agent_task` twice. There is no `agent_task` in `TOOL_DECLARATIONS`. The model either hallucinates the call (wasted turn) or ignores the rule. Either rename it to `dev_agent` or delete the lines.

---

## Phase 1 — Remove what blocks change

> **Sprint 2 shipped 1.1, 1.2 and 1.5.** 1259 dead lines deleted; `core/paths.py` + `core/settings.py` + `core/log.py` added; 240 `print()` calls migrated to loggers. 1.3 (registry), 1.4 (`ui/` split) and 1.6 (tests/CI) remain.

### 1.1 Delete the dead `core/` LLM stack — ✅ done

`core/llm_client.py` (586), `core/tts.py` (442), `core/stt.py` (93), `core/installer.py` (138) — **1259 lines that nothing imports.** Verified: no file outside `core/` references them. They are leftovers from the pre-Live-API era when the assistant ran on Ollama with local TTS/STT.

Either delete them, or — if you want the local-LLM fallback back as a feature — move them to `providers/ollama/` and wire them behind a real `llm_provider` switch. Leaving them in place costs every future reader ~20 minutes of confusion.

### 1.2 One config module, one path module — ✅ done

`get_base_dir()` is copy-pasted in **11 files**. `_get_api_key()` in **10 files**, in three different flavours (some raise, some return `""`, some read a cached dict).

Create `core/paths.py` and `core/settings.py`:

```python
# core/settings.py
@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    assistant_name: str = "JARVIS"
    user_name: str = ""
    morning_brief_enabled: bool = True
    live_model: str = "models/gemini-2.5-flash-native-audio-preview-12-2025"
    fast_model: str = "gemini-2.5-flash"
    lite_model: str = "gemini-2.5-flash-lite"

def get_settings() -> Settings: ...   # cached, invalidated on file mtime change
```

This also solves the fact that **model IDs are hardcoded in 12 files**. Today, upgrading the model means 12 edits and you *will* miss one — there is already a stray uncommitted `gemini-3.6-flash` in `main.py:1239` while everything else says `gemini-2.5-flash`. Pin model names in one place and let config override them.

### 1.3 Tool registry instead of a 20-branch `if/elif` — ✅ done

`main.py` currently holds a 448-line `TOOL_DECLARATIONS` literal (lines 100–548) and a 183-line `_execute_tool` chain (lines 706–889). Adding a tool means editing two places in a 1535-line file, and every tool has a slightly different call signature, which is why each branch needs its own lambda.

Target shape — each tool owns its own schema, colocated with its code:

```python
# core/registry.py
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[[dict, "ToolContext"], str]
    timeout: float = 30.0
    silent: bool = False

_REGISTRY: dict[str, Tool] = {}

def tool(name, description, parameters, **kw):
    def deco(fn):
        _REGISTRY[name] = Tool(name, description, parameters, fn, **kw)
        return fn
    return deco
```

```python
# actions/weather_report.py
@tool(name="weather_report",
      description="Gives the weather report to user",
      parameters={"type": "OBJECT", "properties": {"city": {...}}, "required": ["city"]})
def weather_report(params: dict, ctx: ToolContext) -> str:
    ...
```

`ToolContext` replaces the current grab-bag of `player=`, `response=`, `speak=`, `session_memory=` kwargs with one object carrying `ui`, `speak`, `settings`, `memory`, `current_file`.

Then `_execute_tool` collapses to roughly:

```python
async def _execute_tool(self, fc):
    tool = REGISTRY.get(fc.name)
    if tool is None:
        return fr(fc, f"Unknown tool: {fc.name}")
    try:
        result = await asyncio.wait_for(
            self._pool_run(tool.fn, dict(fc.args or {}), self.ctx),
            timeout=tool.timeout,
        )
    except asyncio.TimeoutError:
        result = f"{fc.name} timed out after {tool.timeout}s."
    except Exception as e:
        result = f"Tool '{fc.name}' failed: {e}"
        self.speak_error(fc.name, e)
    return fr(fc, result)
```

Wins: adding a tool is one decorated function; `TOOL_DECLARATIONS` is generated by iterating the registry; **per-tool timeouts become possible** (see 2.3); the plugin system on your roadmap becomes a `for entry_point in importlib.metadata.entry_points(group="markl.tools")` loop rather than a redesign.

Migrate incrementally: keep the old chain as a fallback for unregistered names, port 2–3 tools per sitting.

### 1.4 Split `ui.py`

3338 lines, 136 KB, one file, containing: an animated HUD canvas, system metric bars, a log widget with typewriter animation, a file drop zone, camera preview, four modal overlays, a hue color wheel, QR rendering, autostart registry edits, desktop `.lnk` creation, and theming. Any two people editing the UI collide on every commit.

```
ui/
  __init__.py        JarvisUI facade (public API unchanged)
  theme.py           C, qcol, apply_ui_accent, retheme_all_widgets, current_palette
  widgets/hud.py     HudCanvas
  widgets/metrics.py MetricBar, _SysMetrics
  widgets/log.py     LogWidget
  widgets/dropzone.py FileDropZone, _DropCanvas
  widgets/camera.py  _CameraPreview
  overlays/          SetupOverlay, CustomizeOverlay, RemoteKeyOverlay, ClipboardPanel
  window.py          MainWindow
  platform/          autostart, shortcut creation, icon generation
```

Keep `JarvisUI`'s public surface identical so `main.py` doesn't change.

### 1.5 Real logging — ✅ done

271 `print()` calls with emoji prefixes across the codebase. There is no way to raise the log level in production, no timestamps, no file output, and on Windows the emoji can throw `UnicodeEncodeError` on a non-UTF-8 console.

Replace with `logging` + a rotating file handler, keeping the existing tag convention as logger names (`markl.vision`, `markl.memory`, `markl.dashboard`). A single `core/log.py` and a sed pass gets 90% of it. Route `WARNING`+ to the HUD log panel automatically instead of hand-calling `ui.write_log` everywhere.

### 1.6 Tests and CI (there are currently zero)

Highest value per line of test, in order:

1. `memory_manager` — trim at the 2200-char boundary, unicode truncation, concurrent `update_memory` (this is where the race in 0.4 gets proven).
2. `background_monitor` — `_is_blocked` across languages, `_slug` collisions, hash-change alerting fires exactly once.
3. Tool registry — every registered tool's `parameters` is valid schema; every declared name has an implementation (would have caught the `agent_task` drift in 0.5).
4. `dashboard` auth — `TestClient` over expired keys, reused one-time keys, missing tokens.
5. `_clean_transcript`, `ProactiveEngine.should_trigger` — pure functions, trivial to cover.

GitHub Actions: `ruff check` + `pytest` on push. `requirements.txt` currently pins **nothing** — 29 unpinned packages means a fresh clone six months from now is a different program. Move to `pyproject.toml` with a lockfile, and keep `requirements.txt` generated from it.

---

## Phase 2 — Latency and cost

Voice agents live or die on time-to-first-word. Measure before you tune: add a `perf` logger that records `tool_call → response` per tool and `user_speech_end → first_audio_byte` per turn, and put a rolling p50/p95 on the HUD.

### 2.1 Stop constructing a `genai.Client` per call

`genai.Client(api_key=…)` is instantiated inside the function body in `web_search.py:24` and `:125`, `code_helper.py:28` and `:462`, `computer_control.py:330`, `computer_settings.py:592`, `desktop.py:107`, `dev_agent.py:29`, `file_processor.py:36`, `flight_finder.py:66` and `:160`, `youtube_video.py:173`, `screen_processor.py:257`, `main.py:1236`.

Each construction builds a fresh HTTP connection pool and re-does TLS on first use. For a tool that runs on every search, that is avoidable latency on the critical path.

```python
# core/gemini.py
@lru_cache(maxsize=1)
def client() -> genai.Client:
    return genai.Client(api_key=get_settings().gemini_api_key)
```

Invalidate the cache when the API key changes (`ui.prompt_reconfig` path).

### 2.2 Cache memory reads

`load_memory()` does a full file read + JSON parse. It is called on every proactive trigger, every background-monitor pass, every `save_memory`, and inside `update_memory` — plus a read-modify-write on every save. Cache the parsed dict keyed on `MEMORY_PATH.stat().st_mtime_ns`, invalidate on write. Small win in isolation, but it removes disk I/O from the tool-dispatch path.

### 2.3 Bound the executor and time out tools — ✅ done

Two related problems:

- `run_in_executor(None, …)` uses the default pool, which on Python 3.11+ sizes to `min(32, cpu_count + 4)`. A tool that hangs (`browser_control` waiting on a Playwright selector, `game_updater` waiting on Steam, `flight_finder` scraping) holds a thread indefinitely.
- **Nothing times out.** Gemini sits waiting for a function response that may never come, and the user hears silence with no recovery path.

Fix both in the registry (1.3): a dedicated `ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool")` and `asyncio.wait_for` per tool with a sensible per-tool default (`web_search` 15 s, `dev_agent` 120 s, `computer_settings` 5 s). A timeout should return a *speakable* string so the model can tell the user rather than hanging.

### 2.4 Move CPU work off the event loop

`_receive_audio` base64-encodes the full screenshot inline (`main.py:1001`) on the asyncio thread. A 1280×720 JPEG is small, but this runs in the same coroutine that drains audio — any stall here stutters playback. `await asyncio.to_thread(b64encode, img_b)`.

Same pattern: `DashboardServer.broadcast` (`server.py:437`) awaits each client sequentially. With one phone it's irrelevant; with three it's three serial round-trips inside the audio loop's thread. `asyncio.gather(*(ws.send_json(msg) for ws in clients), return_exceptions=True)`.

### 2.5 Persist the session-resumption handle

`_build_config` already passes `types.SessionResumptionConfig()`, but the returned handle is never captured or reused. That means every reconnect (backoff 3→60 s on any network blip) starts a **completely fresh conversation** — the model loses everything said in the last few minutes and only gets back what was in `long_term.json`. Capture the handle from `session_resumption_update` messages, store it, and pass it on reconnect. This is the single biggest continuity improvement available, and it directly serves the "never fully left" goal in the README.

### 2.6 Route by cost

Three tools already use `gemini-2.5-flash-lite` for classification; the rest use full flash. Audit: session summarization, language detection, desktop task parsing, and YouTube query extraction are all lite-tier work. Every model choice should come from `Settings` (1.2) so this is one config edit, not a code change.

### 2.7 Trim the system prompt

Every reconnect re-sends: the full `prompt.txt`, the identity block, the time block, formatted memory (up to 2000 chars), **and all 23 tool declarations (~7 KB of JSON)**. That is the prefill cost of every single session. Two levers:

- Tool descriptions are verbose and partly redundant with `prompt.txt` routing rules. Pick one place to state routing.
- Consider a two-tier tool set: always-loaded core tools, plus a `load_toolset(domain)` meta-tool that pulls in the rarely-used ones (`game_updater`, `flight_finder`, `dev_agent`) on demand. Worth it once you pass ~30 tools.

---

## Phase 3 — Making it bigger

### 3.1 Plugin system (README roadmap LI+)

Once 1.3 exists, this is small. Two loading paths:

- **Entry points** — `[project.entry-points."markl.tools"]` in a third-party package's `pyproject.toml`. Installed with pip, discovered automatically.
- **Drop-in folder** — `~/.markl/plugins/*.py`, imported at startup, same `@tool` decorator.

What it needs beyond loading: a manifest with `name`/`version`/`requires`, a capability declaration (`filesystem`, `network`, `input_control`) surfaced in the UI before first use, and failure isolation so one bad plugin logs and unregisters rather than killing the session.

### 3.2 Memory that scales past 2200 characters — ✅ done (sprint 5)

The current hard cap exists because everything is stuffed into the system prompt. It means MARK L structurally cannot remember more than about a page about you, and old facts get silently evicted by `_trim_to_limit`.

Next level:

- Keep the full store (SQLite, not JSON — you get atomic writes, which also solves 0.4).
- Add embeddings per entry (`text-embedding-004`) in a `memories` table.
- At session start, inject only *stable identity* facts (name, language, city) plus the **top-k entries semantically closest to the recent conversation**.
- Add a `recall_memory(query)` tool so the model can pull facts mid-session instead of only at connect time — this also fixes the current limitation that memory is frozen for the whole session.
- Keep an `access_count`/`last_used` column and decay unused entries rather than evicting by age.

This is the change that most moves "assistant" toward "presence."

### 3.3 Structured tool results

Every tool returns a prose string today, so the UI can only `show_content(label, text)`. If tools returned `ToolResult(speech: str, display: dict | None, artifacts: list[Path])`, the HUD could render weather as a card, flights as a table, and system status as gauges — and the model would get a shorter speech string, cutting tokens. The registry (1.3) is the natural place to introduce this without touching all 20 tools at once: allow either `str` or `ToolResult`.

### 3.4 Cross-platform reality check

The README claims Windows/macOS/Linux. The code is Windows-first: the `Popen` monkeypatch at `main.py:6`, `pycaw`/`comtypes`/`pywinauto`/`win10toast` deps, registry autostart, `.lnk` creation, `netsh` firewall rules. Before advertising parity, add a `platform/` abstraction with an explicit capability matrix per OS, and a CI job that at minimum *imports* every module on macOS and Ubuntu runners. Right now an import error on Linux is only discovered by a user.

### 3.5 Observability

You cannot optimize a voice agent you can't measure. Minimum: per-turn timings (speech-end → first audio, tool duration, reconnect count), token counts per session, and error rates per tool, written to a local SQLite file with a small stats view in the HUD. This also tells you which of Phase 2's items actually mattered on your hardware.

---

## Suggested sequencing

| Sprint | Contents | Why here |
|---|---|---|
| ~~1~~ | ~~0.1–0.5~~ **shipped** | Leaks and corruption first |
| ~~2~~ | ~~1.1, 1.2, 1.5~~ **shipped** | Small, mechanical, unblocks everything else |
| ~~3~~ | ~~1.3 registry + 2.3 timeouts, all 23 tools~~ **shipped** | The keystone refactor |
| 4 | 1.6 rest of tests + CI, pin `requirements.txt` | Lock in the refactor |
| 5 | 2.1, 2.2, 2.4 (client reuse, caching, off-loop work) — **after** the perf logger in §2 | Measurable latency work. 2.5 shipped separately |
| 6 | 1.4 `ui/` split | Big diff, no behaviour change — do it when the branch is otherwise quiet |
| 7 | 3.1 plugins, 3.2 memory v2 | The actual "next level" |

2.5 (session resumption) shipped outside this sequence: the handle is captured at `main.py:1091` and replayed in `_build_config`.

Two rules that make this survivable in a codebase with no tests yet: **land Phase 0 and 1.2 before touching anything else**, and **never combine a refactor commit with a behaviour change** — for the registry migration in particular, port tools one at a time with the old chain still in place as a fallback.
