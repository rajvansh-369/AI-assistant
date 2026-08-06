"""Build a webpage and let the user watch it being typed.

`code_helper` can already write an HTML file, but the user only ever sees the
finished artifact on disk — the interesting part (the page taking shape) is
invisible.  This module keeps the same one-file-HTML output and adds a live
view: a tiny local HTTP server streams the model's output to a browser tab as
it is generated, and the tab types the code out character by character next to
a preview iframe that re-renders while the markup grows.

Flow:

    web_builder({"description": "a portfolio page for a photographer"})
        1. pick a save path on the Desktop
        2. start a loopback HTTP server on an ephemeral port
        3. open the browser at it  ← the tool returns here, within a second
        4. stream Gemini into the server in the background; the page types it
        5. write the finished HTML to disk

The tool returns as soon as the tab is open, so the assistant can say "watch
it, sir" while the writing is still happening.  Generation continues on a
daemon thread; the server shuts itself down after SERVER_TTL so a long session
does not accumulate listeners.
"""
from __future__ import annotations

import json
import re
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from core import budget
from core.log import get_logger
from core.settings import get_api_key as _get_api_key

log = get_logger("web")

OUT_DIR     = Path.home() / "Desktop" / "JarvisWeb"
SERVER_TTL  = 1800.0   #: seconds a viewer stays reachable after it is opened
MAX_SERVERS = 3        #: older viewers are closed when this many are alive
GEN_TIMEOUT = 180.0    #: give up on a stalled stream rather than typing forever


# ── generated-code plumbing ───────────────────────────────────────────────────

class _FenceStripper:
    """Remove markdown fences from a stream that arrives in arbitrary chunks.

    The opening fence can only be recognised once enough text has arrived, and
    the closing fence only at the very end — so a few characters are always held
    back and released by `flush()`.
    """

    def __init__(self) -> None:
        self._started = False
        self._buf     = ""

    def feed(self, chunk: str) -> str:
        self._buf += chunk

        if not self._started:
            m = re.match(r"^\s*```[a-zA-Z]*\n", self._buf)
            if m:
                self._buf = self._buf[m.end():]
                self._started = True
            elif "\n" in self._buf or len(self._buf) > 24:
                self._started = True   # no fence — emit as-is from here on
            else:
                return ""              # too early to tell

        # Hold back the tail so a closing "```" is never emitted as content.
        keep = 4
        if len(self._buf) <= keep:
            return ""
        out, self._buf = self._buf[:-keep], self._buf[-keep:]
        return out

    def flush(self) -> str:
        out, self._buf = re.sub(r"\n?```\s*$", "", self._buf), ""
        return out


class _Build:
    """The text produced so far, plus a way to wait for more of it."""

    def __init__(self, title: str, save_path: Path) -> None:
        self.title     = title
        self.save_path = save_path
        self.text      = ""
        self.done      = False
        self.error     = ""
        self._cv       = threading.Condition()

    def append(self, chunk: str) -> None:
        if not chunk:
            return
        with self._cv:
            self.text += chunk
            self._cv.notify_all()

    def finish(self, error: str = "") -> None:
        with self._cv:
            self.error = error
            self.done  = True
            self._cv.notify_all()

    def follow(self, timeout: float = GEN_TIMEOUT):
        """Yield the text so far, then every later addition, then stop.

        A late subscriber (the browser connects after generation started, or
        the user reloads the tab) gets the whole buffer first, so the view is
        always complete.
        """
        sent     = 0
        deadline = time.monotonic() + timeout
        while True:
            with self._cv:
                while len(self.text) == sent and not self.done:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self.error = self.error or "Generation timed out."
                        self.done  = True
                        break
                    self._cv.wait(min(remaining, 1.0))
                chunk = self.text[sent:]
                sent  = len(self.text)
                done, error = self.done, self.error
            if chunk:
                yield {"t": chunk}
            if done:
                yield {"end": True, "error": error, "path": str(self.save_path)}
                return


# ── model ─────────────────────────────────────────────────────────────────────

_PROMPT = """You are an expert front-end developer.
Build a complete, single-file HTML page for the request below.

Rules:
- Output ONLY the HTML. No explanation, no markdown, no backticks.
- One file: put all CSS in a <style> tag and all JS in a <script> tag.
- No external assets — no CDN links, no web fonts, no remote images. Use
  system font stacks, CSS gradients, inline SVG and emoji instead.
- Responsive, keyboard-accessible, works in both light and dark browsers.
- Real content, not lorem ipsum. Make it look designed, not templated.
{style_line}
Request: {description}

HTML:"""


def _generate(build: _Build, description: str, style: str) -> None:
    """Stream the model into `build`.  Runs on a daemon thread."""
    fence = _FenceStripper()
    try:
        from google import genai

        client = genai.Client(api_key=_get_api_key())
        prompt = _PROMPT.format(
            description=description,
            style_line=f"- Visual direction: {style}\n" if style else "",
        )
        budget.reserve()
        stream = client.models.generate_content_stream(
            model=budget.model("fast"), contents=prompt
        )
        for part in stream:
            text = getattr(part, "text", None)
            if text:
                build.append(fence.feed(text))
        build.append(fence.flush())

        html = build.text.strip()
        if not html:
            build.finish("The model returned nothing.")
            return

        build.save_path.parent.mkdir(parents=True, exist_ok=True)
        build.save_path.write_text(html, encoding="utf-8")
        log.info(f"✅ Page written: {build.save_path}  ({len(html):,} bytes)")
        build.finish()

    except Exception as e:
        budget.report(e)
        log.error(f"Page generation failed: {e}")
        build.append(fence.flush())
        build.finish(str(e))


# ── viewer ────────────────────────────────────────────────────────────────────

_VIEWER = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ — building</title>
<style>
  :root {
    --bg:#0b0f14; --panel:#0e141b; --line:#1c2733; --fg:#c8d6e5;
    --dim:#5b7085; --accent:#39d0ff; --tag:#ff7b93; --attr:#ffcf70;
    --str:#8fe388; --com:#4d6070; --kw:#c792ea;
  }
  * { box-sizing:border-box; }
  html,body { height:100%; margin:0; }
  body {
    background:var(--bg); color:var(--fg); display:flex; flex-direction:column;
    font-family:ui-sans-serif,system-ui,"Segoe UI",sans-serif;
  }
  header {
    display:flex; align-items:center; gap:12px; padding:10px 16px;
    border-bottom:1px solid var(--line); background:var(--panel); flex:0 0 auto;
  }
  .dot { width:10px; height:10px; border-radius:50%; background:var(--accent);
         box-shadow:0 0 12px var(--accent); animation:pulse 1.2s infinite; }
  @keyframes pulse { 50% { opacity:.25; } }
  .name { font-weight:600; letter-spacing:.02em; }
  .meta { color:var(--dim); font-size:13px; margin-left:auto;
          font-variant-numeric:tabular-nums; }
  main { flex:1 1 auto; display:flex; min-height:0; }
  section { flex:1 1 50%; min-width:0; display:flex; flex-direction:column; }
  section + section { border-left:1px solid var(--line); }
  .bar { padding:6px 14px; font-size:12px; letter-spacing:.08em;
         text-transform:uppercase; color:var(--dim);
         border-bottom:1px solid var(--line); background:var(--panel); }
  #code {
    flex:1 1 auto; overflow:auto; margin:0; padding:14px 16px 40vh;
    font-family:ui-monospace,"Cascadia Code",Consolas,monospace;
    font-size:13px; line-height:1.55; white-space:pre-wrap; word-break:break-word;
    tab-size:2;
  }
  #code .t { color:var(--tag); }
  #code .a { color:var(--attr); }
  #code .s { color:var(--str); }
  #code .c { color:var(--com); font-style:italic; }
  #code .k { color:var(--kw); }
  .caret { display:inline-block; width:8px; height:1.05em; background:var(--accent);
           vertical-align:text-bottom; animation:blink .9s steps(1) infinite; }
  @keyframes blink { 50% { opacity:0; } }
  #preview { flex:1 1 auto; border:0; width:100%; background:#fff; }
  footer { padding:8px 16px; border-top:1px solid var(--line); background:var(--panel);
           font-size:12px; color:var(--dim); flex:0 0 auto; }
  footer a { color:var(--accent); text-decoration:none; }
  .err { color:#ff6b6b; }
  @media (max-width:860px) { main { flex-direction:column; }
    section + section { border-left:0; border-top:1px solid var(--line); } }
</style>
</head>
<body>
<header>
  <span class="dot" id="dot"></span>
  <span class="name">__TITLE__</span>
  <span class="meta" id="meta">connecting…</span>
</header>
<main>
  <section>
    <div class="bar">source</div>
    <pre id="code"></pre>
  </section>
  <section>
    <div class="bar">live preview</div>
    <iframe id="preview" sandbox="allow-scripts"></iframe>
  </section>
</main>
<footer id="foot">Saving to <code>__PATH__</code> — <a href="/page" target="_blank">open the finished page</a></footer>

<script>
(function () {
  var target = "", shown = 0, finished = false, savedPath = "";
  var codeEl = document.getElementById("code");
  var metaEl = document.getElementById("meta");
  var footEl = document.getElementById("foot");
  var dotEl  = document.getElementById("dot");
  var frame  = document.getElementById("preview");

  function esc(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // One pass, not a chain of .replace() calls: a chain re-scans the markup it
  // just inserted, so the attribute rule ends up highlighting the class="k" of
  // an earlier span. Alternation order is the precedence — a "//" inside a
  // string is a string, not a comment.
  var TOKEN = new RegExp(
    '(&lt;!--[\\s\\S]*?(?:--&gt;|$))' +          // 1 html comment
    '|("[^"\\n]*"|\'[^\'\\n]*\')' +               // 2 string
    '|(\\/\\*[\\s\\S]*?(?:\\*\\/|$)|\\/\\/[^\\n]*)' + // 3 css/js comment
    '|(&lt;\\/?[a-zA-Z][\\w-]*)' +                // 4 tag
    '|\\b(function|const|let|var|return|if|else|for|while|class|new|await|async)\\b' + // 5 keyword
    '|([A-Za-z_:][\\w:.-]*)(?==")',               // 6 attribute name
    'g');

  function highlight(src) {
    return esc(src).replace(TOKEN, function (m, htmlComment, str, comment, tag, kw) {
      var cls = htmlComment || comment ? "c"
              : str ? "s"
              : tag ? "t"
              : kw  ? "k"
              : "a";
      return '<span class="' + cls + '">' + m + '</span>';
    });
  }

  var lastPaint = 0, lastPreview = 0;
  function paint(now) {
    requestAnimationFrame(paint);

    // Catch up faster when the model outruns the typing, so the view never
    // falls minutes behind the stream.
    var backlog = target.length - shown;
    if (backlog > 0) {
      shown += Math.max(2, Math.ceil(backlog / 45));
      if (shown > target.length) shown = target.length;
    }

    if (now - lastPaint > 40 && (backlog > 0 || !codeEl.dataset.done)) {
      lastPaint = now;
      var atBottom = codeEl.scrollHeight - codeEl.scrollTop - codeEl.clientHeight < 120;
      codeEl.innerHTML = highlight(target.slice(0, shown)) +
                         (finished && shown >= target.length ? "" : '<span class="caret"></span>');
      if (atBottom) codeEl.scrollTop = codeEl.scrollHeight;
      metaEl.textContent = shown.toLocaleString() + " chars" +
                           (finished && shown >= target.length ? " · done" : " · writing…");
      if (finished && shown >= target.length) {
        codeEl.dataset.done = "1";
        dotEl.style.animation = "none";
        dotEl.style.background = "#8fe388";
        dotEl.style.boxShadow = "0 0 12px #8fe388";
      }
    }

    if (now - lastPreview > 350) {
      lastPreview = now;
      var src = target.slice(0, shown);
      if (src && src !== frame.dataset.src) {
        frame.dataset.src = src;
        frame.srcdoc = src;
      }
    }
  }
  requestAnimationFrame(paint);

  var es = new EventSource("/stream");
  es.onmessage = function (ev) {
    var m = JSON.parse(ev.data);
    if (m.t) target += m.t;
    if (m.end) {
      finished = true;
      savedPath = m.path || "";
      es.close();
      if (m.error) {
        footEl.innerHTML = '<span class="err">Failed: ' + esc(m.error) + '</span>';
        dotEl.style.background = "#ff6b6b";
        dotEl.style.boxShadow = "0 0 12px #ff6b6b";
      } else {
        footEl.innerHTML = 'Saved to <code>' + esc(savedPath) + '</code> — ' +
                           '<a href="/page" target="_blank">open the finished page</a>';
      }
    }
  };
  es.onerror = function () { metaEl.textContent = "stream lost"; };
})();
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    build: _Build   # set on the server, mirrored here by _serve()

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):     # keep the assistant's log readable
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        build = self.server.build           # type: ignore[attr-defined]
        path  = self.path.split("?", 1)[0]

        if path in ("/", "/index.html"):
            page = (_VIEWER
                    .replace("__TITLE__", _escape(build.title))
                    .replace("__PATH__", _escape(str(build.save_path))))
            self._send(page.encode("utf-8"), "text/html; charset=utf-8")

        elif path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                for event in build.follow():
                    payload = json.dumps(event, ensure_ascii=False)
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass   # the tab was closed — nothing to recover

        elif path == "/page":
            self._send(build.text.encode("utf-8"), "text/html; charset=utf-8")

        else:
            self._send(b"Not found", "text/plain; charset=utf-8", 404)


def _escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))


_servers: list[ThreadingHTTPServer] = []
_servers_lock = threading.Lock()


def _serve(build: _Build) -> int:
    """Start a viewer server for `build` and return its port."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    server.build = build                    # type: ignore[attr-defined]

    threading.Thread(target=server.serve_forever, daemon=True,
                     name="webviewer").start()

    # Daemon, otherwise a pending TTL timer keeps the whole assistant alive for
    # half an hour after the user asks it to shut down.
    reaper = threading.Timer(SERVER_TTL, lambda: _stop(server))
    reaper.daemon = True
    reaper.start()

    with _servers_lock:
        _servers.append(server)
        while len(_servers) > MAX_SERVERS:
            _stop(_servers[0], locked=True)

    return server.server_address[1]


def _stop(server: ThreadingHTTPServer, locked: bool = False) -> None:
    def drop():
        if server in _servers:
            _servers.remove(server)
    try:
        server.shutdown()
        server.server_close()
    except Exception:
        pass
    if locked:
        drop()
    else:
        with _servers_lock:
            drop()


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "page").strip().lower()).strip("_")
    return (s[:40] or "page")


# ── entry point ───────────────────────────────────────────────────────────────

def web_builder(
    parameters: dict,
    response=None,
    player=None,
    session_memory=None,
    speak=None,
) -> str:
    """Create a webpage and show it being typed in the browser.

    parameters:
        description : What the page should be (required)
        page_name   : Short name for the file / viewer title
        style       : Visual direction, e.g. "dark neon", "minimal editorial"
        output_path : Full path or filename to save to (default: Desktop/JarvisWeb)
        open_view   : Set false to skip opening the browser (default: true)
    """
    p           = parameters or {}
    description = (p.get("description") or "").strip()
    page_name   = (p.get("page_name") or "").strip()
    style       = (p.get("style") or "").strip()
    output_path = (p.get("output_path") or "").strip()
    open_view   = p.get("open_view", True)

    if not description:
        return "Please tell me what the page should be about, sir."

    title = page_name or description[:48]
    if output_path:
        path = Path(output_path)
        if not path.is_absolute():
            path = OUT_DIR / path
        if path.suffix.lower() not in (".html", ".htm"):
            path = path.with_suffix(".html")
    else:
        path = OUT_DIR / f"{_slug(page_name or description)}.html"

    build = _Build(title=title, save_path=path)

    try:
        port = _serve(build)
    except Exception as e:
        log.error(f"Could not start the viewer server: {e}")
        return f"Could not start the live view: {e}"

    url = f"http://127.0.0.1:{port}/"

    threading.Thread(
        target=_generate, args=(build, description, style),
        daemon=True, name="webgen",
    ).start()

    if player:
        player.write_log(f"[Web] Writing page — live at {url}")

    if open_view:
        try:
            webbrowser.open(url)
        except Exception as e:
            log.error(f"Could not open the browser: {e}")
            return (f"I am writing the page, sir, but I could not open the browser. "
                    f"Watch it at {url}. It will be saved to {path}.")

    return (
        f"Writing the page now, sir — you can watch the code being typed at {url}. "
        f"It will be saved to {path} when it is done."
    )


# ── Tool registration ─────────────────────────────────────────────────────────
# Imported at the bottom so the schema sits next to the implementation without
# reordering the module.  Importing this file registers the tool; main.py only
# has to import actions.tools.
from core.registry import ToolContext, tool  # noqa: E402


@tool(
    name="web_builder",
    description=(
        "Creates a webpage / website / HTML page and shows it being written LIVE: "
        "a browser tab opens immediately and types the code out on screen, "
        "character by character, next to a preview that renders as it grows. "
        "Use this for ANY request to make, build, design or create a web page, "
        "landing page, portfolio site, form page, or HTML page — prefer it over "
        "code_helper whenever the output is a webpage. "
        "Returns as soon as the tab is open; the writing continues on screen, "
        "so tell the user they can watch it."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "description": {
                "type": "STRING",
                "description": "What the page should be — content, sections, purpose. Be specific."
            },
            "page_name": {
                "type": "STRING",
                "description": "Short name for the file and the viewer title (e.g. 'portfolio')"
            },
            "style": {
                "type": "STRING",
                "description": "Visual direction, e.g. 'dark neon', 'minimal editorial', 'playful'"
            },
            "output_path": {
                "type": "STRING",
                "description": "Where to save it. Default: Desktop/JarvisWeb/<name>.html"
            },
            "open_view": {
                "type": "BOOLEAN",
                "description": "Open the live typing view in the browser (default: true)"
            },
        },
        "required": ["description"]
    },
    timeout=120,
)
def web_builder_tool(params: dict, ctx: ToolContext) -> str:
    r = web_builder(parameters=params, player=ctx.ui, speak=ctx.speak)
    return r or "Done."
