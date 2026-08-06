#web_search.py
import json
import sys
from pathlib import Path

from core import budget
from core.log import get_logger
from core.settings import get_api_key as _get_api_key

log = get_logger("websearch")


def _gemini_search(query: str) -> str:
    """Grounded search. Raises in free mode / during a cooldown — callers all
    have a DuckDuckGo fallback, which costs nothing and needs no quota."""
    from google import genai

    budget.reserve("grounded_search")

    client = genai.Client(api_key=_get_api_key())
    try:
        response = client.models.generate_content(
            model=budget.model("fast"),
            contents=query,
            config={"tools": [{"google_search": {}}]},
        )
    except Exception as e:
        budget.report(e)
        raise

    text = ""
    for part in response.candidates[0].content.parts:
        if hasattr(part, "text") and part.text:
            text += part.text

    text = text.strip()
    if not text:
        raise ValueError("Gemini returned an empty response.")
    return text


def _ddg_search(query: str, max_results: int = 6) -> list[dict]:
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title":   r.get("title",  ""),
                "snippet": r.get("body",   ""),
                "url":     r.get("href",   ""),
            })
    return results


def _ddg_news(query: str, max_results: int = 8,
              allow_text_fallback: bool = True) -> list[dict]:
    """DDG news search — returns actual articles, not website homepages.

    The text-search fallback exists so a flaky news endpoint still returns
    *something*, but it returns web pages rather than headlines — an air-quality
    page and a YouTube video, for a query about a town's news. Callers that are
    deciding whether a place has news at all must pass
    `allow_text_fallback=False`, or "no news here" is indistinguishable from
    "here are some unrelated websites".
    """
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.news(query, max_results=max_results):
                results.append({
                    "title":   r.get("title",  ""),
                    "snippet": r.get("body",   ""),
                    "url":     r.get("url",    ""),
                    "source":  r.get("source", ""),
                })
    except Exception as e:
        if not allow_text_fallback:
            log.info(f"No news results for {query!r} ({e})")
            return []
        log.error(f"⚠️ DDG news() failed ({e}) — falling back to text search")
        results = _ddg_search(query, max_results=max_results)
    return results


def _format_ddg(query: str, results: list[dict]) -> str:
    if not results:
        return f"No results found for: {query}"

    lines = [f"Search results for: {query}\n"]
    for i, r in enumerate(results, 1):
        if r.get("title"):   lines.append(f"{i}. {r['title']}")
        if r.get("snippet"): lines.append(f"   {r['snippet']}")
        if r.get("url"):     lines.append(f"   Source: {r['url']}")
        lines.append("")
    return "\n".join(lines).strip()


def _format_news(query: str, results: list[dict]) -> str:
    if not results:
        return f"No news found for: {query}"

    lines = [f"Latest news: {query}\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        if not title:
            continue
        src = f"  [{r['source']}]" if r.get("source") else ""
        lines.append(f"{i}. {title}{src}")
        if r.get("snippet"):
            lines.append(f"   {r['snippet'][:140]}")
        if r.get("url"):
            lines.append(f"   {r['url']}")
        lines.append("")
    return "\n".join(lines).strip()


# ── Briefing helper ────────────────────────────────────────────────────────────

def _gemini_headlines(n: int = 5) -> tuple[list[str], str]:
    """
    Fetches current headlines via Gemini grounded search.
    Optimised for speed: minimal prompt + strict token cap.
    Returns (headline_list, raw_text_for_display).
    """
    import re
    from google import genai

    budget.reserve("grounded_search")

    client = genai.Client(api_key=_get_api_key())
    try:
        response = client.models.generate_content(
            model=budget.model("fast"),
            contents=f"Current world news: {n} headlines. Numbered list, titles only.",
            config={"tools": [{"google_search": {}}]},
        )
    except Exception as e:
        budget.report(e)
        raise

    raw = ""
    for part in response.candidates[0].content.parts:
        if hasattr(part, "text") and part.text:
            raw += part.text

    headlines = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Only accept lines that begin with a number — skips preamble/closing sentences
        if not re.match(r'^[\d]+[.\)\-]', line):
            continue
        clean = re.sub(r'^[\d]+[.\)\-]\s*', '', line)
        clean = re.sub(r'^\*+\s*',          '', clean).strip()
        if clean and len(clean) > 10:
            headlines.append(clean)

    return headlines[:n], raw.strip()


# ── Modes ──────────────────────────────────────────────────────────────────────

def _log_fallback(what: str, exc: Exception) -> None:
    """Free mode falling back is normal operation, not an error.

    Only a genuine failure is worth a full stack-length log line; a 429 is
    reported once by core.budget and then handled silently for the cooldown.
    """
    if isinstance(exc, budget.QuotaExhausted):
        log.info(f"[{budget.status()}] {what} → DuckDuckGo ({exc})")
    elif budget.report(exc):
        log.warning(f"{what}: quota exhausted — DuckDuckGo from here.")
    else:
        log.error(f"⚠️ {what} failed ({exc}) — trying DDG...")


def _search(query: str) -> str:
    """Default search — Gemini grounded, DDG fallback."""
    try:
        return _gemini_search(query)
    except Exception as e:
        _log_fallback("Gemini search", e)
        results = _ddg_search(query)
        return _format_ddg(query, results)


def _news(query: str, strict: bool = False) -> str:
    """
    Runs Gemini grounded search AND DDG news in parallel.
    Returns whichever delivers a valid result first; cancels the other.

    `strict` means "only real headlines" — no text-search fallback. Use it when
    an empty result is a meaningful answer, such as deciding whether a small
    town has local news before widening to its region.
    """
    import threading

    gemini_query = f"latest news today: {query}" if query else "top world news today"
    ddg_query    = query if query else "world news today"

    result_box  = [None]   # first valid result lands here
    lock        = threading.Lock()
    done_evt    = threading.Event()
    failures    = [0]

    def _store(r: str) -> None:
        if r and len(r) > 60:
            with lock:
                if result_box[0] is None:
                    result_box[0] = r
            done_evt.set()
        else:
            with lock:
                failures[0] += 1
                if failures[0] >= 2:   # both failed — unblock caller
                    done_evt.set()

    def _try_gemini():
        try:
            _store(_gemini_search(gemini_query))
        except Exception as e:
            _log_fallback("Gemini news", e)
            _store("")

    def _try_ddg():
        try:
            results = _ddg_news(ddg_query, max_results=8,
                                allow_text_fallback=not strict)
            _store(_format_news(ddg_query, results))
        except Exception as e:
            log.error(f"⚠️ DDG news failed ({e})")
            _store("")

    # In free mode the Gemini half would only burn a request to fail, so the
    # race is not started at all — DDG alone counts as one failure, not two.
    if budget.allows("grounded_search"):
        threading.Thread(target=_try_gemini, daemon=True).start()
    else:
        log.info(f"[{budget.status()}] news → DuckDuckGo only")
        failures[0] += 1
    threading.Thread(target=_try_ddg, daemon=True).start()

    done_evt.wait(timeout=10.0)
    return result_box[0] or f"No news found for: {query}"


def _research(query: str) -> str:
    """
    Deep dive — asks Gemini for a comprehensive answer with context.
    Falls back to a wider DDG fetch.
    """
    research_query = (
        f"Comprehensive, detailed explanation of: {query}. "
        "Include background context, key facts, current state, and important nuances."
    )
    try:
        return _gemini_search(research_query)
    except Exception as e:
        _log_fallback("Gemini research", e)
        results = _ddg_search(query, max_results=10)
        return _format_ddg(query, results)


def _price(query: str) -> str:
    """Product price lookup — searches for current market prices."""
    price_query = f"current price of {query} — how much does it cost today"
    try:
        return _gemini_search(price_query)
    except Exception as e:
        _log_fallback("Gemini price lookup", e)
        results = _ddg_search(f"{query} price buy", max_results=6)
        return _format_ddg(query, results)


def _compare(items: list[str], aspect: str) -> str:
    query = (
        f"Compare {', '.join(items)} in terms of {aspect}. "
        "Give specific facts and data."
    )
    try:
        return _gemini_search(query)
    except Exception as e:
        _log_fallback("Gemini compare", e)

    all_results: dict[str, list] = {}
    for item in items:
        try:
            all_results[item] = _ddg_search(f"{item} {aspect}", max_results=3)
        except Exception:
            all_results[item] = []

    lines = [f"Comparison — {aspect.upper()}", "─" * 40]
    for item in items:
        lines.append(f"\n▸ {item}")
        for r in all_results.get(item, [])[:2]:
            if r.get("snippet"):
                lines.append(f"  • {r['snippet']}")
            if r.get("url"):
                lines.append(f"    {r['url']}")
    return "\n".join(lines)


# ── Public entry point ─────────────────────────────────────────────────────────

def web_search(
    parameters:     dict,
    response=None,
    player=None,
    session_memory=None,
) -> str:
    params = parameters or {}
    query  = params.get("query", "").strip()
    mode   = params.get("mode",  "search").lower().strip()
    items  = params.get("items", [])
    aspect = params.get("aspect", "general").strip() or "general"

    if not query and not items:
        return "Please provide a search query."

    if items and mode not in ("compare",):
        mode = "compare"

    if player:
        player.write_log(f"[Search:{mode}] {query or ', '.join(items)}")

    log.info(f"🔍 mode={mode!r}  query={query!r}")

    try:
        if mode == "compare" and items:
            return _compare(items, aspect)
        if mode == "news":
            return _news(query)
        if mode == "research":
            return _research(query)
        if mode == "price":
            return _price(query)
        return _search(query)

    except Exception as e:
        log.error(f"❌ All backends failed: {e}")
        return f"Search failed: {e}"


# ── Tool registration ─────────────────────────────────────────────────────────
# Imported at the bottom so the schema sits next to the implementation without
# reordering the module.  Importing this file registers the tool; main.py only
# has to import actions.tools.
from core.registry import ToolContext, tool  # noqa: E402


@tool(
    name="web_search",
    description=(
        "Searches the web. Use for ANY question about current facts, events, prices, "
        "or topics — always prefer this over guessing. "
        "Modes: 'search' (default), 'news' (latest headlines on a topic), "
        "'research' (deep comprehensive answer), 'price' (product cost lookup), "
        "'compare' (side-by-side comparison of items)."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "query":  {"type": "STRING", "description": "Search query or topic"},
            "mode":   {"type": "STRING", "description": "search | news | research | price | compare"},
            "items":  {"type": "ARRAY",  "items": {"type": "STRING"}, "description": "Items to compare (compare mode)"},
            "aspect": {"type": "STRING", "description": "Comparison aspect: price | specs | reviews | features"},
        },
        "required": ["query"]
    },
    timeout=60,
)
def web_search_tool(params: dict, ctx: ToolContext) -> str:
    r = web_search(parameters=params, player=ctx.ui)
    # Mirror results to the on-screen content panel
    if r and not r.startswith("No results") and not r.startswith("Search failed"):
        mode  = params.get("mode", "search")
        query = params.get("query") or ", ".join(params.get("items", []) or [])
        label = f"{mode.upper()} — {query[:38]}" if query else mode.upper()
        if ctx.ui is not None:
            try:
                ctx.ui.show_content(label, r)
            except Exception:
                pass
    return r or "Done."
