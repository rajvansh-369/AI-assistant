"""
BackgroundMonitor — user-configured topic watching.
Checks DDG news once per day per topic; alerts JARVIS when a new headline appears.
No crypto, no finance, no uninvited tracking.
"""
import hashlib
import re
from datetime import datetime

from core.log import get_logger

log = get_logger("monitor")


# ── Blocked categories (never monitor regardless of what user says) ────────────

_BLOCKED = {
    # Marka / varlık adları — her dilde aynı yazılır
    "bitcoin", "ethereum", "dogecoin", "solana", "binance",
    "nft", "blockchain", "defi", "altcoin", "memecoin", "coin", "token",
    # "kripto" kökünün farklı dillerdeki yazılışları
    "crypto", "kripto", "cripto", "krypto", "крипто", "仮想通貨", "暗号資産",
    "cryptocurrency",
}

def _is_blocked(topic: str) -> bool:
    t = topic.lower()
    return any(word in t for word in _BLOCKED)


# ── Slug / hash helpers ────────────────────────────────────────────────────────

def _slug(topic: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", topic.lower().strip())[:40].strip("_")

def _title_hash(title: str) -> str:
    return hashlib.md5(title.encode("utf-8", errors="ignore")).hexdigest()[:12]


# ── Memory I/O ─────────────────────────────────────────────────────────────────

def _load() -> dict:
    from memory.memory_manager import load_memory
    data = load_memory().get("monitors", {})
    return data if isinstance(data, dict) else {}


def _update_monitors(mutate):
    """Apply `mutate(monitors)` inside a single memory transaction.

    Replaces the old load-then-save pair, which could clobber a concurrent write
    to any other section of long_term.json.  Keep `mutate` fast and I/O-free — it
    runs while the memory lock is held.
    """
    from memory.memory_manager import memory_txn
    with memory_txn() as memory:
        monitors = memory.get("monitors", {})
        if not isinstance(monitors, dict):
            monitors = {}
        result = mutate(monitors)
        memory["monitors"] = monitors
    return result


# ── Public API ─────────────────────────────────────────────────────────────────

def add_monitor(topic: str) -> str:
    topic = topic.strip()
    if not topic:
        return "Please specify a topic to monitor."
    if _is_blocked(topic):
        return "I don't monitor crypto or financial topics."
    slug = _slug(topic)

    def _mutate(monitors: dict) -> str | None:
        if slug in monitors:
            return f"Already monitoring: {monitors[slug]['topic']}"
        monitors[slug] = {
            "topic":      topic,
            "added":      datetime.now().strftime("%Y-%m-%d"),
            "last_check": "",
            "last_hash":  "",
        }
        return None

    already = _update_monitors(_mutate)
    if already:
        return already
    log.info(f"➕ Added: {topic}")
    return f"Now monitoring: {topic}"


def remove_monitor(topic: str) -> str:
    topic = topic.strip().lower()
    slug  = _slug(topic)

    def _mutate(monitors: dict) -> str | None:
        # exact slug match first
        if slug in monitors:
            return monitors.pop(slug)["topic"]
        # partial match fallback
        for key, val in list(monitors.items()):
            if topic in val.get("topic", "").lower():
                return monitors.pop(key)["topic"]
        return None

    label = _update_monitors(_mutate)
    if label:
        return f"Stopped monitoring: {label}"
    return f"Not found in monitored topics: {topic}"


def list_monitors() -> list[str]:
    return [v.get("topic", k) for k, v in _load().items()]


def check_all() -> list[str]:
    """
    Run all pending topic checks (once per day per topic).
    Returns a list of [MONITOR_ALERT] strings — empty if nothing new.
    """
    from actions.web_search import _ddg_news

    monitors = _load()
    if not monitors:
        return []

    today   = datetime.now().strftime("%Y-%m-%d")
    alerts  = []
    # slug → fields to write back.  Collected rather than applied in place because
    # the network calls below must not run while the memory lock is held.
    patches: dict[str, dict] = {}

    for slug, data in monitors.items():
        if data.get("last_check") == today:
            continue                     # already checked today

        topic = data.get("topic", slug)
        try:
            results = _ddg_news(topic, max_results=5)
            if not results:
                patches.setdefault(slug, {})["last_check"] = today
                continue

            top   = results[0]
            title = top.get("title", "").strip()
            if not title:
                continue

            h = _title_hash(title)
            patches.setdefault(slug, {})["last_check"] = today

            if h == data.get("last_hash"):
                continue                 # same headline as last check — no alert

            patches[slug]["last_hash"] = h

            snippet = top.get("snippet", "")[:150]
            source  = top.get("source", "")
            parts   = [f"[MONITOR_ALERT] {topic}", f"Headline: {title}"]
            if snippet:
                parts.append(snippet)
            if source:
                parts.append(f"Source: {source}")
            alerts.append("\n".join(parts))
            log.info(f"🔔 New headline for '{topic}': {title[:60]}")

        except Exception as e:
            log.error(f"⚠️ Check failed for '{topic}': {e}")

    if patches:
        def _mutate(current: dict) -> None:
            for slug, fields in patches.items():
                if slug in current:      # skip topics removed while we were checking
                    current[slug].update(fields)
        _update_monitors(_mutate)

    return alerts


# ── Tool registration ─────────────────────────────────────────────────────────
# Imported at the bottom so the schema sits next to the implementation without
# reordering the module.  Importing this file registers the tool; main.py only
# has to import actions.tools.
from core.registry import ToolContext, tool  # noqa: E402


@tool(
    name="manage_monitor",
    description=(
        "Add, remove, or list background monitoring topics. "
        "JARVIS checks these topics once a day and alerts the user when there is a new development. "
        "Use 'add' when the user says 'monitor X', 'track X', 'follow X'. "
        "Use 'remove' when the user says 'stop monitoring X'. "
        "Use 'list' when the user asks what is being monitored. "
        "Do NOT add crypto, financial, or trading topics."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "action": {
                "type":        "STRING",
                "description": "add | remove | list",
            },
            "topic": {
                "type":        "STRING",
                "description": "Topic to monitor or stop monitoring (e.g. 'space exploration', 'AI news')",
            },
        },
        "required": ["action"],
    },
    timeout=30,
)
def manage_monitor_tool(params: dict, ctx: ToolContext) -> str:
    action = str(params.get("action", "")).lower().strip()
    topic  = str(params.get("topic", "")).strip()
    if action == "add" and topic:
        return add_monitor(topic)
    if action == "remove" and topic:
        return remove_monitor(topic)
    if action == "list":
        topics = list_monitors()
        return ("Monitoring: " + ", ".join(topics)) if topics else "No topics are being monitored."
    return "Specify action (add/remove/list) and a topic."
