"""Scoring facts against a query, with or without an API.

Two backends, same interface:

* **Embeddings** — Gemini `embed_content`, cosine similarity. Understands that
  "what am I building" should surface a fact stored as "side project: a drone
  controller", which no amount of word matching will do.
* **Lexical** — weighted token overlap, entirely local. Not as good, but it
  costs nothing, works offline, and never returns an error mid-conversation.

Lexical is the default in free mode and during a quota cooldown. That is not a
degraded-mode afterthought: this machine runs on a free key, so the local path
is the one that actually runs, and it has to be good enough to be useful on its
own. Embeddings are the upgrade, not the baseline.

Embeddings are computed once per fact and cached in the `embedding` column;
`store.put_fact` nulls it when a value changes, so a stale vector cannot
outlive the text it described.
"""
from __future__ import annotations

import math
import re
import threading

import numpy as np

from core import budget
from core.log import get_logger
from core.settings import get_api_key, get_settings

from memory import store

log = get_logger("memory.embed")

#: Small, cheap, and fine for one-line facts.
EMBED_MODEL = "text-embedding-004"

#: Retrieval quality is not worth an API stall on the connect path.
EMBED_TIMEOUT = 8.0

_lock = threading.Lock()
_failed = False          #: set after a hard failure, so we stop retrying all session


def available() -> bool:
    """Whether the embedding backend should be used at all.

    Degraded covers both free mode and a quota cooldown after a 429.
    """
    if _failed or budget.degraded():
        return False
    return bool(get_settings().gemini_api_key)


# ── embedding backend ─────────────────────────────────────────────────────────

def _client():
    from google import genai
    return genai.Client(api_key=get_api_key())


def embed(texts: list[str]) -> list[np.ndarray] | None:
    """Vectors for `texts`, or None if the backend is unavailable or failed."""
    global _failed
    if not texts or not available():
        return None

    try:
        resp = _client().models.embed_content(model=EMBED_MODEL, contents=texts)
        out = [np.asarray(e.values, dtype=np.float32) for e in resp.embeddings]
        if len(out) != len(texts):
            log.warning(f"Expected {len(texts)} embeddings, got {len(out)}")
            return None
        return out
    except Exception as e:
        budget.report(e)
        # One failure is enough. Retrying per query would put an API round-trip
        # with a known-bad outcome in front of every recall.
        with _lock:
            _failed = True
        log.warning(f"Embeddings unavailable, falling back to lexical: {e}")
        return None


def backfill(limit: int = 64) -> int:
    """Embed facts that have no current vector. Returns how many were done."""
    if not available():
        return 0
    rows = store.facts_missing_embeddings(EMBED_MODEL, limit=limit)
    if not rows:
        return 0

    vectors = embed([_fact_text(r) for r in rows])
    if vectors is None:
        return 0

    for row, vec in zip(rows, vectors):
        store.set_embedding(row["id"], vec.tobytes(), EMBED_MODEL)
    log.debug(f"Embedded {len(rows)} facts")
    return len(rows)


def _fact_text(row) -> str:
    """What actually gets embedded — the key carries meaning the value omits.

    "favorite_food / pizza" is a far better vector than "pizza" alone.
    """
    return f"{row['category']} {row['key'].replace('_', ' ')}: {row['value']}"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── lexical backend ───────────────────────────────────────────────────────────

_WORD = re.compile(r"[a-z0-9]+")

#: Words too common to indicate what a query is about.
_STOP = frozenset("""
a an the and or but if then than that this these those of in on at to for from by with
about into over after is are was were be been being am do does did doing have has had
i me my mine you your yours he him his she her it its we us our they them their
what which who whom whose when where why how all any both each few more most other some
such no nor not only own same so too very can will just should now got get
""".split())


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]


#: Word matching alone cannot connect "what do I do for work" to a fact stored
#: as `current_job`, and that miss is the difference between recall_memory being
#: useful and it returning nothing. These clusters cover the vocabulary people
#: actually use about the things this store holds — the six fact categories and
#: little else.
#:
#: A deliberate stopgap for the local backend, not an ontology. Embeddings get
#: this right without a word list; this is what free mode has instead.
_SYNONYMS: tuple[frozenset[str], ...] = (
    frozenset("job work works working career employment profession occupation role company employer".split()),
    frozenset("city home live lives living location place town country address".split()),
    frozenset("food eat eats eating cuisine dish meal dinner lunch breakfast".split()),
    frozenset("family sister brother mother father parent parents sibling siblings wife husband partner girlfriend boyfriend relative son daughter".split()),
    frozenset("friend friends colleague colleagues mate".split()),
    frozenset("project projects building build builds built making make side app".split()),
    frozenset("travel trip visit vacation holiday going abroad".split()),
    frozenset("name called call".split()),
    frozenset("age old birthday born".split()),
    frozenset("study studies school college university degree student education".split()),
    frozenset("hobby hobbies interest interests fun leisure pastime".split()),
    frozenset("want wants wish wishes plan plans goal goals dream dreams".split()),
    frozenset("code coding programming program developer development software engineer".split()),
    frozenset("framework frameworks stack tech technology technologies tool tools library".split()),
    frozenset("music song songs artist band listen".split()),
    frozenset("film films movie movies show shows watch".split()),
    frozenset("game games gaming play playing".split()),
    frozenset("sport sports team play exercise gym fitness".split()),
)

#: token -> every cluster-mate, built once
_EXPANSION: dict[str, set[str]] = {}
for _cluster in _SYNONYMS:
    for _word in _cluster:
        _EXPANSION.setdefault(_word, set()).update(_cluster - {_word})

#: An expanded term is a weaker signal than one the user actually said, so an
#: exact match always outranks a synonym match.
_SYNONYM_WEIGHT = 0.5


def _expand(tokens: set[str]) -> dict[str, float]:
    """Query terms plus their cluster-mates, weighted lower."""
    weighted = {t: 1.0 for t in tokens}
    for token in tokens:
        for related in _EXPANSION.get(token, ()):
            weighted.setdefault(related, _SYNONYM_WEIGHT)
    return weighted


def lexical_scores(query: str, texts: list[str]) -> list[float]:
    """Overlap between query and each text, rarer words counting for more.

    A plain count would rank a fact containing "project" above one containing
    the actual subject, because common words hit more often. Weighting by
    inverse document frequency across the stored facts fixes that without
    needing a real index — there are tens of facts here, not millions.
    """
    q_tokens = set(_tokens(query))
    if not q_tokens or not texts:
        return [0.0] * len(texts)

    q    = _expand(q_tokens)
    docs = [_tokens(t) for t in texts]
    n    = len(docs)

    df: dict[str, int] = {}
    for doc in docs:
        for word in set(doc):
            df[word] = df.get(word, 0) + 1

    scores = []
    for doc in docs:
        hit = q.keys() & set(doc)
        if not hit:
            scores.append(0.0)
            continue
        weight = sum(q[w] * math.log(1 + n / (1 + df.get(w, 0))) for w in hit)
        # Normalise by what the user actually asked, not by the expansion and
        # not by the document: a long fact that contains the query terms is
        # still a good answer.
        scores.append(weight / (len(q_tokens) + 1e-9))
    return scores


# ── the one entry point callers use ───────────────────────────────────────────

def score(query: str, rows: list) -> list[float]:
    """Relevance of each fact row to `query`, in [0, 1]-ish. Never raises.

    Uses embeddings when they are available for *every* row, and lexical
    otherwise — mixing cosine similarity with overlap scores in one ranking
    would compare two different scales and quietly favour whichever ran.
    """
    if not rows:
        return []

    texts = [_fact_text(r) for r in rows]

    if available():
        try:
            vectors = _vectors_for(rows)
            query_vec = embed([query]) if vectors else None
            if query_vec:
                return [_cosine(query_vec[0], v) for v in vectors]
        except Exception as e:
            log.warning(f"Embedding scoring failed, using lexical: {e}")

    return lexical_scores(query, texts)


def _vectors_for(rows: list) -> list[np.ndarray] | None:
    """Current vectors for every row, or None if any is still missing one.

    Rows are re-read after the backfill: the caller's copies were loaded before
    the embeddings were written, so their `embedding` columns are stale.
    """
    backfill(limit=len(rows))

    fresh = {r["id"]: r for r in store.facts_by_ids([r["id"] for r in rows])}
    vectors = []
    for row in rows:
        r = fresh.get(row["id"])
        if r is None or not r["embedding"] or r["embed_model"] != EMBED_MODEL:
            return None
        vectors.append(np.frombuffer(r["embedding"], dtype=np.float32))
    return vectors
