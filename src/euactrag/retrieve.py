"""Dense, lexical and hybrid retrieval, and hybrid over the parts of a question.

Hybrid uses Reciprocal Rank Fusion rather than a weighted score blend. Cosine
similarities and BM25 scores live on incompatible scales, so blending them needs
a normalisation constant that has to be re-tuned per corpus. RRF only consumes
*ranks*, so it has one parameter (k=60, the value from Cormack et al. 2009) and
no per-corpus tuning, which matters when the point of the project is to report
honest numbers rather than numbers fitted to the eval set.

Every function takes the corpus `version` (see config.VERSIONS); None means
config.CORPUS_VERSION, the latest text unless the environment says otherwise.
"""
from __future__ import annotations

import functools
import re
from datetime import date

from . import config, scope
from .index import get_collection, tokenize
from .ingest import load_chunks


@functools.lru_cache(maxsize=1)
def _encoder():
    from .index import get_encoder

    return get_encoder()


@functools.lru_cache(maxsize=None)
def _bm25(version: str, granularity: str = "article"):
    from rank_bm25 import BM25Okapi

    chunks = load_chunks(version, granularity)
    return BM25Okapi([tokenize(c["text"]) for c in chunks]), chunks


@functools.lru_cache(maxsize=None)
def _collection(version: str, granularity: str = "article"):
    try:
        return get_collection(version=version, granularity=granularity)
    except Exception:
        if granularity != config.GUIDANCE:
            raise
        # `index` does not embed the guidelines, so the first search that
        # asks for them does. Under a minute, once.
        from .index import build

        build(version=version, granularity=granularity)
        return get_collection(version=version, granularity=granularity)


def _as_hit(chunk_id: str, text: str, meta: dict, score: float, rank: int) -> dict:
    hit = {
        "chunk_id": chunk_id, "text": text, "score": float(score),
        "rank": rank, **{k: meta.get(k) for k in
                         ("kind", "unit_id", "citation", "chapter_title", "url")},
    }
    # Chroma metadata is flat; a chunk read from the jsonl keeps it under "meta".
    hit["ref"] = meta.get("ref") or (meta.get("meta") or {}).get("ref") or hit["unit_id"]
    return hit


def dense(query: str, k: int = 10, version: str | None = None,
          granularity: str | None = None) -> list[dict]:
    q = _encoder().encode(
        [config.QUERY_PREFIX + query], normalize_embeddings=True
    )[0].tolist()
    r = _collection(version or config.CORPUS_VERSION,
                    granularity or config.GRANULARITY).query(query_embeddings=[q], n_results=k)
    return [
        # Chroma returns cosine *distance*; flip it so higher is better.
        _as_hit(i, d, m, 1.0 - dist, rank)
        for rank, (i, d, m, dist) in enumerate(
            zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0]), 1
        )
    ]


def bm25(query: str, k: int = 10, version: str | None = None,
         granularity: str | None = None) -> list[dict]:
    model, chunks = _bm25(version or config.CORPUS_VERSION,
                          granularity or config.GRANULARITY)
    scores = model.get_scores(tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [
        _as_hit(chunks[i]["chunk_id"], chunks[i]["text"], chunks[i], scores[i], rank)
        for rank, i in enumerate(order, 1)
    ]


def hybrid(query: str, k: int = 10, pool: int = 30,
           recital_weight: float | None = None,
           version: str | None = None, granularity: str | None = None) -> list[dict]:
    """Fuse dense and lexical rankings by RRF: score = sum w / (K + rank).

    `w` is 1.0 for binding provisions (articles, annexes) and
    `config.RECITAL_WEIGHT` for recitals, see the note there for why.
    """
    w = config.RECITAL_WEIGHT if recital_weight is None else recital_weight
    runs = [dense(query, pool, version, granularity),
            bm25(query, pool, version, granularity)]
    fused: dict[str, dict] = {}
    for run in runs:
        for h in run:
            e = fused.setdefault(h["chunk_id"], {**h, "score": 0.0})
            e["score"] += (w if h["kind"] == "recital" else 1.0) / (
                config.RRF_K + h["rank"])
    # Ties on the fused score are common: any two chunks that sit at the same
    # rank in exactly one run score the same. A sort on score alone then falls
    # back to dict insertion order, which is whatever order the dense index
    # happened to return its own near-ties in, and that is not the same on every
    # machine. The weekly eval caught it: one query's hybrid rank differed between
    # this laptop and the CI runner while both component rankings were identical.
    # chunk_id is a string and sorts the same everywhere.
    out = [h for h in sorted(fused.values(),
                             key=lambda h: (-h["score"], h["chunk_id"]))
           if not (w == 0.0 and h["kind"] == "recital")][:k]
    for rank, h in enumerate(out, 1):
        h["rank"] = rank
    return out


POOL = 30   # candidates read per sub-question
UNIT_CAP = 2  # chunks of any one article allowed in the result

# A clause boundary: the end of a sentence, or a comma / "and" followed by a
# word that starts a new question.
_LEAD = (r"(?:what|when|who|whom|which|how|why|is|are|does|do|did|must|can|may|"
         r"should|from|by|under|if|to)")
_BOUNDARY = re.compile(
    rf"(?<=[.?])\s+|,?\s+and\s+(?={_LEAD}\b)|,\s+(?={_LEAD}\b)", re.I)


def split_question(query: str) -> list[str]:
    """The parts of a question that asks more than one thing, or [] if it
    asks one. "What must a chatbot provider disclose, and what fine applies if
    it fails to?" is two searches: the rule and the penalty sit in different
    articles, and embedded as one query the first half drowns the second."""
    parts = [p.strip() for p in _BOUNDARY.split(query) if p and len(p.split()) >= 3]
    return parts if len(parts) > 1 else []


def split(query: str, k: int = 10, version: str | None = None,
          granularity: str | None = None) -> list[dict]:
    """Hybrid search for the question and for each of its parts, dealt out in
    turn, with at most UNIT_CAP chunks per article.

    Taking turns matters: fusing the runs by rank would reward chunks that
    several parts agree on, and a two-part question needs the chunks they do
    not share. Without the cap, paragraph chunks of one long article can fill
    the whole top-k.
    """
    runs = [hybrid(q, POOL, version=version, granularity=granularity)
            for q in [query] + split_question(query)]
    out, seen, per_unit = [], set(), {}
    for depth in range(POOL):
        for run in runs:
            if depth >= len(run) or run[depth]["chunk_id"] in seen:
                continue
            h = run[depth]
            if per_unit.get(h["unit_id"], 0) >= UNIT_CAP:
                continue
            per_unit[h["unit_id"]] = per_unit.get(h["unit_id"], 0) + 1
            seen.add(h["chunk_id"])
            # Fused scores from different runs are not comparable, so the
            # score reported is the depth at which the chunk was dealt.
            out.append({**h, "rank": len(out) + 1,
                        "score": 1.0 / (config.RRF_K + depth + 1)})
    return out[:k]


FILTER_POOL = 50  # passages ranked before a role or date filter is applied


def search(query: str, k: int | None = None, mode: str | None = None,
           version: str | None = None, granularity: str | None = None,
           role: str | None = None, on: str | None = None,
           guidelines: bool = False) -> list[dict]:
    """Ranked passages, each tagged with `roles`, `applies_from` and
    `applies_rule` (see scope.py). `role` drops passages that name other roles
    but not that one; `on`, an ISO date, drops articles that do not apply yet
    on that date. `guidelines` adds, after the k passages of the Act, up to
    config.GUIDELINE_K paragraphs of the Commission's guidelines (kind
    "guideline"), found by a hybrid search of the guidelines alone."""
    k = k or config.TOP_K
    mode = mode or config.RETRIEVAL_MODE
    version = version or config.CORPUS_VERSION
    granularity = granularity or config.GRANULARITY
    if role and role not in scope.ROLES:
        raise ValueError(f"role must be one of {', '.join(scope.ROLES)}, not {role!r}")
    on = on and date.fromisoformat(on).isoformat()
    fn = {"dense": dense, "bm25": bm25, "hybrid": hybrid, "split": split}[mode]
    # ponytail: a filter runs after ranking, over the best FILTER_POOL
    # passages, so a narrow one can return fewer than k. Filter inside each
    # run instead if that starts to bite.
    hits = fn(query, max(k, FILTER_POOL) if role or on else k,
              version=version, granularity=granularity)
    tags = scope.tags(version, granularity)
    # .get: an index built before the last ingest can hold a chunk the corpus
    # no longer has. It goes untagged, and the filters keep untagged passages.
    hits = [{**h, **tags.get(h["chunk_id"], {"roles": []})} for h in hits]
    if role or on:
        hits = [{**h, "rank": rank} for rank, h in enumerate(
            (h for h in hits if scope.keep(h, role, on)), 1)][:k]
    if guidelines:
        # One index for every version of the Act: the guidelines do not change
        # with it. Three of the four predate the 2026 amendments.
        extra = hybrid(query, FILTER_POOL if role else config.GUIDELINE_K,
                       version=config.LATEST, granularity=config.GUIDANCE)
        tags = scope.tags(config.LATEST, config.GUIDANCE)
        extra = [{**h, **tags.get(h["chunk_id"], {"roles": []})} for h in extra]
        extra = [h for h in extra if scope.keep(h, role)][:config.GUIDELINE_K]
        hits += [{**h, "rank": len(hits) + i} for i, h in enumerate(extra, 1)]
    return hits
