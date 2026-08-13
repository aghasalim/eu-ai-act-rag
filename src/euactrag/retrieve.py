"""Dense, lexical and hybrid retrieval.

Hybrid uses Reciprocal Rank Fusion rather than a weighted score blend. Cosine
similarities and BM25 scores live on incompatible scales, so blending them needs
a normalisation constant that has to be re-tuned per corpus. RRF only consumes
*ranks*, so it has one parameter (k=60, the value from Cormack et al. 2009) and
no per-corpus tuning -- which matters when the point of the project is to report
honest numbers rather than numbers fitted to the eval set.
"""
from __future__ import annotations

import functools

from . import config
from .index import get_collection, tokenize
from .ingest import load_chunks


@functools.lru_cache(maxsize=1)
def _encoder():
    from .index import get_encoder

    return get_encoder()


@functools.lru_cache(maxsize=1)
def _bm25():
    from rank_bm25 import BM25Okapi

    chunks = load_chunks()
    return BM25Okapi([tokenize(c["text"]) for c in chunks]), chunks


@functools.lru_cache(maxsize=1)
def _collection():
    return get_collection()


def _as_hit(chunk_id, text, meta, score, rank):
    return {
        "chunk_id": chunk_id, "text": text, "score": float(score),
        "rank": rank, **{k: meta.get(k) for k in
                         ("kind", "unit_id", "citation", "chapter_title", "url")},
    }


def dense(query: str, k: int = 10) -> list[dict]:
    q = _encoder().encode(
        [config.QUERY_PREFIX + query], normalize_embeddings=True
    )[0].tolist()
    r = _collection().query(query_embeddings=[q], n_results=k)
    return [
        # Chroma returns cosine *distance*; flip it so higher is better.
        _as_hit(i, d, m, 1.0 - dist, rank)
        for rank, (i, d, m, dist) in enumerate(
            zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0]), 1
        )
    ]


def bm25(query: str, k: int = 10) -> list[dict]:
    model, chunks = _bm25()
    scores = model.get_scores(tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [
        _as_hit(chunks[i]["chunk_id"], chunks[i]["text"], chunks[i], scores[i], rank)
        for rank, i in enumerate(order, 1)
    ]


def hybrid(query: str, k: int = 10, pool: int = 30) -> list[dict]:
    """Fuse dense and lexical rankings by RRF: score = sum 1 / (K + rank)."""
    runs = [dense(query, pool), bm25(query, pool)]
    fused: dict[str, dict] = {}
    for run in runs:
        for h in run:
            e = fused.setdefault(h["chunk_id"], {**h, "score": 0.0})
            e["score"] += 1.0 / (config.RRF_K + h["rank"])
    out = sorted(fused.values(), key=lambda h: -h["score"])[:k]
    for rank, h in enumerate(out, 1):
        h["rank"] = rank
    return out


def search(query: str, k: int | None = None, mode: str | None = None) -> list[dict]:
    k = k or config.TOP_K
    mode = mode or config.RETRIEVAL_MODE
    fn = {"dense": dense, "bm25": bm25, "hybrid": hybrid}[mode]
    return fn(query, k)
