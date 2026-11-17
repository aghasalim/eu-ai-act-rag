"""Retrieval variants side by side: question splitting, a reranker, other embedders.

Everything here is retrieval, so it needs no API key. Each variant is scored on
the same questions as the baseline and compared with an exact paired test on
full recall, because on ~100 questions a two-question gap is not a result.

    python eval/experiments.py                       # the default grid
    python eval/experiments.py --embed path/to/model # add a fine-tuned embedder

Dense search runs in memory here instead of through Chroma, so that trying
another embedder does not mean rebuilding the index. The first row repeats the
published baseline to show the two agree.
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.euactrag import config, retrieve  # noqa: E402
from src.euactrag.ingest import load_chunks  # noqa: E402
from eval import metrics as M  # noqa: E402
from eval.run_eval import load_qa, mean  # noqa: E402

BASE = "BAAI/bge-small-en-v1.5"
RERANKER = "BAAI/bge-reranker-base"
# (label, corpus, granularity, question set). The first is the published
# headline setting, the second is what search and the app use today.
SETTINGS = [
    ("original 45, 2024 text, article chunks", "2024-07-12", "article", "original"),
    ("all 122, 2026 text, paragraph chunks", "2026-07-27", "paragraph", "all"),
]


@functools.lru_cache(maxsize=None)
def memory_dense(model_name: str):
    """A drop-in for retrieve.dense backed by a numpy matrix."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(model_name)
    cache: dict = {}

    def dense(query, k=10, version=None, granularity=None):
        key = (version or config.CORPUS_VERSION, granularity or config.GRANULARITY)
        if key not in cache:
            chunks = load_chunks(*key)
            cache[key] = (chunks, enc.encode([c["text"] for c in chunks],
                                             normalize_embeddings=True, batch_size=32))
        chunks, vecs = cache[key]
        q = enc.encode([config.QUERY_PREFIX + query], normalize_embeddings=True)[0]
        scores = vecs @ q
        order = np.argsort(-scores, kind="stable")[:k]
        return [retrieve._as_hit(chunks[i]["chunk_id"], chunks[i]["text"], chunks[i],
                                 scores[i], rank) for rank, i in enumerate(order, 1)]

    return dense


@functools.lru_cache(maxsize=None)
def _reranker(model: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model)


def reranked(query, k, version, granularity):
    """Hybrid top 30 re-sorted by a cross-encoder that reads each candidate
    next to the question. Lives here, not in the package: it did not earn its
    latency, see the table."""
    hits = retrieve.hybrid(query, retrieve.POOL, version=version, granularity=granularity)
    scores = _reranker(RERANKER).predict([(query, h["text"]) for h in hits])
    order = sorted(range(len(hits)), key=lambda i: (-scores[i], hits[i]["chunk_id"]))
    return [{**hits[i], "rank": rank} for rank, i in enumerate(order[:k], 1)]


def variant(embed: str, fn=None):
    """`fn` (default: plain hybrid) run with `embed` as the dense model."""
    fn = fn or (lambda q, k, version, granularity:
                retrieve.hybrid(q, k, version=version, granularity=granularity))

    def search(query, k, version, granularity):
        old = retrieve.dense
        retrieve.dense = memory_dense(embed)
        try:
            return fn(query, k, version, granularity)
        finally:
            retrieve.dense = old
    return search


def split(query, k, version, granularity):
    return retrieve.split(query, k, version=version, granularity=granularity)


def sign_test(a: list[float], b: list[float]) -> tuple[int, int, float]:
    """Exact two-sided sign test on paired 0/1 outcomes: (b wins, a wins, p)."""
    up = sum(y > x for x, y in zip(a, b))
    down = sum(y < x for x, y in zip(a, b))
    n = up + down
    if not n:
        return 0, 0, 1.0
    tail = sum(comb(n, i) for i in range(min(up, down) + 1)) / 2 ** n
    return up, down, min(1.0, 2 * tail)


def score(search, qa: list[dict], k: int, version: str, granularity: str) -> dict:
    answerable = [q for q in qa if q["gold_units"]]
    per_q = {}
    for q in answerable:
        hits = search(q["question"], k, version, granularity)
        per_q[q["id"]] = M.retrieval_scores(hits, q["gold_units"], k)
        if q.get("gold_refs"):
            per_q[q["id"]].update(M.paragraph_scores(hits, q["gold_refs"], k))
    multi = [q["id"] for q in answerable if q["type"] == "multi_hop"]
    with_refs = [q["id"] for q in answerable if q.get("gold_refs")]
    return {
        "n": len(answerable),
        **{m: mean([per_q[i][m] for i in per_q]) for m in ("hit_rate", "full_recall", "mrr")},
        "multi_hop_full_recall": mean([per_q[i]["full_recall"] for i in multi]),
        "para_full_recall": mean([per_q[i].get("para_full_recall") for i in with_refs]),
        "per_question": {i: per_q[i]["full_recall"] for i in per_q},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--embed", action="append", default=[],
                   help="extra embedding model (name or path); repeatable")
    p.add_argument("--k", type=int, default=config.TOP_K)
    p.add_argument("--out", default=str(config.RESULTS_DIR / "experiments.json"))
    a = p.parse_args()

    variants = {"hybrid, bge-small (baseline)": variant(BASE),
                "+ reranker": variant(BASE, reranked),
                "+ split and cap (mode: split)": variant(BASE, split)}
    for e in a.embed:
        name = Path(e).name if Path(e).exists() else e.split("/")[-1]
        variants[f"hybrid, {name}"] = variant(e)
        variants[f"+ split and cap, {name}"] = variant(e, split)

    out = {"k": a.k, "reranker": RERANKER, "settings": {}}
    for label, version, gran, qa_name in SETTINGS:
        qa = load_qa(qa_name)
        print(f"\n{label}, k={a.k}")
        print(f"  {'variant':<46} {'hit':>6} {'full':>6} {'MRR':>6} {'multi':>6} "
              f"{'para':>6}  vs baseline (better/worse, p)")
        rows = {}
        for name, search in variants.items():
            r = rows[name] = score(search, qa, a.k, version, gran)
            base = next(iter(rows.values()))["per_question"]
            ids = sorted(base)
            up, down, pv = sign_test([base[i] for i in ids],
                                     [r["per_question"][i] for i in ids])
            r["vs_baseline"] = {"better": up, "worse": down, "p": round(pv, 4)}
            para = "n/a" if r["para_full_recall"] is None else f"{r['para_full_recall']:.3f}"
            print(f"  {name:<46} {r['hit_rate']:>6.3f} {r['full_recall']:>6.3f} "
                  f"{r['mrr']:>6.3f} {r['multi_hop_full_recall']:>6.3f} {para:>6}"
                  f"  +{up} -{down} p={pv:.3f}", flush=True)
        out["settings"][label] = rows
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
