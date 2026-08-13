"""Retrieval and attribution metrics. No LLM involved -- these are exact.

Everything is scored at *unit* level (art_6, anx_III, rct_27) rather than chunk
level. A long article is split across several chunks, and retrieving any chunk of
the right article is a retrieval success for a citation task: the user gets sent
to the right provision. Scoring at chunk level would punish the system for a
split that we chose ourselves, which would flatter or damage the numbers for
reasons unrelated to retrieval quality.
"""
from __future__ import annotations

import math


def _units(hits: list[dict], k: int) -> list[str]:
    """Top-k retrieved chunks collapsed to unique unit ids, order preserved."""
    seen, out = set(), []
    for h in hits[:k]:
        u = h["unit_id"]
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def hit_rate(hits, gold, k) -> float:
    """Did we surface at least one relevant provision? The floor for usefulness."""
    return float(bool(set(_units(hits, k)) & set(gold)))


def recall(hits, gold, k) -> float:
    """Fraction of the gold provisions retrieved."""
    return len(set(_units(hits, k)) & set(gold)) / len(gold)


def full_recall(hits, gold, k) -> float:
    """All gold provisions retrieved. This is the metric that matters for
    multi-hop questions: retrieving 1 of 2 required articles yields an answer
    that is confidently half-right, which is worse than a visible miss."""
    return float(set(gold) <= set(_units(hits, k)))


def precision(hits, gold, k) -> float:
    u = _units(hits, k)
    return len(set(u) & set(gold)) / len(u) if u else 0.0


def mrr(hits, gold, k) -> float:
    for i, u in enumerate(_units(hits, k), 1):
        if u in gold:
            return 1.0 / i
    return 0.0


def ndcg(hits, gold, k) -> float:
    """Binary-gain nDCG@k. Rewards ranking gold provisions near the top, which
    matters because the generator only sees what fits in its context window."""
    dcg = sum(1.0 / math.log2(i + 1)
              for i, u in enumerate(_units(hits, k), 1) if u in gold)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / idcg if idcg else 0.0


RETRIEVAL_METRICS = {
    "hit_rate": hit_rate, "recall": recall, "full_recall": full_recall,
    "precision": precision, "mrr": mrr, "ndcg": ndcg,
}


def retrieval_scores(hits: list[dict], gold: list[str], k: int) -> dict:
    return {name: fn(hits, gold, k) for name, fn in RETRIEVAL_METRICS.items()}


def citation_validity(cited: list[str], hits: list[dict]) -> float | None:
    """Fraction of the answer's inline citations that point at a provision that
    was actually retrieved. A citation to something outside the context window is
    a fabricated attribution -- the most dangerous failure in a legal assistant,
    because it looks like evidence. Costs nothing to compute and needs no judge.

    Returns None when the answer cited nothing (undefined, not zero).
    """
    if not cited:
        return None
    retrieved = {h["unit_id"] for h in hits}
    return sum(c in retrieved for c in cited) / len(cited)
