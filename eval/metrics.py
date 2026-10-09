"""Retrieval and attribution metrics. No LLM involved, these are exact.

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


def _norm(s: str) -> str:
    return " ".join(s.replace("’", "'").split()).lower()


def evidence_hit(hits: list[dict], evidence: list[str], k: int) -> float:
    """Is every gold quote inside some top-k chunk? Unit-level hit rate cannot
    tell a current provision from a superseded one with the same number, which
    is the whole problem with a stale corpus. Whitespace and case are ignored."""
    texts = [_norm(h["text"]) for h in hits[:k]]
    return float(all(any(_norm(ev) in t for t in texts) for ev in evidence))


def retrieval_scores(hits: list[dict], gold: list[str], k: int) -> dict:
    return {name: fn(hits, gold, k) for name, fn in RETRIEVAL_METRICS.items()}


def citation_validity(cited: list[str], hits: list[dict]) -> float | None:
    """Fraction of the answer's inline citations that point at a provision that
    was actually retrieved. A citation to something outside the context window is
    a fabricated attribution, the most dangerous failure in a legal assistant,
    because it looks like evidence. Costs nothing to compute and needs no judge.

    Returns None when the answer cited nothing (undefined, not zero).
    """
    if not cited:
        return None
    retrieved = {h["unit_id"] for h in hits}
    return sum(c in retrieved for c in cited) / len(cited)


# --------------------------------------------------------------------------
# Paragraph level
# --------------------------------------------------------------------------
# A reference is a unit id followed by outline markers: art_6(3), art_5(1)(f),
# anx_III(4)(a), art_113(c)(i). Paragraph-level scoring compares the unit plus
# the first marker, so art_5(1)(f) and art_5(1)(h) are the same paragraph and
# art_6(3) and art_6(4) are not. A citation with no marker (a whole article)
# only counts against gold that has no marker either: citing "Article 6" for an
# answer in Article 6(3) is not a paragraph-level citation.

def para_key(ref: str) -> str:
    """art_5(1)(f) -> art_5(1). art_113 -> art_113."""
    i = ref.find("(")
    if i < 0:
        return ref
    return ref[: ref.index(")", i) + 1]


def _cited_refs(hits: list[dict], k: int) -> list[str]:
    """Top-k chunks collapsed to unique paragraph keys of what they are cited as.
    An article-level chunk is cited as its unit."""
    seen, out = set(), []
    for h in hits[:k]:
        key = para_key(h.get("ref") or h["unit_id"])
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def citation_precision(cited: list[str], gold_refs: list[str]) -> float | None:
    """Share of cited references that name a gold paragraph. None if nothing
    was cited, as with citation_validity."""
    if not cited:
        return None
    gold = {para_key(g) for g in gold_refs}
    keys = list(dict.fromkeys(para_key(c) for c in cited))
    return sum(c in gold for c in keys) / len(keys)


def paragraph_scores(hits: list[dict], gold_refs: list[str], k: int) -> dict:
    """Paragraph-level hit rate, recall, MRR and citation precision of the top k,
    treating each retrieved chunk as a citation of what it is labelled as."""
    cited = _cited_refs(hits, k)
    gold = list(dict.fromkeys(para_key(g) for g in gold_refs))
    found = [g for g in gold if g in cited]
    first = next((i for i, c in enumerate(cited, 1) if c in gold), 0)
    return {
        "para_hit_rate": float(bool(found)),
        "para_recall": len(found) / len(gold),
        "para_full_recall": float(len(found) == len(gold)),
        "para_mrr": 1.0 / first if first else 0.0,
        "citation_precision": citation_precision(cited, gold_refs) or 0.0,
    }


PARAGRAPH_METRICS = ("para_hit_rate", "para_recall", "para_full_recall",
                     "para_mrr", "citation_precision")
