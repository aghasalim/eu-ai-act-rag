"""End-to-end RAG: retrieve -> ground -> generate -> attribute.

Two prompt decisions carry most of the faithfulness result, so they are stated
here rather than buried:

1. The model is given a literal abstention token and told the corpus boundary.
   Without an explicit escape hatch an instruction-tuned model will almost always
   produce *something*, and on out-of-scope questions that "something" is a
   hallucination. Measuring abstention separately (see eval/) is the only way to
   see this.
2. Every excerpt is labelled with its citation, and the model is required to cite
   the label. That makes an unsupported claim visible to a human reader instead
   of merely plausible.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config, llm, retrieve

SYSTEM = """You are a careful legal research assistant answering questions about \
Regulation (EU) 2024/1689 (the EU AI Act). You have no knowledge outside the \
excerpts provided to you.

Rules:
1. Answer ONLY from the numbered excerpts below. Never use outside knowledge, and \
never infer a rule that the excerpts do not state.
2. Cite the source of every claim inline using its bracketed label exactly as \
given, e.g. [Article 6 - Classification rules for high-risk AI systems].
3. If the excerpts do not contain enough information to answer, reply with exactly \
{abstain} and nothing else. This includes questions about other laws (GDPR, the \
Digital Services Act), about facts not in the Regulation, and about anything the \
excerpts do not cover.
4. Recitals are non-binding interpretive context. Prefer Articles and Annexes for \
operative rules, and say so when you rely on a Recital.
5. Be precise and concise. Quote the operative wording where the exact phrasing \
matters. Do not speculate about intent."""

USER = """Excerpts:

{context}

---
Question: {question}

Answer using only the excerpts above, citing each claim. If the answer is not in \
the excerpts, reply exactly {abstain}."""


@dataclass
class Answer:
    question: str
    answer: str
    contexts: list[dict] = field(default_factory=list)
    abstained: bool = False
    cited_units: list[str] = field(default_factory=list)
    mode: str = ""
    model: str = ""

    @property
    def retrieved_units(self) -> list[str]:
        seen, out = set(), []
        for c in self.contexts:
            if c["unit_id"] not in seen:
                seen.add(c["unit_id"])
                out.append(c["unit_id"])
        return out


def format_context(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[{h['citation']}]\n{h['text'].split(chr(10), 1)[-1]}" for h in hits
    )


# Case-insensitive on purpose: the Official Journal renders annex headings in
# caps ("ANNEX III"), so the model cites them that way, while articles and
# recitals are title case. A case-sensitive pattern silently drops every annex
# citation, which would quietly inflate citation-validity scores.
#
# Both bracket families are accepted because the model chooses, not us:
# gpt-oss-120b emits full-width 【...】 while llama emits [...]. Matching only
# ASCII brackets silently parsed zero citations from one of them, which does not
# fail loudly -- it just reports citation validity as "no citations made".
_CITE = re.compile(
    r"[\[【]\s*(Article|Annex|Recital)[^\]】]*?(\d+|[IVXLC]+)[^\]】]*?[\]】]", re.I
)
_KIND = {"article": "art", "annex": "anx", "recital": "rct"}


def parse_citations(text: str) -> list[str]:
    """Map inline citation labels back to corpus unit ids, order-preserving."""
    seen, out = set(), []
    for m in _CITE.finditer(text):
        kind = _KIND[m.group(1).lower()]
        ref = m.group(2)
        uid = f"{kind}_{ref.upper() if kind == 'anx' else ref}"
        if uid not in seen:
            seen.add(uid)
            out.append(uid)
    return out


def answer(
    question: str,
    k: int | None = None,
    mode: str | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> Answer:
    mode = mode or config.RETRIEVAL_MODE
    hits = retrieve.search(question, k=k, mode=mode)
    res = Answer(question=question, answer="", contexts=hits, mode=mode,
                 model=model or config.LLM_MODEL)

    if not llm.available(provider):
        # Retrieval-only degradation: no key, no invented answer.
        res.answer = (
            "_No generation backend configured (set GROQ_API_KEY in .env). "
            "Showing retrieved passages only._"
        )
        res.model = "extractive"
        return res

    out = llm.chat(
        [
            {"role": "system", "content": SYSTEM.format(abstain=config.ABSTAIN_STRING)},
            {"role": "user", "content": USER.format(
                context=format_context(hits), question=question,
                abstain=config.ABSTAIN_STRING)},
        ],
        model=model,
        provider=provider,
    )
    res.answer = out
    res.abstained = config.ABSTAIN_STRING in out
    res.cited_units = parse_citations(out)
    return res


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What are the prohibited AI practices?"
    a = answer(q)
    print(f"\nQ: {a.question}\n\nA: {a.answer}\n")
    print("Sources:")
    for c in a.contexts:
        print(f"  [{c['rank']}] {c['citation']}  (score={c['score']:.4f})")
