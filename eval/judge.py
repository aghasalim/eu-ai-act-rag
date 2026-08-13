"""LLM-as-judge scoring for faithfulness and answer correctness.

Two guards against the obvious criticism of LLM-graded evaluation:

* The judge is a *different model* from the generator (see `pick_judge`), so the
  system is not grading its own homework. The judge actually used is recorded in
  the results file.
* Faithfulness is scored per *claim*, not per answer. Asking "is this answer
  faithful?" invites a vague holistic yes; asking "is this specific sentence
  supported by these excerpts?" is a question with a defensible answer, and it
  localises which sentence went wrong.

`eval/run_eval.py` additionally cross-checks these numbers against RAGAS when it
is installed, so the headline faithfulness figure is never from a single source.
"""
from __future__ import annotations

import json
import os
import re

import httpx

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.euactrag import config, llm  # noqa: E402

# Preference order for the judge, most capable first. Anything from a different
# family than the generator is acceptable.
JUDGE_PREFERENCE = [
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
    "gemma2-9b-it",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]


def list_models(provider: str | None = None) -> list[str]:
    provider = provider or config.LLM_PROVIDER
    base, key_env = llm.PROVIDERS[provider]
    key = os.getenv(key_env) if key_env else "ollama"
    try:
        r = httpx.get(f"{base}/models",
                      headers={"Authorization": f"Bearer {key}"}, timeout=30)
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]
    except Exception:
        return []


def pick_judge(generator_model: str, provider: str | None = None) -> str:
    """Choose the best available judge that is not the generator itself."""
    available = set(list_models(provider))
    for m in JUDGE_PREFERENCE:
        if m != generator_model and (not available or m in available):
            return m
    return config.JUDGE_MODEL


def _json(text: str) -> dict | list | None:
    """Tolerant JSON extraction -- judges sometimes wrap output in prose or fences."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"[\[{].*[\]}]", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


CLAIMS_PROMPT = """Break the ANSWER into atomic factual claims. A claim is one \
verifiable assertion. Ignore hedging, citations and connective phrases.

Return ONLY a JSON array of strings, e.g. ["claim one", "claim two"].
Return [] if the answer makes no factual assertions.

ANSWER:
{answer}"""

VERIFY_PROMPT = """You are checking whether a claim is supported by source excerpts \
from Regulation (EU) 2024/1689.

A claim is SUPPORTED only if it follows from the excerpts alone. If it requires \
outside knowledge, contradicts the excerpts, or adds specifics (numbers, dates, \
conditions) not present in them, it is NOT supported. Being true in the real world \
is irrelevant -- only the excerpts count.

EXCERPTS:
{context}

CLAIMS:
{claims}

Return ONLY a JSON array, one object per claim in order:
[{{"claim_index": 0, "supported": true, "reason": "short quote or reason"}}]"""

CORRECTNESS_PROMPT = """Compare a CANDIDATE answer against a REFERENCE answer for a \
question about Regulation (EU) 2024/1689.

Grade:
- "correct": conveys all the substantive content of the reference; wording may differ.
- "partial": correct as far as it goes but omits a required element (e.g. states \
the rule but omits its exception, or answers one hop of a two-hop question).
- "incorrect": contradicts the reference, or states the wrong figure, date or rule.

Abstention ("{abstain}") is "incorrect" when the reference is a real answer.

QUESTION: {question}
REFERENCE: {reference}
CANDIDATE: {candidate}

Return ONLY JSON: {{"grade": "correct|partial|incorrect", "reason": "one sentence"}}"""


def faithfulness(answer: str, contexts: list[str], model: str) -> dict:
    """RAGAS-style claim-level faithfulness, implemented directly so the harness
    has no hard dependency on RAGAS being installable."""
    raw = llm.chat([{"role": "user", "content": CLAIMS_PROMPT.format(answer=answer)}],
                   model=model, temperature=0.0, max_tokens=800)
    claims = _json(raw)
    if not isinstance(claims, list) or not claims:
        return {"score": None, "n_claims": 0, "claims": []}
    claims = [str(c) for c in claims][:20]

    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims))
    raw = llm.chat(
        [{"role": "user", "content": VERIFY_PROMPT.format(
            context="\n\n".join(contexts), claims=numbered)}],
        model=model, temperature=0.0, max_tokens=1500)
    verdicts = _json(raw)
    if not isinstance(verdicts, list) or not verdicts:
        return {"score": None, "n_claims": len(claims), "claims": claims}

    supported = [bool(v.get("supported")) for v in verdicts if isinstance(v, dict)]
    if not supported:
        return {"score": None, "n_claims": len(claims), "claims": claims}
    return {
        "score": sum(supported) / len(supported),
        "n_claims": len(supported),
        "claims": claims,
        "verdicts": verdicts,
    }


def correctness(question: str, reference: str, candidate: str, model: str) -> dict:
    raw = llm.chat(
        [{"role": "user", "content": CORRECTNESS_PROMPT.format(
            question=question, reference=reference, candidate=candidate,
            abstain=config.ABSTAIN_STRING)}],
        model=model, temperature=0.0, max_tokens=300)
    d = _json(raw)
    if not isinstance(d, dict) or d.get("grade") not in {"correct", "partial", "incorrect"}:
        return {"grade": "unscored", "reason": "judge returned unparseable output"}
    return d
