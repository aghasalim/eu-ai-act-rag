"""How far to trust the LLM judge and the rule-based metrics.

eval/human_labels.jsonl holds 41 generated answers that I graded by hand from
the text of the Act, without looking at the judge's grade: correct, partial or
incorrect against the reference answer, and which provisions each answer cites
as a reader sees them. They are every non-empty, non-abstaining answer in two
runs of the same generator (rows_latest.jsonl and rows_cap800_superseded.jsonl),
with answers identical across the two runs counted once. The four empty
answers from the token-cap bug are left out, since grading them tests nothing.

Three comparisons, all on stored outputs, so no API key is needed:

1. Judge correctness grade against mine: accuracy and Cohen's kappa, on the
   three grades and on correct against not correct.
2. parse_citations (a regex) against the provisions I read as cited.
3. The abstention rule (the answer contains NOT_IN_CORPUS) against my reading
   of whether the answer refuses, on all 45 rows of the published run.

`--rejudge MODEL` grades the same 41 answers again with a live judge and
compares that too. It needs an API key, see .env.example.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.euactrag import config  # noqa: E402
from src.euactrag.pipeline import parse_citations  # noqa: E402

LABELS = ROOT / "eval" / "human_labels.jsonl"
RUNS = {"latest": "rows_latest.jsonl", "cap800": "rows_cap800_superseded.jsonl"}
OUT = config.RESULTS_DIR / "agreement.json"


def cohen_kappa(a: list, b: list) -> float | None:
    """Agreement beyond chance for two raters over the same items."""
    n = len(a)
    if n == 0 or n != len(b):
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in ca.keys() | cb.keys()) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def agree(a: list, b: list) -> dict:
    return {"n": len(a), "accuracy": round(sum(x == y for x, y in zip(a, b)) / len(a), 4),
            "kappa": None if (k := cohen_kappa(a, b)) is None else round(k, 4)}


def load_rows() -> dict[tuple[str, str], dict]:
    out = {}
    for src, name in RUNS.items():
        for line in open(config.RESULTS_DIR / name, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out[(src, r["id"])] = r
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rejudge", metavar="MODEL",
                   help="also grade the labelled answers with this judge model (needs a key)")
    a = p.parse_args()

    labels = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
    rows = load_rows()
    items = [(lab, rows[(lab["source"], lab["id"])]) for lab in labels]
    human = [lab["human_grade"] for lab, _ in items]
    judge = [r["grade"] for _, r in items]
    binary = lambda gs: ["correct" if g == "correct" else "not correct" for g in gs]

    out = {
        "judge_model": sorted({r.get("judge_model") or "unknown" for _, r in items}),
        "judge_grade_3way": agree(human, judge),
        "judge_grade_binary": agree(binary(human), binary(judge)),
        "confusion_human_by_judge": {f"{h} / {j}": n for (h, j), n in
                                     sorted(Counter(zip(human, judge)).items())},
        "disagreements": [{"item": f"{lab['source']}:{lab['id']}", "human": lab["human_grade"],
                           "judge": r["grade"], "judge_reason": r.get("grade_reason", "")}
                          for lab, r in items if lab["human_grade"] != r["grade"]],
    }

    parsed = [set(parse_citations(r["answer"])) for _, r in items]
    hand = [set(lab["human_cited_units"]) for lab, _ in items]
    tp = sum(len(p & h) for p, h in zip(parsed, hand))
    out["citation_parser"] = {
        "n": len(items),
        "exact_match": round(sum(p == h for p, h in zip(parsed, hand)) / len(items), 4),
        "precision": round(tp / max(1, sum(map(len, parsed))), 4),
        "recall": round(tp / max(1, sum(map(len, hand))), 4),
        "misses": [{"item": f"{lab['source']}:{lab['id']}", "parsed": sorted(p),
                    "human": sorted(h)} for (lab, _), p, h in zip(items, parsed, hand) if p != h],
    }

    # Abstention: on the published run every abstaining answer is exactly the
    # token and every other answer is a real attempt (checked by reading all 45).
    latest = [r for (src, _), r in rows.items() if src == "latest"]
    rule = [config.ABSTAIN_STRING in r["answer"] for r in latest]
    read = [r["answer"].strip() == config.ABSTAIN_STRING for r in latest]
    out["abstention_rule"] = agree(rule, read)

    if a.rejudge:
        out["rejudge"] = rejudge(items, human, a.rejudge)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("disagreements",)}, indent=2, ensure_ascii=False))
    print(f"-> {OUT}")


def rejudge(items, human, model: str) -> dict:
    from dotenv import load_dotenv

    from eval import judge as judge_mod
    from src.euactrag import llm

    load_dotenv()
    if not llm.available():
        print("[skipped rejudge] no LLM key configured, see .env.example")
        return {"skipped": "no API key"}
    qa = {json.loads(l)["id"]: json.loads(l)
          for l in open(ROOT / "eval" / "qa_set.jsonl", encoding="utf-8") if l.strip()}
    grades = [judge_mod.correctness(r["question"], qa[r["id"]]["reference"], r["answer"],
                                    model)["grade"] for _, r in items]
    return {"model": model, "grades": grades, "3way": agree(human, grades)}


if __name__ == "__main__":
    main()
