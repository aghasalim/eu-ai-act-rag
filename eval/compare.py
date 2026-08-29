"""Compare a fresh eval against the committed one.

The two halves of this evaluation behave differently and are treated
differently here.

Retrieval is deterministic: the same index and the same questions give the same
hit rate every time, so any movement is a real change and this exits non-zero.

Generation is not. The answering model and the judge are both sampled, so a
number moving is not evidence of anything on its own. With 45 questions one
flipped answer is 2.2 points, and faithfulness currently sits at 0.902, which is
one answer away from a 0.90 line. So the generation half is reported and never
gates. It cannot honestly gate until someone runs the eval several times and
measures the spread, which `--repeat` on run_eval.py would be the way to do.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"

DETERMINISTIC = ["hit_rate", "recall", "full_recall", "mrr", "ndcg"]
SAMPLED = ["answer_accuracy_strict", "answer_accuracy_lenient", "faithfulness",
           "citation_validity", "correct_abstention_rate",
           "false_abstention_rate", "hallucination_rate_unanswerable"]


def load(name: str) -> dict:
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return json.loads(path.read_text())


def retrieval_rows(baseline: dict, fresh: dict):
    """Retrieval is nested mode -> at_k -> k -> overall -> metric, so walk it."""
    out = []
    b, f = baseline.get("retrieval", {}), fresh.get("retrieval", {})
    for mode in sorted(set(b) & set(f)):
        bk = b[mode].get("at_k", {})
        fk = f[mode].get("at_k", {})
        for k in sorted(set(bk) & set(fk), key=int):
            bo, fo = bk[k].get("overall", {}), fk[k].get("overall", {})
            for m in DETERMINISTIC:
                if isinstance(bo.get(m), (int, float)) and isinstance(fo.get(m), (int, float)):
                    out.append((f"{mode}@{k} {m}", bo[m], fo[m], fo[m] - bo[m]))
    return out


def summary_rows(baseline: dict, fresh: dict):
    out = []
    b, f = baseline.get("summary", {}), fresh.get("summary", {})
    for k in SAMPLED:
        if isinstance(b.get(k), (int, float)) and isinstance(f.get(k), (int, float)):
            out.append((k, b[k], f[k], f[k] - b[k]))
    return out


def main() -> int:
    baseline = load(sys.argv[1] if len(sys.argv) > 1 else "eval_latest.json")
    fresh = load(sys.argv[2] if len(sys.argv) > 2 else "eval_fresh.json")

    lines, failed = [], False

    det = retrieval_rows(baseline, fresh)
    if det:
        lines.append("## Retrieval, deterministic\n")
        lines.append("| metric | committed | fresh | change |")
        lines.append("|---|---:|---:|---:|")
        for k, b, f, d in det:
            flag = "" if abs(d) < 1e-9 else "  <- moved"
            lines.append(f"| {k} | {b:.4f} | {f:.4f} | {d:+.4f}{flag} |")
            if abs(d) > 1e-9:
                failed = True
        lines.append("")

    sam = summary_rows(baseline, fresh)
    if sam:
        n = fresh.get("summary", {}).get("n") or baseline.get("summary", {}).get("n")
        step = f"{100 / n:.1f}" if n else "?"
        lines.append("## Generation, sampled\n")
        lines.append(f"Reported, not gated. n={n}, so one flipped answer is {step} points.\n")
        lines.append("| metric | committed | fresh | change |")
        lines.append("|---|---:|---:|---:|")
        for k, b, f, d in sam:
            lines.append(f"| {k} | {b:.4f} | {f:.4f} | {d:+.4f} |")
        lines.append("")

    if failed:
        lines.append("Retrieval moved. That half is deterministic, so this is a real "
                     "change: either the index, the corpus or the question set differs "
                     "from what is committed.")

    report = "\n".join(lines)
    print(report)
    dest = os.environ.get("GITHUB_STEP_SUMMARY")
    if dest:
        Path(dest).write_text(report)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
