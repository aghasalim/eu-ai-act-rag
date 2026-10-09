"""Check every question set against the text of the Act.

For each question with an answer:
* every gold reference must exist in the paragraph-level corpus it is scored on;
* every evidence quote must appear, verbatim up to whitespace and case, inside a
  chunk cited to one of the question's gold paragraphs;
* in qa_extended.jsonl every gold paragraph must hold at least one quote, so no
  reference is there without the wording that supports it.

It also reports which questions quote wording that is absent from the 2024 text,
i.e. questions that only the amended Act can answer. Exits 1 on any problem.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.metrics import para_key  # noqa: E402
from src.euactrag import config  # noqa: E402
from src.euactrag.ingest import load_chunks  # noqa: E402

# Which text each set is checked against.
SETS = {"qa_set.jsonl": config.ORIGINAL, "qa_amended.jsonl": config.LATEST,
        "qa_extended.jsonl": config.LATEST}


def _norm(s: str) -> str:
    return " ".join(s.replace("’", "'").split()).lower()


def check(name: str, version: str) -> tuple[list[str], list[dict]]:
    chunks = load_chunks(version, "paragraph")
    known = {c["unit_id"] for c in chunks} | {
        r for c in chunks for r in c["meta"].get("refs", [])}
    texts = [(para_key(c["meta"].get("ref", c["unit_id"])), _norm(c["text"]))
             for c in chunks]
    old = [_norm(c["text"]) for c in load_chunks(config.ORIGINAL, "paragraph")]
    problems, rows = [], []
    for q in (json.loads(l) for l in open(ROOT / "eval" / name, encoding="utf-8")
              if l.strip()):
        gold = {para_key(r) for r in q.get("gold_refs", [])}
        for r in q.get("gold_refs", []):
            if r not in known:
                problems.append(f"{name} {q['id']}: {r} is not in the {version} text")
            if r.split("(")[0] not in q["gold_units"]:
                problems.append(f"{name} {q['id']}: {r} is outside gold_units")
        supported = set()
        for ev in q.get("evidence", []):
            where = {k for k, t in texts if _norm(ev) in t}
            if not where & gold:
                problems.append(f"{name} {q['id']}: quote not in a gold paragraph "
                                f"(found in {sorted(where) or 'nothing'}): {ev[:60]}")
            supported |= where & gold
        if name == "qa_extended.jsonl" and supported != gold:
            problems.append(f"{name} {q['id']}: no quote for {sorted(gold - supported)}")
        new_only = bool(q.get("evidence")) and any(
            not any(_norm(ev) in t for t in old) for ev in q["evidence"])
        if "new_in_2026" in q and q["new_in_2026"] != new_only:
            problems.append(f"{name} {q['id']}: new_in_2026 says {q['new_in_2026']}, "
                            f"the text says {new_only}")
        rows.append({"id": q["id"], "topic": q.get("topic", ""), "new_only": new_only})
    return problems, rows


def main() -> int:
    bad = 0
    for name, version in SETS.items():
        problems, rows = check(name, version)
        bad += len(problems)
        for p in problems:
            print("FAIL", p)
        topics = Counter(r["topic"] for r in rows if r["topic"])
        new = sum(r["new_only"] for r in rows)
        print(f"{name}: {len(rows)} questions checked against {version}, "
              f"{new} quote wording the 2024 text does not have"
              + (f", topics {dict(topics)}" if topics else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
