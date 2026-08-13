"""Render RESULTS.md from an eval json, so the reported numbers are always
generated from the artefact rather than typed by hand."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.euactrag import config  # noqa: E402

FAILURE_GLOSS = {
    "ok": "Answered correctly (or correctly refused).",
    "retrieval_miss": "No gold provision in the top-k. The generator never had a chance.",
    "partial_retrieval": "Multi-hop question where only some required provisions were retrieved.",
    "false_abstention": "Refused although the evidence was retrieved.",
    "generation_error": "Right passages retrieved, wrong answer produced.",
    "incomplete_answer": "Right direction, missing a required element (e.g. an exception).",
    "hallucination_no_abstain": "Out-of-scope question answered instead of refused.",
}


def fmt(v, pct=False):
    if v is None:
        return "—"
    return f"{v * 100:.1f}%" if pct else f"{v:.3f}"


def main(tag: str = "latest") -> None:
    path = config.RESULTS_DIR / f"eval_{tag}.json"
    d = json.loads(path.read_text())
    cfg, out = d["config"], []
    A = out.append

    A("# Evaluation results\n")
    A(f"Generated from `{path.relative_to(ROOT)}`. "
      "Regenerate with `make eval && make report`.\n")
    A("| setting | value |\n|---|---|")
    for k in ("embed_model", "gen_model", "judge_model", "gen_mode", "top_k",
              "rrf_k", "n_chunks", "n_questions"):
        if k in cfg:
            A(f"| {k} | `{cfg[k]}` |")
    A("")

    # --- retrieval -------------------------------------------------------
    A("## 1. Retrieval quality\n")
    A("Scored at *provision* level: retrieving any chunk of the correct article "
      "counts as a hit, since the user is directed to the right provision. "
      "`full_recall` requires **every** gold provision — the metric that matters "
      "for multi-hop questions.\n")
    k = str(cfg["top_k"])
    A(f"### All strategies @ k={k}\n")
    A("| strategy | hit rate | recall | full recall | precision | MRR | nDCG | s/query |")
    A("|---|---|---|---|---|---|---|---|")
    for mode, md in d["retrieval"].items():
        s = md["at_k"][k]["overall"]
        A(f"| **{mode}** | {fmt(s['hit_rate'], 1)} | {fmt(s['recall'], 1)} | "
          f"{fmt(s['full_recall'], 1)} | {fmt(s['precision'], 1)} | "
          f"{fmt(s['mrr'])} | {fmt(s['ndcg'])} | {md['latency_s_per_query']:.3f} |")
    A("")

    A("### Single-hop vs multi-hop\n")
    A("| strategy | type | hit rate | recall | full recall | MRR |")
    A("|---|---|---|---|---|---|")
    for mode, md in d["retrieval"].items():
        for t, s in md["at_k"][k]["by_type"].items():
            A(f"| {mode} | {t} | {fmt(s['hit_rate'], 1)} | {fmt(s['recall'], 1)} | "
              f"{fmt(s['full_recall'], 1)} | {fmt(s['mrr'])} |")
    A("")

    ks = sorted(next(iter(d["retrieval"].values()))["at_k"], key=int)
    A(f"### Sensitivity to k ({', '.join(ks)})\n")
    A("| strategy | " + " | ".join(f"hit@{x}" for x in ks) + " | "
      + " | ".join(f"full@{x}" for x in ks) + " |")
    A("|---" * (1 + 2 * len(ks)) + "|")
    for mode, md in d["retrieval"].items():
        hits = [fmt(md["at_k"][x]["overall"]["hit_rate"], 1) for x in ks]
        full = [fmt(md["at_k"][x]["overall"]["full_recall"], 1) for x in ks]
        A(f"| {mode} | " + " | ".join(hits) + " | " + " | ".join(full) + " |")
    A("")

    # --- generation ------------------------------------------------------
    if "summary" not in d:
        A("## 2. Generation\n\n_Not run: no LLM key was configured._\n")
        (ROOT / "RESULTS.md").write_text("\n".join(out))
        print(f"-> RESULTS.md (retrieval only)")
        return

    s = d["summary"]
    A("## 2. Answer quality, faithfulness and abstention\n")
    A("| metric | value | what it means |")
    A("|---|---|---|")
    A(f"| Answer accuracy (strict) | {fmt(s['answer_accuracy_strict'], 1)} | "
      "judged fully equivalent to the hand-written reference |")
    A(f"| Answer accuracy (incl. partial) | {fmt(s['answer_accuracy_lenient'], 1)} | "
      "correct but possibly missing an element |")
    A(f"| **Faithfulness** | {fmt(s['faithfulness'], 1)} | "
      "share of atomic claims entailed by the retrieved passages |")
    A(f"| **Citation validity** | {fmt(s['citation_validity'], 1)} | "
      "share of inline citations pointing at a passage actually retrieved |")
    A(f"| **Correct abstention** | {fmt(s['correct_abstention_rate'], 1)} | "
      "out-of-scope questions correctly refused |")
    A(f"| Hallucination rate (out-of-scope) | {fmt(s['hallucination_rate_unanswerable'], 1)} | "
      "out-of-scope questions answered anyway |")
    A(f"| False abstention | {fmt(s['false_abstention_rate'], 1)} | "
      "answerable questions wrongly refused |")
    A(f"| Mean latency | {s['mean_latency_s']:.2f}s | end-to-end per question |")
    A("")

    A("### By question type\n")
    A("| type | n | accuracy (strict) | faithfulness |")
    A("|---|---|---|---|")
    for t, v in s["by_type"].items():
        A(f"| {t} | {v['n']} | {fmt(v['accuracy_strict'], 1)} | {fmt(v['faithfulness'], 1)} |")
    A("")

    if d.get("ragas") and not d["ragas"].get("skipped") and not d["ragas"].get("error"):
        A("### RAGAS cross-check\n")
        A("Independent second measurement of the same answers.\n")
        A("| ragas metric | value |\n|---|---|")
        for kk, vv in d["ragas"].items():
            A(f"| {kk} | {fmt(vv)} |")
        A("")
    elif d.get("ragas"):
        A(f"_RAGAS cross-check unavailable: {d['ragas'].get('skipped') or d['ragas'].get('error')}_\n")

    # --- failures --------------------------------------------------------
    A("## 3. Where it fails\n")
    A("| failure mode | n | meaning |")
    A("|---|---|---|")
    for mode, n in sorted(s["failure_modes"].items(), key=lambda x: -x[1]):
        A(f"| `{mode}` | {n} | {FAILURE_GLOSS.get(mode, '')} |")
    A("")

    rows = d["generation"]["rows"]
    bad = [r for r in rows if r["failure"] != "ok"]
    if bad:
        A("### Every failing question\n")
        A("| id | type | failure | question | why |")
        A("|---|---|---|---|---|")
        for r in bad:
            why = (r.get("grade_reason") or "").replace("|", "/")[:150]
            q = r["question"].replace("|", "/")[:90]
            A(f"| `{r['id']}` | {r['type']} | `{r['failure']}` | {q} | {why} |")
        A("")

    unsup = [(r["id"], v.get("claim_index"), (v.get("reason") or "")[:120])
             for r in rows for v in r.get("unsupported_claims", [])]
    if unsup:
        A("### Claims the judge could not trace to a retrieved passage\n")
        A("| question | claim # | judge reason |\n|---|---|---|")
        for qid, ci, reason in unsup[:25]:
            A(f"| `{qid}` | {ci} | {reason.replace('|', '/')} |")
        A("")

    (ROOT / "RESULTS.md").write_text("\n".join(out))
    print("-> RESULTS.md")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "latest")
