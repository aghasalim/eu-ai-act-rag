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


def _emit(out: list[str], check: bool) -> None:
    """Write RESULTS.md, or in check mode fail if it has drifted.

    RESULTS.md is generated, so hand editing it is silently undone by the next
    run. That has happened twice: a punctuation pass ate the space after the
    pipe in 42 table cells, and an arrow was typed into the recital table by
    hand. `report.py --check` is what catches it, and CI runs it.
    """
    text = "\n".join(out)
    path = ROOT / "RESULTS.md"
    if not check:
        path.write_text(text)
        return
    current = path.read_text() if path.exists() else ""
    if current == text:
        print("RESULTS.md is up to date")
        return
    cur, want = current.split("\n"), text.split("\n")
    for i in range(max(len(cur), len(want))):
        a = cur[i] if i < len(cur) else "<end of file>"
        b = want[i] if i < len(want) else "<end of file>"
        if a != b:
            raise SystemExit(
                f"RESULTS.md has drifted from eval/report.py at line {i + 1}.\n"
                f"  committed: {a}\n  generated: {b}\n"
                "Run `python eval/report.py` and commit the result. Do not edit "
                "RESULTS.md by hand, it is generated."
            )


def fmt(v: float | None, pct: bool = False) -> str:
    # "n/a" for a value that does not exist, never a bare hyphen: a hyphen in a
    # table cell reads as a minus sign next to columns that hold numbers.
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%" if pct else f"{v:.3f}"


README_START = "<!-- RETRIEVAL_TABLE:START -->"
README_END = "<!-- RETRIEVAL_TABLE:END -->"

#: Labels the README uses for the strategies, which are friendlier than the keys.
README_LABELS = {"dense": "dense (bge-small)", "bm25": "BM25", "hybrid": "**hybrid (RRF)**"}


def readme_table(d: dict, k: str) -> list[str]:
    """The README's retrieval table, from the same json RESULTS.md is built from.

    This existed as hand-typed markdown until the numbers drifted: the README
    claimed 0.402 s/query for dense while RESULTS.md said 0.245, and a later eval
    run moved hybrid MRR without the table following. Generating both from one
    artefact is the only version of this that stays true.
    """
    rows = [
        "| strategy | hit rate | recall | full recall | MRR | nDCG | s/query |",
        "|---|---|---|---|---|---|---|",
    ]
    for mode, md in d["retrieval"].items():
        s = md["at_k"][k]["overall"]
        label = README_LABELS.get(mode, mode)
        bold = "**{}**".format if label.startswith("**") else str
        rows.append(
            f"| {label} | {bold(fmt(s['hit_rate'], 1))} | {bold(fmt(s['recall'], 1))} | "
            f"{bold(fmt(s['full_recall'], 1))} | {bold(fmt(s['mrr']))} | "
            f"{bold(fmt(s['ndcg']))} | {md['latency_s_per_query']:.3f} |"
        )
    return rows


STALE_START = "<!-- STALE_TABLE:START -->"
STALE_END = "<!-- STALE_TABLE:END -->"

#: The four retrieval runs behind the stale-law comparison: which questions,
#: which text of the Act they searched, and the results file. The first is the
#: published run. All four come from `make eval-retrieval` / `make eval-versions`.
STALE_RUNS = [
    ("original", config.ORIGINAL, "latest"),
    ("original", config.LATEST, f"original_{config.LATEST}"),
    ("amended", config.ORIGINAL, f"amended_{config.ORIGINAL}"),
    ("amended", config.LATEST, f"amended_{config.LATEST}"),
]


def _stale_runs() -> list[tuple[str, str, dict]] | None:
    paths = [config.RESULTS_DIR / f"eval_{tag}.json" for _, _, tag in STALE_RUNS]
    if not all(p.exists() for p in paths):
        return None
    return [(qs, ver, json.loads(p.read_text()))
            for (qs, ver, _), p in zip(STALE_RUNS, paths)]


def stale_table(k: str) -> list[str]:
    """Hybrid retrieval on each question set against each text of the Act.

    "current wording in top k" is the evidence check: every quote the answer
    rests on appears in a retrieved chunk. Only the amended questions have
    quotes, so it is n/a for the original set.
    """
    runs = _stale_runs()
    if runs is None:
        return ["_Not run yet: `make eval-versions`._"]
    rows = [
        "| questions | text searched | answerable | hit rate | full recall | MRR "
        "| current wording in top k |",
        "|---|---|---|---|---|---|---|",
    ]
    for qs, ver, d in runs:
        r = d["retrieval"]["hybrid"]["at_k"][k]
        s = r["overall"]
        rows.append(
            f"| {qs} | {ver} | {len(r['per_question'])} | {fmt(s['hit_rate'], 1)} | "
            f"{fmt(s['full_recall'], 1)} | {fmt(s['mrr'])} | "
            f"{fmt(s.get('evidence_hit'), 1)} |")
    old = runs[2][2]["retrieval"]["hybrid"]["at_k"][k]["overall"]
    new = runs[3][2]["retrieval"]["hybrid"]["at_k"][k]["overall"]
    rows += [
        "",
        f"On the amended questions the 2024 text still finds a provision with the "
        f"right number for {fmt(old['hit_rate'], 1)} of them, and the current "
        f"wording for {fmt(old['evidence_hit'], 1)}. On the 2026 text that is "
        f"{fmt(new['hit_rate'], 1)} and {fmt(new['evidence_hit'], 1)}.",
    ]
    return rows


def stale_section(k: str) -> list[str]:
    runs = _stale_runs()
    if runs is None:
        return []
    out = [
        "## 4. Stale law: the 2024 text against the amended text\n",
        "Regulation (EU) 2026/1744 amended the Act on 27 July 2026. Hybrid "
        f"retrieval at k={k} on both question sets and both texts. The amended "
        "questions (`eval/qa_amended.jsonl`) can only be answered from the 2026 "
        "text, and each carries the exact wording its answer rests on. A hit "
        "means a chunk with the right article number came back; on a stale "
        "corpus that article can hold the old rule, so the last column checks "
        "for the current wording instead. Files: "
        + ", ".join(f"`eval_{tag}.json`" for _, _, tag in STALE_RUNS) + ".\n",
    ]
    out += stale_table(k)
    out += ["", "### Every amended question\n",
            "| id | type | gold | hit, 2024 text | wording, 2024 text "
            "| hit, 2026 text | wording, 2026 text |",
            "|---|---|---|---|---|---|---|"]
    yn = lambda v: "yes" if v else "no"
    old = runs[2][2]["retrieval"]["hybrid"]["at_k"][k]["per_question"]
    new = runs[3][2]["retrieval"]["hybrid"]["at_k"][k]["per_question"]
    qa_path = ROOT / "eval" / "qa_amended.jsonl"
    qa = [json.loads(l) for l in qa_path.read_text().splitlines() if l.strip()]
    for q in qa:
        o, n = old[q["id"]], new[q["id"]]
        out.append(
            f"| `{q['id']}` | {q['type']} | {', '.join(q['gold_units'])} | "
            f"{yn(o['hit_rate'])} | {yn(o['evidence_hit'])} | "
            f"{yn(n['hit_rate'])} | {yn(n['evidence_hit'])} |")
    out += ["", "Generated answers were not re-run for these: no API key on the "
            "machine that produced these files. Every number in this section is "
            "retrieval, which needs no model.\n"]
    return out


def _replace_between(text: str, start: str, end: str, body: str) -> str | None:
    if start not in text or end not in text:
        return None
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{body}\n{end}{tail}"


def update_readme(d: dict, k: str, check: bool = False) -> None:
    """Rewrite the README tables in place between their markers.

    In check mode this writes nothing and fails on drift, so the check cannot
    quietly repair the thing it is meant to be reporting.
    """
    path = ROOT / "README.md"
    text = path.read_text()
    want = text
    for name, start, end, body in (
        ("retrieval", README_START, README_END, readme_table(d, k)),
        ("stale-law", STALE_START, STALE_END, stale_table(k)),
    ):
        new = _replace_between(want, start, end, "\n".join(body))
        if new is None:
            print(f"!! README {name} markers missing, table not updated")
            continue
        want = new
    if check:
        if want != text:
            raise SystemExit(
                "The README tables have drifted from eval/report.py. "
                "Run `python eval/report.py` and commit the result."
            )
        print("README tables are up to date")
        return
    path.write_text(want)
    print("-> README.md (tables)")


def main(tag: str = "latest", check: bool = False) -> None:
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
      "`full_recall` requires **every** gold provision, the metric that matters "
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

    if d.get("ablation_recital_weight"):
        A("### Ablation: down-weighting recitals\n")
        A("Recitals restate the operative rules in flowing prose, so they match a "
          "natural-language question *better* than the terse article that actually "
          "contains the rule, and were crowding binding provisions out of the "
          "top-k. `w` is the weight a recital carries in rank fusion relative to a "
          "binding provision.\n")
        A(f"| w | hit rate | recall | full recall | MRR | nDCG |")
        A("|---|---|---|---|---|---|")
        abl = d["ablation_recital_weight"]
        for w, s in abl.items():
            # Plain ASCII, so it survives a copy into a terminal or a diff.
            star = " (default)" if float(w) == d["config"].get("recital_weight") else ""
            A(f"| {w}{star} | {fmt(s['hit_rate'], 1)} | {fmt(s['recall'], 1)} | "
              f"{fmt(s['full_recall'], 1)} | {fmt(s['mrr'])} | {fmt(s['ndcg'])} |")
        A("")
        # Read the span off the sweep rather than typing it: the weights below the
        # top one are close, but they are not equal, and the sentence has to say so.
        ws = sorted(abl, key=float, reverse=True)
        lo = min(abl[w]["ndcg"] for w in ws[1:])
        hi = max(abl[w]["ndcg"] for w in ws[1:])
        A(f"The gain is a **step, not a peak**: nearly all of it comes from dropping "
          f"below `w={ws[0]}`, and the weights under that barely differ, nDCG only "
          f"{fmt(lo)} to {fmt(hi)}. So the default is not an argmax fitted to this "
          "question set, it is doing something structural, pushing non-binding text "
          "below binding text. `w=0.5` is kept rather than `w=0.0` because it ties "
          "on every column above while leaving recitals retrievable for interpretive "
          "questions.\n")
        A("> **Honest caveat.** Every gold label in this eval set is an article or "
          "annex, so an eval containing recital-answerable questions would show a "
          "smaller benefit. The measured gain is an upper bound.\n")

    # --- generation ------------------------------------------------------
    if "summary" not in d:
        A("## 2. Generation\n\n_Not run: no LLM key was configured._\n")
        out += stale_section(k)
        _emit(out, check)
        update_readme(d, k, check)
        if not check:
            print("-> RESULTS.md (retrieval only)")
        return

    s = d["summary"]
    A("## 2. Answer quality, faithfulness and abstention\n")
    cov = s.get("coverage")
    if cov and cov["scored"] < cov["total"]:
        A(f"> **Coverage: {cov['scored']} of {cov['total']} questions.** Groq's free "
          f"tier meters tokens *per day*, and a full run exceeds that allowance. "
          f"Missing: `{'`, `'.join(cov['missing'])}`. The harness checkpoints every "
          f"row, so rerunning `make eval` after the quota resets completes the set "
          f"without repeating work. All 12 out-of-scope questions and all 21 "
          f"single-hop questions were scored.\n")
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
    A(f"| Mean latency | {s['mean_latency_s']:.2f}s | _not a real latency figure_, "
      "the client sleeps to stay under the free tier's token-per-minute cap, so "
      "this is dominated by throttling, not by the model |")
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

    out += stale_section(k)
    _emit(out, check)
    update_readme(d, k, check)
    if not check:
        print("-> RESULTS.md")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--check"]
    main(args[0] if args else "latest", check="--check" in sys.argv[1:])
