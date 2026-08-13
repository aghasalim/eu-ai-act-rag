"""Run the evaluation and write reproducible results.

Two independent passes:

1. Retrieval (exact, no LLM). Runs for every retrieval mode so the dense vs
   hybrid comparison is measured rather than asserted.
2. Generation (needs an API key). Faithfulness, correctness, abstention and
   citation validity, plus a RAGAS cross-check when RAGAS is installed.

Every failure is bucketed into a cause, because "78% correct" is not actionable
but "of the 10 failures, 6 are retrieval misses on definitional queries" is.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.euactrag import config, llm, pipeline, retrieve  # noqa: E402
from src.euactrag.ingest import load_chunks  # noqa: E402
from eval import judge as judge_mod  # noqa: E402
from eval import metrics as M  # noqa: E402

QA_PATH = Path(__file__).parent / "qa_set.jsonl"


def load_qa() -> list[dict]:
    return [json.loads(l) for l in open(QA_PATH, encoding="utf-8") if l.strip()]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 4) if xs else None


# --------------------------------------------------------------------------
# Pass 1: retrieval (exact)
# --------------------------------------------------------------------------
def score_retrieval(qa: list[dict], modes: list[str], ks: list[int]) -> dict:
    answerable = [q for q in qa if q["gold_units"]]
    out: dict = {}
    for mode in modes:
        t0 = time.time()
        hits_by_q = {q["id"]: retrieve.search(q["question"], k=max(ks), mode=mode)
                     for q in answerable}
        elapsed = (time.time() - t0) / len(answerable)
        out[mode] = {"latency_s_per_query": round(elapsed, 4), "at_k": {}}
        for k in ks:
            per_q = {q["id"]: M.retrieval_scores(hits_by_q[q["id"]], q["gold_units"], k)
                     for q in answerable}
            agg = {name: mean([per_q[q["id"]][name] for q in answerable])
                   for name in M.RETRIEVAL_METRICS}
            by_type = {}
            for t in ("single_hop", "multi_hop"):
                sub = [q for q in answerable if q["type"] == t]
                by_type[t] = {name: mean([per_q[q["id"]][name] for q in sub])
                              for name in M.RETRIEVAL_METRICS}
            out[mode]["at_k"][str(k)] = {"overall": agg, "by_type": by_type,
                                         "per_question": per_q}
    return out


# --------------------------------------------------------------------------
# Pass 2: generation (needs an LLM)
# --------------------------------------------------------------------------
def classify_failure(q: dict, r: dict) -> str:
    """Deterministic failure taxonomy. Order matters: the earliest cause wins,
    because a generation error downstream of a retrieval miss is not a
    generation problem."""
    if q["type"] == "unanswerable":
        return "ok" if r["abstained"] else "hallucination_no_abstain"
    if not r["retrieval"]["hit_rate"]:
        return "retrieval_miss"
    if r["abstained"]:
        return "false_abstention"
    if not r["retrieval"]["full_recall"] and q["type"] == "multi_hop":
        return "partial_retrieval"
    if r["grade"] == "incorrect":
        return "generation_error"
    if r["grade"] == "partial":
        return "incomplete_answer"
    return "ok"


def score_generation(qa: list[dict], k: int, mode: str, gen_model: str,
                     judge_model: str) -> dict:
    rows = []
    for i, q in enumerate(qa, 1):
        t0 = time.time()
        a = pipeline.answer(q["question"], k=k, mode=mode, model=gen_model)
        latency = time.time() - t0
        contexts = [h["text"] for h in a.contexts]

        row = {
            "id": q["id"], "type": q["type"], "question": q["question"],
            "answer": a.answer, "abstained": a.abstained,
            "gold_units": q["gold_units"],
            "retrieved_units": a.retrieved_units,
            "cited_units": a.cited_units,
            "contexts": contexts,
            "citation_validity": M.citation_validity(a.cited_units, a.contexts),
            "latency_s": round(latency, 3),
            "retrieval": (M.retrieval_scores(a.contexts, q["gold_units"], k)
                          if q["gold_units"] else {}),
        }

        if q["type"] == "unanswerable":
            row["grade"] = "correct" if a.abstained else "incorrect"
            row["faithfulness"] = None
            row["n_claims"] = 0
        elif a.abstained:
            row["grade"] = "incorrect"
            row["faithfulness"] = None
            row["n_claims"] = 0
        else:
            f = judge_mod.faithfulness(a.answer, contexts, judge_model)
            c = judge_mod.correctness(q["question"], q["reference"], a.answer,
                                      judge_model)
            row["faithfulness"] = f["score"]
            row["n_claims"] = f["n_claims"]
            row["unsupported_claims"] = [
                v for v in f.get("verdicts", []) if not v.get("supported")
            ]
            row["grade"] = c["grade"]
            row["grade_reason"] = c.get("reason", "")

        row["failure"] = classify_failure(q, row)
        rows.append(row)
        print(f"  [{i}/{len(qa)}] {q['id']} {row['grade']:<9} {row['failure']}")
    return {"rows": rows}


def summarise(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["type"] != "unanswerable"]
    una = [r for r in rows if r["type"] == "unanswerable"]
    graded = [r for r in ans if r["grade"] in {"correct", "partial", "incorrect"}]
    return {
        "n": len(rows),
        "answer_accuracy_strict": mean([r["grade"] == "correct" for r in graded]),
        "answer_accuracy_lenient": mean(
            [r["grade"] in {"correct", "partial"} for r in graded]),
        "faithfulness": mean([r["faithfulness"] for r in ans]),
        "citation_validity": mean([r["citation_validity"] for r in rows]),
        "correct_abstention_rate": mean([r["abstained"] for r in una]),
        "false_abstention_rate": mean([r["abstained"] for r in ans]),
        "hallucination_rate_unanswerable": mean([not r["abstained"] for r in una]),
        "mean_latency_s": mean([r["latency_s"] for r in rows]),
        "by_type": {
            t: {
                "n": len([r for r in rows if r["type"] == t]),
                "accuracy_strict": mean(
                    [r["grade"] == "correct" for r in rows if r["type"] == t]),
                "faithfulness": mean(
                    [r["faithfulness"] for r in rows if r["type"] == t]),
            }
            for t in ("single_hop", "multi_hop", "unanswerable")
        },
        "failure_modes": dict(Counter(r["failure"] for r in rows)),
    }


def ragas_crosscheck(rows: list[dict], qa: list[dict]) -> dict | None:
    """Independent faithfulness measurement. Optional: RAGAS pulls a large
    dependency tree, so its absence degrades the report rather than breaking it."""
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import answer_relevancy, faithfulness
    except Exception as e:
        return {"skipped": f"ragas not installed ({type(e).__name__})"}
    try:
        by_id = {q["id"]: q for q in qa}
        keep = [r for r in rows if r["type"] != "unanswerable" and not r["abstained"]]
        ds = Dataset.from_dict({
            "question": [r["question"] for r in keep],
            "answer": [r["answer"] for r in keep],
            "contexts": [r["contexts"] or [""] for r in keep],
            "ground_truth": [by_id[r["id"]]["reference"] for r in keep],
        })
        res = ragas_evaluate(ds, metrics=[faithfulness, answer_relevancy])
        return {k: round(float(v), 4) for k, v in res.items()}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--modes", default="dense,bm25,hybrid")
    p.add_argument("--ks", default="3,5,10")
    p.add_argument("--k", type=int, default=config.TOP_K)
    p.add_argument("--gen-mode", default=config.RETRIEVAL_MODE)
    p.add_argument("--model", default=config.LLM_MODEL)
    p.add_argument("--judge-model", default=None)
    p.add_argument("--no-generation", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--tag", default="latest")
    a = p.parse_args()

    qa = load_qa()
    if a.limit:
        qa = qa[: a.limit]
    modes = a.modes.split(",")
    ks = [int(x) for x in a.ks.split(",")]
    n_chunks = len(load_chunks())

    print(f"corpus: {n_chunks} chunks | qa: {len(qa)} questions")
    print(f"\n== retrieval ({', '.join(modes)}) ==")
    ret = score_retrieval(qa, modes, ks)
    for mode in modes:
        s = ret[mode]["at_k"][str(a.k)]["overall"]
        print(f"  {mode:<7} @{a.k}  hit={s['hit_rate']}  recall={s['recall']}  "
              f"full={s['full_recall']}  mrr={s['mrr']}  ndcg={s['ndcg']}")

    out = {
        "config": {
            "embed_model": config.EMBED_MODEL, "top_k": a.k,
            "gen_mode": a.gen_mode, "rrf_k": config.RRF_K,
            "n_chunks": n_chunks, "n_questions": len(qa),
        },
        "retrieval": ret,
    }

    if not a.no_generation and llm.available():
        judge_model = a.judge_model or judge_mod.pick_judge(a.model)
        print(f"\n== generation (gen={a.model}, judge={judge_model}) ==")
        out["config"]["gen_model"] = a.model
        out["config"]["judge_model"] = judge_model
        gen = score_generation(qa, a.k, a.gen_mode, a.model, judge_model)
        out["summary"] = summarise(gen["rows"])
        out["ragas"] = ragas_crosscheck(gen["rows"], qa)
        # Contexts are large and only needed for the RAGAS pass; drop before saving.
        for r in gen["rows"]:
            r.pop("contexts", None)
        out["generation"] = gen
        print("\n== summary ==")
        for k_, v in out["summary"].items():
            if not isinstance(v, dict):
                print(f"  {k_:<34} {v}")
        print(f"  failure_modes: {out['summary']['failure_modes']}")
    else:
        print("\n[skipped generation] no LLM key configured "
              "(set GROQ_API_KEY in .env). Retrieval metrics above are complete.")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / f"eval_{a.tag}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
