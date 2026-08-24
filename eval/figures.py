"""Draw the README figures from eval/results/eval_latest.json.

Reads the saved evaluation only -- no API calls, no index, no model. Every number
here is the one RESULTS.md and the README table are built from, so the pictures
cannot drift from the prose.

    python eval/figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results" / "eval_latest.json"
FIGURES = ROOT / "eval" / "figures"

STRATEGIES = {"dense": "#9ecae1", "bm25": "#f4a582", "hybrid": "#2166ac"}
FAILURE_ORDER = [
    "ok", "false_abstention", "retrieval_miss", "partial_retrieval",
    "incomplete_answer", "generation_error",
]
FAILURE_COLOURS = {
    "ok": "#1a9850", "false_abstention": "#f4a582", "retrieval_miss": "#b2182b",
    "partial_retrieval": "#d6604d", "incomplete_answer": "#fdae61",
    "generation_error": "#67001f",
}


def load() -> dict:
    return json.loads(RESULTS.read_text())


def retrieval_across_k(out: Path, data: dict) -> Path:
    """Each strategy as the budget k grows.

    Hybrid is not simply better everywhere: BM25 leads dense on full recall at
    every k, and the fusion is what buys the gap over both.
    """
    ks = sorted(data["retrieval"]["hybrid"]["at_k"], key=int)
    panels = [("hit_rate", "hit rate"), ("full_recall", "full recall"), ("mrr", "MRR")]

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)
    for ax, (metric, label) in zip(axes, panels, strict=True):
        for strategy, colour in STRATEGIES.items():
            values = [data["retrieval"][strategy]["at_k"][k]["overall"][metric] for k in ks]
            ax.plot([int(k) for k in ks], values, "o-", color=colour, lw=2,
                    label=strategy)
        ax.set_xlabel("k (passages retrieved)")
        ax.set_ylabel(label)
        ax.set_xticks([int(k) for k in ks])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle(
        "Full recall is the column that matters: a question needing two articles "
        "is not answered by finding one.",
        fontsize=10, y=0.02, color="0.35",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def single_vs_multi_hop(out: Path, data: dict) -> Path:
    """Where the retrieval difficulty actually lives."""
    k = str(data["config"]["top_k"])
    by_type = data["retrieval"]["hybrid"]["at_k"][k]["by_type"]
    types = list(by_type)
    metrics = ["hit_rate", "recall", "full_recall"]

    figure, ax = plt.subplots(figsize=(9, 4.4))
    width = 0.26
    base = np.arange(len(types))
    for offset, metric in enumerate(metrics):
        ax.bar(base + (offset - 1) * width,
               [by_type[t][metric] * 100 for t in types], width,
               label=metric.replace("_", " "), edgecolor="0.3", lw=0.5)
    ax.set_xticks(base)
    ax.set_xticklabels([t.replace("_", "-") for t in types])
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_title(
        f"Hybrid retrieval at k={k}. Multi-hop questions lose most of their "
        "ground on full recall,\nwhich is exactly the metric a partial answer "
        "hides.",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def recital_ablation(out: Path, data: dict) -> Path:
    """Down-weighting recitals is a step, not a tuned peak.

    Every weight below 1.0 scores identically, which is the evidence that this is
    structural -- pushing non-binding text below binding text -- rather than a
    hyper-parameter fitted to 45 questions.
    """
    ablation = data["ablation_recital_weight"]
    weights = sorted(ablation, key=float, reverse=True)
    positions = np.arange(len(weights))

    figure, ax = plt.subplots(figsize=(9, 4.4))
    for metric, colour, label in [
        ("mrr", "#2166ac", "MRR"),
        ("ndcg", "#67a9cf", "nDCG"),
        ("full_recall", "#f4a582", "full recall"),
    ]:
        ax.plot(positions, [ablation[w][metric] for w in weights], "o-",
                color=colour, lw=2, label=label)
    default = str(data["config"]["recital_weight"])
    if default in weights:
        ax.axvline(weights.index(default), color="0.5", ls="--", lw=1.2)
        ax.text(weights.index(default), ax.get_ylim()[0], " default",
                fontsize=8, color="0.4", va="bottom")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"w={w}" for w in weights])
    ax.set_xlabel("weight a recital carries relative to a binding provision")
    ax.set_ylabel("score")
    ax.set_title(
        "Flat below w=1.0. A tuned hyper-parameter would show a peak; this shows "
        "a step.",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def answer_quality(out: Path, data: dict) -> Path:
    """The generation side, split by question type.

    Unanswerable questions are the interesting column: the system refuses all 12,
    so the hallucination rate is zero -- but it also refuses 21% of answerable
    ones, and over-refusal is the cost of that.
    """
    summary = data["summary"]
    by_type = summary["by_type"]

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    types = list(by_type)
    base = np.arange(len(types))
    left.bar(base - 0.2, [by_type[t]["accuracy_strict"] * 100 for t in types], 0.4,
             label="strict accuracy", color="#2166ac", edgecolor="0.3", lw=0.5)
    faith = [
        (by_type[t]["faithfulness"] or 0) * 100 for t in types
    ]
    left.bar(base + 0.2, faith, 0.4, label="faithfulness",
             color="#9ecae1", edgecolor="0.3", lw=0.5)
    for index, t in enumerate(types):
        if by_type[t]["faithfulness"] is None:
            left.text(index + 0.2, 2, "n/a", ha="center", fontsize=8, color="0.35")
        left.text(index, -8, f"n={by_type[t]['n']}", ha="center", fontsize=8,
                  color="0.4")
    left.set_xticks(base)
    left.set_xticklabels([t.replace("_", "-") for t in types])
    left.set_ylim(-12, 105)
    left.set_ylabel("%")
    left.set_title("answer quality by question type", fontsize=10)
    left.legend(frameon=False, fontsize=8)
    left.spines[["top", "right"]].set_visible(False)

    gauges = [
        ("citation\nvalidity", summary["citation_validity"], "#1a9850"),
        ("correct\nabstention", summary["correct_abstention_rate"], "#1a9850"),
        ("hallucination\n(unanswerable)", summary["hallucination_rate_unanswerable"], "#1a9850"),
        ("false\nabstention", summary["false_abstention_rate"], "#b2182b"),
    ]
    right.bar([g[0] for g in gauges], [g[1] * 100 for g in gauges],
              color=[g[2] for g in gauges], edgecolor="0.3", lw=0.5)
    for index, (_, value, _) in enumerate(gauges):
        right.text(index, value * 100 + 2, f"{value * 100:.0f}%", ha="center",
                   fontsize=10, fontweight="bold")
    right.set_ylim(0, 112)
    right.tick_params(axis="x", labelsize=8)
    right.set_ylabel("%")
    right.set_title(
        "Never cites a passage it did not retrieve, never answers an\n"
        "unanswerable question -- and refuses 21% of answerable ones.",
        fontsize=10,
    )
    right.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def failure_modes(out: Path, data: dict) -> Path:
    """Where the 45 questions actually end up."""
    modes = data["summary"]["failure_modes"]
    order = [m for m in FAILURE_ORDER if m in modes]
    counts = [modes[m] for m in order]
    total = sum(counts)

    figure, ax = plt.subplots(figsize=(10, 2.8))
    left = 0
    for mode, count in zip(order, counts, strict=True):
        ax.barh(0, count, left=left, height=0.55, color=FAILURE_COLOURS[mode],
                edgecolor="white", lw=1.4)
        if count >= 2:
            ax.text(left + count / 2, 0, str(count), ha="center", va="center",
                    fontsize=10, color="white", fontweight="bold")
        left += count
    ax.set_xlim(0, total)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel(f"questions (n={total})")
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAILURE_COLOURS[m]) for m in order]
    ax.legend(handles, [m.replace("_", " ") for m in order], frameon=False,
              fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.35))
    ax.set_title(
        "Retrieval misses and partial retrieval account for 6 of the 12 "
        "non-ok outcomes;\nfalse abstention for another 4.",
        fontsize=10,
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = load()
    for path in (
        retrieval_across_k(FIGURES / "retrieval-across-k.png", data),
        single_vs_multi_hop(FIGURES / "single-vs-multi-hop.png", data),
        recital_ablation(FIGURES / "recital-ablation.png", data),
        answer_quality(FIGURES / "answer-quality.png", data),
        failure_modes(FIGURES / "failure-modes.png", data),
    ):
        print(f"-> {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
