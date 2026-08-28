"""Draw the README figures from eval/results/eval_latest.json.

Reads the saved evaluation only, no API calls, no index, no model. Every number
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

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.style import PALETTE, titled  # noqa: E402

RESULTS = ROOT / "eval" / "results" / "eval_latest.json"
QA_PATH = ROOT / "eval" / "qa_set.jsonl"
FIGURES = ROOT / "eval" / "figures"

# One colour per retrieval strategy, the same in every figure, so hybrid is the
# same blue wherever a reader meets it.
STRATEGY = {"dense": PALETTE[4], "bm25": PALETTE[3], "hybrid": PALETTE[0]}

# Coverage of the articles a question needs: all of them, some, none. The same
# three colours mean the same thing in the still figures and in the animation.
ALL_FOUND, SOME_FOUND, NONE_FOUND = PALETTE[2], PALETTE[3], "#d3d3d3"

# Every non-ok outcome, in the order they are drawn, with the part of the
# pipeline that caused it.
FAILURE_CAUSE = {
    "retrieval_miss": "retrieval",
    "partial_retrieval": "retrieval",
    "false_abstention": "abstention",
    "generation_error": "generation",
    "incomplete_answer": "generation",
}
CAUSE_COLOUR = {
    "retrieval": PALETTE[1],
    "abstention": PALETTE[3],
    "generation": PALETTE[4],
}


def load() -> dict:
    return json.loads(RESULTS.read_text())


def load_types() -> dict[str, str]:
    """Question id to type, so the grid can be split without parsing ids."""
    return {
        row["id"]: row["type"]
        for row in (json.loads(line) for line in QA_PATH.read_text().splitlines() if line.strip())
    }


def retrieval_across_k(out: Path, data: dict) -> Path:
    """Each strategy as the budget k grows.

    Hybrid is not simply better everywhere. Dense is ahead of BM25 on full
    recall at k=3 and behind it from k=5 on, and the fusion is what buys the
    gap over both.
    """
    ks = sorted(data["retrieval"]["hybrid"]["at_k"], key=int)
    x = [int(k) for k in ks]
    panels = [
        ("hit_rate", "share of questions",
         "Hybrid is ahead at every budget",
         "hit rate: at least one required article in the top k"),
        ("full_recall", "share of questions",
         "BM25 overtakes the embeddings from k=5 on",
         "full recall: all of a question's required articles in the top k"),
        ("mrr", "mean reciprocal rank",
         "Fusion is what puts the right article first",
         "MRR: 1 / rank of the first required article, averaged"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    for ax, (metric, ylabel, claim, gloss) in zip(axes, panels, strict=True):
        for strategy, colour in STRATEGY.items():
            values = [data["retrieval"][strategy]["at_k"][k]["overall"][metric] for k in ks]
            ax.plot(x, values, "o-", color=colour, label=strategy)
        ax.set_xlabel("k (passages retrieved per question)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        titled(ax, claim, gloss)
    # Lower right is the empty corner in all three panels: every curve rises.
    axes[0].legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def single_vs_multi_hop(out: Path, data: dict) -> Path:
    """Where the retrieval difficulty actually lives."""
    k = str(data["config"]["top_k"])
    by_type = data["retrieval"]["hybrid"]["at_k"][k]["by_type"]
    counts = data["summary"]["by_type"]
    types = list(by_type)
    metrics = [
        ("hit_rate", "hit rate", PALETTE[5]),
        ("recall", "recall", PALETTE[0]),
        ("full_recall", "full recall", PALETTE[2]),
    ]

    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    pitch, width = 0.2, 0.18
    # Groups pulled closer than one unit apart, so the panel is not mostly the
    # white gap between two question types.
    base = np.arange(len(types)) * 0.72
    for offset, (metric, label, colour) in enumerate(metrics):
        pos = base + (offset - 1) * pitch
        values = [by_type[t][metric] * 100 for t in types]
        ax.bar(pos, values, width, label=label, color=colour)
        for px, value in zip(pos, values, strict=True):
            ax.text(px, value + 1.5, f"{value:.1f}", ha="center", va="bottom",
                    fontsize=9, color="#333333")

    ax.set_xticks(base)
    ax.set_xticklabels([f"{t.replace('_', '-')}\n{counts[t]['n']} questions" for t in types])
    ax.set_xlim(base[0] - 0.4, base[-1] + 0.4)
    ax.set_ylabel("share of questions (%)")
    ax.set_ylim(0, 112)
    ax.grid(axis="x", visible=False)
    titled(ax,
           "Multi-hop loses its ground on full recall, not on hit rate",
           f"hybrid retrieval at k={k}; full recall is the column a partial answer hides")
    # Under the axes: at 100% the bars reach the top of the panel, so any
    # in-axes legend would sit on data.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def recital_ablation(out: Path, data: dict) -> Path:
    """Down-weighting recitals is a step, not a tuned peak.

    Every weight below 1.0 scores identically, which is the evidence that this is
    structural, pushing non-binding text below binding text, rather than a
    hyper-parameter fitted to 45 questions.
    """
    ablation = data["ablation_recital_weight"]
    weights = sorted(ablation, key=float, reverse=True)
    positions = np.arange(len(weights))

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for metric, colour, label in [
        ("mrr", PALETTE[0], "MRR"),
        ("ndcg", PALETTE[4], "nDCG"),
        ("full_recall", PALETTE[3], "full recall"),
    ]:
        ax.plot(positions, [ablation[w][metric] for w in weights], "o-",
                color=colour, label=label)

    default = str(data["config"]["recital_weight"])
    if default in weights:
        at = weights.index(default)
        ax.axvline(at, color="#8a8a8a", ls=(0, (5, 3)), lw=1.1, zorder=0)
        ax.annotate("shipped default", xy=(at, 0.62), xytext=(6, 0),
                    textcoords="offset points", fontsize=9, color="#5a5a5a",
                    va="center")

    ax.set_xticks(positions)
    ax.set_xticklabels(weights)
    ax.set_xlabel("recital weight (multiplier on a recital's fused rank score)")
    ax.set_ylabel("score (0 to 1)")
    ax.set_ylim(0.55, 0.83)
    titled(ax,
           "Every weight below 1.0 scores the same, so this is a step and not a peak",
           f"hybrid retrieval at k={data['config']['top_k']}, sweeping only the recital weight")
    # Every curve is flat and high on the right, so the free corner is bottom right.
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def answer_quality(out: Path, data: dict) -> Path:
    """The generation side, split by question type, then the four headline rates.

    Unanswerable questions are the interesting column: the system refuses all 12,
    so the hallucination rate is zero, but it also refuses 21% of answerable
    ones, and over-refusal is the cost of that.
    """
    summary = data["summary"]
    by_type = summary["by_type"]
    types = list(by_type)
    base = np.arange(len(types))

    fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.0))

    left.bar(base - 0.19, [by_type[t]["accuracy_strict"] * 100 for t in types], 0.38,
             label="strict accuracy", color=PALETTE[0])
    left.bar(base + 0.19, [(by_type[t]["faithfulness"] or 0) * 100 for t in types], 0.38,
             label="faithfulness of the answer", color=PALETTE[2])
    for index, t in enumerate(types):
        left.text(index - 0.19, by_type[t]["accuracy_strict"] * 100 + 1.5,
                  f"{by_type[t]['accuracy_strict'] * 100:.1f}", ha="center",
                  va="bottom", fontsize=9, color="#333333")
        faith = by_type[t]["faithfulness"]
        # A refusal makes no claims, so there is nothing to check for entailment.
        left.text(index + 0.19, (faith or 0) * 100 + 4.0,
                  f"{faith * 100:.1f}" if faith is not None else "not defined",
                  ha="center", va="bottom", fontsize=9,
                  color="#333333" if faith is not None else "#7a7a7a")
    left.set_xticks(base)
    left.set_xticklabels([f"{t.replace('_', '-')}\n{by_type[t]['n']} questions" for t in types])
    left.set_ylabel("share (%)")
    left.set_ylim(0, 112)
    left.grid(axis="x", visible=False)
    titled(left, "Multi-hop is the weak half on both counts",
           "answers from gpt-oss-20b, graded by qwen3.6-27b, a different family")
    left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

    gauges = [
        ("citation\nvalidity", "answers given", summary["citation_validity"], PALETTE[2]),
        ("correct\nabstention", "12 unanswerable", summary["correct_abstention_rate"], PALETTE[2]),
        ("hallucination", "12 unanswerable", summary["hallucination_rate_unanswerable"], PALETTE[2]),
        ("false\nabstention", "33 answerable", summary["false_abstention_rate"], PALETTE[1]),
    ]
    right.bar(range(len(gauges)), [g[2] * 100 for g in gauges], 0.6,
              color=[g[3] for g in gauges])
    for index, (_, _, value, _) in enumerate(gauges):
        right.text(index, value * 100 + 1.5, f"{value * 100:.0f}%", ha="center",
                   va="bottom", fontsize=10.5, fontweight="bold")
    right.set_xticks(range(len(gauges)))
    right.set_xticklabels([f"{g[0]}\n{g[1]}" for g in gauges], fontsize=9)
    right.set_ylabel("share of that denominator (%)")
    right.set_ylim(0, 112)
    right.grid(axis="x", visible=False)
    titled(right, "No hallucination out of scope, paid for with 21% false abstention",
           "each bar has its own denominator, printed under its label")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def failure_modes(out: Path, data: dict) -> Path:
    """The questions that did not end ok, by the stage that caused it.

    The old version was one stacked strip, where a mode with a single question
    was a sliver too thin to label. One bar per mode is the same twelve
    questions and can actually be read.
    """
    modes = data["summary"]["failure_modes"]
    ok = modes.get("ok", 0)
    order = [m for m in FAILURE_CAUSE if modes.get(m)]
    counts = [modes[m] for m in order]
    causes = [FAILURE_CAUSE[m] for m in order]
    y = np.arange(len(order))[::-1]

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.barh(y, counts, height=0.6, color=[CAUSE_COLOUR[c] for c in causes])
    for yy, count in zip(y, counts, strict=True):
        ax.text(count + 0.08, yy, str(count), va="center", ha="left", fontsize=10,
                color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels([m.replace("_", " ") for m in order])
    ax.set_xlim(0, max(counts) + 0.8)
    ax.set_xticks(range(max(counts) + 1))
    ax.set_xlabel(f"questions (of the {sum(counts)} that did not end ok)")
    ax.grid(axis="y", visible=False)

    handles = [Rectangle((0, 0), 1, 1, color=CAUSE_COLOUR[c])
               for c in ("retrieval", "abstention", "generation")]
    # Bars grow to the right from a left axis, so the far right is always clear.
    ax.legend(handles, ["retrieval", "abstention", "generation"],
              loc="lower right", title="stage at fault")
    titled(ax, "Half the failures are retrieval, only two are the generator's fault",
           f"{ok} of the {ok + sum(counts)} questions ended ok; these are the other "
           f"{sum(counts)}, at k={data['config']['top_k']}")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _coverage_grid(data: dict) -> tuple[list[str], dict[str, list[float]], list[str]]:
    """Per-question share of required articles retrieved, at each committed k.

    Straight out of the saved per-question retrieval metrics. Tiles are ordered
    by the k at which a question first reaches full recall, so the fill sweeps
    left to right and the questions that never complete sit at the end.
    """
    ks = sorted(data["retrieval"]["hybrid"]["at_k"], key=int)
    per_k = {k: data["retrieval"]["hybrid"]["at_k"][k]["per_question"] for k in ks}
    coverage = {q: [per_k[k][q]["recall"] for k in ks] for q in per_k[ks[0]]}

    def rank(q: str) -> tuple[int, float, str]:
        series = coverage[q]
        first = next((i for i, v in enumerate(series) if v >= 1.0), len(ks))
        return first, -series[-1], q

    return ks, coverage, sorted(coverage, key=rank)


def anim_coverage_across_k(out: Path, data: dict) -> Path:
    """One tile per answerable question, filling in as the budget k grows.

    The still figures give the averages. This gives the questions: single-hop
    is mostly done at k=3, and the multi-hop row is still filling at k=10.
    """
    ks, coverage, ordered = _coverage_grid(data)
    types = load_types()
    rows = [
        ("single-hop", [q for q in ordered if types[q] == "single_hop"]),
        ("multi-hop", [q for q in ordered if types[q] == "multi_hop"]),
    ]
    width = max(len(qs) for _, qs in rows)

    # Tile height chosen against the axes aspect so a question reads as a square
    # rather than a bar: a bar length would suggest a quantity, and every tile
    # here carries the same weight.
    pitch, tile = 0.75, 0.34
    fig, ax = plt.subplots(figsize=(8.8, 3.6))
    ax.set_xlim(-0.3, width + 0.3)
    ax.set_ylim(-0.25, pitch * len(rows) - 0.1)
    ax.set_xticks([])
    ax.set_yticks([pitch * (len(rows) - 1 - i) + tile / 2 for i in range(len(rows))])
    ax.set_yticklabels([f"{label}\n{len(qs)} questions" for label, qs in rows])
    ax.grid(visible=False)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.tick_params(length=0)

    tiles: dict[str, Rectangle] = {}
    for row, (_, qs) in enumerate(rows):
        y = pitch * (len(rows) - 1 - row)
        for column, q in enumerate(qs):
            patch = Rectangle((column + 0.06, y), 0.88, tile,
                              facecolor=NONE_FOUND, edgecolor="white", lw=1.0)
            ax.add_patch(patch)
            tiles[q] = patch

    handles = [Rectangle((0, 0), 1, 1, color=c)
               for c in (ALL_FOUND, SOME_FOUND, NONE_FOUND)]
    ax.legend(handles, ["all required articles found", "some found", "none found"],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)

    # Same layout as style.titled, but the second line changes every frame.
    ax.set_title("Multi-hop questions are the last to fill in", pad=26)
    ticker = ax.text(0.0, 1.012, "", transform=ax.transAxes, fontsize=9.3,
                     color="#5a5a5a", va="bottom", ha="left")

    hold = 15  # frames per k, so each budget is on screen for 1.5 s at 15 fps

    def draw(frame: int):
        index = min(frame // hold, len(ks) - 1)
        k = ks[index]
        done = 0
        for q, patch in tiles.items():
            value = coverage[q][index]
            patch.set_facecolor(ALL_FOUND if value >= 1.0
                                else SOME_FOUND if value > 0 else NONE_FOUND)
            done += value >= 1.0
        ticker.set_text(
            f"k = {k} passages: {done} of {len(tiles)} answerable questions have "
            "every article they need"
        )
        return (*tiles.values(), ticker)

    fig.tight_layout()
    anim = FuncAnimation(fig, draw, frames=hold * len(ks), blit=False)
    # dpi is passed on purpose. The style saves stills at 170, and sixty frames
    # at that size would be a several MB GIF for no extra detail.
    anim.save(out, writer=PillowWriter(fps=10), dpi=100)
    plt.close(fig)
    _shrink_gif(out)
    return out


def _shrink_gif(path: Path, colours: int = 64) -> None:
    """Rewrite the GIF on one shared palette.

    PillowWriter gives every frame its own full palette, which is most of the
    file size and is wasted here: consecutive frames differ only in a few tiles,
    so one palette taken from a middle frame covers all of them and lets the
    encoder store just the changes.
    """
    from PIL import Image

    source = Image.open(path)
    frames, durations = [], []
    try:
        while True:
            frames.append(source.convert("RGB"))
            durations.append(source.info.get("duration", 62))
            source.seek(source.tell() + 1)
    except EOFError:
        pass
    shared = frames[len(frames) // 2].quantize(colours, method=Image.Quantize.MEDIANCUT)
    quantised = [f.quantize(palette=shared, dither=Image.Dither.NONE) for f in frames]
    # No disposal method: leaving it unset lets the encoder store only the region
    # that changed. Setting disposal=2 forces a full redraw per frame and makes
    # the file bigger than the one it replaces.
    quantised[0].save(path, save_all=True, append_images=quantised[1:], loop=0,
                      duration=durations, optimize=True)


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = load()
    for path in (
        retrieval_across_k(FIGURES / "retrieval-across-k.png", data),
        single_vs_multi_hop(FIGURES / "single-vs-multi-hop.png", data),
        recital_ablation(FIGURES / "recital-ablation.png", data),
        answer_quality(FIGURES / "answer-quality.png", data),
        failure_modes(FIGURES / "failure-modes.png", data),
        anim_coverage_across_k(FIGURES / "coverage-across-k.gif", data),
    ):
        print(f"-> {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
