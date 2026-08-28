"""Smallest checks that fail loudly if the logic breaks.

Deliberately not a per-function suite: these cover the parts where a silent bug
would corrupt the evaluation numbers, chunk integrity, the metric maths, and
citation parsing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import metrics as M  # noqa: E402
from src.euactrag import config, pipeline  # noqa: E402

CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
QA = ROOT / "eval" / "qa_set.jsonl"

pytestmark = pytest.mark.skipif(
    not CHUNKS.exists(), reason="run `make corpus` first"
)


@pytest.fixture(scope="module")
def chunks():
    return [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]


@pytest.fixture(scope="module")
def qa():
    return [json.loads(l) for l in open(QA, encoding="utf-8") if l.strip()]


# --- corpus integrity -----------------------------------------------------
def test_all_113_articles_present(chunks):
    arts = {c["unit_id"] for c in chunks if c["kind"] == "article"}
    assert arts == {f"art_{i}" for i in range(1, 114)}


def test_chunks_fit_the_encoder_window(chunks):
    """A chunk over the encoder's 512-token window is silently truncated at
    embedding time, which loses text without any error. Allow a small margin
    over the 420 budget for unsplittable single sentences."""
    over = [c["chunk_id"] for c in chunks if c["n_tokens"] > 500]
    assert not over, f"chunks would be truncated by the encoder: {over}"


def test_chunk_ids_unique_and_text_nonempty(chunks):
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(c["body"].strip() for c in chunks)


def test_breadcrumb_makes_split_chunks_self_describing(chunks):
    """Part 2 of an article must still say which article it is."""
    for c in chunks:
        if c["n_parts"] > 1:
            assert c["citation"] in c["text"].split("\n", 1)[0]


def test_superscript_exponent_survives_parsing(chunks):
    """The 10^25 FLOP threshold must not be flattened to "10 25"."""
    art51 = " ".join(c["body"] for c in chunks if c["unit_id"] == "art_51")
    assert "10^25" in art51


# --- eval set integrity ---------------------------------------------------
def test_gold_units_exist(qa, chunks):
    units = {c["unit_id"] for c in chunks}
    missing = [(q["id"], u) for q in qa for u in q["gold_units"] if u not in units]
    assert not missing


def test_qa_set_shape(qa):
    assert len(qa) >= 40
    assert len({q["id"] for q in qa}) == len(qa)
    types = {q["type"] for q in qa}
    assert types == {"single_hop", "multi_hop", "unanswerable"}
    # unanswerable questions must have no gold, answerable must have some
    for q in qa:
        assert bool(q["gold_units"]) == (q["type"] != "unanswerable")


# --- metrics --------------------------------------------------------------
def _hits(*units):
    return [{"unit_id": u, "chunk_id": u} for u in units]


def test_retrieval_metrics_maths():
    hits = _hits("art_1", "art_6", "art_99")
    gold = ["art_6", "art_99"]
    assert M.hit_rate(hits, gold, 3) == 1.0
    assert M.recall(hits, gold, 3) == 1.0
    assert M.full_recall(hits, gold, 3) == 1.0
    assert M.mrr(hits, gold, 3) == 0.5           # first gold at rank 2
    assert M.recall(hits, gold, 2) == 0.5        # only art_6 within k=2
    assert M.full_recall(hits, gold, 2) == 0.0
    assert M.hit_rate(_hits("art_2"), gold, 1) == 0.0


def test_duplicate_chunks_of_one_unit_count_once():
    """Two chunks of the same article are one retrieved provision, not two."""
    hits = [{"unit_id": "art_6", "chunk_id": "art_6#p1"},
            {"unit_id": "art_6", "chunk_id": "art_6#p2"}]
    assert M.precision(hits, ["art_6"], 2) == 1.0
    assert M.recall(hits, ["art_6", "art_99"], 2) == 0.5


def test_ndcg_rewards_higher_rank():
    gold = ["art_99"]
    top = M.ndcg(_hits("art_99", "art_1", "art_2"), gold, 3)
    low = M.ndcg(_hits("art_1", "art_2", "art_99"), gold, 3)
    assert top == 1.0 and low < top


def test_citation_validity_flags_fabrication():
    hits = _hits("art_6", "art_99")
    assert M.citation_validity(["art_6"], hits) == 1.0
    assert M.citation_validity(["art_6", "art_50"], hits) == 0.5
    assert M.citation_validity([], hits) is None


# --- citation parsing -----------------------------------------------------
def test_parse_citations():
    txt = ("Fines reach EUR 35 000 000 [Article 99 - Penalties] and the system is "
           "high-risk [ANNEX III - High-risk AI systems] per [Recital (53)].")
    assert pipeline.parse_citations(txt) == ["art_99", "anx_III", "rct_53"]


def test_parse_citations_accepts_full_width_brackets():
    """gpt-oss models cite with 【...】; llama uses [...]. Both must parse, or
    citation validity silently reads as 'no citations made'."""
    txt = "The cap is EUR 35 000 000 【Article 99 - Penalties】 and 【ANNEX III - x】."
    assert pipeline.parse_citations(txt) == ["art_99", "anx_III"]


def test_parse_citations_dedupes_and_ignores_prose():
    txt = "[Article 6 - X] then [Article 6 - X] again, but Article 7 is not cited."
    assert pipeline.parse_citations(txt) == ["art_6"]


def test_abstain_token_is_detectable():
    assert config.ABSTAIN_STRING in f"prefix {config.ABSTAIN_STRING}"


def test_judge_is_never_from_the_generator_family():
    """The README claims the judge "isn't marking its own work". Enforce it.

    This regressed once: pick_judge only excluded the generator's exact name, so
    `openai/gpt-oss-120b` was chosen to grade `openai/gpt-oss-20b`, same vendor,
    same lineage, while the claim of independence stayed in the README.
    """
    from eval import judge as J

    for generator in ("openai/gpt-oss-20b", "openai/gpt-oss-120b",
                      "llama-3.3-70b-versatile", "qwen/qwen3.6-27b"):
        picked = J.pick_judge(generator)
        assert J.family(picked) != J.family(generator), (
            f"{picked} judges {generator} but shares its family"
        )


def test_defaults_reproduce_the_published_evaluation():
    """`make eval` with no flags must reproduce the models the README reports.

    The default generator used to be a model that was never evaluated, so the
    documented command produced different numbers than the documentation it was
    meant to reproduce.
    """
    from eval import judge as J

    results = config.RESULTS_DIR / "eval_latest.json"
    if not results.exists():
        pytest.skip("no eval run on disk")
    recorded = json.loads(results.read_text())["config"]
    assert config.LLM_MODEL == recorded["gen_model"]
    assert J.pick_judge(config.LLM_MODEL) == recorded["judge_model"]
