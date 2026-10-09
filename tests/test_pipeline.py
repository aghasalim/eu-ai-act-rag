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

from eval import judge  # noqa: E402
from eval import metrics as M  # noqa: E402
from src.euactrag import config, pipeline  # noqa: E402

QA = ROOT / "eval" / "qa_set.jsonl"

pytestmark = pytest.mark.skipif(
    not all(config.chunks_path(v).exists() for v in config.VERSIONS),
    reason="run `make corpus` first",
)


def _load(version):
    return [json.loads(l) for l in open(config.chunks_path(version), encoding="utf-8")]


@pytest.fixture(scope="module", params=sorted(config.VERSIONS))
def chunks(request):
    """Every corpus-integrity test runs on every version of the text."""
    return _load(request.param)


@pytest.fixture(scope="module")
def original():
    return _load(config.ORIGINAL)


@pytest.fixture(scope="module")
def qa():
    return [json.loads(l) for l in open(QA, encoding="utf-8") if l.strip()]


# --- corpus integrity -----------------------------------------------------
# Articles the Digital Omnibus on AI (Regulation (EU) 2026/1744) inserted.
INSERTED_2026 = {"art_4a", "art_60a", "art_75a", "art_75b", "art_75c", "art_75d"}


def test_all_articles_present(chunks):
    arts = {c["unit_id"] for c in chunks if c["kind"] == "article"}
    want = {f"art_{i}" for i in range(1, 114)}
    if chunks[0]["version"] != config.ORIGINAL:
        want |= INSERTED_2026
    assert arts == want


def test_every_chunk_is_stamped_with_its_version(chunks):
    assert len({c["version"] for c in chunks}) == 1
    assert chunks[0]["version"] in config.VERSIONS


def test_amended_text_differs_only_where_it_was_amended(original):
    """The two renderings use different markup. If the parser read them
    differently, unamended articles would differ too; most must be identical.
    Article 113 must not be, it carries the new dates."""
    amended = _load(config.LATEST)
    body = lambda cs, u: " ".join(c["body"] for c in cs if c["unit_id"] == u)
    units = {c["unit_id"] for c in original}
    same = sum(body(original, u) == body(amended, u) for u in units)
    assert same / len(units) > 0.75
    assert "2 December 2027" in body(amended, "art_113")
    assert "2 December 2027" not in body(original, "art_113")


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
def test_gold_units_exist(qa, original):
    units = {c["unit_id"] for c in original}
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


# --- judge output parsing -------------------------------------------------
# The judge is a model, so its wrapper text is not under our control. Every
# shape below has come back from a real judge at some point. If _json stops
# handling one, faithfulness silently becomes None for those rows and the mean
# is computed over a smaller set without saying so, which is the quiet failure
# worth a test.

def test_judge_json_reads_a_plain_array():
    assert judge._json('["one", "two"]') == ["one", "two"]


def test_judge_json_strips_a_fenced_block():
    assert judge._json('```json\n{"score": 1}\n```') == {"score": 1}
    assert judge._json('```\n[1, 2]\n```') == [1, 2]


def test_judge_json_digs_the_object_out_of_prose():
    text = 'Sure, here is my assessment:\n{"score": 0, "why": "unsupported"}\nHope that helps.'
    assert judge._json(text) == {"score": 0, "why": "unsupported"}


def test_judge_json_returns_none_rather_than_guessing():
    # None is the signal the caller checks. Returning {} or [] here would look
    # like a real judgement of "no claims" and quietly drag the mean around.
    assert judge._json("I could not evaluate this.") is None
    assert judge._json('{"score": ') is None
    assert judge._json("") is None


def test_hybrid_breaks_rrf_ties_the_same_way_everywhere(monkeypatch):
    # Two chunks at the same rank in exactly one run each get the identical RRF
    # score, and the fusion used to leave their order to dict insertion, which
    # is the order the dense index happened to return them in. That differed
    # between this machine and the CI runner and moved one query's MRR. Feed the
    # two runs in both insertion orders and require the same fused order.
    from src.euactrag import retrieve
    a = {"chunk_id": "art_2", "kind": "article", "rank": 1}
    b = {"chunk_id": "art_1", "kind": "article", "rank": 1}
    for first, second in ((a, b), (b, a)):
        monkeypatch.setattr(retrieve, "dense", lambda q, p, v=None, g=None, f=first: [dict(f)])
        monkeypatch.setattr(retrieve, "bm25", lambda q, p, v=None, g=None, s=second: [dict(s)])
        out = retrieve.hybrid("q", k=2, recital_weight=1.0)
        assert [h["chunk_id"] for h in out] == ["art_1", "art_2"]


# --- amended question set -------------------------------------------------
AMENDED = ROOT / "eval" / "qa_amended.jsonl"


def test_amended_questions_are_only_answerable_from_the_amended_text(original):
    """Each amended question quotes the provision its answer rests on. The quote
    must sit inside a gold unit of the 2026 text and nowhere in the 2024 text,
    otherwise the question does not test what it claims to."""
    latest = _load(config.LATEST)
    norm = lambda s: " ".join(s.split()).lower()
    qa = [json.loads(l) for l in open(AMENDED, encoding="utf-8") if l.strip()]
    assert len({q["id"] for q in qa}) == len(qa) >= 10
    for q in qa:
        assert q["type"] in {"single_hop", "multi_hop"} and q["gold_units"]
        assert (len(q["gold_units"]) > 1) == (q["type"] == "multi_hop"), q["id"]
        for ev in q["evidence"]:
            where = {c["unit_id"] for c in latest if norm(ev) in norm(c["text"])}
            assert where and where <= set(q["gold_units"]), (q["id"], ev)
            assert not any(norm(ev) in norm(c["text"]) for c in original), (q["id"], ev)


def test_stale_marks_on_the_original_set(qa):
    for q in qa:
        if "stale" in q:
            assert q["stale"]["version"] in config.VERSIONS
            assert q["stale"]["status"] in {"superseded", "incomplete"}
            assert q["stale"]["note"]


def test_evidence_hit_needs_every_quote():
    hits = [{"text": "It shall apply from 2 December  2027."}, {"text": "other"}]
    assert M.evidence_hit(hits, ["2 december 2027"], 2) == 1.0
    assert M.evidence_hit(hits, ["2 December 2027", "2 August 2028"], 2) == 0.0
    assert M.evidence_hit(hits, ["2 December 2027"], 0) == 0.0


# --- paragraph-level citations --------------------------------------------
def _paragraph_chunks(version):
    path = config.chunks_path(version, "paragraph")
    if not path.exists():
        pytest.skip("run `make corpus` first")
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def test_outline_paths_tell_letters_from_roman_numerals():
    from src.euactrag.ingest import outline_paths
    lines = ["1. The following shall be prohibited:",
             "(h) the use of real-time systems, unless:",
             "    (i) the targeted search for victims;",
             "    (ii) the prevention of a threat;",
             "(i) a letter point after (h);",
             "The first subparagraph shall apply.",
             "2. Next paragraph."]
    assert outline_paths(lines) == [("1",), ("1", "h"), ("1", "h", "i"),
                                    ("1", "h", "ii"), ("1", "i"), ("1",), ("2",)]


def test_paragraph_chunks_have_unique_ids_and_known_refs():
    for v in config.VERSIONS:
        ch = _paragraph_chunks(v)
        assert len({c["chunk_id"] for c in ch}) == len(ch)
        art6 = {c["meta"]["ref"] for c in ch if c["unit_id"] == "art_6"}
        assert {"art_6(1)", "art_6(2)", "art_6(3)", "art_6(4)"} <= art6
        # Same text as the article chunks, cut differently.
        arts = [json.loads(l) for l in open(config.chunks_path(v), encoding="utf-8")]
        words = lambda cs: sum(len(c["body"].split()) for c in cs)
        assert words(ch) == words(arts)


def test_every_gold_ref_exists_in_the_text_it_is_scored_on():
    """A gold reference that names a paragraph the corpus does not have would
    score as a miss forever and nobody would notice."""
    for path, version in ((QA, config.ORIGINAL), (AMENDED, config.LATEST)):
        known = {r for c in _paragraph_chunks(version)
                 for r in c["meta"].get("refs", [c["unit_id"]])}
        known |= {c["unit_id"] for c in _paragraph_chunks(version)}
        for q in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip()):
            assert bool(q["gold_refs"]) == bool(q["gold_units"]), q["id"]
            for r in q["gold_refs"]:
                assert r in known, (q["id"], r)
                assert r.split("(")[0] in q["gold_units"], (q["id"], r)


def test_paragraph_metrics():
    hits = [{"unit_id": "art_6", "ref": "art_6(4)"},
            {"unit_id": "art_6", "ref": "art_6(3)"},
            {"unit_id": "art_5", "ref": "art_5(1)"}]
    s = M.paragraph_scores(hits, ["art_6(3)", "art_5(1)(f)"], 3)
    assert s["para_full_recall"] == 1.0 and s["para_mrr"] == 0.5
    assert s["citation_precision"] == pytest.approx(2 / 3)
    # An article-level chunk is cited as the whole article: not a paragraph hit.
    art = [{"unit_id": "art_6", "ref": "art_6"}]
    assert M.paragraph_scores(art, ["art_6(3)"], 1)["para_hit_rate"] == 0.0
    assert M.paragraph_scores(art, ["art_6"], 1)["para_hit_rate"] == 1.0
    assert M.citation_precision([], ["art_6(3)"]) is None


def test_parse_refs_reads_paragraphs_and_points():
    text = ("Prohibited [Article 5(1)(f) - Prohibited AI practices], high-risk "
            "【ANNEX III(4)(a) - High-risk AI systems】, see [Article 6 - X] and "
            "[Recital (27)] and again [Article 5(1)(f) - Prohibited AI practices].")
    assert pipeline.parse_refs(text) == ["art_5(1)(f)", "anx_III(4)(a)", "art_6"]


def test_every_question_set_matches_the_text():
    """Gold references exist, quotes sit in a gold paragraph, and every gold
    paragraph of the extended set is backed by a quote. See eval/check_qa.py."""
    from eval import check_qa
    for name, version in check_qa.SETS.items():
        if not config.chunks_path(version, "paragraph").exists():
            pytest.skip("run `make corpus` first")
        problems, rows = check_qa.check(name, version)
        assert not problems, problems[:5]
        assert rows
