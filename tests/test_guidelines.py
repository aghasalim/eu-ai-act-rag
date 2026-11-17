"""The Commission's guidelines: the committed chunks, the PDF reader and the
way search adds them. No embedding model."""
import collections

import pytest

from src.euactrag import config, retrieve
from src.euactrag.ingest import load_chunks


def test_every_numbered_paragraph_is_a_citable_chunk():
    chunks = load_chunks(None, config.GUIDANCE)
    refs = collections.Counter(c["unit_id"] for c in {c["meta"]["ref"]: c for c in chunks}.values())
    assert refs == {f"gl_{k}": g["paragraphs"] for k, g in config.GUIDELINES.items()}
    assert sum(refs.values()) == 793
    assert all(c["kind"] == "guideline" and "#page=" in c["url"] for c in chunks)

    by_id = {c["chunk_id"]: c for c in chunks}
    c = by_id["gl_prohibited(253)"]
    assert c["citation"] == "Guidelines on prohibited AI practices, para. 253"
    assert c["title"].startswith("7.2.2. Limitation of the prohibition to workplace")
    assert c["body"].startswith("The prohibition in Article 5(1)(f) AI Act is limited to")
    assert c["text"].startswith("[Commission Guidelines on prohibited AI practices "
                                "(C(2025) 5052 final) > 7.2.2.")
    # An exponent is kept, a footnote marker is not: this sentence ends in one.
    assert "10^23 FLOP is typical" in by_id["gl_gpai(18)"]["body"]
    assert by_id["gl_prohibited(14)"]["body"].split("misuse")[1].startswith("), deployers")
    # Small capitals in a heading are read as the heading, not dropped.
    assert by_id["gl_transparency(1)"]["title"] == "1. BACKGROUND AND OBJECTIVES"


def test_a_pdf_reads_into_the_same_paragraphs():
    pytest.importorskip("pypdf")
    from src.euactrag import guidelines

    got = guidelines.parse("definition")
    want = [c for c in load_chunks(None, config.GUIDANCE) if c["unit_id"] == "gl_definition"]
    assert [c.chunk_id for c in got] == [c["chunk_id"] for c in want]
    assert got[8].body.startswith("That definition comprises seven main elements: (1) a")


def test_a_pdf_that_reads_differently_is_refused(monkeypatch):
    pytest.importorskip("pypdf")
    from src.euactrag import guidelines

    monkeypatch.setitem(config.GUIDELINES["definition"], "paragraphs", 65)
    with pytest.raises(ValueError, match="read 64 numbered paragraphs, expected 65"):
        guidelines.parse("definition")


def test_search_adds_guidelines_after_the_act(monkeypatch):
    act = [{"chunk_id": c, "rank": i, "kind": "article"}
           for i, c in enumerate(["art_5(1)#p3", "art_26(6)"], 1)]
    found = ["gl_prohibited(253)", "gl_prohibited(254)", "gl_gpai(18)", "gl_definition(9)"]
    asked = {}

    def hybrid(q, k, version=None, granularity=None):
        asked.update(version=version, granularity=granularity)
        return [{"chunk_id": c, "rank": i, "kind": "guideline"}
                for i, c in enumerate(found[:k], 1)]

    monkeypatch.setattr(retrieve, "split", lambda q, k, **kw: act[:k])
    monkeypatch.setattr(retrieve, "hybrid", hybrid)

    def ids(**kw):
        return [(h["rank"], h["chunk_id"]) for h in retrieve.search(
            "emotion recognition at work", k=2, mode="split", version=config.ORIGINAL,
            granularity="paragraph", **kw)]

    assert ids() == [(1, "art_5(1)#p3"), (2, "art_26(6)")]
    assert ids(guidelines=True) == ids() + [
        (3, "gl_prohibited(253)"), (4, "gl_prohibited(254)"), (5, "gl_gpai(18)")]
    # One guideline index serves every version of the Act.
    assert asked == {"version": config.LATEST, "granularity": config.GUIDANCE}
    # gl_gpai(18) names nobody and stays; the Act is filtered as before.
    assert ("gl_gpai(18)" in dict((c, r) for r, c in ids(guidelines=True, role="deployer")))
