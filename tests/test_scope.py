"""Role and date tags, read from the committed corpus. No embedding model."""
import pytest

from src.euactrag import config, retrieve, scope
from src.euactrag.ingest import load_chunks


def test_article_113_is_read_the_way_it_is_written():
    by_scope, general = scope.rules(config.ORIGINAL)
    assert general == ("2026-08-02", "art_113")
    assert by_scope == {
        ("CHAPTER I", ""): ("2025-02-02", "art_113(a)"),
        ("CHAPTER II", ""): ("2025-02-02", "art_113(a)"),
        ("CHAPTER III", "SECTION 4"): ("2025-08-02", "art_113(b)"),
        ("CHAPTER V", ""): ("2025-08-02", "art_113(b)"),
        ("CHAPTER VII", ""): ("2025-08-02", "art_113(b)"),
        ("CHAPTER XII", ""): ("2025-08-02", "art_113(b)"),
        "art_78": ("2025-08-02", "art_113(b)"),
        "art_101": ("2026-08-02", "art_113"),       # the exception in (b)
        "art_6(1)": ("2027-08-02", "art_113(c)"),
    }
    by_scope, general = scope.rules(config.LATEST)
    assert general == ("2026-08-02", "art_113")
    # What the 2026 amendments changed in Article 113.
    assert by_scope["art_5(1)(ba)"] == by_scope["art_5(1b)"] == ("2026-12-02", "art_113(a)")
    assert by_scope[("CHAPTER III", "SECTION 2")] == ("2027-12-02", "art_113(c)")
    assert by_scope["art_6(5)"] == ("2026-08-02", "art_113")
    assert by_scope["art_102"] == by_scope["art_110"] == ("2026-07-27", "art_113(d)")
    assert "art_6(1)" not in by_scope and "art_111" not in by_scope


def test_passages_are_tagged_with_roles_and_dates():
    t = scope.tags(config.LATEST, "paragraph")
    assert t["art_26(1)"] == {"roles": ["deployer"], "applies_from": "2027-12-02",
                              "applies_rule": "art_113(c)"}
    assert t["art_23(1)"]["roles"] == ["provider", "importer"]
    assert t["art_99(1)"]["roles"] == list(scope.ROLES)      # "operators"
    assert t["art_9(1)"]["roles"] == []
    # The new prohibition applies later than the rest of Article 5, and a
    # chunk that holds old and new points gets the earlier date.
    assert t["art_5(1a)"]["applies_from"] == "2026-12-02"
    assert t["art_5(1)#p1"]["applies_from"] == "2025-02-02"
    assert "applies_from" not in t["anx_III"] and "applies_from" not in t["rct_27#p1"]
    # A point of Article 16 names nobody, so it takes the article's title.
    point = next(c for c in load_chunks(config.LATEST, "paragraph")
                 if c["unit_id"] == "art_16" and "provider" not in c["body"].lower())
    assert scope.roles(point) == ["provider"]
    old = scope.tags(config.ORIGINAL, "paragraph")
    assert old["art_26(1)"]["applies_from"] == "2026-08-02"
    assert old["art_6(1)"]["applies_from"] == "2027-08-02"


def test_search_filters_after_ranking(monkeypatch):
    ranked = ["art_19(1)", "art_12(1)", "art_26(6)", "art_50(1)", "rct_27#p1", "anx_III"]
    monkeypatch.setattr(retrieve, "split", lambda q, k, **kw: [
        {"chunk_id": c, "rank": i} for i, c in enumerate(ranked, 1)])

    def refs(**kw):
        return [h["chunk_id"] for h in retrieve.search(
            "logs", k=6, mode="split", version=config.LATEST, granularity="paragraph", **kw)]

    assert refs() == ranked
    # Articles 19(1) and 50(1) name providers only; Article 12 names nobody and stays.
    assert refs(role="deployer") == ["art_12(1)", "art_26(6)", "rct_27#p1", "anx_III"]
    # High-risk rules do not apply yet in 2026; undated passages stay.
    assert refs(on="2026-11-17") == ["art_50(1)", "rct_27#p1", "anx_III"]
    assert refs(on="2027-12-02") == ranked
    hits = retrieve.search("logs", k=2, mode="split", version=config.LATEST,
                           granularity="paragraph", role="deployer")
    assert [h["rank"] for h in hits] == [1, 2] and hits[1]["applies_rule"] == "art_113(c)"
    with pytest.raises(ValueError):
        refs(role="user")
    with pytest.raises(ValueError):
        refs(on="17/11/2026")
