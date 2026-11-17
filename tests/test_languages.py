"""French, German and Dutch. The texts are fetched on demand and not committed,
so the tests that read them are skipped where they are missing (CI); the rules
that make them line up with English are tested on lines written here."""
import pytest

from src.euactrag import config, retrieve, scope
from src.euactrag.index import tokenize
from src.euactrag.ingest import load_chunks, outline_paths

OTHERS = [lang for lang in config.LANGUAGES if lang != "en"]


def fetched(lang):
    return config.chunks_path(config.localized(config.LATEST, lang), "paragraph").exists()


def test_a_language_is_part_of_the_corpus_key():
    assert config.localized("2026-07-27", None) == config.localized("2026-07-27", "en") == "2026-07-27"
    assert config.localized("2026-07-27", "fr") == "2026-07-27.fr"
    assert config.split_version("2026-07-27.fr") == ("2026-07-27", "fr")
    assert config.split_version("2026-07-27") == ("2026-07-27", "en")
    with pytest.raises(ValueError):
        config.localized("2026-07-27", "xx")
    src = config.source("2026-07-27.de")
    assert src["raw"].name == "ai_act_2026-07-27.de.xhtml" and src["token"] == "deu"
    assert src["eli"].endswith("/DE/TXT/?uri=CELEX:02024R1689-20260727")
    assert config.source("2024-07-12.nl")["eli"].endswith("/oj/nld")
    assert config.source("2024-07-12") == {**config.VERSIONS["2024-07-12"], "lang": "en", "token": "eng"}


def test_each_language_gets_an_embedder_that_reads_it():
    assert config.embedder("2026-07-27") == (config.EMBED_MODEL, config.QUERY_PREFIX, "")
    assert config.embedder("2026-07-27.fr") == (config.MULTILINGUAL_EMBED, "query: ", "passage: ")
    # A fine-tuned bge keeps the bge prefix whatever it is called.
    assert config.embedder("2026-07-27", "train/bge-small-euaiact")[1:] == (config.QUERY_PREFIX, "")


def test_keyword_search_keeps_accented_words_whole_outside_english():
    assert tokenize("le système d'IA, Geldbuße") == ["le", "syst", "me", "d", "ia", "geldbu", "e"]
    assert tokenize("le système d'IA, Geldbuße", True) == ["le", "système", "d", "ia", "geldbuße"]


def test_every_language_numbers_into_the_same_references():
    french = ["1. Les pratiques suivantes sont interdites:", "a) la mise sur le marché",
              "b bis ) la mise sur le marché", "1 bis . Aux fins du paragraphe 1", "a) la mise",
              "2. L'utilisation"]
    assert outline_paths(french) == [("1",), ("1", "a"), ("1", "ba"), ("1a",), ("1a", "a"), ("2",)]
    dutch = ["1. Ongeacht", "1 bis. Voor de toepassing", "1 quater. Een product", "2. Naast"]
    assert outline_paths(dutch) == [("1",), ("1a",), ("1c",), ("2",)]
    # German writes paragraphs "(1)"; its consolidated text slips into "1b." once.
    german = ["(1) Folgende Praktiken sind verboten:", "a) das Inverkehrbringen",
              "(1a) Für die Zwecke", "1b. Für die Zwecke", "(2) Die Verwendung"]
    assert outline_paths(german) == [("1",), ("1", "a"), ("1a",), ("1b",), ("2",)]
    # Definitions: "(1)" in English, "1)" in French, "1." in German, all art_3(1).
    assert outline_paths(["Aux fins du présent règlement, on entend par:",
                          "14 bis ) «micro, petite ou moyenne entreprise»"]) == [(), ("14a",)]
    # English is read as before: a bracketed number indented under a paragraph stays under it.
    assert outline_paths(["1. A", "    (1) nested", "2. B"]) == [("1",), ("1", "1"), ("2",)]


def test_tags_are_carried_over_from_english_by_reference(monkeypatch):
    real = scope.load_chunks

    def fake(version=None, granularity=None):
        if version != "2026-07-27.fr":
            return real(version, granularity)
        return [{"chunk_id": "art_26(6)", "unit_id": "art_26", "meta": {"ref": "art_26(6)"}},
                {"chunk_id": "art_5(1)", "unit_id": "art_5", "meta": {"ref": "art_5(1)"}},
                {"chunk_id": "rct_27#p1", "unit_id": "rct_27", "meta": {}},
                {"chunk_id": "art_999", "unit_id": "art_999", "meta": {"ref": "art_999"}}]

    monkeypatch.setattr(scope, "load_chunks", fake)
    t = scope.tags.__wrapped__("2026-07-27.fr", "paragraph")
    assert t["art_26(6)"] == scope.tags("2026-07-27", "paragraph")["art_26(6)"]
    # English cuts Article 5(1) into four chunks; the French one gets the earliest date.
    assert t["art_5(1)"]["applies_from"] == "2025-02-02"
    # A recital has no paragraph reference, so its parts share the unit's tags.
    en = scope.tags("2026-07-27", "paragraph")
    both = en["rct_27#p1"]["roles"] + en["rct_27#p2"]["roles"]
    assert t["rct_27#p1"] == {"roles": [r for r in scope.ROLES if r in both]}
    assert t["art_999"] == {"roles": []}


def test_search_in_a_language_uses_its_corpus_and_refuses_the_guidelines(monkeypatch):
    seen = {}
    monkeypatch.setattr(retrieve, "split",
                        lambda q, k, **kw: seen.update(kw) or [])
    monkeypatch.setattr(scope, "tags", lambda version, granularity: {})
    assert retrieve.search("q", mode="split", version="2024-07-12", lang="nl") == []
    assert seen["version"] == "2024-07-12.nl"
    with pytest.raises(ValueError, match="English only"):
        retrieve.search("q", mode="split", lang="fr", guidelines=True)


@pytest.mark.parametrize("lang", OTHERS)
def test_the_fetched_text_lines_up_with_english(lang):
    if not fetched(lang):
        pytest.skip(f"run: euactrag fetch --lang {lang} && euactrag ingest --lang {lang}")

    def refs(version):
        return {r for c in load_chunks(version, "paragraph")
                for r in c["meta"].get("refs") or [c["unit_id"]]}

    for base in config.VERSIONS:
        missing = refs(base) - refs(config.localized(base, lang))
        # The Dutch 2024 text numbers Article 73(10) as a second "11".
        assert missing == ({"art_73(10)"} if (lang, base) == ("nl", config.ORIGINAL) else set())
    c = next(c for c in load_chunks(config.localized(config.LATEST, lang), "paragraph")
             if c["chunk_id"] == "art_5(1a)")
    assert f"/{lang.upper()}/TXT/" in c["url"] and c["citation"].startswith("Arti")
    tags = scope.tags(config.localized(config.LATEST, lang), "paragraph")
    assert tags["art_26(6)"] == scope.tags(config.LATEST, "paragraph")["art_26(6)"]
