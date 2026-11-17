"""The MCP server, called the way a client calls it but in-process. Search is
left out: it needs the embedding model, which CI does not install."""
import asyncio

import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402

from src.euactrag import config  # noqa: E402
from src.euactrag.mcp_server import parse_ref, server  # noqa: E402


def call(name, **args):
    async def go():
        async with Client(server) as c:
            return await c.call_tool(name, args)
    return asyncio.run(go())


def test_references_are_read_however_they_are_written():
    assert parse_ref("Article 6(3)") == ("art_6", "art_6(3)")
    assert parse_ref("art. 5 (1)(f)") == ("art_5", "art_5(1)(f)")
    assert parse_ref("annex iii") == ("anx_III", "anx_III")
    assert parse_ref("Recital (27)") == ("rct_27", "rct_27")
    assert parse_ref("art_6(1a)") == ("art_6", "art_6(1a)")


def test_the_three_tools_are_listed_and_read_only():
    async def go():
        async with Client(server) as c:
            return (await c.list_tools()).tools
    tools = asyncio.run(go())
    assert [t.name for t in tools] == ["eu_ai_act_search", "eu_ai_act_get_provision",
                                       "eu_ai_act_list_provisions"]
    assert all(t.annotations.read_only_hint and t.output_schema for t in tools)


def test_a_provision_comes_back_whole_or_by_paragraph():
    whole = call("eu_ai_act_get_provision", ref="Article 6").structured_content
    para = call("eu_ai_act_get_provision", ref="art_6(3)").structured_content
    assert whole["citation"].startswith("Article 6 - ")
    assert para["refs"] == ["art_6(3)"] and "art_6(3)" in whole["refs"]
    assert para["text"] in whole["text"]
    # A point returns the paragraph it sits in.
    point = call("eu_ai_act_get_provision", ref="Article 5(1)(f)").structured_content
    assert point["refs"] == ["art_5(1)"]


def test_both_texts_are_served():
    old = call("eu_ai_act_get_provision", ref="Article 113", version=config.ORIGINAL)
    new = call("eu_ai_act_get_provision", ref="Article 113", version=config.LATEST)
    assert "2 December 2027" in new.structured_content["text"]
    assert "2 December 2027" not in old.structured_content["text"]


def test_a_wrong_reference_says_what_to_do_instead():
    missing = call("eu_ai_act_get_provision", ref="Article 60a", version=config.ORIGINAL)
    assert missing.is_error and f"exists in the {config.LATEST} text" in missing.content[0].text
    no_para = call("eu_ai_act_get_provision", ref="Article 6(99)")
    assert no_para.is_error and "art_6(3)" in no_para.content[0].text
    unreadable = call("eu_ai_act_get_provision", ref="section five")
    assert unreadable.is_error and "Annex III" in unreadable.content[0].text


def test_the_list_is_the_table_of_contents():
    articles = call("eu_ai_act_list_provisions",
                    version=config.ORIGINAL).structured_content["result"]
    assert len(articles) == 113 and articles[0]["ref"] == "art_1"
    assert articles[4] == {"ref": "art_5", "title": "Prohibited AI practices",
                           "chapter": "CHAPTER II - PROHIBITED AI PRACTICES", "section": ""}
    annexes = call("eu_ai_act_list_provisions", kind="annex",
                   version=config.ORIGINAL).structured_content["result"]
    assert len(annexes) == 13


def test_search_takes_a_role_and_a_date(monkeypatch):
    from src.euactrag import mcp_server, retrieve
    monkeypatch.setattr(mcp_server, "_ensure_index", lambda version: None)
    monkeypatch.setattr(retrieve, "split", lambda q, k, **kw: [
        {"chunk_id": c, "ref": c, "citation": c, "kind": "article", "text": "", "url": "",
         "rank": i} for i, c in enumerate(["art_19(1)", "art_26(6)", "art_50(4)"], 1)])
    out = call("eu_ai_act_search", query="what logs must I keep", role="deployer")
    assert [p["ref"] for p in out.structured_content["result"]] == ["art_26(6)", "art_50(4)"]
    out = call("eu_ai_act_search", query="what logs must I keep", applies_on="2026-11-17")
    assert out.structured_content["result"] == [
        {"ref": "art_50(4)", "citation": "art_50(4)", "kind": "article", "text": "", "url": "",
         "roles": ["deployer"], "applies_from": "2026-08-02", "applies_rule": "art_113"}]
    assert call("eu_ai_act_search", query="what logs", applies_on="soon").is_error


def test_search_can_add_guidelines(monkeypatch):
    from src.euactrag import mcp_server, retrieve
    monkeypatch.setattr(mcp_server, "_ensure_index", lambda version: None)
    hit = {"citation": "c", "text": "", "url": "", "rank": 1}
    monkeypatch.setattr(retrieve, "split", lambda q, k, **kw: [
        {**hit, "chunk_id": "art_50(4)", "ref": "art_50(4)", "kind": "article"}])
    monkeypatch.setattr(retrieve, "hybrid", lambda q, k, **kw: [
        {**hit, "chunk_id": "gl_transparency(12)", "ref": "gl_transparency(12)",
         "kind": "guideline"}])
    out = call("eu_ai_act_search", query="who is a deployer", guidelines=True)
    assert [(p["ref"], p["kind"]) for p in out.structured_content["result"]] == [
        ("art_50(4)", "article"), ("gl_transparency(12)", "guideline")]
    assert "applies_from" not in out.structured_content["result"][1]
