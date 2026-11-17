"""The Act as tools for an AI assistant, over the Model Context Protocol.

    pip install "euactrag[mcp]"
    euactrag mcp                   # speaks MCP on stdin/stdout

Three read-only tools: search, fetch a provision by its reference, and list
what there is. The assistant on the other end writes the answer; this only
hands it the text, cited to the paragraph, so there is no LLM key involved.
Both texts of the Act are served, so "what did Article 6 say before the 2026
amendments?" is a second call with the other version.
"""
from __future__ import annotations

import functools
import re
from datetime import date
from typing import Annotated, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field
from typing_extensions import NotRequired, TypedDict

from . import __version__, config
from .ingest import load_chunks
from .scope import ROLES

Version = Annotated[
    Literal[tuple(sorted(config.VERSIONS, reverse=True))],  # type: ignore[valid-type]
    Field(description=f"Which text of the Act, by date. {config.LATEST} is the "
                      f"consolidated text after the 2026 amendments (Regulation (EU) "
                      f"2026/1744) and the default. {config.ORIGINAL} is the text as "
                      f"first published.")]



class Passage(TypedDict):
    ref: str
    citation: str
    kind: str
    text: str
    url: str
    roles: list[str]
    applies_from: NotRequired[str]
    applies_rule: NotRequired[str]


class Provision(TypedDict):
    citation: str
    version: str
    text: str
    url: str
    refs: list[str]


class Entry(TypedDict):
    ref: str
    title: str
    chapter: NotRequired[str]
    section: NotRequired[str]


READ_ONLY = dict(read_only_hint=True, destructive_hint=False, idempotent_hint=True,
                 open_world_hint=False)

server = MCPServer(
    "eu_ai_act_mcp", version=__version__,
    instructions=(
        "The text of the EU AI Act (Regulation (EU) 2024/1689), in two versions: as "
        f"published ({config.ORIGINAL}) and as amended by Regulation (EU) 2026/1744 "
        f"({config.LATEST}, the default). Use eu_ai_act_search for a question, "
        "eu_ai_act_get_provision when you know the article, and cite the `citation` "
        "and `url` of what you use. Recitals are not binding. This is the text of "
        "the law, not legal advice."))


@functools.lru_cache(maxsize=None)
def _chunks(version: str) -> list[dict]:
    try:
        return load_chunks(version, "paragraph")
    except FileNotFoundError:
        raise ToolError(
            f"The {version} text has not been downloaded on this machine. Run "
            f"`euactrag fetch && euactrag ingest && euactrag index` once, then try "
            f"again.") from None


@functools.lru_cache(maxsize=None)
def _ensure_index(version: str) -> None:
    """Build the vector index on first use, as the Streamlit app does."""
    from . import index

    try:
        index.get_collection(version=version, granularity="paragraph").count()
    except Exception:
        _chunks(version)
        index.build(version=version, granularity="paragraph")


_REF = re.compile(r"^\s*(art|anx|annex|rct|rec)[a-z]*[\s_.]*\(?(\d+[a-z]?|[ivx]+)\)?"
                  r"((?:\s*\(\w+\))*)\s*$", re.I)


def parse_ref(ref: str) -> tuple[str, str]:
    """("art_6", "art_6(3)") for "Article 6(3)", "art_6(3)" or "art. 6 (3)"."""
    m = _REF.match(ref)
    if not m:
        raise ToolError(
            f"Could not read {ref!r} as a reference. Write it like 'Article 6', "
            f"'Article 5(1)(f)', 'Annex III', 'Recital 27' or 'art_6(3)'.")
    kind = {"ar": "art", "an": "anx", "rc": "rct", "re": "rct"}[m.group(1)[:2].lower()]
    num = m.group(2).upper() if kind == "anx" else m.group(2).lower()
    unit = f"{kind}_{num}"
    return unit, unit + re.sub(r"\s+", "", m.group(3))


@server.tool(annotations=ToolAnnotations(title="Search the EU AI Act", **READ_ONLY))
def eu_ai_act_search(
    query: Annotated[str, Field(min_length=3, max_length=500, description=(
        "A question or topic in plain English, e.g. 'Is emotion recognition at "
        "work prohibited?'. Two-part questions are fine."))],
    k: Annotated[int, Field(ge=1, le=20, description="How many passages.")] = config.TOP_K,
    version: Version = config.LATEST,
    role: Annotated[Literal[ROLES] | None, Field(  # type: ignore[valid-type]
        description="Who is asking. Leaves out passages that name other roles but "
                    "not this one. Passages that name no role are kept.")] = None,
    applies_on: Annotated[date | None, Field(
        description="A date, e.g. 2026-12-01. Leaves out articles that do not apply "
                    "yet on that date under Article 113.")] = None,
    guidelines: Annotated[bool, Field(
        description="After the passages of the Act, add up to three paragraphs of the "
                    "Commission's guidelines on it (AI system definition, prohibited "
                    "practices, general-purpose AI models, Article 50 transparency). "
                    "They explain the law and are not binding.")] = False,
) -> list[Passage]:
    """Find the passages of the EU AI Act most relevant to a question.

    Returns up to `k` passages, best first, each one paragraph of an article,
    annex or recital, with `ref` (e.g. art_5(1), usable with
    eu_ai_act_get_provision), `citation`, `kind` (article, annex, recital or,
    with `guidelines`, guideline: one numbered paragraph of a Commission
    guideline, listed after the Act, to cite as guidance and not as law),
    `text` and the EUR-Lex `url`. `roles` lists which of provider, deployer,
    importer and distributor the passage names, by a word match. Articles
    also carry `applies_from`, the earliest date Article 113 sets for them,
    and `applies_rule`, the point of Article 113 to read for the detail (the
    2026 text gives high-risk rules two dates). Neither covers the
    transitional periods in Article 111. Search is hybrid (embeddings plus keywords);
    it finds one right provision for almost every question but can miss the
    second article of a two-article question, and dates of application
    (Article 113) in particular, so fetch those directly when they matter.
    """
    from . import retrieve

    # Indexing prints progress. That is safe: the SDK's stdio transport keeps
    # the protocol on its own descriptor and sends stray prints to stderr.
    _ensure_index(version)
    try:
        hits = retrieve.search(query, k=k, version=version, granularity="paragraph",
                               role=role, on=applies_on and applies_on.isoformat(),
                               guidelines=guidelines)
    except FileNotFoundError as e:  # the guidelines have not been downloaded
        raise ToolError(str(e)) from None
    keys = ("ref", "citation", "kind", "text", "url", "roles", "applies_from", "applies_rule")
    return [{key: h[key] for key in keys if key in h} for h in hits]


@server.tool(annotations=ToolAnnotations(title="Get a provision of the EU AI Act",
                                         **READ_ONLY))
def eu_ai_act_get_provision(
    ref: Annotated[str, Field(min_length=3, max_length=40, description=(
        "An article, annex or recital, or one paragraph of it: 'Article 6', "
        "'Article 5(1)(f)', 'Annex III', 'Recital 27', or a `ref` from a search "
        "result such as 'art_6(3)'."))],
    version: Version = config.LATEST,
) -> Provision:
    """The exact text of one provision of the EU AI Act.

    Returns `citation`, `version`, `text`, the EUR-Lex `url` and `refs`, the
    paragraphs included. Asking for a point such as Article 5(1)(f) returns
    the paragraph it sits in. Call it twice with the two versions to see what
    the 2026 amendments changed.
    """
    unit, want = parse_ref(ref)
    chunks = [c for c in _chunks(version) if c["unit_id"] == unit]
    if not chunks:
        other = next((v for v in config.VERSIONS if v != version
                      and any(c["unit_id"] == unit for c in _chunks(v))), None)
        raise ToolError(
            f"There is no {ref} in the {version} text."
            + (f" It exists in the {other} text." if other else
               " Use eu_ai_act_list_provisions to see what there is."))
    if want != unit:
        part = [c for c in chunks if want in c["meta"].get("refs", [])]
        if not part:
            have = sorted({c["meta"].get("ref", unit) for c in chunks})
            raise ToolError(f"{unit} has no {want} in the {version} text. Its "
                             f"paragraphs are: {', '.join(have)}.")
        chunks = part
    whole = want == unit and len(chunks) > 1
    citation = chunks[0]["citation"]
    if whole:  # "Article 6(1) - Title" -> "Article 6 - Title"
        citation = re.sub(r"^(\S+ \(?\w+\)?)\(\w+\)", r"\1", citation)
    return {
        "citation": citation, "version": version,
        "text": "\n".join(c["body"] for c in chunks), "url": chunks[0]["url"],
        "refs": list(dict.fromkeys(c["meta"].get("ref", unit) for c in chunks)),
    }


@server.tool(annotations=ToolAnnotations(title="List the provisions of the EU AI Act",
                                         **READ_ONLY))
def eu_ai_act_list_provisions(
    kind: Annotated[Literal["article", "annex", "recital"], Field(
        description="Articles (with chapter and title), annexes or recitals.")] = "article",
    version: Version = config.LATEST,
) -> list[Entry]:
    """The table of contents: every article, annex or recital in one version.

    Each entry has `ref` (usable with eu_ai_act_get_provision), `title` and,
    for articles, the `chapter` and `section` it sits in. Recitals have no
    titles, so they are listed by number only.
    """
    seen: dict[str, Entry] = {}
    for c in _chunks(version):
        if c["kind"] == kind and c["unit_id"] not in seen:
            seen[c["unit_id"]] = {"ref": c["unit_id"], "title": c["title"]}
            if kind == "article":
                seen[c["unit_id"]] |= {"chapter": c["chapter_title"],
                                       "section": c["section_title"]}
    return list(seen.values())


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
