"""Parse the official EU AI Act XHTML into structure-aware, citable chunks.

Why not fixed-size chunks
-------------------------
The AI Act is authored as self-contained normative units. Legal questions are
answered with a *citation* ("Article 6(2)", "Annex III, point 5(b)"), so the
retrieval unit should be the unit a lawyer would cite. A 512-token sliding
window would routinely cut an article's scope clause away from its exceptions
paragraph, and in this corpus the exception *is* the answer. So we chunk on the
document's own ELI markup (`art_N`, `rct_N`, `anx_X`).

The three problems that naive article-chunking creates, and how we handle them:

1. Articles vary from 2 lines (Art. 4) to ~9k tokens (Art. 3, 68 definitions).
   Oversized articles are split at *numbered-paragraph* boundaries, never
   mid-sentence, and never mid-paragraph.
2. A split chunk retrieved on its own loses its identity ("...shall not apply"
, what shall not apply?). Every chunk therefore carries a breadcrumb header
   (Chapter > Section > Article N, Title) that is embedded along with the body.
3. Enumerations are encoded as nested two-column HTML tables. `get_text()`
   flattens them into unreadable soup, so we render them recursively back into
   "1. ... (a) ..." outline text.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, replace

from bs4 import BeautifulSoup, NavigableString, Tag

from . import config

# Roman numeral -> we keep the literal label from the document, no conversion.
_WS = re.compile(r"[ \t ]+")
_MULTINL = re.compile(r"\n{3,}")


def _clean(s: str) -> str:
    s = s.replace(" ", " ").replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = _WS.sub(" ", s).strip()
    # Rejoin exponents split by the inline <sup> tag: "10 ^25" -> "10^25".
    return re.sub(r"(\d)\s+\^(\d)", r"\1^\2", s)


def _fix_superscripts(soup: BeautifulSoup) -> None:
    """The 10^25 FLOP threshold in Article 51(2) is marked up as <sup>25</sup>.
    Flattening it yields "10 25", which is wrong and unsearchable. Only convert
    a digit superscript that directly follows a digit, the other 58 plain
    superscripts in the document are footnote reference markers, not exponents.
    """
    # "oj-super" in the Official Journal markup, "superscript" in the consolidated.
    for sp in soup.find_all("span", class_=("oj-super", "superscript")):
        if "oj-note-tag" in (sp.get("class") or []):
            continue
        txt = sp.get_text().strip()
        prev = sp.previous_sibling
        prev_txt = str(prev) if isinstance(prev, NavigableString) else ""
        if txt.isdigit() and prev_txt.rstrip()[-1:].isdigit():
            parent = sp.parent
            sp.replace_with(f"^{txt}")
            # The consolidated text holds this sentence in a <div>, whose text
            # nodes the renderer emits one per line; merge "10" and "^25" there.
            # Only there: the OJ rendering already reads "10^25 ." and the
            # original corpus is kept byte-identical to what was evaluated.
            if parent.name == "div":
                parent.smooth()


# Heading nodes are captured into structured fields, so the body renderer drops
# them rather than repeating them inline. `oj-note*` are footnote markers and OJ
# bibliographic footnotes: noise for retrieval. The second group is the same set
# in the consolidated-text markup, plus its "▼M1" amendment markers, which say
# which act changed a passage and are not part of the law's wording.
_SKIP_CLASSES = ("oj-note", "oj-ti-art", "oj-sti-art", "oj-doc-ti",
                 "title-article-norm", "stitle-article-norm", "title-annex-",
                 "footnote", "modref", "arrow")


def _classes(el: Tag) -> str:
    # A decomposed tag has attrs=None; treat it as classless rather than crashing.
    return " ".join(getattr(el, "attrs", None) and el.attrs.get("class") or [])


def _is_note(el: Tag) -> bool:
    cls = _classes(el)
    return any(s in cls for s in _SKIP_CLASSES)


def _row(marker: str, body: str, depth: int) -> list[str]:
    """One enumerated item: marker and first line joined, the rest indented."""
    pad = "    " * depth
    if marker and body:
        first, *rest = body.split("\n")
        return [f"{pad}{marker} {first}"] + [f"{pad}    {r}" for r in rest if r.strip()]
    return [f"{pad}{body}"] if body else []


def render(el: Tag, depth: int = 0) -> str:
    """Recursively render a node to outline text, preserving enumeration markers.

    EUR-Lex encodes "(a) some text" as a 2-column table row: a narrow marker cell
    and a wide content cell. We rejoin them and indent nested levels so that the
    hierarchy survives into the embedded text. The consolidated text says the
    same thing with a two-div "grid-list" for points and a "no-parag" span for
    paragraph numbers, which get the same treatment.
    """
    out: list[str] = []
    for child in el.children:
        if isinstance(child, NavigableString):
            txt = _clean(str(child))
            if txt:
                out.append(txt)
            continue
        if not isinstance(child, Tag):
            continue
        if _is_note(child):
            continue

        cls = _classes(child)
        if "grid-container" in cls:
            cols = child.find_all("div", recursive=False)
            if len(cols) == 2:
                out.extend(_row(_clean(cols[0].get_text(" ")),
                                render(cols[1], depth + 1).strip(), depth))
                continue
        num = child.find("span", class_="no-parag", recursive=False)
        if num is not None:
            marker = _clean(num.extract().get_text(" "))
            body = render(child, depth).strip()
            first, *rest = body.split("\n") if body else [""]
            out.append(f"{'    ' * depth}{marker} {first}".rstrip())
            out.extend(r for r in rest if r.strip())
            continue

        if child.name == "table":
            for row in child.find_all("tr", recursive=True):
                # only rows that belong to *this* table, not nested ones
                if row.find_parent("table") is not child:
                    continue
                cells = [c for c in row.find_all("td", recursive=False)]
                if len(cells) == 2:
                    out.extend(_row(_clean(cells[0].get_text(" ")),
                                    render(cells[1], depth + 1).strip(), depth))
                else:
                    joined = " ".join(_clean(c.get_text(" ")) for c in cells)
                    if joined.strip():
                        out.append("    " * depth + joined)
        elif child.name == "p":
            txt = _clean(child.get_text(" "))
            if txt:
                out.append("    " * depth + txt)
        else:
            sub = render(child, depth)
            if sub.strip():
                out.append(sub)
    return "\n".join(o for o in out if o.strip())


def approx_tokens(text: str) -> int:
    """Word-count proxy for tokens. Legal English runs ~1.35 tokens/word for
    bge's WordPiece vocab; we use 1.4 to stay conservative against the 512 cap."""
    return int(len(text.split()) * 1.4) + 1


@dataclass
class Chunk:
    chunk_id: str
    kind: str  # article | recital | annex
    unit_id: str  # art_6 / rct_27 / anx_III, the retrieval-scoring unit
    citation: str
    title: str = ""
    chapter: str = ""
    chapter_title: str = ""
    section: str = ""
    section_title: str = ""
    part: int = 1
    n_parts: int = 1
    body: str = ""
    text: str = ""  # breadcrumb + body; this is what gets embedded
    n_tokens: int = 0
    url: str = ""
    version: str = ""  # corpus version, the date of the text (config.VERSIONS)
    meta: dict = field(default_factory=dict)


def _breadcrumb(c: Chunk) -> str:
    bits = [b for b in (c.chapter_title or c.chapter, c.section_title or c.section) if b]
    head = " > ".join(bits)
    part = f" (part {c.part} of {c.n_parts})" if c.n_parts > 1 else ""
    return f"[{head}] {c.citation}{part}\n" if head else f"{c.citation}{part}\n"


# A line that opens a new enumerated item: "1.", "(12)", "(a)", "(iv)", "Section A."
_MARKER = re.compile(
    r"^(?:\d{1,3}\.|\(\d{1,3}\)|\([a-z]{1,3}\)|\([ivxlcdm]{1,7}\)|Section\s+[A-Z]\.)\s"
)
# Sentence boundary for legal prose: after . ; or : followed by a capital or "(".
# Fixed-width lookbehinds guard the abbreviations that actually recur in OJ text
# ("No. 5", "Art. 6"); "p. 24" is safe already since a digit follows.
_SENT = re.compile(r"(?<!\bNo\.)(?<!\bArt\.)(?<=[.;:])\s+(?=[A-Z(\"'])")


def _atoms(body: str) -> list[str]:
    """Group lines into indivisible semantic units (one enumerated item, or one
    paragraph). Indentation is preserved; matching ignores it so that nested
    points are still recognised as boundaries."""
    atoms: list[list[str]] = []
    for ln in body.split("\n"):
        if not atoms or _MARKER.match(ln.lstrip()):
            atoms.append([ln])
        else:
            atoms[-1].append(ln)
    return ["\n".join(a) for a in atoms]


def _sentence_split(text: str, max_tokens: int) -> list[str]:
    """Last-resort split for unenumerated prose (long recitals are a single
    paragraph). Packs whole sentences up to the budget."""
    sents = _SENT.split(text)
    out, cur, tok = [], [], 0
    for s in sents:
        st = approx_tokens(s)
        if cur and tok + st > max_tokens:
            out.append(" ".join(cur))
            cur, tok = [], 0
        cur.append(s)
        tok += st
    if cur:
        out.append(" ".join(cur))
    return out or [text]


def _split_paragraphs(body: str, max_tokens: int) -> list[str]:
    """Split an oversized unit, always at a semantic boundary.

    Order of preference: enumerated-item boundary > sentence boundary. We never
    split mid-sentence, so a definition or a paragraph always stays whole.
    Anything left over budget after both passes is a single sentence, which we
    keep intact and let the encoder truncate, that is rare and visible in the
    ingest stats rather than silent.
    """
    units: list[str] = []
    for atom in _atoms(body):
        units.extend(
            [atom] if approx_tokens(atom) <= max_tokens
            else _sentence_split(atom, max_tokens)
        )

    parts: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for u in units:
        ut = approx_tokens(u)
        if cur and cur_tok + ut > max_tokens:
            parts.append("\n".join(cur))
            cur, cur_tok = [], 0
        cur.append(u)
        cur_tok += ut
    if cur:
        parts.append("\n".join(cur))

    # Fold a runt tail back into its predecessor rather than emitting a stub.
    if len(parts) > 1 and approx_tokens(parts[-1]) < config.MIN_CHUNK_TOKENS:
        # Pop first: written as one expression, the pop runs before the
        # assignment and [-2] then points one part too far back.
        tail = parts.pop()
        parts[-1] = parts[-1] + "\n" + tail
    return parts or [body]


def _emit(base: Chunk, body: str, out: list[Chunk]) -> None:
    parts = _split_paragraphs(body, config.MAX_CHUNK_TOKENS)
    for i, p in enumerate(parts, 1):
        c = Chunk(**{**asdict(base), "part": i, "n_parts": len(parts)})
        c.body = p.strip()
        c.chunk_id = base.unit_id if len(parts) == 1 else f"{base.unit_id}#p{i}"
        c.text = _breadcrumb(c) + c.body
        c.n_tokens = approx_tokens(c.text)
        out.append(c)


# ---- Paragraph granularity -----------------------------------------------
# A legal answer is cited as "Article 6(3)" or "Annex III, point 4(a)", not as
# "Article 6". The paragraph chunker labels every line of a unit with its place
# in the outline (paragraph, point, sub-point) and cuts one chunk per top-level
# item, so a retrieved chunk can be cited to its paragraph.
#
# Reference format: the unit id followed by each outline marker in brackets.
# art_6(3), art_5(1)(ba), art_3(12), art_113(c)(i), anx_III(4)(a), anx_I(A)(1).
# Unnumbered paragraphs (Article 113's first lines, Article 4 in 2024) have no
# marker of their own, so text there is cited to the unit alone.
_M_PARA = re.compile(r"^(\d+[a-z]*)\.\s")           # 1.  1a.
_M_NUM = re.compile(r"^\((\d+[a-z]*)\)\s")          # (1) (12a), definitions
_M_ALPHA = re.compile(r"^\(([a-z]{1,3})\)\s")        # (a) (ba) (iv)
_M_SECTION = re.compile(r"^Section\s+([A-Z]|\d+)(?:\.|\s+[—-]\s|$)")  # "Section A.", "Section A —", "Section 1"
_ROMAN = re.compile(r"^(?=[ivx])x{0,3}(ix|iv|v?i{0,3})$")


def outline_paths(lines: list[str]) -> list[tuple[str, ...]]:
    """The outline position of every line, e.g. ("3", "a") for Article 6(3)(a).

    Ranks: Section A. < 1. < (1) < (a) < (i). "(i)", "(v)" and "(x)" are read as
    roman sub-points only when indented under a letter point, since the renderer
    indents nested items; at the same depth they are the next letter. An
    unmarked line with no indent closes the points above it: it is the next
    subparagraph of the same paragraph, not a continuation of the last point.
    """
    stack: list[tuple[int, str, int]] = []  # (rank, marker, indent)
    out = []
    for ln in lines:
        s = ln.lstrip()
        indent = len(ln) - len(s)
        rank = val = None
        for r, rx in ((-1, _M_SECTION), (0, _M_PARA), (1, _M_NUM)):
            m = rx.match(s)
            if m:
                rank, val = r, m.group(1)
                break
        else:
            m = _M_ALPHA.match(s)
            if m:
                val = m.group(1)
                letters = [i for r, _, i in stack if r == 2]
                nested = bool(letters) and indent > letters[-1]
                rank = 3 if (_ROMAN.match(val) and nested) else 2
        if rank is not None:
            while stack and stack[-1][0] >= rank:
                stack.pop()
            stack.append((rank, val, indent))
        elif indent == 0:
            while stack and stack[-1][0] >= 2:
                stack.pop()
        out.append(tuple(v for _, v, _ in stack))
    return out


def ref_of(unit_id: str, path: tuple[str, ...]) -> str:
    return unit_id + "".join(f"({p})" for p in path)


def _common(paths: list[tuple[str, ...]]) -> tuple[str, ...]:
    out: list[str] = []
    for level in zip(*paths):
        if len(set(level)) > 1:
            break
        out.append(level[0])
    return tuple(out)


def _emit_paragraphs(base: Chunk, body: str, out: list[Chunk]) -> None:
    """One chunk per top-level outline item, split further only if oversized.

    Each chunk records `ref`, what it is cited as (the deepest outline position
    all its lines share), and `refs`, every position it contains.
    """
    lines = [ln for ln in body.split("\n") if ln.strip()]
    paths = outline_paths(lines)
    groups: list[list[int]] = []
    for i, p in enumerate(paths):
        if groups and paths[groups[-1][0]][:1] == p[:1]:
            groups[-1].append(i)
        else:
            groups.append([i])
    label = base.citation.split(" - ", 1)[0]
    seen: dict[str, int] = {}
    for g in groups:
        text = "\n".join(lines[i] for i in g)
        parts = _split_paragraphs(text, config.MAX_CHUNK_TOKENS)
        top = ref_of(base.unit_id, paths[g[0]][:1])
        # Unnumbered text can come back after the points (Article 75b), so the
        # same reference can open two groups; the id has to stay unique.
        seen[top] = seen.get(top, 0) + 1
        cid = top if seen[top] == 1 else f"{top}~{seen[top]}"
        for n, p in enumerate(parts, 1):
            # ponytail: a line is matched to its part by its opening words; a
            # sentence-split line only claims the first part, later parts fall
            # back to the paragraph's own reference.
            own = [paths[i] for i in g if lines[i].strip()[:60] in p]
            path = _common(own) if own else paths[g[0]][:1]
            ref = ref_of(base.unit_id, path)
            c = replace(base, part=n, n_parts=len(parts),
                        meta={**base.meta, "ref": ref,
                              "refs": sorted({ref_of(base.unit_id, q) for q in own} | {ref})})
            c.citation = (f"{label}{ref[len(base.unit_id):]} - {base.title}"
                          if base.title else f"{label}{ref[len(base.unit_id):]}")
            c.body = p.strip()
            c.chunk_id = cid if len(parts) == 1 else f"{cid}#p{n}"
            c.text = _breadcrumb(c) + c.body
            c.n_tokens = approx_tokens(c.text)
            out.append(c)


def parse(version: str = config.ORIGINAL, granularity: str = "article") -> list[Chunk]:
    eli = config.VERSIONS[version]["eli"]
    xhtml_path = config.VERSIONS[version]["raw"]
    soup = BeautifulSoup(open(xhtml_path, encoding="utf-8").read(), "lxml")
    _fix_superscripts(soup)
    chunks: list[Chunk] = []
    # Recitals are one short paragraph each, so they stay whole either way.
    emit = _emit_paragraphs if granularity == "paragraph" else _emit

    # Chapter/section headings are plain <p> in document order, not wrapped in a
    # container we can nest on: so we sweep the document once and remember the
    # most recent heading seen before each article.
    chapter = chapter_title = section = section_title = ""
    pending: str | None = None  # "chapter" or "section" awaiting its title line

    for el in soup.find_all(["p", "div"]):
        cls = _classes(el)
        eid = (getattr(el, "attrs", None) or {}).get("id", "") or ""

        if "oj-ti-section-1" in cls or "title-division-1" in cls:
            t = _clean(el.get_text(" "))
            if t.upper().startswith("CHAPTER"):
                chapter, chapter_title, pending = t, "", "chapter"
                section = section_title = ""
            elif t.upper().startswith("SECTION"):
                section, section_title, pending = t, "", "section"
            continue
        if ("oj-ti-section-2" in cls or "title-division-2" in cls) and pending:
            t = _clean(el.get_text(" "))
            if pending == "chapter":
                chapter_title = f"{chapter} - {t}"
            else:
                section_title = f"{section} - {t}"
            pending = None
            continue

        if el.name != "div":
            continue

        # ---- Articles -------------------------------------------------
        if "eli-subdivision" in cls and eid.startswith("art_"):
            num = eid.split("_", 1)[1]
            ti = el.find("p", class_=("oj-ti-art", "title-article-norm"))
            sti = el.find("p", class_=("oj-sti-art", "stitle-article-norm"))
            label = _clean(ti.get_text(" ")) if ti else f"Article {num}"
            # Both renderings leave a stray closing quote on a few titles.
            title = _clean(sti.get_text(" ")).rstrip("`'") if sti else ""
            body = render(el)
            base = Chunk(
                chunk_id=eid, kind="article", unit_id=eid,
                citation=f"{label} - {title}" if title else label,
                title=title, chapter=chapter, chapter_title=chapter_title,
                section=section, section_title=section_title,
                url=f"{eli}#{eid}", version=version, meta={"article": num},
            )
            emit(base, body, chunks)

        # ---- Recitals -------------------------------------------------
        elif "eli-subdivision" in cls and eid.startswith("rct_"):
            num = eid.split("_", 1)[1]
            body = render(el)
            body = re.sub(r"^\(\d+\)\s*", "", body).strip()
            base = Chunk(
                chunk_id=eid, kind="recital", unit_id=eid,
                citation=f"Recital ({num})", chapter="Recitals",
                chapter_title="Recitals (non-binding interpretive context)",
                url=f"{eli}#{eid}", version=version, meta={"recital": num},
            )
            _emit(base, body, chunks)

        # ---- Annexes --------------------------------------------------
        # The consolidated text puts annexes in a bare <div id="anx_I">.
        elif eid.startswith("anx_") and (not cls or "eli-container" in cls):
            roman = eid.split("_", 1)[1]
            titles = el.find_all(
                "p", class_=("oj-doc-ti", "title-annex-1", "title-annex-2"), limit=2)
            label = _clean(titles[0].get_text(" ")) if titles else f"ANNEX {roman}"
            subtitle = _clean(titles[1].get_text(" ")) if len(titles) > 1 else ""
            body = render(el)
            base = Chunk(
                chunk_id=eid, kind="annex", unit_id=eid,
                citation=f"{label} - {subtitle}" if subtitle else label,
                title=subtitle, chapter="Annexes", chapter_title="Annexes",
                url=f"{eli}#{eid}", version=version, meta={"annex": roman},
            )
            emit(base, body, chunks)

    # A consolidated text is published without the preamble. The recitals of
    # 2024/1689 are not amended (the Omnibus has recitals of its own, which
    # explain the amendments but are not part of this Regulation), so the
    # original ones are carried over and still link to the Official Journal.
    if version != config.ORIGINAL and not any(c.kind == "recital" for c in chunks):
        chunks = [replace(c, version=version) for c in parse(config.ORIGINAL)
                  if c.kind == "recital"] + chunks
    return chunks


def write_jsonl(chunks: list[Chunk], version: str, granularity: str = "article") -> None:
    path = config.chunks_path(version, granularity)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_chunks(version: str | None = None, granularity: str | None = None) -> list[dict]:
    path = config.chunks_path(version or config.CORPUS_VERSION,
                              granularity or config.GRANULARITY)
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> None:
    from collections import Counter

    for version, gran in [(v, g) for v in sorted(config.VERSIONS)
                          for g in config.GRANULARITIES]:
        chunks = parse(version, gran)
        write_jsonl(chunks, version, gran)
        kinds = Counter(c.kind for c in chunks)
        toks = sorted(c.n_tokens for c in chunks)
        units = len({c.unit_id for c in chunks})
        print(f"[{version} {gran}] chunks={len(chunks)} units={units} {dict(kinds)}")
        print(
            f"  tokens: min={toks[0]} p50={toks[len(toks)//2]} "
            f"p95={toks[int(len(toks)*.95)]} max={toks[-1]}"
        )
        print(f"  -> {config.chunks_path(version, gran)}")


if __name__ == "__main__":
    main()
