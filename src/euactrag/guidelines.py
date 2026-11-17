"""The Commission's guidelines on the Act, cut into their numbered paragraphs.

    python -m src.euactrag.guidelines      # or: euactrag guidelines

downloads the four PDFs (config.GUIDELINES) and writes one chunk per numbered
paragraph, cited as "Guidelines on prohibited AI practices, para. 14" with a
link to the page. Needs pypdf: pip install "euactrag[guidelines]".

How the PDF is read. Body text, footnotes and footnote markers differ in font
size, so anything smaller than the body size is dropped, with two exceptions:
small digits straight after "10" are exponents (10^25 FLOP), and small bold
text is the small capitals of a heading. A paragraph starts at a
line that opens with the next number in sequence, "(15)" after "(14)", and
sits left of the text column. Section headings are the bold or italic lines
that open with a section number, or with "a)" under one. The paragraph count is checked against
config.GUIDELINES, so a file that reads differently raises.
"""
from __future__ import annotations

import collections
import json
import re
from dataclasses import asdict, replace

from . import config
from .ingest import Chunk, _breadcrumb, _split_paragraphs, approx_tokens

_PARA = re.compile(r"\((\d+)\)\s*")
_HEADING = re.compile(r"(\d+\.)+(\d+\.?)?\s*\S")
_SUBHEADING = re.compile(r"[a-z]\)\s+\S")  # "a) 'Workplace'", under a numbered heading


def fetch(key: str, force: bool = False) -> int:
    import httpx

    raw = config.GUIDELINES_RAW / f"{key}.pdf"
    if raw.exists() and not force:
        return raw.stat().st_size
    r = httpx.get(config.GUIDELINE_URL.format(doc=config.GUIDELINES[key]["doc"]),
                  timeout=120, follow_redirects=True)
    r.raise_for_status()
    if not r.content.startswith(b"%PDF-"):
        raise RuntimeError(f"{r.url} did not return a PDF; the Commission moved the file.")
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(r.content)
    print(f"-> {raw} ({len(r.content):,} bytes)")
    return len(r.content)


def lines(path) -> list[tuple[int, float, bool, str]]:
    """(page, x, emphasised, text) for every line of body text in the PDF."""
    from pypdf import PdfReader

    frags = []
    for page_no, page in enumerate(PdfReader(path).pages, 1):
        def visit(text, cm, tm, font, size, page_no=page_no):
            if text:
                name = str((font or {}).get("/BaseFont", ""))
                frags.append((page_no, round(size * abs(tm[3] or tm[0]), 1), tm[4], tm[5],
                              "Bold" in name, "Bold" in name or "Italic" in name, text))
        page.extract_text(visitor_text=visit)
    sizes: collections.Counter = collections.Counter()
    for _, size, _, _, _, _, text in frags:
        sizes[size] += len(text.strip())
    body = sizes.most_common(1)[0][0]

    out: list[list] = []  # [(page, y), x, [emphasised per char], [text]]
    for page_no, size, x, y, bold, emph, text in frags:
        if y < 72:  # the page number
            continue
        last = out[-1] if out else None
        if size < body and not bold:
            if (last and last[0][0] == page_no and abs(y - last[0][1]) < 8
                    and text.strip().isdigit() and "".join(last[3]).endswith("10")):
                last[3].append("^" + text.strip())
            continue
        if last and last[0] == (page_no, round(y)):
            last[3].append(text)
            last[2] += [emph] * len(text.strip())
        elif text.strip():
            out.append([(page_no, round(y)), x, [emph] * len(text.strip()), [text]])
    # "( )" is what is left of a footnote marker that was set in brackets.
    return [(key[0], x, sum(emph) * 2 > len(emph),
             re.sub(r"\s+", " ", re.sub(r"\(\s*\)", "", "".join(text))).strip())
            for key, x, emph, text in out]


def parse(key: str) -> list[Chunk]:
    g = config.GUIDELINES[key]
    rows = lines(config.GUIDELINES_RAW / f"{key}.pdf")
    column = collections.Counter(round(x) for _, x, _, _ in rows).most_common(1)[0][0]

    paras: list[dict] = []
    numbered, heading, in_heading = "", "", False
    for page, x, emph, text in rows:
        m = _PARA.match(text)
        if m and int(m[1]) == len(paras) + 1 and x < column - 5:
            paras.append({"n": len(paras) + 1, "page": page, "heading": heading,
                          "lines": [text[m.end():]]})
            in_heading = False
        elif emph and "...." not in text and _HEADING.match(text):
            numbered = heading = text
            in_heading = True
        elif emph and _SUBHEADING.match(text):
            heading = f"{numbered} {text}".strip()
            in_heading = True
        elif emph and in_heading:  # a heading that wraps
            heading = f"{heading} {text}"
            numbered = heading if heading.startswith(numbered) and " a) " not in heading else numbered
        else:
            in_heading = False
            if paras:
                paras[-1]["lines"].append(text)
    if len(paras) != g["paragraphs"]:
        raise ValueError(f"{key}: read {len(paras)} numbered paragraphs, expected "
                         f"{g['paragraphs']}. The PDF changed or reads differently.")

    out: list[Chunk] = []
    url = config.GUIDELINE_URL.format(doc=g["doc"])
    for p in paras:
        ref = f"gl_{key}({p['n']})"
        base = Chunk(
            chunk_id=ref, kind="guideline", unit_id=f"gl_{key}",
            citation=f"{g['title']}, para. {p['n']}", title=p["heading"],
            chapter_title=f"Commission {g['title']} ({g['number']})",
            section_title=p["heading"], url=f"{url}#page={p['page']}",
            version=g["adopted"], meta={"ref": ref, "refs": [ref], "page": p["page"]})
        parts = _split_paragraphs(" ".join(p["lines"]), config.MAX_CHUNK_TOKENS)
        for i, part in enumerate(parts, 1):
            c = replace(base, part=i, n_parts=len(parts), body=part.strip(),
                        chunk_id=ref if len(parts) == 1 else f"{ref}#p{i}")
            c.text = _breadcrumb(c) + c.body
            c.n_tokens = approx_tokens(c.text)
            out.append(c)
    return out


def main(force: bool = False) -> None:
    chunks: list[Chunk] = []
    for key in config.GUIDELINES:
        fetch(key, force)
        got = parse(key)
        toks = sorted(c.n_tokens for c in got)
        print(f"[{key}] paragraphs={config.GUIDELINES[key]['paragraphs']} chunks={len(got)} "
              f"tokens: p50={toks[len(toks) // 2]} max={toks[-1]}")
        chunks += got
    config.GUIDELINES_CHUNKS.parent.mkdir(parents=True, exist_ok=True)
    with open(config.GUIDELINES_CHUNKS, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    print(f"  -> {config.GUIDELINES_CHUNKS}")


if __name__ == "__main__":
    main()
