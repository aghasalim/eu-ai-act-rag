"""Build the static site from the chunked corpus. Standard library only.

One page per article and annex, a recitals page, an index, a sitemap, and the
risk checker. Nothing on the site is written by hand or by a model: every
sentence of law is the corpus text, and the checker's questions are the points
of Article 5(1), Annex III, Article 6 and Article 50 as they stand in the
corpus version being built. When the law is amended and the corpus is
re-ingested, the site follows.

    python site/build.py                 # -> _site/
    python site/build.py --base-url https://aghasalim.github.io/eu-ai-act-rag
"""
from __future__ import annotations

import argparse
import difflib
import html
import itertools
import json
import math
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.euactrag import config  # noqa: E402

HERE = Path(__file__).parent
REPO = "https://github.com/aghasalim/eu-ai-act-rag"
DEMO = "https://eu-ai-act-rag-eval.streamlit.app/"
ACT = "Regulation (EU) 2024/1689"
esc = html.escape
# Twelve dots in a ring, the header mark.
STARS = ('<svg class="stars" viewBox="-12 -12 24 24" aria-hidden="true">' + "".join(
    f'<circle cx="{7.5 * math.cos(math.radians(30 * i)):.2f}" '
    f'cy="{7.5 * math.sin(math.radians(30 * i)):.2f}" r="1.1"/>' for i in range(12)) + "</svg>")


# --------------------------------------------------------------------------
# Corpus -> units
# --------------------------------------------------------------------------
def slug(unit_id: str) -> str:
    kind, num = unit_id.split("_", 1)
    return {"art": "article-", "anx": "annex-"}[kind] + num.lower()


def name(unit_id: str) -> str:
    kind, num = unit_id.split("_", 1)
    return {"art": "Article ", "anx": "Annex ", "rct": "Recital "}[kind] + num


def load_units(version: str) -> dict[str, dict]:
    """unit_id -> {title, chapter, section, url, lines}, in the Act's order.
    A line is (depth, anchor, text): depth 0 is a paragraph, 1 a point, 2 a
    sub-point, and anchor is "3-a" for Article 6(3)(a) or None for a line that
    continues the one above."""
    units: dict[str, dict] = {}
    for raw in open(config.chunks_path(version, "paragraph"), encoding="utf-8"):
        c = json.loads(raw)
        u = units.setdefault(c["unit_id"], {
            "kind": c["kind"], "title": c["title"], "chapter": c["chapter_title"],
            "section": c["section_title"], "url": c["url"], "lines": []})
        refs = list(c["meta"].get("refs", []))
        depth, indent = 0, 0
        for raw_line in c["body"].split("\n"):
            line, anchor = raw_line.strip(), None
            if not line:
                continue
            m = re.match(r"(?:\(([0-9a-z]+)\)|(\d+[a-z]*)\.)\s", line)
            label = m and (m.group(1) or m.group(2))
            hit = label and next((r for r in refs if r.endswith(f"({label})")), None)
            if hit:
                refs = refs[refs.index(hit) + 1:]
                outline = re.findall(r"\(([^)]+)\)", hit)
                depth, anchor = len(outline) - 1, "-".join(outline)
                indent = len(raw_line) - len(raw_line.lstrip())
            elif len(raw_line) - len(raw_line.lstrip()) <= indent:
                # An unlabelled line that is not indented under the point above
                # is a new subparagraph of the paragraph, not more of the point:
                # the last sentence of Article 6(3) is not part of point (d).
                depth = 0
            u["lines"].append((depth, anchor, line))
    return units


# --------------------------------------------------------------------------
# Cross-references
# --------------------------------------------------------------------------
# "Article 9(1) of Regulation (EU) 2016/679" points at another law. Linking it
# to this Act's Article 9 would be a wrong citation, so anything followed
# shortly by "of <some other instrument>" or by TFEU/TEU is left alone.
_OTHER_LAW = (r"(?![^.;]{0,60}?\bof\s+(?:Regulation|Directive|Decision|Council|"
              r"Framework|Implementing|Delegated|the\s+Charter|the\s+Treaty|"
              r"that\s+(?:Regulation|Directive|Decision))|\s+T?F?EU\b)")
_XREF = re.compile(
    r"\b(Articles?|Annex(?:es)?|ANNEX)\s+(\d+[a-z]?|[IVX]+)\b(?:\((\d+[a-z]?)\))?"
    + _OTHER_LAW)


def link_refs(text: str, units: dict, here: str, found: set[str]) -> str:
    """Escape `text` and turn references to this Act's articles and annexes
    into links. Targets are added to `found` for the backlinks."""
    def sub(m: re.Match) -> str:
        kind = "art" if m.group(1).lower().startswith("art") else "anx"
        target = f"{kind}_{m.group(2)}"
        if target not in units or (kind == "art") != m.group(2)[0].isdigit():
            return m.group(0)
        if target != here:
            found.add(target)
        para = f"#{m.group(3)}" if m.group(3) else ""
        return f'<a href="{slug(target)}.html{para}">{m.group(0)}</a>'
    return _XREF.sub(sub, esc(text, quote=False))


# --------------------------------------------------------------------------
# What the amendments changed
# --------------------------------------------------------------------------
def _words(unit: dict) -> list[str]:
    """The unit's text as words, with a "\n" between lines. The two versions
    come from different markup, so footnote markers, curly quotes and a space
    before punctuation are evened out first: none of those is an amendment."""
    out: list[str] = []
    for _, _, text in unit["lines"]:
        text = text.replace("\u2019", "'").replace("\u2018", "'")
        text = re.sub(r"\(\s*\*\d*\s*\)|\(\s+\d+\s+\)", "", text)
        out += re.sub(r"\s+([.,;:)])", r"\1", text).split() + ["\n"]
    return out


def changes(old: dict | None, new: dict) -> str | None:
    """The paragraphs of `new` that differ from `old`, word by word, as HTML
    with <ins> and <del>. None when the unit is new or the words agree."""
    if old is None:
        return None
    a, b = _words(old), _words(new)
    pairs: list[tuple[str, str]] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            pairs += [("", w) for w in a[i1:i2]]
        if op in ("delete", "replace"):
            pairs += [("del", w) for w in a[i1:i2]]
        if op in ("insert", "replace"):
            pairs += [("ins", w) for w in b[j1:j2]]
    paras = [list(g) for brk, g in itertools.groupby(pairs, key=lambda x: x[1] == "\n")
             if not brk]
    out = []
    for para in paras:
        if not any(tag for tag, _ in para):
            continue
        runs = []
        for tag, group in itertools.groupby(para, key=lambda x: x[0]):
            words = esc(" ".join(w for _, w in group), quote=False)
            runs.append(f"<{tag}>{words}</{tag}>" if tag else words)
        out.append(f"<p>{' '.join(runs)}</p>")
    return "\n".join(out) or None


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page(title: str, desc: str, body: str, path: str, base: str,
         jsonld: dict | None = None, script: str = "") -> str:
    canonical = f'<link rel="canonical" href="{base}/{path}">' if base else ""
    ld = (f'<script type="application/ld+json">{json.dumps(jsonld)}</script>'
          if jsonld else "")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{esc(desc)}">
<meta property="og:title" content="{esc(title)}">
<meta property="og:description" content="{esc(desc)}">
<meta property="og:type" content="article">
{canonical}
<link rel="stylesheet" href="style.css">
{ld}
</head>
<body>
<header class="top">
  <a class="brand" href="index.html">{STARS}EU AI Act, by the paragraph</a>
  <nav>
    <a href="index.html#articles">Articles</a>
    <a href="index.html#annexes">Annexes</a>
    <a href="recitals.html">Recitals</a>
    <a href="changes.html">2026 changes</a>
    <a class="cta" href="checker.html">Risk checker</a>
  </nav>
</header>
<main>
{body}
</main>
<footer>
  <p>Text: {ACT} as amended, consolidated version of {esc(VERSION)}, from the
  EU Publications Office. Reuse is authorised under Decision 2011/833/EU. Only
  the texts published in the Official Journal are authentic.</p>
  <p>This is a student project and not legal advice. It is not affiliated with
  or endorsed by the EU. <a href="{REPO}">Source and evaluation on GitHub</a> ·
  <a href="{DEMO}">Ask a question</a> · <a href="act.json">This text as JSON</a></p>
</footer>
{script}
</body>
</html>
"""


def render_lines(lines, units, here, found) -> str:
    out = []
    for depth, anchor, text in lines:
        ident = f' id="{anchor}"' if anchor else ""
        pin = (f'<a class="pin" href="#{anchor}" aria-label="Link to this provision">#</a>'
               if anchor else "")
        out.append(f'<p class="d{min(depth, 2)}"{ident}>'
                   f'{link_refs(text, units, here, found)}{pin}</p>')
    return "\n".join(out)


def status(uid: str, units: dict, original: dict) -> str:
    """ "added", "amended" or "" against the original text."""
    if uid not in original:
        return "added"
    return "amended" if changes(original[uid], units[uid]) else ""


def unit_page(uid: str, units: dict, order: list[str], refs_to: dict, base: str,
              original: dict) -> str:
    u, found = units[uid], set()
    state = status(uid, units, original)
    badge = f' · <span class="badge {state}">{state.capitalize()} in {VERSION[:4]}</span>' if state else ""
    diff = ""
    if state == "amended":
        diff = f"""<details class="diff"><summary>What changed against the {config.ORIGINAL[:4]} text</summary>
<p class="hint">Paragraphs that differ from the original Official Journal text,
compared word by word: <ins>added</ins>, <del>removed</del>. The two texts come
from different source documents, so a difference in layout can show up here too.</p>
{changes(original[uid], u)}</details>"""
    text = render_lines(u["lines"], units, uid, found)
    refs_to[uid] = found
    i = order.index(uid)
    prev = (f'<a href="{slug(order[i - 1])}.html">← {name(order[i - 1])}</a>'
            if i else "<span></span>")
    nxt = (f'<a href="{slug(order[i + 1])}.html">{name(order[i + 1])} →</a>'
           if i + 1 < len(order) else "<span></span>")
    crumbs = " › ".join(esc(x) for x in (u["chapter"], u["section"]) if x)
    plain = " ".join(t for _, _, t in u["lines"])
    title = f"{name(uid)}: {u['title']} | EU AI Act"
    body = f"""<article class="law">
<p class="crumbs">{crumbs}</p>
<h1><span class="num">{name(uid)}</span> {esc(u['title'])}</h1>
<p class="meta">Consolidated text of {esc(VERSION)} · <a href="{esc(u['url'])}">EUR-Lex source</a>{badge}</p>
{diff}
{text}
<!--BACKLINKS:{uid}-->
<nav class="pager">{prev}{nxt}</nav>
</article>"""
    ld = {"@context": "https://schema.org", "@type": "Legislation",
          "name": f"{name(uid)} - {u['title']}",
          "legislationIdentifier": f"{ACT}, {name(uid)}",
          "legislationDate": VERSION, "inLanguage": "en",
          "isPartOf": {"@type": "Legislation",
                       "name": f"{ACT} (Artificial Intelligence Act)",
                       "sameAs": config.VERSIONS[VERSION]["eli"]}}
    if base:
        ld["url"] = f"{base}/{slug(uid)}.html"
    return page(title, plain[:155].rsplit(" ", 1)[0] + "…", body,
                f"{slug(uid)}.html", base, ld)


def backlinks(uid: str, refs_to: dict, units: dict) -> str:
    src = [s for s in units if uid in refs_to.get(s, ())]
    if not src:
        return ""
    links = ", ".join(f'<a href="{slug(s)}.html">{name(s)}</a>' for s in src)
    return f'<aside class="back"><h2>Referred to by</h2><p>{links}</p></aside>'


def changes_page(units: dict, original: dict, base: str) -> str:
    def items(want: str) -> str:
        return "".join(
            f'<li><a href="{slug(i)}.html"><span class="num">{name(i)}</span> '
            f'{esc(u["title"])}</a></li>' for i, u in units.items()
            if u["kind"] != "recital" and status(i, units, original) == want)
    body = f"""<section class="hero"><h1>What changed in <em>{VERSION[:4]}</em></h1>
<p>The articles and annexes whose consolidated text of {esc(VERSION)} differs from
the original text of {esc(config.ORIGINAL)}. Each amended page shows the changed
paragraphs word by word. The list is computed by comparing the two texts, so it
is as good as the two source documents and no better.</p></section>
<h2>Added</h2><div class="toc"><section class="chapter"><ul>{items('added')}</ul></section></div>
<h2>Amended</h2><div class="toc"><section class="chapter"><ul>{items('amended')}</ul></section></div>"""
    return page(f"EU AI Act: what the {VERSION[:4]} amendments changed, article by article",
                "Every article and annex of the EU AI Act that the 2026 amendments added "
                "or changed, with a word-by-word comparison against the original text.",
                body, "changes.html", base)


def index_page(units: dict, base: str) -> str:
    groups: dict[str, list[str]] = {}
    for uid, u in units.items():
        if u["kind"] == "article":
            groups.setdefault(u["chapter"], []).append(uid)
    toc = []
    for chapter, ids in groups.items():
        items = "".join(
            f'<li><a href="{slug(i)}.html"><span class="num">{name(i)}</span> '
            f'{esc(units[i]["title"])}</a></li>' for i in ids)
        toc.append(f'<section class="chapter"><h3>{esc(chapter)}</h3><ul>{items}</ul></section>')
    annexes = "".join(
        f'<li><a href="{slug(i)}.html"><span class="num">{name(i)}</span> '
        f'{esc(u["title"])}</a></li>' for i, u in units.items() if u["kind"] == "annex")
    n_art = sum(u["kind"] == "article" for u in units.values())
    n_anx = sum(u["kind"] == "annex" for u in units.values())
    body = f"""<section class="hero">
<h1>The EU AI Act, <em>by the paragraph</em></h1>
<p>All {n_art} articles and {n_anx} annexes of {ACT} as amended, with every
cross-reference linked and every paragraph citable. Text of {esc(VERSION)}.</p>
<p class="actions"><a class="btn primary" href="checker.html">Is my AI system high-risk?</a>
<a class="btn" href="changes.html">What changed in {VERSION[:4]}</a>
<a class="btn" href="{DEMO}">Ask a question</a></p>
<input id="filter" type="search" placeholder="Filter articles, e.g. high-risk, penalties, 50" aria-label="Filter articles">
</section>
<h2 id="articles">Articles</h2>
<div class="toc">{''.join(toc)}</div>
<h2 id="annexes">Annexes</h2>
<div class="toc"><section class="chapter"><ul>{annexes}</ul></section></div>"""
    script = """<script>
const f = document.getElementById('filter');
f.addEventListener('input', () => {
  const q = f.value.trim().toLowerCase();
  document.querySelectorAll('.toc li').forEach(li => {
    li.hidden = !!q && !li.textContent.toLowerCase().includes(q);
  });
  document.querySelectorAll('.chapter').forEach(c => {
    c.hidden = !c.querySelector('li:not([hidden])');
  });
});
</script>"""
    return page("EU AI Act, by the paragraph: every article, linked and citable",
                f"All {n_art} articles and {n_anx} annexes of the EU AI Act as amended "
                f"in 2026, with linked cross-references and a high-risk checker.",
                body, "index.html", base, script=script)


def recitals_page(units: dict, base: str) -> str:
    found: set[str] = set()
    items = "".join(
        f'<p id="{uid.split("_")[1]}">{link_refs(" ".join(t for _, _, t in u["lines"]), units, uid, found)}'
        f'<a class="pin" href="#{uid.split("_")[1]}" aria-label="Link to this recital">#</a></p>'
        for uid, u in units.items() if u["kind"] == "recital")
    body = f"""<article class="law"><h1>Recitals</h1>
<p class="meta">Recitals explain the reasons for the articles. They are not binding.</p>
{items}</article>"""
    return page("Recitals | EU AI Act", "The recitals of the EU AI Act: the non-binding "
                "explanations that precede the articles.", body, "recitals.html", base)


# --------------------------------------------------------------------------
# Risk checker
# --------------------------------------------------------------------------
def points(unit: dict, depth: int, under: str = "") -> list[tuple[str, str]]:
    """(anchor, text) for each outline item at `depth` whose anchor starts with
    `under`, with the lines beneath it folded in."""
    out: list[list[str]] = []
    open_ = False
    for d, anchor, text in unit["lines"]:
        if d == depth and anchor and anchor.startswith(under):
            out.append([anchor, text])
            open_ = True
        elif open_ and (d > depth or (d == depth and not anchor)):
            out[-1][1] += " " + text
        else:
            open_ = False
    return [(a, t) for a, t in out]


def paragraph(unit: dict, anchor: str) -> str:
    """The full text of one numbered paragraph."""
    keep, on = [], False
    for d, a, text in unit["lines"]:
        if d == 0 and a:
            on = a == anchor
        if on:
            keep.append(text)
    assert keep, f"paragraph {anchor} not found"
    return " ".join(keep)


def checker_page(units: dict, base: str) -> str:
    found: set[str] = set()

    def quote(uid: str, anchor: str, must_hold: str) -> str:
        """A provision quoted in a result. `must_hold` is a phrase the quoted
        text has to contain: the result panels are tied to specific points of
        Articles 6 and 113, and if an amendment re-letters them this fails the
        build instead of quoting the wrong rule."""
        text = (paragraph(units[uid], anchor) if "-" not in anchor
                else dict(points(units[uid], 1))[anchor])
        assert must_hold in text, f"{uid} {anchor} no longer says {must_hold!r}"
        ref = f"{name(uid)}({anchor.replace('-', ')(')})"
        return (f'<blockquote>{link_refs(text, units, "", found)}'
                f'<cite><a href="{slug(uid)}.html#{anchor}">{ref}</a></cite></blockquote>')

    def box(group: str, uid: str, anchor: str, text: str) -> str:
        ref = f"{name(uid)}({anchor.replace('-', ')(')})"
        return (f'<label class="opt"><input type="checkbox" data-group="{group}" '
                f'data-ref="{ref}" data-href="{slug(uid)}.html#{anchor}" '
                f'value="{group[0]}{anchor}"><span>{esc(text)} '
                f'<a href="{slug(uid)}.html#{anchor}">{ref}</a></span></label>')

    def see(*ids: str) -> str:
        return "".join(f'<li><a href="{slug(i)}.html">{name(i)}: '
                       f'{esc(units[i]["title"])}</a></li>' for i in ids)

    prohibited = "".join(box("prohibited", "art_5", a, t)
                         for a, t in points(units["art_5"], 1, "1-"))
    areas = []
    for a, text in points(units["anx_III"], 0):
        subs = points(units["anx_III"], 1, a + "-")
        head = text.split(":")[0] if subs else text
        opts = "".join(box("annex3", "anx_III", sa, st) for sa, st in subs) \
            or box("annex3", "anx_III", a, text)
        areas.append(f"<h4>{esc(head)}</h4>{opts}")
    conditions = "".join(box("derogation", "art_6", a, t)
                         for a, t in points(units["art_6"], 1, "3-"))
    transparency = "".join(box("transparency", "art_50", a, paragraph(units["art_50"], a))
                           for a in ("1", "2", "3", "4"))

    body = f"""<article class="law checker">
<h1>Is my AI system prohibited, high-risk, or neither?</h1>
<p class="meta">Tick what applies. Every option is the wording of the Act itself
(text of {esc(VERSION)}), so read each one to the end: most carry exceptions.
The result shows which provisions your answers point to. It is a reading aid
and not legal advice, and <a href="article-2.html">Article 2</a> excludes some
uses from the Act altogether.</p>

<form id="form">
<fieldset><legend>1. Prohibited practices <a href="article-5.html">Article 5(1)</a></legend>
<p class="hint">Does the system do any of the following?</p>{prohibited}</fieldset>

<fieldset><legend>2. Products under EU safety law <a href="article-6.html#1">Article 6(1)</a></legend>
<label class="opt"><input type="checkbox" data-group="annex1" data-ref="Article 6(1)"
data-href="article-6.html#1" value="x1"><span>Both conditions of Article 6(1) are met:
{esc(paragraph(units['art_6'], '1'))}</span></label></fieldset>

<fieldset><legend>3. Listed high-risk uses <a href="annex-iii.html">Annex III</a></legend>
<p class="hint">Is the system intended for any of these uses?</p>{''.join(areas)}</fieldset>

<fieldset><legend>4. The Annex III exception <a href="article-6.html#3">Article 6(3)</a></legend>
<p class="hint">Only matters if you ticked something in step 3.</p>{conditions}
<label class="opt"><input type="checkbox" data-group="profiling" data-ref="Article 6(3)"
data-href="article-6.html#3" value="xp"><span>The AI system performs profiling of natural persons.</span></label></fieldset>

<fieldset><legend>5. Transparency <a href="article-50.html">Article 50</a></legend>
<p class="hint">Do any of these describe you?</p>{transparency}</fieldset>

<fieldset><legend>6. General-purpose AI models <a href="article-3.html#63">Article 3(63)</a></legend>
<label class="opt"><input type="checkbox" data-group="gpai" data-ref="Article 3(63)"
data-href="article-3.html#63" value="xg"><span>I provide a general-purpose AI model:
{esc(paragraph(units['art_3'], '63'))}</span></label></fieldset>
</form>
<div id="bar" aria-live="polite"><b>Result</b><span id="tags"></span><a href="#result">Details ↓</a></div>

<section id="result">
<h2>What your answers point to</h2>

<div class="verdict none" data-show="none"><h3>Nothing matched</h3>
<p>None of the options above is ticked. That is not a finding that the Act
does not apply to you: these questions cover Article 5, Article 6 with
Annexes I and III, Article 50 and general-purpose AI models, and nothing else.</p></div>

<div class="verdict bad" data-show="prohibited" hidden><h3>A prohibited practice</h3>
<p>You ticked: <span data-list="prohibited"></span>. Read the full point,
including its exceptions.</p>
<p>When it applies:</p>{quote('art_113', 'a', 'Chapters I and II')}
<ul>{see('art_5', 'art_99')}</ul></div>

<div class="verdict warn" data-show="highrisk" hidden><h3>A high-risk AI system</h3>
<p>You ticked: <span data-list="annex1 annex3 profiling"></span>.</p>
<p>When the high-risk rules apply:</p>{quote('art_113', 'c', 'Article 6(2) and Annex III')}
<ul>{see('art_6', 'art_8', 'art_16', 'art_26', 'art_27', 'art_43', 'art_49')}</ul></div>

<div class="verdict maybe" data-show="derogation" hidden><h3>Listed in Annex III, but possibly not high-risk</h3>
<p>You ticked <span data-list="annex3"></span> together with
<span data-list="derogation"></span>. Article 6(3) can take such a system out
of the high-risk category, and Article 6(4) then applies:</p>
{quote('art_6', '3', 'shall always be considered to be high-risk')}
{quote('art_6', '4', 'shall document its assessment')}
<ul>{see('art_6', 'art_49')}</ul></div>

<div class="verdict info" data-show="transparency" hidden><h3>Transparency obligations</h3>
<p>You ticked: <span data-list="transparency"></span>. These apply on top of
anything above.</p><ul>{see('art_50', 'art_99')}</ul></div>

<div class="verdict info" data-show="gpai" hidden><h3>A general-purpose AI model</h3>
<p>Chapter V applies to providers of general-purpose AI models.</p>
<p>When it applies:</p>{quote('art_113', 'b', 'Chapter V')}
<ul>{see('art_51', 'art_53', 'art_55')}</ul></div>

<p class="share"><button type="button" id="copy">Copy a link to this result</button>
<button type="reset" form="form">Clear</button></p>
</section>
</article>"""
    return page("EU AI Act risk checker: prohibited, high-risk or neither",
                "Tick what your AI system does and see which provisions of the EU AI Act "
                "your answers point to. Every option is the wording of the Act.",
                body, "checker.html", base, script='<script src="checker.js"></script>')


# --------------------------------------------------------------------------
def build(out: Path, base: str, version: str) -> int:
    global VERSION
    VERSION = version
    units = load_units(version)
    original = load_units(config.ORIGINAL)
    order = [u for u in units if units[u]["kind"] != "recital"]
    refs_to: dict[str, set[str]] = {}
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)

    pages = {f"{slug(u)}.html": unit_page(u, units, order, refs_to, base, original)
             for u in order}
    for u in order:  # backlinks need every page rendered first
        path = f"{slug(u)}.html"
        pages[path] = pages[path].replace(f"<!--BACKLINKS:{u}-->",
                                          backlinks(u, refs_to, units))
    pages["index.html"] = index_page(units, base)
    pages["recitals.html"] = recitals_page(units, base)
    pages["checker.html"] = checker_page(units, base)
    pages["changes.html"] = changes_page(units, original, base)
    # The same text for programs: every unit, every line with its anchor.
    (out / "act.json").write_text(json.dumps({
        "regulation": ACT, "version": version, "source": config.VERSIONS[version]["eli"],
        "units": [{"id": i, "name": name(i), "title": u["title"], "chapter": u["chapter"],
                   "section": u["section"], "status": status(i, units, original),
                   "lines": [{"depth": d, "anchor": a, "text": t} for d, a, t in u["lines"]]}
                  for i, u in units.items()]}, ensure_ascii=False), encoding="utf-8")
    for path, text in pages.items():
        (out / path).write_text(text, encoding="utf-8")
    for f in (HERE / "static").iterdir():
        shutil.copy(f, out / f.name)
    if base:
        urls = "".join(f"<url><loc>{base}/{p}</loc><lastmod>{version}</lastmod></url>"
                       for p in pages)
        (out / "sitemap.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?><urlset xmlns='
            f'"http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>')
        (out / "robots.txt").write_text(f"User-agent: *\nAllow: /\nSitemap: {base}/sitemap.xml\n")
    return len(pages)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(config.ROOT / "_site"))
    p.add_argument("--base-url", default="", help="public URL, for canonical links and the sitemap")
    p.add_argument("--corpus", default=config.LATEST, choices=sorted(config.VERSIONS))
    a = p.parse_args()
    n = build(Path(a.out), a.base_url.rstrip("/"), a.corpus)
    print(f"{n} pages -> {a.out}")
