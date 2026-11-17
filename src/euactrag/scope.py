"""Who a passage names, and the date it applies from.

Two tags for every chunk, worked out from the text when a search asks, so
nothing is re-ingested or re-indexed:

- `roles`: which of provider, deployer, importer and distributor the passage
  names. "Operator" counts as all four (Article 3(8)).
- `applies_from`: the date Article 113 sets for the passage, read from
  Article 113 in the same version of the corpus, with `applies_rule`, the
  point of Article 113 that sets it.

Only Article 113 is read. The transitional periods in Article 111, for systems
already on the market, are not applied.
"""
from __future__ import annotations

import functools
import re
from datetime import datetime

from . import config
from .config import ROLES
from .ingest import load_chunks

_ROLE = re.compile(r"\b(provider|deployer|importer|distributor|operator)s?\b", re.I)


def roles(chunk: dict) -> list[str]:
    # ponytail: a word match, not a reading of who an obligation falls on.
    # Articles 9 to 15 bind providers through Article 16(a) and name nobody,
    # which is why `keep` lets unnamed passages through. The article's title
    # is used only when the body names nobody: the points of Article 16 are
    # the provider's, and Article 50 has paragraphs for each role.
    named = {m.lower() for m in _ROLE.findall(chunk["body"]) or _ROLE.findall(chunk["title"])}
    return list(ROLES) if "operator" in named else [r for r in ROLES if r in named]


_DATE = re.compile(r"\d{1,2} [A-Z][a-z]+ \d{4}")
_CHAPTER = re.compile(r"Chapters? ([IVX]+(?:(?:, | and )[IVX]+)*)"
                      r"(?:,? Sections? (\d+(?:(?:, |,? and )\d+)*))?")


def _dates(text: str) -> list[str]:
    return sorted(datetime.strptime(d, "%d %B %Y").date().isoformat()
                  for d in _DATE.findall(text))


def _scopes(text: str) -> list:
    """What a clause of Article 113 covers: ("CHAPTER III", "SECTION 4"),
    ("CHAPTER V", "") or a reference such as "art_78" or "art_6(5)"."""
    out: list = []
    for m in _CHAPTER.finditer(text):
        for ch in re.findall(r"[IVX]+", m[1]):
            out += ([(f"CHAPTER {ch}", f"SECTION {s}") for s in re.findall(r"\d+", m[2] or "")]
                    or [(f"CHAPTER {ch}", "")])
    for piece in re.split(r"(?=Articles? \d)", text)[1:]:
        m = re.match(r"Articles? (\d+[a-z]?)(?: to (\d+))?", piece)
        groups = re.findall(r"\((\w+)\)", piece)
        if m[2]:                   # "Articles 102 to 110"
            out += [f"art_{n}" for n in range(int(m[1]), int(m[2]) + 1)]
        elif "points" in piece:    # "Article 5(1), first subparagraph, points (ba) and (bb)"
            out += [f"art_{m[1]}({groups[0]})({p})" for p in groups[1:]]
        else:                      # "Article 78", "Article 6(5)", "Article 5(1a) and (1b)"
            out += [f"art_{m[1]}({g})" for g in groups] or [f"art_{m[1]}"]
    return out


@functools.lru_cache(maxsize=None)
def rules(version: str) -> tuple[dict, tuple[str, str]]:
    """Article 113 as {scope: (date, point)}, and the general (date, point).

    Each point reads "<scopes> [with the exception of <scopes>] shall apply
    from <dates> [with the exception of <scopes> [which shall apply from
    <date>]]". An exception with no date of its own falls back to the general
    one. A point that does not read that way raises, so an amendment that
    rewords Article 113 fails here instead of dating passages wrongly.
    """
    out: dict = {}
    general = None
    for c in load_chunks(version, "paragraph"):
        if c["unit_id"] != "art_113":
            continue
        ref = c["meta"]["ref"]
        head, _, rest = re.sub(r"\s+", " ", c["body"]).partition("shall apply from")
        main, _, exc_head = head.partition("with the exception of")
        dates, _, exc_rest = rest.partition("with the exception of")
        exc, _, exc_date = exc_rest.partition("which shall apply from")
        when, scopes = _dates(dates), _scopes(main)
        if not when or bool(scopes) != bool(general):
            raise ValueError(f"cannot read {ref} of the {version} text: {c['body']!r}")
        if not scopes:
            general = (when[0], ref)
            continue
        # Two dates, as in 113(c) of the 2026 text, means "from the earlier one
        # for some systems": the point is named so the reader can see which.
        out.update({s: (when[0], ref) for s in scopes})
        later = _dates(exc_date)
        out.update({s: (later[0], ref) if later else general
                    for s in _scopes(exc_head) + _scopes(exc)})
    if general is None:
        raise ValueError(f"no Article 113 in the {version} text")
    return out, general


def _when(chunk: dict, ref: str) -> tuple[str, str]:
    by_scope, general = rules(chunk["version"])
    # Most specific first: the point, its paragraph, its article, then the
    # section and the chapter the chunk sits in.
    prefixes = [ref[:i] for i in range(len(ref), 0, -1) if i == len(ref) or ref[i] == "("]
    for key in [*prefixes, (chunk["chapter"], chunk["section"]), (chunk["chapter"], "")]:
        if key in by_scope:
            return by_scope[key]
    return general


def tag(chunk: dict) -> dict:
    """`roles` and, for an article, `applies_from` (an ISO date) and
    `applies_rule`. A chunk holding points with different dates gets the
    earliest. Recitals are not rules and annexes take effect through the
    articles that refer to them, so neither carries a date."""
    out: dict = {"roles": roles(chunk)}
    if chunk["kind"] == "article":
        refs = chunk["meta"].get("refs") or [chunk["unit_id"]]
        out["applies_from"], out["applies_rule"] = min(_when(chunk, r) for r in refs)
    return out


@functools.lru_cache(maxsize=None)
def tags(version: str, granularity: str) -> dict[str, dict]:
    base, lang = config.split_version(version)
    if lang == "en":
        return {c["chunk_id"]: tag(c) for c in load_chunks(version, granularity)}
    # The law is the same in every language, so the tags are worked out on the
    # English text and carried over by reference: art_26(6) is art_26(6).
    # Where English cut a paragraph into parts, the parts' tags are merged.
    ref = lambda c: c["meta"].get("ref", c["unit_id"])  # noqa: E731
    by_ref: dict[str, dict] = {}
    for c in load_chunks(base, granularity):
        t, old = tag(c), by_ref.get(ref(c))
        if old:
            t = {**old, **t, "roles": [r for r in ROLES if r in old["roles"] + t["roles"]]}
            if "applies_from" in old:
                t["applies_from"], t["applies_rule"] = min(
                    (old["applies_from"], old["applies_rule"]),
                    (t["applies_from"], t["applies_rule"]))
        by_ref[ref(c)] = t
    return {c["chunk_id"]: by_ref.get(ref(c), {"roles": []})
            for c in load_chunks(version, granularity)}


def keep(t: dict, role: str | None = None, on: str | None = None) -> bool:
    """False for a passage that names other roles but not `role`, or that
    does not apply yet on the ISO date `on`. Undated passages are kept."""
    if role and t["roles"] and role not in t["roles"]:
        return False
    return not on or t.get("applies_from", "") <= on
