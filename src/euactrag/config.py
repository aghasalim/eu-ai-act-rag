"""Central configuration. Everything tunable lives here so experiments are reproducible."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# In a checkout the corpus and index live in ./data. An installed package has no
# checkout around it, so it keeps them in ~/.cache/euactrag unless EUACTRAG_DATA
# says where.
DATA = Path(os.getenv("EUACTRAG_DATA") or (
    ROOT / "data" if (ROOT / "data").is_dir() else Path.home() / ".cache" / "euactrag"))
INDEX_DIR = DATA / "index"
EVAL_DIR = ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"

# --- Corpus versions --------------------------------------------------------
# The Act is amended law now, so the corpus is versioned by the date of the text
# it holds. "2024-07-12" is the original Official Journal text of Regulation (EU)
# 2024/1689. "2026-07-27" is the consolidated text after Regulation (EU) 2026/1744
# (the Digital Omnibus on AI), which entered into force that day. Both are fetched
# from the Publications Office Cellar by CELEX number; the consolidated one has a
# different CELEX and different markup, see ingest.py.
VERSIONS = {
    "2024-07-12": {
        "celex": "32024R1689",
        "raw": DATA / "raw" / "ai_act.xhtml",
        "eli": "https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng",
    },
    "2026-07-27": {
        "celex": "02024R1689-20260727",
        "raw": DATA / "raw" / "ai_act_2026-07-27.xhtml",
        "eli": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:02024R1689-20260727",
    },
}
ORIGINAL = "2024-07-12"
LATEST = max(VERSIONS)
# What the app and search use unless told otherwise. Eval takes --corpus.
CORPUS_VERSION = os.getenv("CORPUS_VERSION", LATEST)


# --- Languages ---------------------------------------------------------------
# The Act is equally authentic in 24 languages. English is the one everything
# here was measured on and the only one committed. The others listed are read
# by the same parser and fetched on demand: `euactrag fetch --lang fr`.
# ponytail: three of the other 23. Adding one is its Publications Office token
# and its word for a recital here, then a run of tests/test_languages.py.
LANGUAGES = {"en": "eng", "fr": "fra", "de": "deu", "nl": "nld"}
RECITAL = {"en": "Recital", "fr": "Considérant", "de": "Erwägungsgrund", "nl": "Overweging"}


def localized(version: str | None, lang: str | None) -> str:
    """The corpus key for a version in a language: "2026-07-27.fr". English
    keeps the bare date, so every key written before languages still holds."""
    version = version or CORPUS_VERSION
    if lang not in (None, "en") and lang not in LANGUAGES:
        raise ValueError(f"language must be one of {', '.join(LANGUAGES)}, not {lang!r}")
    return version if lang in (None, "en") else f"{version}.{lang}"


def split_version(version: str) -> tuple[str, str]:
    base, _, lang = version.partition(".")
    return base, lang or "en"


def source(version: str) -> dict:
    """Where a corpus key comes from: CELEX number, raw file, EUR-Lex link and
    the language token the Publications Office wants."""
    base, lang = split_version(version)
    v = VERSIONS[base]
    if lang == "en":
        return {**v, "lang": lang, "token": LANGUAGES[lang]}
    return {
        "celex": v["celex"], "lang": lang, "token": LANGUAGES[lang],
        "raw": v["raw"].with_suffix(f".{lang}.xhtml"),
        "eli": v["eli"].replace("/oj/eng", f"/oj/{LANGUAGES[lang]}")
                       .replace("/EN/TXT/", f"/{lang.upper()}/TXT/"),
    }


# "article" is the unit the published numbers were measured on. "paragraph",
# the default for search and the app since it measured better on both sets,
# cuts one chunk per numbered paragraph (or top-level point) so that a retrieved
# chunk can be cited as Article 6(3) rather than Article 6. See ingest.py.
GRANULARITIES = ("article", "paragraph")
GRANULARITY = os.getenv("GRANULARITY", "paragraph")


def chunks_path(version: str, granularity: str = "article") -> Path:
    suffix = "" if granularity == "article" else f"_{granularity}"
    return DATA / "processed" / f"chunks_{version}{suffix}.jsonl"


# --- Chunking -------------------------------------------------------------
# bge-small-en-v1.5 has a 512-token window. We keep a margin for the breadcrumb
# header that gets prepended to every chunk.
MAX_CHUNK_TOKENS = 420
MIN_CHUNK_TOKENS = 24  # below this we merge into the previous chunk

# --- Embeddings -----------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# bge models are trained with an asymmetric query instruction; documents get none.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# bge-small-en reads English only, so the other languages get a multilingual
# model of about the same size. The e5 family wants a prefix on both sides.
MULTILINGUAL_EMBED = os.getenv("MULTILINGUAL_EMBED", "intfloat/multilingual-e5-small")


def embedder(version: str, model: str | None = None) -> tuple[str, str, str]:
    """(model, query prefix, passage prefix) for a corpus key."""
    model = model or (EMBED_MODEL if split_version(version)[1] == "en" else MULTILINGUAL_EMBED)
    return (model, "query: ", "passage: ") if "e5" in model.lower() else (model, QUERY_PREFIX, "")


# --- Retrieval ------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "6"))
RRF_K = 60  # reciprocal-rank-fusion damping constant (Cormack et al. 2009)
# "split" is hybrid run on the question and on each of its parts, see
# retrieve.split. It is the default for search and the app because it measured
# better on two-part questions (eval/experiments.py). The published numbers were
# measured on plain "hybrid", and run_eval.py still defaults to that.
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "split")  # dense | bm25 | hybrid | split

# Recitals restate the operative rules in flowing prose, which makes them score
# *higher* against a natural-language question than the terse article that
# actually contains the rule: they were crowding binding provisions out of the
# top-k. The Regulation itself treats recitals as non-binding interpretive aids,
# so down-weighting them in the ranking is a domain prior, not a tuned constant:
# 0.5 = "half the evidential weight of a binding provision". Set to 1.0 to
# disable, 0.0 to drop recitals from the ranking entirely (both reported in
# RESULTS.md).
RECITAL_WEIGHT = float(os.getenv("RECITAL_WEIGHT", "0.5"))

# --- Generation -----------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # groq|openai|gemini|ollama|extractive
# The evaluated model, so that `make eval` with no flags reproduces the numbers
# in the README. The previous default (llama-3.3-70b-versatile) was never the one
# measured, which meant the documented command produced different figures than
# the documentation it was supposed to reproduce: and the hosted demo answered
# with one model while advertising another's scores.
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
# 800 was too small and it cost me four answers. On a reasoning model this cap is
# a *shared* budget: the <think> block is billed against it before any content is
# emitted, so a question that reasons for 800 tokens returns an empty string with
# no error and a ~60s latency. It bit exactly the hardest questions: all four
# empties were multi-hop: which is the worst possible bias, because it silently
# scores the model's weakest category as wrong for a harness reason.
# Not fixable with reasoning_effort="none": gpt-oss-20b rejects that with a 400
# (the judge's qwen model accepts it, which is why judge.py can use it).
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2500"))

# Judge model for faithfulness scoring. Kept separate from the answering model so
# the system is never grading its own homework with the identical config.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen/qwen3.6-27b")

ABSTAIN_STRING = "NOT_IN_CORPUS"

# The roles a search can be filtered by (scope.py). Here, not there, because
# the command line lists them and must start without the parsing stack.
ROLES = ("provider", "deployer", "importer", "distributor")

# --- Commission guidelines ---------------------------------------------------
# The guidelines the Commission has adopted on the Act, as the PDFs it
# publishes: only the Article 50 ones are in the Official Journal, so one
# source format serves all four. Drafts are left out. `paragraphs` is how many
# numbered paragraphs the text has; the parser checks it, so a changed file
# fails instead of being cut wrongly. They are not law and not part of what the
# published numbers measure: search reads them only when asked.
GUIDELINES = {
    "definition": {
        "title": "Guidelines on the definition of an AI system",
        "number": "C(2025) 5053 final", "adopted": "2025-07-29",
        "doc": 112455, "paragraphs": 64,
    },
    "prohibited": {
        "title": "Guidelines on prohibited AI practices",
        "number": "C(2025) 5052 final", "adopted": "2025-07-29",
        "doc": 112367, "paragraphs": 433,
    },
    "gpai": {
        "title": "Guidelines on the obligations for providers of general-purpose AI models",
        "number": "C(2025) 7719 final", "adopted": "2025-11-19",
        "doc": 118340, "paragraphs": 141,
    },
    "transparency": {
        "title": "Guidelines on the transparency obligations in Article 50",
        "number": "C(2026) 7702 final", "adopted": "2026-11-05",
        "doc": 131215, "paragraphs": 155,
    },
}
GUIDELINE_URL = "https://ec.europa.eu/newsroom/dae/redirection/document/{doc}"
GUIDELINES_RAW = DATA / "raw" / "guidelines"
GUIDELINES_CHUNKS = DATA / "processed" / "chunks_guidelines.jsonl"
# The guidelines as a chunk set of their own, next to GRANULARITIES. Kept out
# of that tuple so `index`, `ingest` and the Docker image do not build it; the
# first search that asks for the guidelines does. They are searched apart from
# the Act and added after it, GUIDELINE_K passages at most: mixed into one
# ranking, some 850 chunks of commentary pushed the article they comment on out
# of the candidate pool.
GUIDANCE = "guidance"
GUIDELINE_K = 3
