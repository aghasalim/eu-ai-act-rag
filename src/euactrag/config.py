"""Central configuration. Everything tunable lives here so experiments are reproducible."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
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

# --- Retrieval ------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "6"))
RRF_K = 60  # reciprocal-rank-fusion damping constant (Cormack et al. 2009)
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")  # dense | bm25 | hybrid

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
