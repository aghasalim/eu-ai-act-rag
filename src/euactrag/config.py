"""Central configuration. Everything tunable lives here so experiments are reproducible."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW_XHTML = DATA / "raw" / "ai_act.xhtml"
CHUNKS = DATA / "processed" / "chunks.jsonl"
INDEX_DIR = DATA / "index"
EVAL_DIR = ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"

# Canonical source: Publications Office Cellar, Regulation (EU) 2024/1689.
CELEX = "32024R1689"
SOURCE_URL = f"http://publications.europa.eu/resource/celex/{CELEX}"
ELI_BASE = "https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng"

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
# actually contains the rule -- they were crowding binding provisions out of the
# top-k. The Regulation itself treats recitals as non-binding interpretive aids,
# so down-weighting them in the ranking is a domain prior, not a tuned constant:
# 0.5 = "half the evidential weight of a binding provision". Set to 1.0 to
# disable, 0.0 to drop recitals from the ranking entirely (both reported in
# RESULTS.md).
RECITAL_WEIGHT = float(os.getenv("RECITAL_WEIGHT", "0.5"))

# --- Generation -----------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # groq|openai|gemini|ollama|extractive
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
# 800 was too small and it cost me four answers. On a reasoning model this cap is
# a *shared* budget: the <think> block is billed against it before any content is
# emitted, so a question that reasons for 800 tokens returns an empty string with
# no error and a ~60s latency. It bit exactly the hardest questions -- all four
# empties were multi-hop -- which is the worst possible bias, because it silently
# scores the model's weakest category as wrong for a harness reason.
# Not fixable with reasoning_effort="none": gpt-oss-20b rejects that with a 400
# (the judge's qwen model accepts it, which is why judge.py can use it).
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2500"))

# Judge model for faithfulness scoring. Kept separate from the answering model so
# the system is never grading its own homework with the identical config.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.1-8b-instant")

ABSTAIN_STRING = "NOT_IN_CORPUS"
