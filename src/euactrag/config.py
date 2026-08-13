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

# --- Generation -----------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # groq|openai|gemini|ollama|extractive
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))

# Judge model for faithfulness scoring. Kept separate from the answering model so
# the system is never grading its own homework with the identical config.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.1-8b-instant")

ABSTAIN_STRING = "NOT_IN_CORPUS"
