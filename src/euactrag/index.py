"""Build the dense (Chroma) index over the chunked corpus.

The lexical BM25 index is not persisted: at 464 chunks it rebuilds from
chunks.jsonl in a few milliseconds, which is cheaper than serialising it and
avoids shipping a pickle in the image. See `retrieve.load_bm25`.
"""
from __future__ import annotations

import re

from . import config
from .ingest import load_chunks

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lexical tokenizer for BM25. Lowercase alphanumeric runs; we deliberately
    keep digits so that "Article 6" and "Annex III" stay searchable, which is
    exactly where dense retrieval is weakest."""
    return _TOKEN.findall(text.lower())


def get_encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL)


def get_collection(create: bool = False):
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(config.INDEX_DIR),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )
    name = "ai_act"
    if create:
        try:
            client.delete_collection(name)
        except Exception:
            pass
        # Cosine, not the default L2: bge embeddings are meant to be compared by
        # angle. Normalised vectors make the two rank-equivalent anyway, but
        # being explicit avoids depending on that coincidence.
        return client.create_collection(name, metadata={"hnsw:space": "cosine"})
    return client.get_collection(name)


META_KEYS = ("kind", "unit_id", "citation", "title", "chapter_title",
             "section_title", "part", "n_parts", "url", "n_tokens")


def build(include_recitals: bool = True) -> int:
    chunks = load_chunks()
    if not include_recitals:
        chunks = [c for c in chunks if c["kind"] != "recital"]

    encoder = get_encoder()
    texts = [c["text"] for c in chunks]
    print(f"encoding {len(texts)} chunks with {config.EMBED_MODEL} ...")
    vecs = encoder.encode(
        texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True
    )

    col = get_collection(create=True)
    col.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=[v.tolist() for v in vecs],
        documents=texts,
        metadatas=[{k: c[k] for k in META_KEYS} for c in chunks],
    )
    print(f"indexed {len(chunks)} chunks -> {config.INDEX_DIR}")
    return len(chunks)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--no-recitals", action="store_true",
                   help="ablation: index binding articles + annexes only")
    a = p.parse_args()
    build(include_recitals=not a.no_recitals)
