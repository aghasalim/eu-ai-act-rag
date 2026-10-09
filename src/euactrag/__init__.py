"""Question answering over the EU AI Act, cited to the paragraph.

    import euactrag
    for h in euactrag.search("Is emotion recognition at work prohibited?", k=3):
        print(h["citation"], h["ref"])    # e.g. "Article 50(3) - ...", "art_50(3)"

Build the corpus and index once first: `euactrag fetch && euactrag ingest &&
euactrag index`. Heavy dependencies are imported on first use, not on import.
"""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["search", "answer", "__version__"]


def search(query: str, k: int | None = None, mode: str | None = None,
           version: str | None = None, granularity: str | None = None) -> list[dict]:
    """Ranked passages for a question. Each hit has `citation`, `ref` (e.g.
    art_5(1)), `unit_id`, `text`, `score` and `url`."""
    from .retrieve import search as _search

    return _search(query, k=k, mode=mode, version=version, granularity=granularity)


def answer(question: str, **kwargs):
    """Retrieve, then generate a cited answer. Without an LLM key configured the
    answer says so and only the retrieved passages are returned. Keyword
    arguments go to `euactrag.pipeline.answer`."""
    from .pipeline import answer as _answer

    return _answer(question, **kwargs)
