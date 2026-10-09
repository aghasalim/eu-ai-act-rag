"""The `euactrag` command.

    euactrag fetch                 download both texts of the Act
    euactrag ingest                chunk them, by article and by paragraph
    euactrag index                 embed the chunks into the local vector store
    euactrag search "question"     ranked passages with citations
    euactrag ask "question"        a generated answer (needs an LLM key)
"""
from __future__ import annotations

import argparse

from . import __version__, config


def _corpus(p: argparse.ArgumentParser, default=None) -> None:
    p.add_argument("--corpus", choices=sorted(config.VERSIONS), default=default,
                   help="which text of the Act, by date (default: "
                        + (default or "all of them") + ")")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="euactrag", description="Question answering over the EU AI Act.",
        epilog=f"Data directory: {config.DATA} (set EUACTRAG_DATA to move it).")
    p.add_argument("-V", "--version", action="version", version=f"euactrag {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download the text of the Act")
    _corpus(f)
    f.add_argument("--force", action="store_true", help="download again")

    sub.add_parser("ingest", help="chunk every downloaded text")

    i = sub.add_parser("index", help="embed the chunks")
    _corpus(i)
    i.add_argument("--granularity", choices=config.GRANULARITIES)

    for name, helptext in (("search", "ranked passages"), ("ask", "a generated answer")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("query")
        s.add_argument("-k", type=int, default=config.TOP_K)
        s.add_argument("--mode", choices=("hybrid", "dense", "bm25"),
                       default=config.RETRIEVAL_MODE)
        s.add_argument("--granularity", choices=config.GRANULARITIES,
                       default=config.GRANULARITY)
        _corpus(s, config.CORPUS_VERSION)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    versions = [a.corpus] if getattr(a, "corpus", None) else sorted(config.VERSIONS)

    if a.cmd == "fetch":
        from .fetch import fetch
        for v in versions:
            fetch(v, force=a.force)
    elif a.cmd == "ingest":
        from .ingest import main as ingest_main
        ingest_main()
    elif a.cmd == "index":
        from .index import build
        for v in versions:
            for g in [a.granularity] if a.granularity else config.GRANULARITIES:
                build(version=v, granularity=g)
    elif a.cmd == "search":
        from .retrieve import search
        for h in search(a.query, k=a.k, mode=a.mode, version=a.corpus,
                        granularity=a.granularity):
            print(f"{h['rank']:>2}. {h['citation']}  [{h['ref']}]  {h['score']:.4f}")
            print(f"    {h['url']}")
    elif a.cmd == "ask":
        from .pipeline import answer
        res = answer(a.query, k=a.k, mode=a.mode, version=a.corpus,
                     granularity=a.granularity)
        print(res.answer, "\nSources:")
        for h in res.contexts:
            print(f"  {h['rank']}. {h['citation']}  {h['url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
