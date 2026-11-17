"""The `euactrag` command.

    euactrag fetch                 download both texts of the Act
    euactrag ingest                chunk them, by article and by paragraph
    euactrag index                 embed the chunks into the local vector store
    euactrag search "question"     ranked passages with citations
                                   (--role deployer, --on 2026-12-01 to filter)
    euactrag ask "question"        a generated answer (needs an LLM key)
    euactrag mcp                   serve the Act to an AI assistant over MCP (stdio)
"""
from __future__ import annotations

import argparse
from datetime import date

from . import __version__, config
from .scope import ROLES


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

    sub.add_parser("mcp", help="serve search and the text over MCP on stdin/stdout")

    for name, helptext in (("search", "ranked passages"), ("ask", "a generated answer")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("query")
        s.add_argument("-k", type=int, default=config.TOP_K)
        s.add_argument("--mode", choices=("split", "hybrid", "dense", "bm25"),
                       default=config.RETRIEVAL_MODE)
        s.add_argument("--granularity", choices=config.GRANULARITIES,
                       default=config.GRANULARITY)
        _corpus(s, config.CORPUS_VERSION)
        s.add_argument("--role", choices=ROLES,
                       help="leave out passages that name other roles but not this one")
        s.add_argument("--on", metavar="YYYY-MM-DD", type=date.fromisoformat,
                       help="leave out articles that do not apply yet on this date")
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
        filters = {"role": a.role, "on": a.on and a.on.isoformat()}
        for h in search(a.query, k=a.k, mode=a.mode, version=a.corpus,
                        granularity=a.granularity, **filters):
            print(f"{h['rank']:>2}. {h['citation']}  [{h['ref']}]  {h['score']:.4f}")
            if h.get("applies_from"):
                print(f"    applies from {h['applies_from']}  [{h['applies_rule']}]")
            print(f"    {h['url']}")
    elif a.cmd == "mcp":
        try:
            from .mcp_server import main as mcp_main
        except ModuleNotFoundError as e:
            if e.name != "mcp":
                raise
            raise SystemExit('The MCP server needs one more package: '
                             'pip install "euactrag[mcp]"') from None
        mcp_main()
    elif a.cmd == "ask":
        from .pipeline import answer
        res = answer(a.query, k=a.k, mode=a.mode, version=a.corpus,
                     granularity=a.granularity, role=a.role,
                     on=a.on and a.on.isoformat())
        print(res.answer, "\nSources:")
        for h in res.contexts:
            print(f"  {h['rank']}. {h['citation']}  {h['url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
