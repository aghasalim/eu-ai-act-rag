"""Download the authentic AI Act text.

Source note: eur-lex.europa.eu sits behind a bot challenge that returns an
HTTP 202 interstitial to scripted clients, so scraping it is unreliable. The EU
Publications Office "Cellar" content API serves the same official XHTML rendering
with stable ELI markup (`art_6`, `rct_27`, `anx_III`), which is what the
structure-aware chunker keys on. The language token must be sent as an
Accept-Language header or Cellar refuses the request.
"""
from __future__ import annotations

import httpx

from . import config

HEADERS = {"Accept": "application/xhtml+xml", "Accept-Language": "eng"}


def fetch(force: bool = False) -> int:
    if config.RAW_XHTML.exists() and not force:
        n = config.RAW_XHTML.stat().st_size
        print(f"already present: {config.RAW_XHTML} ({n:,} bytes), use --force to refetch")
        return n
    config.RAW_XHTML.parent.mkdir(parents=True, exist_ok=True)
    r = httpx.get(config.SOURCE_URL, headers=HEADERS, timeout=120, follow_redirects=True)
    r.raise_for_status()
    if b"eli-subdivision" not in r.content:
        raise RuntimeError(
            "downloaded document has no ELI markup, the source format changed; "
            "the chunker keys on `eli-subdivision` ids."
        )
    config.RAW_XHTML.write_bytes(r.content)
    print(f"-> {config.RAW_XHTML} ({len(r.content):,} bytes)")
    return len(r.content)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    fetch(force=p.parse_args().force)
