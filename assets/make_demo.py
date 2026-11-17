"""Write assets/demo.svg: the terminal animation at the top of the README.

The lines in it are what `euactrag search` prints for that question today, so
re-run this after a change to retrieval:

    python assets/make_demo.py
"""
from __future__ import annotations

import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.euactrag import retrieve  # noqa: E402

QUESTION = "A deployer publishes a deep fake without disclosing it. What fine can it face?"
K, DUR, LINE, CHAR = 4, 14.0, 24, 7.83  # seconds per loop, px per line, px per char


def fade(at: float) -> str:
    """Invisible until `at` seconds, then visible until the loop restarts."""
    a, b = at / DUR, (at + 0.3) / DUR
    return (f'<animate attributeName="opacity" dur="{DUR}s" repeatCount="indefinite" '
            f'keyTimes="0;{a:.3f};{b:.3f};0.96;1" values="0;0;1;1;0"/>')


def main() -> None:
    hits = retrieve.search(QUESTION, k=K)
    parts = retrieve.split_question(QUESTION)
    cmd = f'euactrag search "{QUESTION}" -k {K}'
    rows = [(f'{h["rank"]:>2}. ', h["citation"][:78], f'  [{h["ref"]}]') for h in hits]
    height = 86 + LINE * (len(rows) + 3)
    typed = len(cmd) * CHAR

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 880 {height}" width="880" '
           f'role="img" aria-label="Terminal: euactrag search on a two-part question '
           f'returns the disclosure rule and the penalty article">',
           '<style>text{font:13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;'
           'white-space:pre}.d{fill:#9aa1b2}.g{fill:#f5b700}.b{fill:#8fb0ff}</style>',
           f'<rect width="880" height="{height}" rx="12" fill="#0f1218"/>',
           '<circle cx="24" cy="22" r="6" fill="#ff5f57"/><circle cx="44" cy="22" r="6" '
           'fill="#febc2e"/><circle cx="64" cy="22" r="6" fill="#28c840"/>',
           f'<clipPath id="type"><rect x="40" y="48" height="24" width="0"><animate '
           f'attributeName="width" dur="{DUR}s" repeatCount="indefinite" '
           f'keyTimes="0;0.04;0.26;0.96;1" values="0;0;{typed:.0f};{typed:.0f};0"/></rect></clipPath>',
           '<text x="24" y="66" class="g">$</text>',
           f'<text x="40" y="66" fill="#e8e6df" clip-path="url(#type)">{escape(cmd)}</text>']
    y = 66 + LINE + 8
    out.append(f'<text x="24" y="{y}" class="d" opacity="0"># searched as two parts: '
               f'{escape(" | ".join(parts))}{fade(4.0)}</text>')
    for i, (rank, citation, ref) in enumerate(rows):
        y += LINE
        out.append(f'<text x="24" y="{y}" fill="#e8e6df" opacity="0">{rank}'
                   f'<tspan class="b">{escape(citation)}</tspan>'
                   f'<tspan class="d">{escape(ref)}</tspan>{fade(4.7 + 0.5 * i)}</text>')
    out.append("</svg>\n")
    path = Path(__file__).parent / "demo.svg"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
