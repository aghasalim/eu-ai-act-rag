"""The installable package: public API, command line, data directory.

Runs without the corpus or the ML stack, so it also runs against a freshly
installed wheel in CI.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import euactrag  # noqa: E402
from euactrag import cli  # noqa: E402


def test_public_api():
    assert euactrag.__all__ == ["search", "answer", "__version__"]
    assert euactrag.__version__.count(".") == 2


def test_search_goes_through_the_retriever(monkeypatch):
    from euactrag import retrieve
    seen = {}
    monkeypatch.setattr(retrieve, "search",
                        lambda q, **kw: seen.update(q=q, **kw) or [{"ref": "art_5(1)"}])
    assert euactrag.search("emotion recognition", k=3) == [{"ref": "art_5(1)"}]
    assert seen["q"] == "emotion recognition" and seen["k"] == 3


def test_cli_search_prints_paragraph_citations(monkeypatch, capsys):
    from euactrag import retrieve
    hit = {"rank": 1, "citation": "Article 5(1) - Prohibited AI practices",
           "ref": "art_5(1)", "score": 0.03, "url": "https://example.org/#art_5"}
    monkeypatch.setattr(retrieve, "search", lambda q, **kw: [hit])
    assert cli.main(["search", "is social scoring banned", "-k", "1"]) == 0
    assert "Article 5(1)" in capsys.readouterr().out


def test_cli_version_and_help():
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    with pytest.raises(SystemExit) as e:
        cli.main([])
    assert e.value.code == 2


def test_data_directory_can_be_moved(tmp_path):
    code = "from euactrag import config; print(config.DATA)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={"EUACTRAG_DATA": str(tmp_path), "PYTHONPATH": str(ROOT / "src")},
                         check=True).stdout.strip()
    assert out == str(tmp_path)
