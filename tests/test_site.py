"""The static site is generated from the corpus; these check the generator."""
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("site_build", ROOT / "site" / "build.py")
site_build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(site_build)


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    out = tmp_path_factory.mktemp("site")
    site_build.build(out, "https://example.org", site_build.config.LATEST)
    return out


def test_every_article_and_annex_has_a_page_and_no_link_dangles(site):
    units = site_build.load_units(site_build.config.LATEST)
    for uid, u in units.items():
        if u["kind"] != "recital":
            assert (site / f"{site_build.slug(uid)}.html").exists(), uid
    for page in site.glob("*.html"):
        for href in re.findall(r'href="([a-z0-9-]+\.html)', page.read_text()):
            assert (site / href).exists(), f"{page.name} links to missing {href}"
    assert "<loc>https://example.org/article-6.html</loc>" in (site / "sitemap.xml").read_text()


def test_references_to_other_laws_are_not_linked_to_this_act(site):
    art5 = (site / "article-5.html").read_text()
    # Article 5(1) cites Article 9 of the GDPR, which is not this Act's Article 9.
    assert "Article 9 of Regulation (EU) 2016/679" in art5
    assert '<a href="article-9.html">Article 9</a> of Regulation (EU) 2016/679' not in art5


def test_checker_offers_every_prohibition_and_keeps_points_apart(site):
    units = site_build.load_units(site_build.config.LATEST)
    html = (site / "checker.html").read_text()
    prohibited = site_build.points(units["art_5"], 1, "1-")
    assert len(prohibited) >= 8
    for anchor, _ in prohibited:
        assert f'value="p{anchor}"' in html
    # The profiling sentence closes Article 6(3); it is not part of point (d).
    conditions = dict(site_build.points(units["art_6"], 1, "3-"))
    assert not any("profiling" in t for t in conditions.values())
    values = re.findall(r'<input type="checkbox"[^>]*value="([^"]+)"', html, re.S)
    assert len(values) == len(set(values)), "two options share a value"


def test_amended_articles_show_the_change_and_layout_noise_does_not(site):
    art113 = (site / "article-113.html").read_text()
    assert 'class="badge amended"' in art113
    assert re.search(r"<ins>[^<]*2 December 2027", art113)
    # Article 51 differs between the two source documents only by a space
    # before a full stop, which is not an amendment.
    assert 'class="badge' not in (site / "article-51.html").read_text()
    assert 'class="badge added"' in (site / "article-4a.html").read_text()
