from unittest.mock import patch

import pytest

from haive.tools import files, git, ripgrep, web
from haive.types import WebResult


@pytest.mark.parametrize("use_rg", [True, False])
def test_ripgrep_and_fallback(sample_repo, use_rg):
    if use_rg and not ripgrep.have_rg():
        pytest.skip("rg not installed")
    roots = {"v": sample_repo}
    hits = ripgrep.search("fetch_user", roots, fixed=True, use_rg=use_rg)
    assert any(h.path == "app/api/client.py" for h in hits)
    assert all(h.repo == "v" for h in hits)
    # secrets are never returned, gitignored files are skipped, dotfiles are included
    assert not ripgrep.search("sk-or-v1", roots, fixed=True, use_rg=use_rg)
    assert not ripgrep.search("print('ignored')", roots, fixed=True, use_rg=use_rg)
    assert ripgrep.search("actions/checkout", roots, fixed=True, use_rg=use_rg)


def test_find_definition(sample_repo):
    hits = ripgrep.find_definition("choose_method", {"v": sample_repo})
    assert any(h.path == "app/valuation/dcf.py" and h.line == 1 for h in hits)
    hits = ripgrep.find_definition("loadUser", {"v": sample_repo})
    assert any(h.path == "web/src/api.ts" for h in hits)


def test_files_read_and_window(sample_repo):
    snip = files.read_range(sample_repo, "app/valuation/dcf.py", 3, 4)
    assert snip.text.splitlines()[0].strip() == 'if company.get("dividends"):'
    w = files.window(sample_repo, "app/valuation/dcf.py", 5, radius=1)
    assert (w.start_line, w.end_line, w.symbol) == (1, 7, "choose_method")
    enc = files.enclosing(sample_repo, "app/api/client.py", 18)
    assert enc[2] == "fetch_user"
    with pytest.raises(files.FileAccessError):
        files.read_range(sample_repo, ".env", 1, 2)
    with pytest.raises(files.FileAccessError):
        files.read_range(sample_repo, "../../etc/passwd", 1, 2)
    assert "    3 | x" in files.numbered("a\nb\nx", 1)


def test_git_helpers(sample_repo):
    assert git.is_repo(sample_repo)
    log = git.log(sample_repo, n=5, numstat=True)
    assert log[0]["subject"] == "Bump web api version"
    assert git.log(sample_repo, pickaxe="retries")[0]["subject"] == "Add retries to the API client"
    assert git.log(sample_repo, grep="RETRIES")[0]["subject"] == "Add retries to the API client"
    hist = git.log(sample_repo, line_range=("15,21", "app/api/client.py"))
    assert [h["subject"] for h in hist] == ["Add retries to the API client", "Initial valuation app"]
    bl = git.blame(sample_repo, "app/api/client.py", 1, 3)
    assert len(bl) == 3 and bl[0].author == "Dev" and bl[0].time > 0
    head = git.head(sample_repo)
    assert git.rev_list(sample_repo, "HEAD") and git.is_ancestor(sample_repo, "HEAD~1", head)
    assert git.changed_since(sample_repo, "HEAD") == []


def test_web_search_providers(monkeypatch):
    class FakeDDGS:
        def text(self, q, max_results=5):
            return [{"href": "https://a", "title": "A", "body": "x"}, {"href": "https://a", "title": "dup"}]

    with patch("ddgs.DDGS", FakeDDGS):
        assert web.search("q", "duckduckgo") == [WebResult(url="https://a", title="A", snippet="x")]
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-x")
    assert web.pick_provider("auto") == "tavily"
    monkeypatch.delenv("TAVILY_API_KEY")
    assert web.pick_provider("auto") == "duckduckgo"

    class Boom:
        def text(self, *a, **k):
            raise OSError("network down")

    with patch("ddgs.DDGS", Boom), pytest.raises(web.WebSearchError):
        web.search("q", "duckduckgo")
