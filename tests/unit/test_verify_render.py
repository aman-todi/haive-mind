import json

from rich.console import Console

from haive.graph.verify import parse_diff, verify
from haive.render import fmt_duration, render_answer, render_trace, to_json
from haive.types import Answer, Claim, Evidence, Hypothesis, RepoRef, Section, SuggestedFix, WebRef, WebResult


def ev(path="app/valuation/dcf.py"):
    return [Evidence(id="e1", source="code", repo="v", path=path, start_line=1, end_line=7, text="...")]


def ans(*claims, fix=None):
    return Answer(summary="s", sections=[Section(heading="h", claims=list(claims))], suggested_fix=fix)


def ref(**kw):
    base = dict(repo="v", path="app/valuation/dcf.py", start_line=1, end_line=7, symbol="choose_method")
    base.update(kw)
    return RepoRef(**base)


def test_valid_claims(sample_repo):
    a = ans(
        Claim(text="picks method", source="repo", refs=[ref()]),
        Claim(text="per docs", source="web", refs=[WebRef(url="https://x")]),
        Claim(text="general fact", source="model"),
    )
    r = verify(a, ev(), [WebResult(url="https://x")], {"v": sample_repo})
    assert r.ok, r.errors


def test_invalid_claims(sample_repo):
    a = ans(
        Claim(text="no refs", source="repo"),
        Claim(text="missing file", source="repo", refs=[ref(path="nope.py")]),
        Claim(text="bad range", source="repo", refs=[ref(start_line=5, end_line=500)]),
        Claim(text="bad symbol", source="repo", refs=[ref(symbol="other_fn")]),
        Claim(
            text="not in evidence",
            source="repo",
            refs=[ref(path="app/server.py", start_line=1, end_line=3, symbol=None)],
        ),
        Claim(text="bad url", source="web", refs=[WebRef(url="https://evil")]),
        Claim(text="model with ref", source="model", refs=[ref()]),
        Claim(text="unknown repo", source="repo", refs=[ref(repo="zzz")]),
    )
    r = verify(a, ev(), [], {"v": sample_repo})
    assert set(r.bad_claims) == {f"s0.c{i}" for i in (1, 2, 3, 4, 5, 7)}
    assert any("not among the retrieved evidence" in e for e in r.errors)
    assert any("symbol 'other_fn'" in e for e in r.errors)
    # the uncited repo claim and the cited model claim were relabeled instead of sent back
    assert r.relabeled == 2 and a.sections[0].claims[0].source == "model"
    assert a.sections[0].claims[6].source == "repo"


def test_sources_are_relabeled_to_match_refs(sample_repo):
    a = ans(
        Claim(text="cited", source="repo", refs=[ref()]),
        Claim(text="the profile says", source="repo"),
        Claim(text="checks out", source="model", refs=[ref()]),
        Claim(text="does not", source="model", refs=[ref(symbol="other_fn")]),
    )
    r = verify(a, ev(), [], {"v": sample_repo})
    assert r.ok and r.relabeled == 3, r.errors
    assert [(c.source, len(c.refs)) for c in a.all_claims()] == [
        ("repo", 1),
        ("model", 0),
        ("repo", 1),
        ("model", 0),
    ]
    # an answer that cites nothing at all is still sent back for citations
    bare = ans(Claim(text="a", source="repo"), Claim(text="b", source="repo"))
    r = verify(bare, ev(), [], {"v": sample_repo})
    assert len(r.bad_claims) == 2 and r.relabeled == 0


def test_symbol_slack(sample_repo):
    # choose_method is on line 1; citing lines 3-4 is within the ±3 slack
    a = ans(Claim(text="x", source="repo", refs=[ref(start_line=3, end_line=4)]))
    assert verify(a, ev(), [], {"v": sample_repo}).ok


def test_diff_check(sample_repo):
    good = """--- a/app/valuation/dcf.py
+++ b/app/valuation/dcf.py
@@ -4,3 +4,3 @@
         return "ddm"
-    if company["fcf"] > 0:
+    if company.get("fcf", 0) > 0:
         return "fcf"
"""
    bad = good.replace('company["fcf"] > 0', 'company["cash"] > 0')
    hyp = Hypothesis(title="t", confidence="high", claims=[Claim(text="x", source="repo", refs=[ref()])])
    a = Answer(summary="s", hypotheses=[hyp], suggested_fix=SuggestedFix(diff=good))
    assert verify(a, ev(), [], {"v": sample_repo}).approximate_fix is False
    a.suggested_fix = SuggestedFix(diff=bad)
    assert verify(a, ev(), [], {"v": sample_repo}).approximate_fix is True
    assert parse_diff(good)["app/valuation/dcf.py"][1] == '    if company["fcf"] > 0:'


def test_render_and_json():
    a = ans(
        Claim(text="repo claim", source="repo", refs=[ref()]),
        Claim(text="shaky", source="repo", refs=[ref()], unverified=True),
    )
    a.suggested_fix = SuggestedFix(diff="-a\n+b\n", approximate=True)
    con = Console(record=True, width=120)
    render_answer(
        con, a, {"intent": "lookup", "notices": ["web search skipped"], "llm_calls": 3, "scope": ["v"]}
    )
    out = con.export_text()
    assert "[repo] repo claim" in out and "[unverified] shaky" in out
    assert "v/app/valuation/dcf.py:1-7" in out and "approximate" in out
    assert "web search skipped" in out and "LLM calls: 3" in out
    data = json.loads(to_json(a, {"intent": "lookup", "llm_calls": 3}))
    assert data["answer"]["sections"][0]["claims"][0]["source"] == "repo" and data["intent"] == "lookup"
    con2 = Console(record=True, width=120)
    render_trace(
        con2,
        [
            {"type": "node", "node": "router", "ms": 5},
            {"type": "llm", "step": "answer", "model": "m", "ok": True, "ms": 9},
        ],
    )
    assert "router" in con2.export_text()


def test_response_time_is_printed_first():
    console = Console(record=True, width=100)
    render_answer(console, Answer(summary="s"), {"latency_s": 62.4, "llm_calls": 3, "scope": ["a"]})
    lines = [ln for ln in console.export_text().splitlines() if ln.strip()]
    assert lines[0] == "⏱ response time 1m 02s"
    assert lines[-1] == "scope: a · LLM calls: 3"  # not repeated in the footer
    assert fmt_duration(7.25) == "7.2s"


def test_answer_schema_requires_sections_for_models():
    # A paid model filled in only the required `summary`; the schema models see must require the content.
    from langchain_core.utils.function_calling import convert_to_openai_tool

    params = convert_to_openai_tool(Answer)["function"]["parameters"]
    assert set(params["required"]) == {"summary", "sections"}
    assert '"required": ["claims"]' in json.dumps(params)
    assert Answer(summary="s").sections == []  # code may still build an empty answer


def test_symbol_may_be_the_definition_enclosing_the_cited_lines(tmp_path):
    from haive.graph.verify import _Lines, check_repo_ref, symbol_name

    (tmp_path / "m.py").write_text(
        "class Meter:\n"
        "    def __init__(self, limit):\n"
        "        self.limit = limit\n"
        "\n"
        "\n"
        "def verify(\n"
        "    answer, evidence,\n"
        ") -> bool:\n"
        "    keys = list(answer)\n"
        "    for k in keys:\n"
        "        if k:\n"
        "            pass\n"
        "        total = len(evidence)\n"
        "        seen = set()\n"
        "        more = []\n"
        "    return True\n"
    )
    lines, ev = _Lines({"r": tmp_path}), {("r", "m.py")}

    def errs(start, end, symbol):
        return check_repo_ref(
            RepoRef(repo="r", path="m.py", start_line=start, end_line=end, symbol=symbol), lines, ev
        )

    assert not errs(13, 15, "verify")  # deep inside a function with a multi-line signature
    assert not errs(3, 3, "Meter") and not errs(3, 3, "Meter.__init__")  # enclosing class / method
    assert not errs(13, 15, "`verify()`")
    assert "neither in the cited lines nor a definition around them" in errs(13, 15, "check_everything")[0]
    assert not errs(13, 15, "<module>")  # the chunker's label for top-level code
    assert symbol_name("HnswIndex<T>::search") == "search"
    assert symbol_name("BruteForceIndex class") == "BruteForceIndex"


def test_symbol_after_allman_braces_and_on_documented_definitions(tmp_path):
    from haive.graph.verify import _Lines, check_repo_ref

    (tmp_path / "C.cs").write_text(
        "namespace App\n"
        "{\n"
        "    public class UserController\n"
        "    {\n"
        '        [HttpGet("me")]\n'
        "        public IActionResult Me()\n"
        "        {\n"
        "            var name = User.Identity?.Name;\n"
        "            var ok = name != null;\n"
        "            var roles = new List<string>();\n"
        "            var claims = User.Claims;\n"
        "            return Ok(new { ok, name, roles, claims });\n"
        "        }\n"
        "    }\n"
        "}\n"
    )
    (tmp_path / "oauth.ts").write_text(
        "/**\n"
        " * The single redirect target for both sign-in methods.\n"
        " * Magic links land here, and so do OAuth callbacks.\n"
        " * It reads the stored return path.\n"
        " * Falls back to the home page.\n"
        " * Never redirects off-site.\n"
        " */\n"
        "export function postLoginRedirect(): string {\n"
        "  return '/';\n"
        "}\n"
    )
    lines = _Lines({"r": tmp_path})
    ev = {("r", "C.cs"), ("r", "oauth.ts")}

    def errs(path, start, end, symbol):
        return check_repo_ref(
            RepoRef(repo="r", path=path, start_line=start, end_line=end, symbol=symbol), lines, ev
        )

    assert not errs("C.cs", 10, 12, "Me")  # the method above a lone `{`
    assert not errs("C.cs", 10, 12, "UserController")
    assert not errs("oauth.ts", 2, 3, "postLoginRedirect")  # its doc comment
    assert errs("oauth.ts", 2, 3, "completeSignIn")
