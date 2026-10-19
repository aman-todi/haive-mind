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
    assert set(r.bad_claims) == {f"s0.c{i}" for i in range(8)}
    assert any("not among the retrieved evidence" in e for e in r.errors)
    assert any("symbol 'other_fn'" in e for e in r.errors)


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
            {"type": "llm", "role": "fast", "model": "m", "ok": True, "ms": 9},
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
