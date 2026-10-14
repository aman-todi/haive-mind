from pathlib import Path

import pytest

from haive.config import RepoConfig
from haive.graph.router import code_identifiers, heuristic_intent
from haive.types import WebResult
from tests.fakes import api_error
from tests.graph.helpers import DCF_CLAIM, answer, grade, make_runner, nodes, repo_claim, router

TRACES = Path(__file__).parent.parent / "fixtures" / "traces"
Q = "Where does the DCF choose a valuation method?"


def test_lookup_happy_path(cfg, db, store):
    r, fake = make_runner(
        cfg,
        db,
        store,
        {
            "RouterOutput": [router("lookup", identifiers=["choose_method"])],
            "GradeOutput": [grade()],
            "Answer": [answer(DCF_CLAIM)],
        },
    )
    res = r.run(Q, thread_id="t")
    assert nodes(res) == ["freshness", "router", "retrieve", "grade", "answer", "verify", "finalize"]
    assert res.state["llm_calls"] == 3 and fake.calls == ["RouterOutput", "GradeOutput", "Answer"]
    assert not res.answer.sections[0].claims[0].unverified
    assert res.meta()["first_verify"] == {"repo_claims": 1, "repo_claims_ok": 1}
    ev_paths = {e["path"] for e in res.state["evidence"]}
    assert "app/valuation/dcf.py" in ev_paths
    assert ".env" not in ev_paths


def test_verify_retry_then_success(cfg, db, store):
    bad = repo_claim("app/valuation/dcf.py", 900, 950, text="bad lines")
    r, fake = make_runner(
        cfg,
        db,
        store,
        {
            "RouterOutput": [router(identifiers=["choose_method"])],
            "GradeOutput": [grade()],
            "Answer": [answer(bad), answer(DCF_CLAIM)],
        },
    )
    res = r.run(Q, thread_id="t")
    assert nodes(res).count("answer") == 2 and res.state["answer_retries"] == 1
    assert "line range invalid" in fake.messages[-1][-2].content or "line range invalid" in str(
        fake.messages[-1]
    )
    assert not any(c.unverified for c in res.answer.all_claims())
    assert res.meta()["first_verify"] == {"repo_claims": 1, "repo_claims_ok": 0}


def test_retries_exhausted_marks_unverified(cfg, db, store):
    bad = repo_claim("app/valuation/dcf.py", 900, 950, text="bad lines")
    r, _ = make_runner(
        cfg,
        db,
        store,
        {
            "RouterOutput": [router(identifiers=["choose_method"])],
            "GradeOutput": [grade()],
            "Answer": [answer(bad)],
        },
    )
    res = r.run(Q, thread_id="t")
    assert nodes(res).count("answer") == 3  # first try + max_answer_retries (2)
    assert res.answer.sections[0].claims[0].unverified
    assert any("could not be verified" in n for n in res.state["notices"])


def test_grade_missing_symbol_expands(cfg, db, store):
    r, _ = make_runner(
        cfg,
        db,
        store,
        {
            "RouterOutput": [router(identifiers=["create_valuation"])],
            "GradeOutput": [grade("missing_symbol", ["choose_method"]), grade()],
            "Answer": [answer(DCF_CLAIM)],
        },
    )
    res = r.run("How does create_valuation work?", thread_id="t")
    assert nodes(res)[:6] == ["freshness", "router", "retrieve", "grade", "expand", "grade"]
    assert any("definition of choose_method" in e["why"] for e in res.state["evidence"])
    assert res.state["loops"] == 1


def test_grade_loop_limit(cfg, db, store):
    r, _ = make_runner(
        cfg,
        db,
        store,
        {
            "RouterOutput": [router(identifiers=["choose_method"])],
            "GradeOutput": [grade("need_context", ["app/server.py:10"])],
            "Answer": [answer(DCF_CLAIM)],
        },
    )
    res = r.run(Q, thread_id="t")
    assert nodes(res).count("expand") == 2 and "exhausted" in nodes(res)
    assert res.state["low_confidence"]
    assert any(
        e["path"] == "app/server.py" and e["why"] == "read app/server.py" for e in res.state["evidence"]
    )


def test_nothing_relevant_rewrites_then_general(cfg, db, store):
    general = {
        "summary": "general",
        "sections": [{"claims": [{"text": "use limit/offset", "source": "model"}]}],
    }
    r, fake = make_runner(
        cfg,
        db,
        store,
        {
            "RouterOutput": [router("lookup")],
            "GradeOutput": [grade("nothing_relevant")],
            "RewriteOutput": [{"query": "pagination cursor", "identifiers": []}],
            "Answer": [general],
        },
    )
    res = r.run("how does pagination work", thread_id="t", web_mode="off")
    ns = nodes(res)
    assert ns[:7] == ["freshness", "router", "retrieve", "grade", "rewrite", "retrieve", "grade"]
    assert "to_general" in ns and "maybe_web" in ns
    assert res.state["intent"] == "general"
    assert any("Web search is off" in n for n in res.state["notices"])
    assert any("Nothing relevant" in n for n in res.state["notices"])


def test_need_external_web_auto(cfg, db, store):
    web = [WebResult(url="https://fastapi.tiangolo.com/x", title="FastAPI", snippet="path params")]
    ans = answer(
        DCF_CLAIM,
        {
            "text": "FastAPI parses path params",
            "source": "web",
            "refs": [{"url": "https://fastapi.tiangolo.com/x"}],
        },
    )
    r, _ = make_runner(
        cfg,
        db,
        store,
        {
            "RouterOutput": [router(identifiers=["choose_method"])],
            "GradeOutput": [grade("need_external", ["fastapi path parameters"]), grade()],
            "Answer": [ans],
        },
        web=web,
    )
    res = r.run(Q, thread_id="t", web_mode="auto")
    assert ["grade", "web_grade", "maybe_web", "grade"] == nodes(res)[3:7]
    assert r.deps.extra["web_calls"] == ["fastapi path parameters"]
    assert not any(c.unverified for c in res.answer.all_claims())


def test_web_url_not_in_results_is_unverified(cfg, db, store):
    ans = answer({"text": "made up", "source": "web", "refs": [{"url": "https://invented.example"}]})
    r, _ = make_runner(
        cfg,
        db,
        store,
        {
            "RouterOutput": [router(identifiers=["choose_method"])],
            "GradeOutput": [grade()],
            "Answer": [ans],
        },
    )
    res = r.run(Q, thread_id="t")
    assert res.answer.all_claims()[0].unverified


@pytest.mark.parametrize("approve", [True, False])
def test_web_ask_interrupt(cfg, db, store, approve):
    web = [WebResult(url="https://docs.example/a", title="A")]
    r, _ = make_runner(
        cfg,
        db,
        store,
        {
            "RouterOutput": [router("general")],
            "Answer": [{"summary": "s", "sections": []}],
        },
        web=web,
    )
    asked = []
    res = r.run(
        "how do I paginate in fastapi?",
        thread_id="t",
        web_mode="ask",
        on_interrupt=lambda p: asked.append(p) or approve,
    )
    assert asked and asked[0]["kind"] == "web"
    assert bool(res.state.get("web_results")) is approve
    if not approve:
        assert any("declined" in n for n in res.state["notices"])


def test_non_interactive_defaults(cfg, db, store):
    r, _ = make_runner(
        cfg,
        db,
        store,
        {
            "RouterOutput": [router("general")],
            "Answer": [{"summary": "s", "sections": []}],
        },
    )
    res = r.run("how do I paginate?", thread_id="t", web_mode="ask", interactive=False)
    assert not r.deps.extra["web_calls"]
    assert any("Non-interactive: web search declined" in n for n in res.state["notices"])


def test_clarify_interrupt(cfg, db, store, other_repo):
    cfg.repos["frontend"] = RepoConfig(path=str(other_repo))
    r, fake = make_runner(
        cfg,
        db,
        store,
        {
            "RouterOutput": [router("lookup", repos=["zzqx-unknown"], identifiers=["requestValuation"])],
            "GradeOutput": [grade()],
            "Answer": [{"summary": "s", "sections": []}],
        },
    )
    asked = []
    res = r.run(
        "how does zzqx-unknown request valuations?",
        thread_id="t",
        on_interrupt=lambda p: asked.append(p) or "frontend",
    )
    assert asked[0]["kind"] == "clarify" and "frontend" in asked[0]["options"]
    assert res.state["scope_repos"] == ["frontend"]
    assert fake.calls.count("RouterOutput") == 1  # router output is reused after clarification
    assert nodes(res)[:4] == ["freshness", "router", "clarify", "router"]


def test_requested_scope_wins(cfg, db, store, other_repo):
    cfg.repos["frontend"] = RepoConfig(path=str(other_repo))
    r, _ = make_runner(
        cfg,
        db,
        store,
        {
            "RouterOutput": [router("lookup", identifiers=["choose_method"])],
            "GradeOutput": [grade()],
            "Answer": [answer(DCF_CLAIM)],
        },
    )
    res = r.run(Q, thread_id="t", scope=["Valuation"])
    assert res.state["scope_repos"] == ["valuation"]
    assert all(e.get("repo") in (None, "valuation") for e in res.state["evidence"])


def test_budget_guard(cfg, db, store):
    cfg.limits.max_llm_calls_per_question = 2
    bad = repo_claim("app/valuation/dcf.py", 900, 950)
    r, fake = make_runner(
        cfg,
        db,
        store,
        {
            "RouterOutput": [router(identifiers=["choose_method"])],
            "GradeOutput": [grade()],
            "Answer": [answer(bad)],
        },
    )
    res = r.run(Q, thread_id="t")
    assert fake.calls == ["RouterOutput", "Answer"]  # grade skipped to keep one call for the answer
    assert res.state["llm_calls"] == 2
    assert nodes(res).count("answer") == 1  # no budget left to retry
    assert res.answer.all_claims()[0].unverified


def test_budget_zero_gives_evidence_only_answer(cfg, db, store):
    cfg.limits.max_llm_calls_per_question = 1
    r, fake = make_runner(cfg, db, store, {"RouterOutput": [router(identifiers=["choose_method"])]})
    res = r.run(Q, thread_id="t")
    assert fake.calls == ["RouterOutput"]
    assert "No model answer" in res.answer.summary
    assert res.answer.sections and not any(c.unverified for c in res.answer.all_claims())


def test_router_failure_uses_heuristics(cfg, db, store):
    r, _ = make_runner(
        cfg,
        db,
        store,
        {
            "RouterOutput": [api_error(500)],
            "GradeOutput": [grade()],
            "Answer": [answer(DCF_CLAIM)],
        },
    )
    res = r.run("when did I add retries to the API client?", thread_id="t")
    assert res.state["intent"] == "history"
    assert any("keyword routing" in n for n in res.state["notices"])


def test_answer_model_failure_falls_back(cfg, db, store):
    r, _ = make_runner(
        cfg,
        db,
        store,
        {
            "RouterOutput": [router(identifiers=["choose_method"])],
            "GradeOutput": [grade()],
            "Answer": ["not json"],
        },
    )
    res = r.run(Q, thread_id="t")
    assert "No model answer" in res.answer.summary
    assert any(c.refs for c in res.answer.all_claims())


def test_history_uses_git(cfg, db, store):
    r, _ = make_runner(
        cfg,
        db,
        store,
        {
            "RouterOutput": [router("history", identifiers=["retries"])],
            "GradeOutput": [grade()],
            "Answer": [{"summary": "s", "sections": []}],
        },
    )
    res = r.run("When and why did I add retries to the API client?", thread_id="t")
    whys = [e["why"] for e in res.state["evidence"]]
    assert any("git log -Sretries" in w for w in whys)
    texts = " ".join(e["text"] for e in res.state["evidence"] if e["source"] == "commit")
    assert "Add retries to the API client" in texts and "flaky" in texts


def test_recall_groups_by_repo(cfg, db, store, other_repo):
    cfg.repos["frontend"] = RepoConfig(path=str(other_repo))
    r, _ = make_runner(
        cfg,
        db,
        store,
        {
            "RouterOutput": [router("recall")],
            "GradeOutput": [grade()],
            "Answer": [{"summary": "s", "sections": []}],
        },
    )
    res = r.run("how did I call the valuations API before?", thread_id="t")
    by_repo = {}
    for e in res.state["evidence"]:
        if e["why"].startswith("best in"):
            by_repo[e["repo"]] = by_repo.get(e["repo"], 0) + 1
    assert by_repo and all(n <= 2 for n in by_repo.values())


def test_debug_python_trace(cfg, db, store):
    diff = (
        "--- a/app/valuation/dcf.py\n+++ b/app/valuation/dcf.py\n@@ -4,3 +4,3 @@\n"
        '         return "ddm"\n-    if company["fcf"] > 0:\n+    if company.get("fcf", 0) > 0:\n'
        '         return "fcf"\n'
    )
    diag = {
        "summary": "company dict lacks fcf",
        "sections": [
            {
                "heading": "How to verify",
                "claims": [{"text": "POST a company without fcf", "source": "model"}],
            }
        ],
        "hypotheses": [
            {
                "title": "missing key",
                "confidence": "high",
                "claims": [repo_claim("app/valuation/dcf.py", 5, 5, text="indexes company['fcf'] directly")],
            }
        ],
        "suggested_fix": {"diff": diff, "approximate": False},
    }
    r, fake = make_runner(cfg, db, store, {"GradeOutput": [grade()], "Answer": [diag]})
    trace = (TRACES / "python_chained.txt").read_text()
    res = r.run("", trace_text=trace, thread_id="t", web_mode="off")
    assert res.state["intent"] == "debug" and "RouterOutput" not in fake.calls
    assert nodes(res)[:4] == ["freshness", "router", "parse_trace", "gather_debug"]
    pt = res.state["parsed_trace"]
    assert any(f["repo_path"] == "app/valuation/dcf.py" for f in pt["frames"])
    frame_ev = [e for e in res.state["evidence"] if e["why"].startswith("frame #")]
    assert any(e["path"] == "app/valuation/dcf.py" for e in frame_ev)
    assert any(e["path"] == "app/server.py" for e in frame_ev)
    assert res.answer.suggested_fix.approximate is False
    assert res.answer.hypotheses[0].confidence == "high"
    assert not any(c.unverified for c in res.answer.all_claims())
    assert "Parsed error" in fake.messages[-1][0].content


def test_debug_external_failure_searches_web(cfg, db, store):
    trace = (
        'Traceback (most recent call last):\n  File "/srv/app/server.py", line 15, in create_valuation\n    x\n'
        '  File "/venv/lib/python3.12/site-packages/requests/api.py", line 59, in get\n    y\n'
        "requests.exceptions.ConnectionError: Max retries exceeded with url: /users/1\n"
    )
    r, _ = make_runner(
        cfg, db, store, {"GradeOutput": [grade()], "Answer": [{"summary": "s", "sections": []}]}
    )
    res = r.run("", trace_text=trace, thread_id="t", web_mode="auto")
    q = r.deps.extra["web_calls"][0]
    assert q.startswith("requests requests.exceptions.ConnectionError") and "Max retries exceeded" in q
    assert "web_grade" in nodes(res)


def test_compare_fanout(cfg, db, store, other_repo):
    cfg.repos["frontend"] = RepoConfig(path=str(other_repo))
    r, _ = make_runner(
        cfg,
        db,
        store,
        {
            "RouterOutput": [router("compare", compare_targets=["valuation", "frontend"])],
            "GradeOutput": [grade()],
            "Answer": [{"summary": "s", "sections": []}],
        },
    )
    res = r.run("How do valuation and frontend differ on deploy?", thread_id="t")
    assert nodes(res).count("compare_sub") == 2 and "merge" in nodes(res)
    tags = {e["why"].split("]")[0] + "]" for e in res.state["evidence"]}
    assert {"[valuation]", "[frontend]"} <= tags


def test_compare_single_repo_becomes_lookup(cfg, db, store):
    r, _ = make_runner(
        cfg,
        db,
        store,
        {
            "RouterOutput": [router("compare")],
            "GradeOutput": [grade()],
            "Answer": [{"summary": "s", "sections": []}],
        },
    )
    res = r.run("compare things", thread_id="t")
    assert res.state["intent"] == "lookup" and "retrieve" in nodes(res)


def test_architecture_group(cfg, db, store, other_repo):
    cfg.repos["frontend"] = RepoConfig(path=str(other_repo))
    cfg.groups["valuation-project"] = ["valuation", "frontend"]
    summary = {"repo": "x", "purpose": "p", "components": ["c"], "interfaces": [], "deploy": ""}
    r, fake = make_runner(
        cfg,
        db,
        store,
        {
            "RouterOutput": [router("architecture", repos=["valuation-project"])],
            "RepoSummary": [summary],
            "GradeOutput": [grade()],
            "Answer": [{"summary": "s", "sections": []}],
        },
    )
    res = r.run("Break down the valuation project", thread_id="t")
    assert res.state["scope_repos"] == ["valuation", "frontend"]
    assert fake.calls.count("RepoSummary") == 2 and nodes(res).count("summarize_repo") == 2
    conn = next(e for e in res.state["evidence"] if e["id"] == "connections")
    assert (
        "frontend/src/valuations.js:4 calls POST /valuations defined at valuation/app/server.py:13"
        in conn["text"]
    )
    assert "compose service api" in conn["text"]
    assert any(e["path"] == "src/valuations.js" for e in res.state["evidence"] if e.get("path"))


def test_architecture_single_repo_skips_summaries(cfg, db, store):
    r, fake = make_runner(
        cfg,
        db,
        store,
        {
            "RouterOutput": [router("architecture")],
            "GradeOutput": [grade()],
            "Answer": [{"summary": "s", "sections": []}],
        },
    )
    res = r.run("Break down the valuation app", thread_id="t")
    assert "RepoSummary" not in fake.calls and "connect_map" in nodes(res)
    assert any(e["id"] == "profile:valuation" for e in res.state["evidence"])


def test_freshness_reindexes_inline(cfg, db, store, sample_repo):
    r, _ = make_runner(
        cfg,
        db,
        store,
        {
            "RouterOutput": [router(identifiers=["brand_new_function"])],
            "GradeOutput": [grade()],
            "Answer": [{"summary": "s", "sections": []}],
        },
    )
    (sample_repo / "app" / "fresh.py").write_text("def brand_new_function():\n    return 42\n")
    res = r.run("where is brand_new_function?", thread_id="t")
    assert any("Re-indexed 1 changed file" in n for n in res.state["notices"])
    assert any(e.get("path") == "app/fresh.py" for e in res.state["evidence"])


def test_unindexed_repo_notice(cfg, db, store):
    r, _ = make_runner(
        cfg,
        db,
        store,
        {
            "RouterOutput": [router()],
            "GradeOutput": [grade()],
            "Answer": [{"summary": "s", "sections": []}],
        },
        index=False,
    )
    res = r.run("anything", thread_id="t")
    assert any("never been indexed" in n for n in res.state["notices"])


def test_ablations(cfg, db, store):
    for ablation in ("vector_only", "ripgrep_only", "hybrid"):
        r, fake = make_runner(
            cfg,
            db,
            store,
            {
                "RouterOutput": [router(identifiers=["choose_method"])],
                "Answer": [answer(DCF_CLAIM)],
                "GradeOutput": [grade()],
            },
            ablation=ablation,
            index=ablation == "vector_only",
        )
        res = r.run(Q, thread_id=f"t-{ablation}")
        whys = [e["why"] for e in res.state["evidence"]]
        if ablation == "vector_only":
            assert all(w.startswith("vector") for w in whys)
        if ablation == "ripgrep_only":
            assert not any(w.startswith("vector") for w in whys)
        assert "GradeOutput" not in fake.calls  # loop disabled for all three


def test_heuristics_and_identifiers():
    assert heuristic_intent("when did I add retries?") == "history"
    assert heuristic_intent("how do weatherpruf and valuation differ on deploy") == "compare"
    assert heuristic_intent("break down the valuation project") == "architecture"
    assert heuristic_intent("how did I implement OAuth before?") == "recall"
    assert heuristic_intent("where is X defined") == "lookup"
    assert code_identifiers("why does `fetch_user` call ApiClient.fetch_user with API_URL and loadUser?") == [
        "fetch_user",
        "ApiClient.fetch_user",
        "API_URL",
        "loadUser",
    ]


def test_expand_with_model_planned_tool_calls(cfg, db, store):
    r, fake = make_runner(cfg, db, store, {
        "RouterOutput": [router(identifiers=["create_valuation"])],
        "GradeOutput": [grade("missing_symbol", ["choose_method"]), grade()],
        "tools": [[{"name": "find_definition", "args": {"symbol": "choose_method"}},
                   {"name": "recent_commits", "args": {"path": "app/api/client.py"}}]],
        "Answer": [answer(DCF_CLAIM)],
    }, supports_tools=True)  # fmt: skip
    res = r.run("How does create_valuation work?", thread_id="t")
    assert "tools" in fake.calls  # one planning call by the fast model
    expand = next(e for e in res.state["node_log"] if e.get("node") == "expand")
    assert expand["detail"].startswith("model plan: ['find_definition', 'recent_commits']")
    whys = [e["why"] for e in res.state["evidence"]]
    assert "definition of choose_method" in whys
    assert any(w == "recent commit touching app/api/client.py" for w in whys)
    tool_events = [e for e in res.state["node_log"] if e["type"] == "tool" and e["name"] == "find_definition"]
    assert tool_events  # ToolNode runs show up in --trace
    assert not any(c.unverified for c in res.answer.all_claims())


def test_expand_uses_heuristics_when_budget_is_tight(cfg, db, store):
    cfg.limits.max_llm_calls_per_question = 4  # router + grade leave 2: not enough for plan + grade + answer
    r, fake = make_runner(cfg, db, store, {
        "RouterOutput": [router(identifiers=["create_valuation"])],
        "GradeOutput": [grade("missing_symbol", ["choose_method"]), grade()],
        "tools": [AssertionError("the planner must not be called")], "Answer": [answer(DCF_CLAIM)],
    }, supports_tools=True)  # fmt: skip
    res = r.run("How does create_valuation work?", thread_id="t")
    assert "tools" not in fake.calls
    expand = next(e for e in res.state["node_log"] if e.get("node") == "expand")
    assert expand["detail"].startswith("heuristic plan")


def test_chat_history_reaches_router_as_messages(cfg, db, store):
    r, fake = make_runner(cfg, db, store, {
        "RouterOutput": [router(identifiers=["choose_method"])], "GradeOutput": [grade()],
        "Answer": [answer(DCF_CLAIM)],
    })  # fmt: skip
    r.run(Q, thread_id="chat")
    r.run("and where is discount?", thread_id="chat")
    router_calls = [msgs for msgs, name in zip(fake.seen, fake.calls, strict=True) if name == "RouterOutput"]
    second = router_calls[1]
    kinds = [m.type for m in second]
    assert kinds[:4] == ["system", "human", "ai", "human"]  # system, history (Q + answer), new question
    assert second[1].content == Q and "choose_method" in second[2].content
