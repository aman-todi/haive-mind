"""Graph assembly and the run loop (interrupt handling, metadata for rendering/eval).

START -> freshness -> router -> {clarify | retrieve | parse_trace -> gather_debug | compare_sub* -> merge |
load_profiles -> summarize_repo* -> connect_map | general} -> grade -> {answer | expand | web | rewrite |
exhausted | to_general} ... -> answer -> verify -> {answer | finalize} -> END
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from haive.graph import answer as answer_mod
from haive.graph import debug, expand, fanout, finalize, grade, router, web_node
from haive.graph.retrieve import make_retrieve
from haive.graph.state import Deps, GraphState, NodeCtx, node, turn_input
from haive.index import freshness as fresh
from haive.types import Answer

Interrupt = Callable[[dict[str, Any]], Any]


def make_freshness(deps: Deps):
    def freshness(state: GraphState, ctx: NodeCtx) -> dict[str, Any]:
        cfg = deps.cfg
        repos, _ = (
            cfg.resolve_names(state.get("requested_scope") or [])
            if state.get("requested_scope")
            else (list(cfg.repos), [])
        )
        rep = fresh.check(cfg, deps.db, repos)
        notices: list[str] = []
        for rf in rep.repos:
            if rf.never_indexed:
                notices.append(f"{rf.repo} has never been indexed; run `haive index {rf.repo}`.")
            elif rf.error:
                notices.append(f"Freshness check failed for {rf.repo}: {rf.error}")
            elif len(rf.changed) > fresh.INLINE_LIMIT:
                notices.append(
                    f"{rf.repo} has {len(rf.changed)} changed files since the last index; "
                    f"run `haive index {rf.repo}` (answering from the current index)."
                )
            elif rf.changed or rf.head_moved:
                from haive.index.pipeline import Indexer

                st = ctx.timed(
                    "reindex",
                    rf.repo,
                    lambda rf=rf: Indexer(cfg, deps.db, deps.store).index_paths(rf.repo, rf.changed),
                )
                if st.files_changed or st.files_deleted:
                    notices.append(
                        f"Re-indexed {st.files_changed + st.files_deleted} changed file(s) in {rf.repo}."
                    )
        ctx.detail = f"{len(rep.repos)} repos checked in {rep.ms:.0f} ms"
        return {"notices": notices}

    return freshness


def make_general(deps: Deps):
    def general(state: GraphState, ctx: NodeCtx) -> dict[str, Any]:
        return {"web_return": "answer"}

    return general


def build_graph(deps: Deps, checkpointer: Any = None):
    g = StateGraph(GraphState)

    def add(name: str, factory: Callable[[Deps], Callable]) -> None:
        g.add_node(name, node(name, deps)(factory(deps)))

    add("freshness", make_freshness)
    add("router", router.make_router)
    add("clarify", router.make_clarify)
    add("retrieve", make_retrieve)
    add("parse_trace", debug.make_parse_trace)
    add("gather_debug", debug.make_gather_debug)
    add("compare_sub", fanout.make_compare_sub)
    add("merge", fanout.make_merge)
    add("load_profiles", fanout.make_load_profiles)
    add("summarize_repo", fanout.make_summarize_repo)
    add("connect_map", fanout.make_connect_map)
    add("general", make_general)
    add("grade", grade.make_grade)
    add("expand", expand.make_expand)
    add("rewrite", grade.make_rewrite)
    add("exhausted", grade.make_exhausted)
    add("to_general", grade.make_general_fallback)
    add("web_grade", lambda d: web_node.make_set_web_return("grade"))
    add("maybe_web", web_node.make_maybe_web)
    add("answer", answer_mod.make_answer)
    add("verify", answer_mod.make_verify)
    add("finalize", finalize.make_finalize)

    def after_router(state: GraphState):
        dest = router.route_after_router(state)
        return fanout.compare_fanout(state) if dest == "compare_fanout" else dest

    def after_rewrite(state: GraphState):
        intent = state.get("intent")
        if intent == "compare":
            return fanout.compare_fanout(state)
        if intent == "architecture":
            return "load_profiles"
        return "retrieve"

    route_grade = grade.route_after_grade(deps)

    def after_grade(state: GraphState) -> str:
        dest = route_grade(state)
        if state.get("intent") == "debug" and dest in ("rewrite", "general"):
            return "exhausted"  # a trace is still worth diagnosing without repo matches
        return {"web": "web_grade"}.get(dest, "to_general" if dest == "general" else dest)

    g.add_edge(START, "freshness")
    g.add_edge("freshness", "router")
    g.add_conditional_edges(
        "router",
        after_router,
        ["clarify", "retrieve", "parse_trace", "compare_sub", "load_profiles", "general"],
    )
    g.add_edge("clarify", "router")
    g.add_edge("retrieve", "grade")
    g.add_edge("parse_trace", "gather_debug")
    g.add_conditional_edges(
        "gather_debug",
        lambda s: "web_grade" if debug.route_after_gather(s) == "web" else "grade",
        ["web_grade", "grade"],
    )
    g.add_edge("compare_sub", "merge")
    g.add_edge("merge", "grade")
    g.add_conditional_edges(
        "load_profiles", fanout.route_after_profiles(deps), ["summarize_repo", "connect_map"]
    )
    g.add_edge("summarize_repo", "connect_map")
    g.add_edge("connect_map", "grade")
    g.add_edge("general", "maybe_web")
    g.add_conditional_edges(
        "grade", after_grade, ["answer", "expand", "web_grade", "exhausted", "rewrite", "to_general"]
    )
    g.add_edge("expand", "grade")
    g.add_edge("exhausted", "answer")
    g.add_conditional_edges("rewrite", after_rewrite, ["retrieve", "compare_sub", "load_profiles"])
    g.add_edge("to_general", "maybe_web")
    g.add_edge("web_grade", "maybe_web")
    g.add_conditional_edges("maybe_web", web_node.route_after_web, ["grade", "answer"])
    g.add_edge("answer", "verify")
    g.add_conditional_edges("verify", answer_mod.route_after_verify, ["answer", "finalize"])
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


@dataclass
class RunResult:
    answer: Answer | None
    state: dict[str, Any]
    latency_s: float
    interrupts: list[dict[str, Any]] = field(default_factory=list)

    def meta(self) -> dict[str, Any]:
        s = self.state
        return {
            "intent": s.get("intent"),
            "scope": s.get("scope_repos") or [],
            "llm_calls": s.get("llm_calls", 0),
            "latency_s": round(self.latency_s, 2),
            "notices": s.get("notices") or [],
            "loops": s.get("loops", 0),
            "answer_retries": s.get("answer_retries", 0),
            "first_verify": s.get("first_verify"),
            "title": s.get("title"),
        }


def default_interrupt(payload: dict[str, Any]) -> Any:
    """Non-interactive defaults: scope = all, web declined."""
    return "all" if payload.get("kind") == "clarify" else False


class Runner:
    def __init__(self, deps: Deps, checkpointer: Any = None):
        if checkpointer is None:
            from langgraph.checkpoint.memory import InMemorySaver

            checkpointer = InMemorySaver()
        self.deps = deps
        self.checkpointer = checkpointer
        self.app = build_graph(deps, checkpointer)

    def run(
        self,
        question: str,
        *,
        thread_id: str,
        trace_text: str | None = None,
        scope: list[str] | None = None,
        web_mode: str | None = None,
        on_interrupt: Interrupt | None = None,
        interactive: bool = True,
    ) -> RunResult:
        self.deps.thread_id = thread_id
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}
        t0 = time.monotonic()
        inp = turn_input(
            question, trace_text=trace_text, scope=scope, web_mode=web_mode or self.deps.cfg.web.mode
        )
        result = self.app.invoke(inp, config)
        seen: list[dict[str, Any]] = []
        defaulted: list[str] = []
        while result.get("__interrupt__"):
            payload = result["__interrupt__"][0].value
            seen.append(payload)
            if interactive and on_interrupt is not None:
                value = on_interrupt(payload)
            else:
                value = default_interrupt(payload)
                defaulted.append(
                    "scope defaulted to all repos"
                    if payload.get("kind") == "clarify"
                    else "web search declined (non-interactive)"
                )
            result = self.app.invoke(Command(resume=value), config)
        state = dict(self.app.get_state(config).values)
        if defaulted:
            state["notices"] = [*(state.get("notices") or []), *(f"Non-interactive: {d}." for d in defaulted)]
        ans = Answer.model_validate(state["answer"]) if state.get("answer") else None
        return RunResult(ans, state, time.monotonic() - t0, seen)
