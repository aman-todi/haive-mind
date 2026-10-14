"""maybe_web: web search per mode (off | ask | auto), with an approval interrupt for "ask"."""

from __future__ import annotations

import re
from typing import Any

from langgraph.types import interrupt

from haive.graph.state import Deps, GraphState, NodeCtx
from haive.tools.web import WebSearchError
from haive.types import Evidence


def web_query(state: GraphState) -> str:
    if state.get("debug_web_query"):
        return state["debug_web_query"]
    g = state.get("grade") or {}
    if g.get("status") == "need_external" and g.get("missing"):
        return " ".join(g["missing"][:2])[:300]
    q = state.get("rewritten_query") or state["question"]
    return re.sub(r"\s+", " ", q)[:300]


def make_maybe_web(deps: Deps):
    def maybe_web(state: GraphState, ctx: NodeCtx) -> dict[str, Any]:
        mode = state.get("web_mode") or deps.cfg.web.mode
        from_grade = state.get("web_return") == "grade"
        base: dict[str, Any] = {"web_done": True}
        if from_grade:
            base["loops"] = state.get("loops", 0) + 1
        query = web_query(state)
        if mode == "off":
            ctx.detail = "off"
            return {**base, "notices": ["Web search is off; skipped."]}
        if mode == "ask":
            approved = interrupt({"kind": "web", "query": query})
            if not approved:
                ctx.detail = "declined"
                return {**base, "notices": ["Web search declined."]}
        if deps.web_search is None:
            return {**base, "notices": ["Web search is unavailable."]}
        try:
            results = ctx.timed("web", query, lambda: deps.web_search(query))
        except WebSearchError as e:
            ctx.detail = "failed"
            return {**base, "notices": [f"Web search failed: {e}"]}
        evs = [
            Evidence(
                id=f"web:{r.url}", source="web", text=f"{r.title}\n{r.url}\n{r.snippet}", why="web"
            ).model_dump()
            for r in results
        ]
        ctx.detail = f"{len(results)} results for {query!r}"
        return {**base, "web_results": [r.model_dump() for r in results], "evidence": evs}

    return maybe_web


def route_after_web(state: GraphState) -> str:
    return "grade" if state.get("web_return") == "grade" else "answer"


def make_set_web_return(target: str):
    def set_return(state: GraphState, ctx: NodeCtx) -> dict[str, Any]:
        return {"web_return": target}

    return set_return
