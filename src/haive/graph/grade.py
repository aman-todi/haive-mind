"""Grade (LLM): is the evidence enough? Also rewrite_query for the nothing-relevant path."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from haive import prompts
from haive.graph.state import Deps, GraphState, NodeCtx, render_evidence
from haive.llm.budget import BudgetExceeded
from haive.llm.models import LLMError
from haive.llm.structured import structured
from haive.types import GradeStatus

# One call is always held back for the answer node.
ANSWER_RESERVE = 1


class GradeOutput(BaseModel):
    status: GradeStatus
    missing: list[str] = Field(default_factory=list)
    reason: str = ""


class RewriteOutput(BaseModel):
    query: str
    identifiers: list[str] = Field(default_factory=list)


def make_grade(deps: Deps):
    def grade(state: GraphState, ctx: NodeCtx) -> dict[str, Any]:
        evidence = state.get("evidence") or []
        if not evidence:
            ctx.detail = "nothing_relevant (no evidence)"
            return {"grade": {"status": "nothing_relevant", "missing": [], "reason": "no evidence"}}
        if not deps.use_loop:
            ctx.detail = "sufficient (loop disabled by ablation)"
            return {"grade": {"status": "sufficient", "missing": [], "reason": "ablation"}}
        if deps.remaining(state) <= ANSWER_RESERVE:
            ctx.detail = "budget: skip grading"
            return {
                "grade": {"status": "sufficient", "missing": [], "reason": "budget"},
                "low_confidence": True,
                "notices": ["LLM call budget nearly spent; answered without further checks."],
            }
        ctx.meter.limit = max(0, deps.remaining(state) - ANSWER_RESERVE)
        prompt = prompts.messages(
            "grade",
            intent=state.get("intent"),
            question=state.get("rewritten_query") or state["question"],
            evidence=render_evidence(evidence, max_chars=24_000, abbreviate=18),
        )
        try:
            g = structured(deps.llm, "grade", GradeOutput, prompt, ctx.meter).model_dump()
        except (LLMError, BudgetExceeded) as e:
            g = {"status": "sufficient", "missing": [], "reason": f"grader failed: {type(e).__name__}"}
        ctx.detail = f"{g['status']} {g['missing'][:3]}"
        return {"grade": g}

    return grade


def route_after_grade(deps: Deps):
    def route(state: GraphState) -> str:
        g = state.get("grade") or {}
        status = g.get("status", "sufficient")
        loops = state.get("loops", 0)
        max_loops = deps.cfg.limits.max_grade_loops
        if status == "sufficient":
            return "answer"
        if status in ("missing_symbol", "need_context"):
            return "expand" if loops < max_loops else "exhausted"
        if status == "need_external":
            return "web" if loops < max_loops and not state.get("web_done") else "exhausted"
        # nothing_relevant
        if state.get("rewrites", 0) < 1 and deps.remaining(state) > ANSWER_RESERVE + 1:
            return "rewrite"
        return "general"

    return route


def make_rewrite(deps: Deps):
    def rewrite(state: GraphState, ctx: NodeCtx) -> dict[str, Any]:
        ctx.meter.limit = max(0, deps.remaining(state) - ANSWER_RESERVE)
        prompt = prompts.messages(
            "rewrite",
            repos=", ".join(state.get("scope_repos") or []),
            question=state["question"],
            query=state.get("rewritten_query") or state["question"],
        )
        try:
            r = structured(deps.llm, "rewrite", RewriteOutput, prompt, ctx.meter)
            query, idents = r.query, r.identifiers
        except (LLMError, BudgetExceeded):
            query, idents = state["question"], []
        merged = list(dict.fromkeys([*(state.get("identifiers") or []), *idents]))[:12]
        ctx.detail = query
        return {"rewritten_query": query, "identifiers": merged, "rewrites": state.get("rewrites", 0) + 1}

    return rewrite


def make_exhausted(deps: Deps):
    def exhausted(state: GraphState, ctx: NodeCtx) -> dict[str, Any]:
        return {
            "low_confidence": True,
            "notices": ["Grade loop limit reached; the answer may be incomplete."],
        }

    return exhausted


def make_general_fallback(deps: Deps):
    def to_general(state: GraphState, ctx: NodeCtx) -> dict[str, Any]:
        return {
            "intent": "general",
            "web_return": "answer",
            "notices": ["Nothing relevant was found in the repos; answering from general knowledge."],
        }

    return to_general
