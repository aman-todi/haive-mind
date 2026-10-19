"""Finalize (code): demote unverifiable claims, set the chat title on the first turn, record the turn."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from haive import threads
from haive.graph.answer import fallback_answer
from haive.graph.state import Deps, GraphState, NodeCtx
from haive.graph.verify import EMPTY_ANSWER, claim_keys
from haive.render import fmt_ref
from haive.types import Answer


def answer_as_text(a: Answer, max_claims: int = 8) -> str:
    lines = [a.summary]
    for _, c in claim_keys(a)[:max_claims]:
        refs = ", ".join(fmt_ref(r) for r in c.refs)
        lines.append(
            f"- [{'unverified' if c.unverified else c.source}] {c.text}" + (f" ({refs})" if refs else "")
        )
    return "\n".join(lines)


def make_finalize(deps: Deps):
    def finalize(state: GraphState, ctx: NodeCtx) -> dict[str, Any]:
        a = (
            Answer.model_validate(state["answer"])
            if state.get("answer")
            else fallback_answer(state, "no answer")
        )
        bad = state.get("bad_claims") or {}
        notices: list[str] = []
        if EMPTY_ANSWER in bad:
            # The model summarized but cited nothing, even after a retry: show what it was given instead.
            a.sections = fallback_answer(state, "", max_claims=12).sections
            notices.append("The model's answer listed no claims; showing the evidence it was given.")
        unverified = 0
        for key, claim in claim_keys(a):
            if key in bad:
                claim.unverified = True
                unverified += 1
        if unverified:
            notices.append(f"{unverified} claim(s) could not be verified and are tagged [unverified].")
        if state.get("low_confidence"):
            notices.append("Low confidence: evidence gathering stopped before the grader was satisfied.")
        update: dict[str, Any] = {"answer": a.model_dump(), "messages": [AIMessage(answer_as_text(a))]}
        if deps.persist_thread and deps.thread_id:
            title = None
            row = threads.get(deps.db, deps.thread_id)
            if row is None or not row.get("title"):
                ctx.meter.limit = max(0, deps.remaining(state))
                title = threads.make_title(deps.llm, state["question"], ctx.meter)
                update["title"] = title
            threads.record_turn(deps.db, deps.thread_id, state.get("scope_repos") or [], title)
        update["notices"] = notices
        ctx.detail = f"{unverified} unverified" + (
            "; empty answer replaced by evidence" if EMPTY_ANSWER in bad else ""
        )
        return update

    return finalize
