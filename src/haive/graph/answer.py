"""Answer (LLM, intent template, structured output) and the verify node around verify.py."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from haive import prompts
from haive.graph.state import Deps, GraphState, NodeCtx, history_messages, render_evidence
from haive.graph.verify import claim_keys, verify
from haive.llm.budget import BudgetExceeded
from haive.llm.models import LLMError
from haive.llm.structured import structured
from haive.types import Answer, Claim, CommitRef, Evidence, ParsedTrace, RepoRef, Section, WebResult


class NeedMore(BaseModel):
    status: Literal["missing_symbol", "need_context", "need_external"]
    missing: list[str] = Field(default_factory=list, description="names, paths, path:line, or a web query")
    reason: str = ""


class AnswerOrMore(Answer):
    """Combined mode: an answer, or (``need_more``) a request for more evidence first."""

    need_more: NeedMore | None = Field(
        default=None, description="set only if the evidence can't answer the core of the question"
    )


ASK_FOR_MORE = (
    "If the evidence can't answer the core of the question, don't guess: set `need_more` instead, with "
    "status missing_symbol (definitions to fetch: list the names), need_context (callers, a config file, "
    "the rest of a file: list identifiers, paths, or path:line), or need_external (third-party behavior: "
    "a web search query), and leave sections empty. Answer whenever the core is covered."
)


def render_web(results: list[dict[str, Any]]) -> str:
    if not results:
        return "(none)"
    return "\n".join(f"- {r['url']}\n  {r.get('title', '')}: {r.get('snippet', '')[:500]}" for r in results)


def fallback_answer(state: GraphState, reason: str, max_claims: int = 6) -> Answer:
    """Deterministic answer from the evidence when the model can't (or may not) be called."""
    claims = []
    for e in state.get("evidence") or []:
        sha = Evidence.model_validate(e).sha
        if sha and e.get("repo"):
            subject = (e["text"].split("\n") + [""])[1].strip()
            claims.append(
                Claim(
                    text=f"Commit: {subject or e.get('why') or sha}",
                    source="repo",
                    refs=[CommitRef(repo=e["repo"], sha=sha)],
                )
            )
        elif e.get("path") and e.get("repo") and e.get("start_line"):
            claims.append(
                Claim(
                    text=f"Relevant: {e.get('why') or e['path']}",
                    source="repo",
                    refs=[
                        RepoRef(
                            repo=e["repo"], path=e["path"], start_line=e["start_line"], end_line=e["end_line"]
                        )
                    ],
                )
            )
        if len(claims) >= max_claims:
            break
    summary = f"No model answer ({reason})."
    summary += (
        " Here is the most relevant evidence I found."
        if claims
        else " No relevant evidence was found either."
    )
    return Answer(summary=summary, sections=[Section(heading="Evidence", claims=claims)] if claims else [])


def make_answer(deps: Deps):
    def answer(state: GraphState, ctx: NodeCtx) -> dict[str, Any]:
        intent = state.get("intent") or "lookup"
        if deps.remaining(state) <= 0:
            ctx.detail = "budget exhausted: evidence-only answer"
            a = fallback_answer(state, "LLM call budget for this question is spent")
            return {
                "answer": a.model_dump(),
                "notices": ["LLM call budget reached before answering."],
                "answer_retries": deps.cfg.limits.max_answer_retries,
            }
        if intent == "debug" and state.get("parsed_trace"):
            from haive.graph.debug import trace_summary

            template = prompts.text(
                "diagnose", trace=trace_summary(ParsedTrace.model_validate(state["parsed_trace"]))
            )
        else:
            template = prompts.text(f"answer_{intent}" if intent != "debug" else "answer_lookup")
        errors = ""
        if state.get("verify_errors"):
            errors = (
                "Your previous answer failed verification. Fix these problems: cite only paths and line ranges, or "
                "commit shas, shown in the evidence headers; keep symbols inside the cited lines; and change "
                "unsupported claims to source 'model' with no refs.\n- "
                + "\n- ".join(state["verify_errors"][:15])
            )
        # Combined mode: this call may ask for more evidence, while loops and budget allow another answer.
        may_ask = (
            deps.combined
            and not state.get("answer_retries")
            and not state.get("low_confidence")  # set by `exhausted`: it must answer now
            and state.get("loops", 0) < deps.cfg.limits.max_grade_loops
            and deps.remaining(state) >= 3  # this call, an expand plan, and the final answer
        )
        prompt = prompts.messages(
            "answer_common",
            template=template,
            intent=intent,
            question=state["question"],
            history=history_messages(state),
            errors="\n\n".join(x for x in (errors, ASK_FOR_MORE if may_ask else "") if x),
            evidence=render_evidence(state.get("evidence") or []),
            web=render_web(state.get("web_results") or []),
        )
        try:
            a = structured(deps.llm, "answer", AnswerOrMore if may_ask else Answer, prompt, ctx.meter)
        except (LLMError, BudgetExceeded) as e:
            ctx.detail = f"failed: {type(e).__name__}"
            a = fallback_answer(state, f"answer model failed: {type(e).__name__}")
            return {
                "answer": a.model_dump(),
                "notices": [f"Answer model failed ({e.__class__.__name__})."],
                "answer_retries": deps.cfg.limits.max_answer_retries,
            }
        if isinstance(a, AnswerOrMore):
            if a.need_more is not None:
                more = a.need_more.model_dump()
                ctx.detail = f"needs more: {more['status']} {more['missing'][:3]}"
                return {"grade": more, "answer": None, "answer_wants_more": True}
            a = Answer.model_validate(a.model_dump(exclude={"need_more"}))
        if intent != "debug":
            a.hypotheses = None
            a.suggested_fix = None
        elif a.suggested_fix is not None:
            a.suggested_fix.approximate = False
        ctx.detail = f"{len(a.all_claims())} claims"
        return {"answer": a.model_dump(), "answer_wants_more": False}

    return answer


def make_verify(deps: Deps):
    def verify_node(state: GraphState, ctx: NodeCtx) -> dict[str, Any]:
        a = Answer.model_validate(state["answer"])
        res = verify(
            a,
            [Evidence.model_validate(e) for e in state.get("evidence") or []],
            [WebResult.model_validate(w) for w in state.get("web_results") or []],
            deps.roots(),
        )
        update: dict[str, Any] = {"verify_errors": res.errors, "bad_claims": res.bad_claims}
        if a.suggested_fix is not None:
            a.suggested_fix.approximate = res.approximate_fix
            update["answer"] = a.model_dump()
        if state.get("answer_retries", 0) == 0:
            keys = claim_keys(a)
            repo_claims = [k for k, c in keys if c.source == "repo"]
            update["first_verify"] = {
                "repo_claims": len(repo_claims),
                "repo_claims_ok": len([k for k in repo_claims if k not in res.bad_claims]),
                "errors": res.errors[:20],  # what failed on the first try (eval reports these)
            }
        retries = state.get("answer_retries", 0)
        # A retry that fixed less than half of the remaining problems won't be helped by another one.
        stuck = retries > 0 and len(res.bad_claims) * 2 > len(state.get("bad_claims") or {})
        retry = (
            bool(res.bad_claims)
            and not stuck
            and retries < deps.cfg.limits.max_answer_retries
            and deps.remaining(state) > 0
        )
        update["retry_answer"] = retry
        if retry:
            update["answer_retries"] = retries + 1
        ctx.detail = f"{len(res.bad_claims)} bad claims; retry={retry}" + ("; no progress" if stuck else "")
        return update

    return verify_node


def route_after_answer(route_grade):
    """Combined mode: an answer that asked for more goes down the grader's routes (expand, web, or
    exhausted, which answers anyway); anything else is verified."""

    def route(state: GraphState) -> str:
        if state.get("answer_wants_more") and state.get("answer") is None:
            dest = route_grade(state)
            return (
                {"web": "web_grade"}.get(dest, dest)
                if dest in ("expand", "web", "exhausted")
                else "exhausted"
            )
        return "verify"

    return route


def route_after_verify(state: GraphState) -> str:
    return "answer" if state.get("retry_answer") else "finalize"
