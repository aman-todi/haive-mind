"""Router: intent classification (fast LLM, structured) plus scope validation in code."""

from __future__ import annotations

import re
from typing import Any

from langgraph.types import interrupt
from pydantic import BaseModel, Field

from haive import prompts
from haive.graph.state import Deps, GraphState, NodeCtx, history_messages
from haive.llm.budget import BudgetExceeded
from haive.llm.models import LLMError
from haive.llm.structured import structured
from haive.types import INTENTS


class RouterOutput(BaseModel):
    intent: str = Field(description="one of: " + ", ".join(INTENTS))
    repos: list[str] = Field(default_factory=list)
    compare_targets: list[str] = Field(default_factory=list)
    identifiers: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification: str = ""
    standalone_question: str = Field(
        default="", description="the latest message rewritten to stand alone, if it depends on earlier turns"
    )


IDENT_RE = re.compile(
    r"`([^`]{2,80})`"  # backticked
    r"|\b([A-Za-z_]\w*\.[A-Za-z_]\w*(?:\.\w+)*)\b"  # dotted (module.func, file.py)
    r"|\b([a-z]+_[a-z0-9_]+)\b"  # snake_case
    r"|\b([a-z]+[A-Z]\w*)\b"  # camelCase
    r"|\b([A-Z][a-z0-9]+[A-Z]\w*)\b"  # PascalCase with 2+ humps
    r"|\b([A-Z][A-Z0-9]*_[A-Z0-9_]+)\b"  # ENV_VARS
)
STOP_IDENTS = {"e.g", "i.e", "etc.etc"}


def code_identifiers(text: str) -> list[str]:
    out: list[str] = []
    for m in IDENT_RE.finditer(text):
        tok = next(g for g in m.groups() if g)
        if tok.lower() not in STOP_IDENTS and tok not in out:
            out.append(tok)
    return out[:12]


def heuristic_intent(q: str) -> str:
    s = q.lower()
    if re.search(r"\b(traceback|exception|error:|panic|stack trace|crash)", s):
        return "debug"
    if re.search(r"\b(differ|difference|compare|versus|vs\.?)\b", s):
        return "compare"
    if re.search(r"\b(architecture|break ?down|overview|how .* fit together|components)\b", s):
        return "architecture"
    if re.search(r"\b(when did|why did|why was|history|commit|changed|introduced|added)\b", s):
        return "history"
    if re.search(r"\b(before|previously|how did i|other projects?|again)\b", s):
        return "recall"
    return "lookup"


def make_router(deps: Deps):
    cfg = deps.cfg

    def router(state: GraphState, ctx: NodeCtx) -> dict[str, Any]:
        notices: list[str] = []
        if not cfg.repos:
            ctx.detail = "no repos registered"
            return {
                "intent": "general",
                "scope_repos": [],
                "router": {"intent": "general"},
                "notices": ["No repos are registered (haive add <path>); answering as a general question."],
            }
        out: dict[str, Any] | None = state.get("router")
        if out is None:
            if state.get("trace_text"):
                out = RouterOutput(intent="debug").model_dump()
            else:
                groups = ", ".join(f"{g}={m}" for g, m in cfg.groups.items()) or "(none)"
                prompt = prompts.messages(
                    "router",
                    repos=", ".join(cfg.repos),
                    groups=groups,
                    history=history_messages(state),
                    question=state["question"],
                )
                try:
                    out = structured(deps.llm, "fast", RouterOutput, prompt, ctx.meter).model_dump()
                except (LLMError, BudgetExceeded) as e:
                    out = RouterOutput(intent=heuristic_intent(state["question"])).model_dump()
                    notices.append(f"Router model failed ({type(e).__name__}); used keyword routing.")
        intent = out.get("intent", "lookup")
        if intent not in INTENTS:
            intent = heuristic_intent(state["question"])
        # A follow-up that leans on earlier turns ("and the tests for it?") is rewritten by the router so
        # retrieval, grading, and expand search for the real subject. The answer still sees the original.
        standalone = (out.get("standalone_question") or "").strip()
        has_history = len(state.get("messages") or []) > 1
        if not has_history or standalone.lower() == state["question"].strip().lower():
            standalone = ""
        idents = [*out.get("identifiers", [])]
        for t in code_identifiers(standalone or state["question"]):
            if t not in idents:
                idents.append(t)

        # ---- scope, validated in code
        update: dict[str, Any] = {
            "router": out,
            "intent": intent,
            "identifiers": idents[:12],
            "rewritten_query": standalone or None,
        }
        if standalone:
            notices.append(f"Follow-up read as: {standalone}")
        requested = state.get("requested_scope") or []
        hint = state.get("scope_hint")
        if requested:
            scope, unknown = cfg.resolve_names(requested)
            if unknown:
                notices.append(f"Unknown repos/groups ignored: {', '.join(unknown)}")
        elif hint is not None:
            scope, unknown = cfg.resolve_names(re.split(r"[,\s]+", hint))
            if not scope:
                scope = list(cfg.repos)
                notices.append("Scope defaulted to all repos.")
        else:
            scope, unknown = cfg.resolve_names(out.get("repos", []))
            if (unknown or out.get("needs_clarification")) and not state.get("clarified"):
                q = out.get("clarification") or (
                    f"I don't know {', '.join(repr(u) for u in unknown)}. Which repos or groups did you mean?"
                    if unknown
                    else "Which repos or groups should I look at?"
                )
                ctx.detail = f"clarify: unknown={unknown}"
                return {**update, "clarify_question": q, "notices": notices}
            if not scope:
                scope = list(cfg.repos)
        update["scope_repos"] = scope
        update["clarify_question"] = None

        if intent == "compare":
            targets = []
            for name in out.get("compare_targets", []) or []:
                repos, _ = cfg.resolve_names([name])
                repos = [r for r in repos if r in scope] or repos
                if repos:
                    targets.append({"name": name, "repos": repos})
            if len(targets) < 2:
                targets = [{"name": r, "repos": [r]} for r in scope]
            if len(targets) < 2:
                update["intent"] = intent = "lookup"
                notices.append("Compare needs two or more repos/groups in scope; answered as a lookup.")
            update["compare_targets"] = targets
            update["scope_repos"] = sorted({r for t in targets for r in t["repos"]}) or scope
        ctx.detail = f"intent={intent} scope={update['scope_repos']}"
        update["notices"] = notices
        return update

    return router


def make_clarify(deps: Deps):
    def clarify(state: GraphState, ctx: NodeCtx) -> dict[str, Any]:
        options = list(deps.cfg.repos) + list(deps.cfg.groups)
        answer = interrupt({"kind": "clarify", "question": state.get("clarify_question"), "options": options})
        ctx.detail = f"answer={answer!r}"
        return {"scope_hint": str(answer or "all"), "clarified": True}

    return clarify


def route_after_router(state: GraphState) -> str:
    if state.get("clarify_question"):
        return "clarify"
    return {
        "lookup": "retrieve",
        "recall": "retrieve",
        "history": "retrieve",
        "debug": "parse_trace",
        "compare": "compare_fanout",
        "architecture": "load_profiles",
        "general": "general",
    }.get(state.get("intent", "lookup"), "retrieve")
