"""Graph state, per-run dependencies, node instrumentation, and evidence helpers.

State values are plain JSON-friendly data (pydantic models are stored via ``model_dump()``) so the
SQLite checkpointer can persist chat threads. List/int accumulators use reducers that reset when a
new turn passes ``None``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property, wraps
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

from haive.config import Config
from haive.db import DB
from haive.llm.budget import Meter
from haive.llm.models import LLM
from haive.tools import files
from haive.types import Evidence, Hit, VectorStore, WebResult

# ---------------------------------------------------------------- reducers


def add_or_reset(a: list | None, b: list | None) -> list:
    if b is None:
        return []
    return (a or []) + list(b)


def add_int_or_reset(a: int | None, b: int | None) -> int:
    if b is None:
        return 0
    return (a or 0) + b


def merge_by_key(key: str) -> Callable[[list | None, list | None], list]:
    def merge(a: list | None, b: list | None) -> list:
        if b is None:
            return []
        out = list(a or [])
        seen = {x[key] for x in out}
        for x in b:
            if x[key] not in seen:
                seen.add(x[key])
                out.append(x)
        return out

    return merge


class GraphState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    trace_text: str | None
    requested_scope: list[str]
    web_mode: str
    intent: str
    scope_repos: list[str]
    compare_targets: list[dict[str, Any]]
    identifiers: list[str]
    router: dict[str, Any] | None
    scope_hint: str | None
    clarified: bool
    clarify_question: str | None
    clarify_kind: str | None  # "scope" | "question"
    clarify_options: list[str]
    rewritten_query: str | None
    rewrites: int
    parsed_trace: dict[str, Any] | None
    evidence: Annotated[list[dict[str, Any]], merge_by_key("id")]
    web_results: Annotated[list[dict[str, Any]], merge_by_key("url")]
    web_done: bool
    web_return: str
    debug_web_query: str | None
    summaries: Annotated[list[dict[str, Any]], add_or_reset]
    grade: dict[str, Any] | None
    loops: int
    low_confidence: bool
    llm_calls: Annotated[int, add_int_or_reset]
    answer: dict[str, Any] | None
    verify_errors: list[str]
    bad_claims: dict[str, list[str]]
    answer_retries: int
    retry_answer: bool
    first_verify: dict[str, int] | None
    title: str | None
    notices: Annotated[list[str], add_or_reset]
    node_log: Annotated[list[dict[str, Any]], add_or_reset]
    turn: int


def turn_input(
    question: str, *, trace_text: str | None = None, scope: list[str] | None = None, web_mode: str = "ask"
) -> dict[str, Any]:
    """Input for one turn. Resets per-turn accumulators; keeps ``messages`` (chat history)."""
    return {
        "messages": [HumanMessage(question if not trace_text else f"{question}\n\n{trace_text}".strip())],
        "question": question,
        "trace_text": trace_text,
        "requested_scope": scope or [],
        "web_mode": web_mode,
        "intent": "",
        "scope_repos": [],
        "compare_targets": [],
        "identifiers": [],
        "router": None,
        "scope_hint": None,
        "clarified": False,
        "clarify_question": None,
        "clarify_kind": None,
        "clarify_options": [],
        "rewritten_query": None,
        "rewrites": 0,
        "parsed_trace": None,
        "evidence": None,
        "web_results": None,
        "web_done": False,
        "web_return": "answer",
        "debug_web_query": None,
        "summaries": None,
        "grade": None,
        "loops": 0,
        "low_confidence": False,
        "llm_calls": None,
        "answer": None,
        "verify_errors": [],
        "bad_claims": {},
        "answer_retries": 0,
        "retry_answer": False,
        "first_verify": None,
        "notices": None,
        "node_log": None,
    }


# ---------------------------------------------------------------- dependencies


@dataclass
class Deps:
    cfg: Config
    db: DB
    llm: LLM
    store_factory: Callable[[], VectorStore]
    web_search: Callable[[str], list[WebResult]] | None = None
    thread_id: str | None = None
    persist_thread: bool = False  # chat threads get titles and rows in `threads`
    ablation: str = "hybrid+loop"  # vector_only | ripgrep_only | hybrid | hybrid+loop
    extra: dict[str, Any] = field(default_factory=dict)

    @cached_property
    def store(self) -> VectorStore:
        return self.store_factory()

    def roots(self, repos: list[str] | None = None) -> dict[str, Path]:
        names = repos if repos is not None else list(self.cfg.repos)
        return {r: self.cfg.repo_root(r) for r in names if r in self.cfg.repos}

    def remaining(self, state: GraphState) -> int:
        return self.cfg.limits.max_llm_calls_per_question - int(state.get("llm_calls") or 0)

    @property
    def use_vector(self) -> bool:
        return self.ablation != "ripgrep_only"

    @property
    def use_ripgrep(self) -> bool:
        return self.ablation != "vector_only"

    @property
    def use_loop(self) -> bool:
        return self.ablation == "hybrid+loop"


@dataclass
class NodeCtx:
    meter: Meter
    events: list[dict[str, Any]] = field(default_factory=list)
    detail: str = ""
    notices: list[str] = field(default_factory=list)  # shown to the user under the answer

    def tool(self, name: str, detail: str, ms: float) -> None:
        self.events.append({"type": "tool", "name": name, "detail": detail[:160], "ms": round(ms)})

    def timed(self, name: str, detail: str, fn: Callable[[], Any]) -> Any:
        t0 = time.monotonic()
        try:
            return fn()
        finally:
            self.tool(name, detail, (time.monotonic() - t0) * 1000)


def node(name: str, deps: Deps) -> Callable:
    """Wrap a node ``fn(state, ctx) -> dict`` with timing, LLM metering, and --trace logging."""

    def deco(fn: Callable[[GraphState, NodeCtx], dict[str, Any] | Any]) -> Callable[[GraphState], Any]:
        @wraps(fn)
        def run(state: GraphState) -> Any:
            ctx = NodeCtx(meter=Meter(limit=max(0, deps.remaining(state))))
            t0 = time.monotonic()
            out = fn(state, ctx) or {}
            ms = round((time.monotonic() - t0) * 1000)
            entry = {"type": "node", "node": name, "ms": ms, "detail": ctx.detail}
            if isinstance(out, dict):
                out = dict(out)
                out["node_log"] = [entry, *ctx.meter.log, *ctx.events, *out.get("node_log", [])]
                out["llm_calls"] = ctx.meter.calls + int(out.get("llm_calls", 0) or 0)
                if ctx.notices:
                    out["notices"] = [*(out.get("notices") or []), *ctx.notices]
            return out

        return run

    return deco


# ---------------------------------------------------------------- evidence helpers

MAX_EVIDENCE_LINES = 150


def ev_from_snippet(repo: str, snip: files.Snippet, source: str = "tool", why: str = "") -> dict[str, Any]:
    text = snip.text
    lines = text.split("\n")
    end = snip.end_line
    if len(lines) > MAX_EVIDENCE_LINES:
        text = "\n".join(lines[:MAX_EVIDENCE_LINES])
        end = snip.start_line + MAX_EVIDENCE_LINES - 1
    return Evidence(
        id=f"{repo}:{snip.path}:{snip.start_line}-{end}",
        source=source,
        repo=repo,
        path=snip.path,
        start_line=snip.start_line,
        end_line=end,
        text=text,
        why=why or snip.symbol,
    ).model_dump()


def ev_from_hit(deps: Deps, hit: Hit, why: str = "vector") -> dict[str, Any] | None:
    """Evidence for a vector hit, re-read from the live tree (the index may lag a little)."""
    if hit.kind == "commit":
        return Evidence(
            id=f"commit:{hit.repo}:{hit.symbol}", source="commit", repo=hit.repo, text=hit.text, why=why
        ).model_dump()
    root = deps.roots([hit.repo]).get(hit.repo)
    if root is None:
        return None
    try:
        snip = files.read_range(root, hit.path, hit.start_line, hit.end_line)
    except files.FileAccessError:
        return None
    if snip is None:
        return None
    snip.symbol = hit.symbol
    return ev_from_snippet(hit.repo, snip, "code", f"{why}: {hit.symbol}" if hit.symbol else why)


def render_evidence(evidence: list[dict[str, Any]], max_chars: int = 60_000, abbreviate: int = 0) -> str:
    """Evidence for prompts: a header per item and numbered lines (or ``abbreviate`` lines each)."""
    out: list[str] = []
    used = 0
    for i, e in enumerate(evidence, 1):
        if e.get("path"):
            header = f"[E{i}] repo={e['repo']} path={e['path']} lines={e['start_line']}-{e['end_line']}"
            body = files.numbered(e["text"], e["start_line"] or 1)
        else:
            header = f"[E{i}] {e['source']}" + (f" repo={e['repo']}" if e.get("repo") else "")
            sha = Evidence.model_validate(e).sha
            if sha:
                header += f" sha={sha}"
            body = e["text"]
        if e.get("why"):
            header += f" ({e['why']})"
        if abbreviate:
            body = "\n".join(body.split("\n")[:abbreviate])
        block = f"{header}\n{body}\n"
        if used + len(block) > max_chars:
            out.append(f"... {len(evidence) - i + 1} more evidence items omitted")
            break
        out.append(block)
        used += len(block)
    return "\n".join(out) if out else "(none)"


def history_messages(state: GraphState, turns: int = 6) -> list[AnyMessage]:
    """Recent chat turns before the current question (for MessagesPlaceholder("history"))."""
    return list(state.get("messages") or [])[:-1][-turns:]
