"""Expand: fetch what the grader said was missing (definitions, callers, files, config, commits).

A small LangGraph subgraph: ``plan`` -> ``tools_condition`` -> prebuilt ``ToolNode``. The plan
comes from the fast model with the tools bound (it picks the calls, in parallel), or, when the
model can't call tools or the budget is tight, from deterministic heuristics. Either way the same
``ToolNode`` executes the calls and their artifacts become evidence.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from haive import prompts
from haive.graph.agent_tools import heuristic_calls, make_tools
from haive.graph.grade import ANSWER_RESERVE
from haive.graph.state import Deps, GraphState, NodeCtx
from haive.llm.models import RECOVERABLE

MAX_TOOL_CALLS = 4


class ExpandState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_expand_graph(plan, tools):
    g = StateGraph(ExpandState)
    g.add_node("plan", plan)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", END)
    return g.compile()


def make_expand(deps: Deps):
    def expand(state: GraphState, ctx: NodeCtx) -> dict[str, Any]:
        scope = state.get("scope_repos") or list(deps.cfg.repos)
        missing = (state.get("grade") or {}).get("missing", [])[:4]
        have = sorted({f"{e['repo']}/{e['path']}" for e in state.get("evidence") or [] if e.get("path")})
        tools = make_tools(deps.roots(scope), deps.db.indexed_paths(scope))
        # plan + the grade after it + the answer must all fit in the budget
        use_model = deps.cfg.limits.expand_with_tools and deps.remaining(state) >= ANSWER_RESERVE + 2
        bound = None
        if use_model:
            try:
                runnables = [deps.llm.model(mid).bind_tools(tools) for mid in deps.llm.model_ids("fast")]
                bound = runnables[0].with_fallbacks(runnables[1:], exceptions_to_handle=RECOVERABLE)
            except NotImplementedError:
                bound = None
        how = {"planner": "heuristic"}

        def plan(s: ExpandState) -> dict[str, Any]:
            msg: AIMessage | None = None
            if bound is not None:
                prompt = prompts.messages(
                    "expand", max_calls=MAX_TOOL_CALLS, repos=", ".join(scope), have=", ".join(have[:30]) or "(none)",
                    question=state.get("rewritten_query") or state["question"], missing="; ".join(missing),
                )  # fmt: skip
                try:
                    msg = bound.invoke(prompt)
                except Exception:  # BudgetExceeded or any model failure: use the heuristic plan
                    msg = None
                how["planner"] = (
                    "model" if msg is not None and msg.tool_calls else "heuristic (model gave no calls)"
                )
            if msg is None or not msg.tool_calls:
                calls = heuristic_calls(missing)
                msg = AIMessage(content="", tool_calls=[{**c, "id": f"h{i}"} for i, c in enumerate(calls)])
            msg.tool_calls = msg.tool_calls[:MAX_TOOL_CALLS]
            return {"messages": [msg]}

        graph = build_expand_graph(plan, tools)
        out = graph.invoke({"messages": []}, config=deps.llm.run_config("fast", ctx.meter))
        evs: list[dict[str, Any]] = []
        names = []
        for m in out["messages"]:
            if isinstance(m, ToolMessage):
                names.append(m.name)
                evs.extend(m.artifact or [])
        ctx.detail = f"{how['planner']} plan: {names} -> {len(evs)} evidence"
        return {"evidence": evs, "loops": state.get("loops", 0) + 1}

    return expand
