from __future__ import annotations

from typing import Any

from haive.config import Config
from haive.db import DB
from haive.graph.build import Runner
from haive.graph.state import Deps
from haive.index.pipeline import Indexer
from haive.llm.models import LLM
from haive.types import WebResult
from tests.fakes import FakeChatModel, factory_for


def router(intent: str = "lookup", **kw: Any) -> dict[str, Any]:
    return {"intent": intent, **kw}


def grade(status: str = "sufficient", missing: list[str] | None = None) -> dict[str, Any]:
    return {"status": status, "missing": missing or [], "reason": "test"}


def repo_claim(
    path: str, start: int, end: int, symbol: str | None = None, text: str = "claim", repo: str = "valuation"
):
    ref = {"repo": repo, "path": path, "start_line": start, "end_line": end}
    if symbol:
        ref["symbol"] = symbol
    return {"text": text, "source": "repo", "refs": [ref]}


def answer(*claims: dict[str, Any], summary: str = "summary", **kw: Any) -> dict[str, Any]:
    return {
        "summary": summary,
        "sections": [{"heading": "Details", "claims": list(claims)}],
        "followups": [],
        **kw,
    }


DCF_CLAIM = repo_claim(
    "app/valuation/dcf.py", 1, 7, "choose_method", "choose_method picks ddm, fcf, or multiples"
)


def make_runner(
    cfg: Config,
    db: DB,
    store,
    script: dict[str, list[Any]],
    *,
    web: list[WebResult] | None = None,
    index: bool = True,
    persist: bool = False,
    ablation: str = "hybrid+loop",
    supports_tools: bool = False,
) -> tuple[Runner, FakeChatModel]:
    fake = FakeChatModel(script, supports_tools=supports_tools)
    llm = LLM(cfg, db, factory_for(fake))
    if index:
        idx = Indexer(cfg, db, store)
        for name in cfg.repos:
            idx.index_repo(name)
    calls: list[str] = []

    def web_search(q: str) -> list[WebResult]:
        calls.append(q)
        return web or []

    deps = Deps(
        cfg=cfg,
        db=db,
        llm=llm,
        store_factory=lambda: store,
        web_search=web_search,
        persist_thread=persist,
        ablation=ablation,
    )
    deps.extra["web_calls"] = calls
    return Runner(deps), fake


def nodes(result) -> list[str]:
    return [e["node"] for e in result.state["node_log"] if e["type"] == "node"]
