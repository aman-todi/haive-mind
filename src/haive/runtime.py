"""Wires config, database, embedder, vector store, LLM, and the checkpointer for CLI commands."""

from __future__ import annotations

import sqlite3
from functools import cached_property
from typing import Any

from haive import paths
from haive.config import Config, load_config, load_secrets
from haive.db import DB
from haive.graph.build import Runner
from haive.graph.state import Deps
from haive.index.embed import make_embedder
from haive.llm.models import LLM
from haive.store.base import make_store
from haive.tools import web
from haive.types import Embedder, VectorStore


class Runtime:
    def __init__(self, cfg: Config | None = None):
        load_secrets()
        self.cfg = cfg or load_config()

    @cached_property
    def db(self) -> DB:
        return DB(paths.db_file())

    @cached_property
    def embedder(self) -> Embedder:
        e = self.cfg.embeddings
        return make_embedder(e.model, e.dim, e.batch_size)

    @cached_property
    def store(self) -> VectorStore:
        return make_store(self.cfg, self.embedder)

    @cached_property
    def llm(self) -> LLM:
        return LLM(self.cfg, self.db)

    @cached_property
    def checkpointer(self) -> Any:
        from langgraph.checkpoint.sqlite import SqliteSaver

        paths.data_dir().mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(paths.checkpoints_file()), check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        return saver

    def web_search(self, query: str) -> list:
        w = self.cfg.web
        return web.search(query, w.provider, w.max_results)

    def deps(self, *, persist_thread: bool = False, ablation: str = "hybrid+loop") -> Deps:
        return Deps(
            cfg=self.cfg,
            db=self.db,
            llm=self.llm,
            store_factory=lambda: self.store,
            web_search=self.web_search,
            persist_thread=persist_thread,
            ablation=ablation,
        )

    def runner(self, *, persist_thread: bool = False, ablation: str = "hybrid+loop") -> Runner:
        warm = getattr(self.embedder, "warm", None)
        if warm:
            warm()  # loads the embedding model while the router's LLM call runs
        return Runner(self.deps(persist_thread=persist_thread, ablation=ablation), self.checkpointer)
