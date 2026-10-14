"""Chroma adapter (optional backend): ``langchain_chroma.Chroma``. Everything Chroma-specific lives here.

Install with ``uv tool install '.[chroma]'``. Persists under the data dir's ``chroma/``.
"""

from __future__ import annotations

from typing import Any

from haive.store.base import LangChainStore, normalize_filter
from haive.types import Embedder


def chroma_where(filter: dict[str, Any] | None) -> dict[str, Any] | None:
    """Chroma needs an explicit ``$and`` when more than one key is filtered."""
    f = normalize_filter(filter)
    if not f:
        return None
    clauses = [{k: v} for k, v in f.items()]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class ChromaStore(LangChainStore):
    def __init__(
        self, embedder: Embedder, path: str | None = None, collection: str = "haive", *, client: Any = None
    ):
        import chromadb
        from langchain_chroma import Chroma

        self.embedder = embedder
        if client is None:
            client = chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
        super().__init__(
            Chroma(
                collection_name=collection,
                embedding_function=embedder,
                client=client,
                collection_metadata={"hnsw:space": "cosine"},
            )
        )

    def _filter(self, f: dict[str, Any]) -> Any:
        return chroma_where(f)

    def count(self, filter: dict[str, Any] | None = None) -> int:
        where = chroma_where(filter)
        col = self.vs._collection
        if where is None:
            return col.count()
        return len(col.get(where=where, include=[])["ids"])
