"""Chroma adapter (optional backend). Self-contained: everything Chroma-specific lives here.

Install with ``uv tool install '.[chroma]'``. Persists under the data dir's ``chroma/``.
"""

from __future__ import annotations

from typing import Any

from haive.index.embed import embed_chunks
from haive.store.base import UPSERT_BATCH, clean_metadata, hit_from, normalize_filter
from haive.types import Chunk, Embedder, Hit


def chroma_where(filter: dict[str, Any] | None) -> dict[str, Any] | None:
    """Chroma needs an explicit ``$and`` when more than one key is filtered."""
    f = normalize_filter(filter)
    if not f:
        return None
    clauses = [{k: v} for k, v in f.items()]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class ChromaStore:
    def __init__(
        self, embedder: Embedder, path: str | None = None, collection: str = "haive", *, client: Any = None
    ):
        import chromadb

        self.embedder = embedder
        if client is None:
            client = chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
        self.client = client
        self.col = client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def upsert(self, chunks: list[Chunk]) -> None:
        for i in range(0, len(chunks), UPSERT_BATCH):
            batch = chunks[i : i + UPSERT_BATCH]
            metas = [clean_metadata(c) for c in batch]
            self.col.upsert(
                ids=[c.id for c in batch],
                embeddings=embed_chunks(self.embedder, batch),
                metadatas=metas,
                documents=[m.pop("text") for m in metas],
            )

    def delete(self, ids: list[str]) -> None:
        if ids:
            self.col.delete(ids=ids)

    def search(self, query: str, k: int, filter: dict[str, Any] | None = None) -> list[Hit]:
        n = self.col.count()
        if n == 0:
            return []
        res = self.col.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(k, n),
            where=chroma_where(filter),
            include=["metadatas", "documents", "distances"],
        )
        out = []
        for id_, meta, doc, dist in zip(
            res["ids"][0], res["metadatas"][0], res["documents"][0], res["distances"][0], strict=True
        ):
            out.append(hit_from(id_, 1.0 - float(dist), {**(meta or {}), "text": doc or ""}))
        return out

    def count(self, filter: dict[str, Any] | None = None) -> int:
        where = chroma_where(filter)
        if where is None:
            return self.col.count()
        return len(self.col.get(where=where, include=[])["ids"])
