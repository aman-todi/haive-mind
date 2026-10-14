"""Pinecone adapter (default backend). Uses the ``pinecone`` client directly."""

from __future__ import annotations

import os
from typing import Any

from haive.index.embed import embed_chunks
from haive.store.base import UPSERT_BATCH, clean_metadata, hit_from, normalize_filter
from haive.types import Chunk, Embedder, Hit


class PineconeError(RuntimeError):
    pass


class PineconeStore:
    def __init__(
        self, embedder: Embedder, index_name: str = "haive", cloud: str = "aws", region: str = "us-east-1",
        *, client: Any = None, api_key: str | None = None,
    ):  # fmt: skip
        self.embedder = embedder
        self.index_name = index_name
        self.cloud = cloud
        self.region = region
        if client is None:
            from pinecone import Pinecone

            key = api_key or os.environ.get("PINECONE_API_KEY")
            if not key:
                raise PineconeError("PINECONE_API_KEY is not set")
            client = Pinecone(api_key=key)
        self.pc = client
        self._index = None

    # ------------------------------------------------------------ admin

    def ensure_index(self) -> bool:
        """Create the index if missing and check its dimension. Returns True if it was created."""
        created = False
        if not self.pc.has_index(self.index_name):
            from pinecone import ServerlessSpec

            self.pc.create_index(
                name=self.index_name, dimension=self.embedder.dim, metric="cosine",
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )  # fmt: skip
            created = True
        desc = self.pc.describe_index(self.index_name)
        dim = getattr(desc, "dimension", None) or (desc.get("dimension") if isinstance(desc, dict) else None)
        if dim and int(dim) != self.embedder.dim:
            raise PineconeError(
                f"Pinecone index '{self.index_name}' has dimension {dim}, but the embedder produces "
                f"{self.embedder.dim}. Use a different pinecone_index or delete the old one."
            )
        metric = getattr(desc, "metric", None) or (desc.get("metric") if isinstance(desc, dict) else None)
        if metric and metric != "cosine":
            raise PineconeError(
                f"Pinecone index '{self.index_name}' uses metric {metric}; haive needs cosine."
            )
        return created

    @property
    def index(self) -> Any:
        if self._index is None:
            self._index = self.pc.Index(self.index_name)
        return self._index

    # ------------------------------------------------------------ VectorStore

    def upsert(self, chunks: list[Chunk]) -> None:
        for i in range(0, len(chunks), UPSERT_BATCH):
            batch = chunks[i : i + UPSERT_BATCH]
            vecs = embed_chunks(self.embedder, batch)
            self.index.upsert(
                vectors=[
                    {"id": c.id, "values": v, "metadata": clean_metadata(c)}
                    for c, v in zip(batch, vecs, strict=True)
                ]
            )

    def delete(self, ids: list[str]) -> None:
        for i in range(0, len(ids), 1000):
            self.index.delete(ids=ids[i : i + 1000])

    def search(self, query: str, k: int, filter: dict[str, Any] | None = None) -> list[Hit]:
        f = normalize_filter(filter)
        res = self.index.query(
            vector=self.embedder.embed_query(query), top_k=k, filter=f or None, include_metadata=True
        )
        matches = res["matches"] if isinstance(res, dict) else res.matches
        out = []
        for m in matches:
            mid = m["id"] if isinstance(m, dict) else m.id
            score = m["score"] if isinstance(m, dict) else m.score
            meta = (m["metadata"] if isinstance(m, dict) else m.metadata) or {}
            out.append(hit_from(mid, score, meta))
        return out

    def count(self, filter: dict[str, Any] | None = None) -> int:
        """Total, or per-repo via id prefix (serverless indexes can't count by metadata filter)."""
        f = normalize_filter(filter)
        if not f:
            stats = self.index.describe_index_stats()
            total = stats["total_vector_count"] if isinstance(stats, dict) else stats.total_vector_count
            return int(total)
        if set(f) != {"repo"}:
            raise ValueError("PineconeStore.count only supports a repo filter")
        repos = f["repo"].get("$in") or [f["repo"].get("$eq")]
        n = 0
        for repo in repos:
            for page in self.index.list(prefix=f"{repo}:"):
                n += len(page)
        return n
