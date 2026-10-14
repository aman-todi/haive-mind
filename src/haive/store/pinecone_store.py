"""Pinecone adapter (default backend): ``langchain_pinecone.PineconeVectorStore`` for vectors, plus the
``pinecone`` client for index creation, checks, and counts."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.documents import Document

from haive.store.base import LangChainStore, normalize_filter
from haive.types import Embedder


class PineconeError(RuntimeError):
    pass


class PineconeStore(LangChainStore):
    def __init__(
        self,
        embedder: Embedder,
        index_name: str = "haive",
        cloud: str = "aws",
        region: str = "us-east-1",
        *,
        client: Any = None,
        api_key: str | None = None,
    ):
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
        self._vs = None

    @property
    def vs(self):  # created lazily so `haive init` can create the index first
        if self._vs is None:
            from langchain_pinecone import PineconeVectorStore

            self._vs = PineconeVectorStore(index=self.index, embedding=self.embedder, text_key="text")
        return self._vs

    @property
    def index(self) -> Any:
        return self.pc.Index(self.index_name)

    def ensure_index(self) -> bool:
        """Create the index if missing and check its dimension. Returns True if it was created."""
        created = False
        if not self.pc.has_index(self.index_name):
            from pinecone import ServerlessSpec

            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedder.dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )
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

    def _scored(self, query: str, k: int, filter: Any) -> list[tuple[Document, float]]:
        return self.vs.similarity_search_with_score(query, k=k, filter=filter)  # cosine similarity

    def count(self, filter: dict[str, Any] | None = None) -> int:
        """Total, or per-repo via id prefix (serverless indexes can't count by metadata filter)."""
        f = normalize_filter(filter)
        index = self.index
        if not f:
            stats = index.describe_index_stats()
            total = stats["total_vector_count"] if isinstance(stats, dict) else stats.total_vector_count
            return int(total)
        if set(f) != {"repo"}:
            raise ValueError("PineconeStore.count only supports a repo filter")
        repos = f["repo"].get("$in") or [f["repo"].get("$eq")]
        return sum(len(page) for repo in repos for page in index.list(prefix=f"{repo}:"))
