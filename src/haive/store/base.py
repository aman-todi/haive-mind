"""Vector store interface, filter normalization, factory, and an in-memory reference store.

Filters use ``{"repo": {"$in": [...]}, "kind": {"$in": [...]}}``; several keys are ANDed.
Pinecone accepts that implicitly; Chroma needs an explicit ``$and``. ``normalize_filter`` turns
plain values and lists into ``$eq`` / ``$in`` and drops empty clauses.
"""

from __future__ import annotations

import math
from typing import Any

from haive.types import Chunk, Embedder, Hit, VectorStore

MAX_METADATA_TEXT = 30_000  # bytes; Pinecone allows 40 KB of metadata per vector
UPSERT_BATCH = 100


def normalize_filter(f: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (f or {}).items():
        if v is None:
            continue
        if isinstance(v, dict):
            if "$in" in v and not v["$in"]:
                continue
            out[k] = v
        elif isinstance(v, (list, tuple, set)):
            vals = list(v)
            if vals:
                out[k] = {"$in": vals}
        else:
            out[k] = {"$eq": v}
    return out


def matches(meta: dict[str, Any], f: dict[str, Any]) -> bool:
    for k, cond in f.items():
        val = meta.get(k)
        if "$in" in cond and val not in cond["$in"]:
            return False
        if "$eq" in cond and val != cond["$eq"]:
            return False
        if "$ne" in cond and val == cond["$ne"]:
            return False
    return True


def clean_metadata(chunk: Chunk) -> dict[str, Any]:
    meta = chunk.metadata()
    text = meta["text"].encode("utf-8")
    if len(text) > MAX_METADATA_TEXT:
        meta["text"] = text[:MAX_METADATA_TEXT].decode("utf-8", "ignore")
    return {k: v for k, v in meta.items() if v is not None}


def hit_from(id_: str, score: float, meta: dict[str, Any]) -> Hit:
    return Hit(
        id=id_,
        score=float(score),
        repo=meta.get("repo", ""),
        path=meta.get("path", ""),
        kind=meta.get("kind", "code"),
        language=meta.get("language", ""),
        symbol=meta.get("symbol", ""),
        start_line=int(meta.get("start_line", 0) or 0),
        end_line=int(meta.get("end_line", 0) or 0),
        text=meta.get("text", ""),
    )


class MemoryStore:
    """In-process cosine store. Used by tests and as the reference implementation."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.vectors: dict[str, tuple[list[float], dict[str, Any]]] = {}

    def upsert(self, chunks: list[Chunk]) -> None:
        from haive.index.embed import embed_chunks

        for c, v in zip(chunks, embed_chunks(self.embedder, chunks), strict=True):
            self.vectors[c.id] = (v, clean_metadata(c))

    def delete(self, ids: list[str]) -> None:
        for i in ids:
            self.vectors.pop(i, None)

    def search(self, query: str, k: int, filter: dict[str, Any] | None = None) -> list[Hit]:
        f = normalize_filter(filter)
        q = self.embedder.embed_query(query)
        scored = []
        for id_, (v, meta) in self.vectors.items():
            if f and not matches(meta, f):
                continue
            dot = sum(a * b for a, b in zip(q, v, strict=False))
            nq = math.sqrt(sum(a * a for a in q)) or 1.0
            nv = math.sqrt(sum(b * b for b in v)) or 1.0
            scored.append((dot / (nq * nv), id_, meta))
        scored.sort(key=lambda t: -t[0])
        return [hit_from(i, s, m) for s, i, m in scored[:k]]

    def count(self, filter: dict[str, Any] | None = None) -> int:
        f = normalize_filter(filter)
        return sum(1 for _, meta in self.vectors.values() if not f or matches(meta, f))


def make_store(cfg: Any, embedder: Embedder, *, backend: str | None = None) -> VectorStore:
    """Build the configured store. ``cfg`` is a ``haive.config.Config``."""
    backend = backend or cfg.vector_store.backend
    if backend == "pinecone":
        from haive.store.pinecone_store import PineconeStore

        vs = cfg.vector_store
        return PineconeStore(embedder, vs.pinecone_index, vs.pinecone_cloud, vs.pinecone_region)
    if backend == "chroma":
        from haive import paths
        from haive.store.chroma_store import ChromaStore

        return ChromaStore(embedder, path=str(paths.chroma_dir()))
    if backend == "memory":
        return MemoryStore(embedder)
    raise ValueError(f"unknown vector store backend: {backend}")
