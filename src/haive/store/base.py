"""Vector stores: a thin adapter over LangChain ``VectorStore``s.

haive's code talks to ``upsert / delete / search / count`` (the ``VectorStore`` protocol in
``types.py``); each backend wraps a LangChain store: ``langchain_pinecone.PineconeVectorStore``,
``langchain_chroma.Chroma``, or ``langchain_core.vectorstores.InMemoryVectorStore`` (tests).

Chunks become ``Document``s whose ``page_content`` is the *embedding* text (a ``repo/path :: symbol``
location header, then the raw chunk), so the vector carries location context. The header line
count is stored in metadata and stripped on the way out, so hits return the raw chunk text.

Filters use ``{"repo": {"$in": [...]}, "kind": {"$in": [...]}}``; several keys are ANDed. Pinecone
accepts that implicitly; Chroma needs an explicit ``$and``; the in-memory store takes a callable.
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.vectorstores import VectorStore as LCVectorStore

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


def to_document(chunk: Chunk) -> Document:
    content = chunk.embed_text
    encoded = content.encode("utf-8")
    if len(encoded) > MAX_METADATA_TEXT:
        content = encoded[:MAX_METADATA_TEXT].decode("utf-8", "ignore")
    meta = {k: v for k, v in chunk.metadata().items() if k != "text" and v is not None}
    meta["header_lines"] = chunk.embed_text[: len(chunk.embed_text) - len(chunk.text)].count("\n")
    return Document(id=chunk.id, page_content=content, metadata=meta)


def to_hit(doc: Document, score: float, id_: str | None = None) -> Hit:
    meta = doc.metadata or {}
    skip = int(meta.get("header_lines", 0) or 0)
    text = "\n".join(doc.page_content.split("\n")[skip:])
    return Hit(
        id=id_ or doc.id or "",
        score=float(score),
        repo=meta.get("repo", ""),
        path=meta.get("path", ""),
        kind=meta.get("kind", "code"),
        language=meta.get("language", ""),
        symbol=meta.get("symbol", ""),
        start_line=int(meta.get("start_line", 0) or 0),
        end_line=int(meta.get("end_line", 0) or 0),
        text=text,
    )


class LangChainStore:
    """Adapter from a LangChain VectorStore to haive's ``VectorStore`` protocol."""

    vs: LCVectorStore

    def __init__(self, vs: LCVectorStore):
        self.vs = vs

    def upsert(self, chunks: list[Chunk]) -> None:
        for i in range(0, len(chunks), UPSERT_BATCH):
            batch = chunks[i : i + UPSERT_BATCH]
            self.vs.add_documents([to_document(c) for c in batch], ids=[c.id for c in batch])

    def delete(self, ids: list[str]) -> None:
        if ids:
            self.vs.delete(ids=ids)

    def _filter(self, f: dict[str, Any]) -> Any:
        return f or None

    def _scored(self, query: str, k: int, filter: Any) -> list[tuple[Document, float]]:
        """(document, similarity in [0, 1]) pairs; higher is better."""
        return self.vs.similarity_search_with_relevance_scores(query, k=k, filter=filter)

    def search(self, query: str, k: int, filter: dict[str, Any] | None = None) -> list[Hit]:
        f = normalize_filter(filter)
        return [to_hit(doc, score) for doc, score in self._scored(query, k, self._filter(f))]

    def count(self, filter: dict[str, Any] | None = None) -> int:
        raise NotImplementedError


class MemoryStore(LangChainStore):
    """``InMemoryVectorStore`` (cosine). Used by tests and as the reference implementation."""

    def __init__(self, embedder: Embedder):
        super().__init__(InMemoryVectorStore(embedding=embedder))
        self.embedder = embedder

    def _filter(self, f: dict[str, Any]) -> Any:
        return (lambda doc: matches(doc.metadata, f)) if f else None

    def _scored(self, query: str, k: int, filter: Any) -> list[tuple[Document, float]]:
        return self.vs.similarity_search_with_score(query, k=k, filter=filter)  # already cosine similarity

    def count(self, filter: dict[str, Any] | None = None) -> int:
        f = normalize_filter(filter)
        return sum(1 for d in self.vs.store.values() if not f or matches(d["metadata"], f))


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
