"""Embeddings. Default: fastembed running ``jinaai/jina-embeddings-v2-base-code`` locally (768-d).

``HashEmbedder`` is a deterministic, dependency-free stand-in for tests and offline runs.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterator

from langchain_core.embeddings import Embeddings

from haive.types import Embedder

# Chunks are capped at ~1,500 tokens; cap embedding input too, so attention memory stays bounded.
MAX_EMBED_CHARS = 6000


class FastEmbedder(Embeddings):
    """LangChain ``Embeddings`` over fastembed (ONNX, runs locally)."""

    def __init__(
        self, model: str = "jinaai/jina-embeddings-v2-base-code", dim: int = 768, batch_size: int = 8
    ):
        self.model_name = model
        self.dim = dim
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        texts = [t[:MAX_EMBED_CHARS] for t in texts]
        return [v.tolist() for v in self.model.embed(texts, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed(text[:MAX_EMBED_CHARS]))).tolist()


_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def _tokens(text: str) -> Iterator[str]:
    for tok in _TOKEN.findall(text):
        low = tok.lower()
        yield low
        # split camelCase and snake_case so "getUser" matches "get user"
        for part in re.findall(r"[a-z]+|[A-Z][a-z]*|\d+", tok):
            if part.lower() != low:
                yield part.lower()


class HashEmbedder(Embeddings):
    """Bag-of-tokens feature hashing. Deterministic; good enough for tests and ablations."""

    def __init__(self, dim: int = 768):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in _tokens(text):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def make_embedder(model: str, dim: int, batch_size: int = 8) -> Embedder:
    if model == "hash":
        return HashEmbedder(dim)
    return FastEmbedder(model, dim, batch_size)
