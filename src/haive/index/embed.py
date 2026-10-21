"""Embeddings. Default: fastembed running ``jinaai/jina-embeddings-v2-base-code`` locally (768-d).

``HashEmbedder`` is a deterministic, dependency-free stand-in for tests and offline runs.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import threading
import warnings
from collections.abc import Iterator
from pathlib import Path

from langchain_core.embeddings import Embeddings

from haive import paths
from haive.types import Embedder

# haive draws its own download progress (cli._load_embedding_model). The Hub's bars are switched off
# with its env var because fastembed re-enables them programmatically around each download; this
# must run before huggingface_hub is first imported. A user's own setting still wins.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", message="Cannot enable progress bars")

# Chunks are capped at ~1,500 tokens; cap embedding input too, so attention memory stays bounded.
MAX_EMBED_CHARS = 6000


def model_cache_dir() -> str:
    """Where the embedding model is downloaded: haive's data dir (fastembed's default is the system temp
    dir, which macOS clears periodically). ``FASTEMBED_CACHE_PATH`` still wins if set."""
    return os.environ.get("FASTEMBED_CACHE_PATH") or str(paths.data_dir() / "models")


def cache_bytes() -> int:
    """Bytes in the model cache. Xet-backed downloads put the large files in a ``blobs/`` dir shared
    across models at the cache root, so measure the whole cache, not the model's folder."""
    total = 0
    for root, _, files in os.walk(model_cache_dir()):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


class _HideAnonymousHubNotice(logging.Filter):
    """The Hugging Face Hub asks anonymous downloaders to set HF_TOKEN. haive downloads one public
    model once, so the notice is noise in the progress output. Other Hub warnings still show."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "unauthenticated requests" not in record.getMessage()


logging.getLogger("huggingface_hub.utils._http").addFilter(_HideAnonymousHubNotice())


class FastEmbedder(Embeddings):
    """LangChain ``Embeddings`` over fastembed (ONNX, runs locally)."""

    def __init__(
        self, model: str = "jinaai/jina-embeddings-v2-base-code", dim: int = 768, batch_size: int = 8
    ):
        self.model_name = model
        self.dim = dim
        self.batch_size = batch_size
        self._model = None
        # A question is searched once per repo in scope (in parallel); embed it once.
        self._queries: dict[str, list[float]] = {}
        self._query_lock = threading.Lock()

    @property
    def model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name, cache_dir=model_cache_dir())
        return self._model

    def is_downloaded(self) -> bool:
        name = self.model_name.split("/")[-1].lower()
        return any(
            p.name.lower().endswith(f"--{name}") and any(p.glob("snapshots/*"))
            for p in Path(model_cache_dir()).glob("models--*")
        )

    def download_size(self) -> int | None:
        """Expected bytes on disk, from fastembed's model list."""
        from fastembed import TextEmbedding

        for m in TextEmbedding.list_supported_models():
            if m["model"].lower() == self.model_name.lower() and m.get("size_in_GB"):
                return int(m["size_in_GB"] * 1e9)
        return None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        texts = [t[:MAX_EMBED_CHARS] for t in texts]
        return [v.tolist() for v in self.model.embed(texts, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        with self._query_lock:
            if text not in self._queries:
                if len(self._queries) > 64:
                    self._queries.clear()
                self._queries[text] = next(iter(self.model.query_embed(text[:MAX_EMBED_CHARS]))).tolist()
            return self._queries[text]


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
