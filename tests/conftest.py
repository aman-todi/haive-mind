from __future__ import annotations

from pathlib import Path

import pytest

from haive.config import Config, RepoConfig
from haive.db import DB
from haive.index.embed import HashEmbedder
from haive.store.base import MemoryStore
from tests.fixtures.repo_builder import build_repo, build_small_repo


@pytest.fixture(autouse=True)
def haive_home(tmp_path, monkeypatch) -> Path:
    home = tmp_path / "haive_home"
    monkeypatch.setenv("HAIVE_HOME", str(home))
    for k in ("OPENROUTER_API_KEY", "PINECONE_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    return home


@pytest.fixture(scope="session")
def _repo_template(tmp_path_factory) -> Path:
    return build_repo(tmp_path_factory.mktemp("tpl") / "valuation")


@pytest.fixture
def sample_repo(tmp_path, _repo_template) -> Path:
    """A fresh copy of the sample repo (tests may modify it)."""
    import shutil

    dest = tmp_path / "valuation"
    shutil.copytree(_repo_template, dest, symlinks=True)
    return dest


@pytest.fixture
def other_repo(tmp_path) -> Path:
    return build_small_repo(tmp_path / "frontend", "frontend")


@pytest.fixture
def cfg(sample_repo) -> Config:
    c = Config()
    c.repos["valuation"] = RepoConfig(path=str(sample_repo))
    c.embeddings.model = "hash"
    c.vector_store.backend = "memory"
    c.models.requests_per_minute = 0
    return c


@pytest.fixture
def db() -> DB:
    return DB(":memory:")


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder(768)


@pytest.fixture
def store(embedder) -> MemoryStore:
    return MemoryStore(embedder)
