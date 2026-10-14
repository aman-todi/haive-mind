"""Configuration: pydantic models, TOML load/save, secrets, and scope resolution."""

from __future__ import annotations

import difflib
import os
import tomllib
from pathlib import Path
from typing import Literal

import tomli_w
from pydantic import BaseModel, Field

from haive import paths


class RepoConfig(BaseModel):
    path: str
    remote_url: str | None = None

    @property
    def root(self) -> Path:
        return Path(os.path.expanduser(self.path)).resolve()


class IndexConfig(BaseModel):
    max_commits: int = 1000
    small_diff_lines: int = 200
    max_file_kb: int = 512
    extra_ignore: list[str] = Field(default_factory=lambda: ["*.ipynb"])


class VectorStoreConfig(BaseModel):
    backend: Literal["pinecone", "chroma", "memory"] = "pinecone"  # memory: tests only
    pinecone_index: str = "haive"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"


class EmbeddingsConfig(BaseModel):
    model: str = "jinaai/jina-embeddings-v2-base-code"
    dim: int = 768
    batch_size: int = 8  # lower it if indexing runs out of memory


class ModelsConfig(BaseModel):
    fast: str = "google/gemma-4-26b-a4b-it:free"
    strong: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    fallback: list[str] = Field(default_factory=lambda: ["openrouter/free"])
    base_url: str = "https://openrouter.ai/api/v1"
    provider: str = "openai"  # langchain model_provider; any OpenAI-compatible endpoint works
    api_key_env: str = "OPENROUTER_API_KEY"
    requests_per_minute: int = 20


class LimitsConfig(BaseModel):
    max_llm_calls_per_question: int = 12
    max_grade_loops: int = 2
    max_answer_retries: int = 2
    daily_call_budget: int = 1000


class WebConfig(BaseModel):
    mode: Literal["off", "ask", "auto"] = "ask"
    provider: Literal["auto", "tavily", "duckduckgo"] = "auto"
    max_results: int = 5


class Config(BaseModel):
    repos: dict[str, RepoConfig] = Field(default_factory=dict)
    groups: dict[str, list[str]] = Field(default_factory=dict)
    index: IndexConfig = Field(default_factory=IndexConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    web: WebConfig = Field(default_factory=WebConfig)

    # ------------------------------------------------------------ scope

    def repo_root(self, name: str) -> Path:
        return self.repos[name].root

    def expand(self, names: list[str]) -> list[str]:
        """Expand exact repo/group names to repo names (order-preserving, deduped)."""
        out: list[str] = []
        for n in names:
            members = self.groups.get(n, [n])
            for m in members:
                if m in self.repos and m not in out:
                    out.append(m)
        return out

    def resolve_names(self, names: list[str]) -> tuple[list[str], list[str]]:
        """Resolve user-supplied repo or group names, case-insensitively and fuzzily.

        Returns ``(repos, unknown)``. Group names expand to their member repos.
        """
        lookup: dict[str, str] = {}
        for n in list(self.repos) + list(self.groups):
            lookup[n.lower()] = n
        repos: list[str] = []
        unknown: list[str] = []
        for raw in names:
            key = raw.strip().lower()
            if not key:
                continue
            if key in ("all", "*"):
                return list(self.repos), []
            match = lookup.get(key)
            if match is None:
                # Substring match first ("valuation" -> "valuation-backend" only if unique).
                subs = [v for k, v in lookup.items() if key in k or k in key]
                if len(subs) == 1:
                    match = subs[0]
                else:
                    close = difflib.get_close_matches(key, list(lookup), n=2, cutoff=0.75)
                    if len(close) == 1 or (len(close) == 2 and _clear_winner(key, close)):
                        match = lookup[close[0]]
            if match is None:
                unknown.append(raw)
                continue
            for r in self.expand([match]):
                if r not in repos:
                    repos.append(r)
        return repos, unknown


def _clear_winner(key: str, close: list[str]) -> bool:
    a = difflib.SequenceMatcher(None, key, close[0]).ratio()
    b = difflib.SequenceMatcher(None, key, close[1]).ratio()
    return a - b > 0.1


# ---------------------------------------------------------------- load / save

TEMPLATE_HEADER = """\
# haive configuration. Secrets (OPENROUTER_API_KEY, PINECONE_API_KEY, TAVILY_API_KEY)
# belong in the environment or in .env next to this file, never here.
# Repos are added with `haive add`; groups are edited by hand:
#
# [groups]
# valuation = ["valuation-backend", "valuation-frontend"]

"""


def load_config(path: Path | None = None) -> Config:
    path = path or paths.config_file()
    if not path.exists():
        return Config()
    with path.open("rb") as f:
        data = tomllib.load(f)
    return Config.model_validate(data)


def save_config(cfg: Config, path: Path | None = None) -> Path:
    path = path or paths.config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.model_dump(exclude_none=True)
    path.write_text(TEMPLATE_HEADER + tomli_w.dumps(data))
    return path


def load_secrets() -> None:
    """Load ~/.config/haive/.env into the environment (existing env vars win)."""
    from dotenv import load_dotenv

    env = paths.env_file()
    if env.exists():
        load_dotenv(env, override=False)


def secret(name: str) -> str | None:
    val = os.environ.get(name)
    return val or None
