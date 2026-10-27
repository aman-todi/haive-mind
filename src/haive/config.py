"""Configuration: pydantic models, TOML load/save, secrets, and scope resolution."""

from __future__ import annotations

import difflib
import json
import os
import tomllib
from pathlib import Path
from typing import Any, ClassVar, Literal

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
    max_commits: int = Field(
        300, description="newest commits indexed per repo (messages and file lists; git covers the rest)"
    )
    max_file_kb: int = 512
    extra_ignore: list[str] = Field(
        default_factory=lambda: ["*.ipynb"], description="glob patterns never indexed, on top of .gitignore"
    )


class VectorStoreConfig(BaseModel):
    backend: Literal["pinecone", "chroma", "memory"] = Field(
        "pinecone", description="pinecone | chroma (local); changing it needs a full re-index"
    )
    pinecone_index: str = "haive"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"


class EmbeddingsConfig(BaseModel):
    model: str = Field("jinaai/jina-embeddings-v2-base-code", description="changing it needs a full re-index")
    dim: int = 768
    batch_size: int = Field(8, description="lower it if indexing runs out of memory")


class ModelsConfig(BaseModel):
    """One model answers every step of a question (router, grader, expand, answer, ...). Each tier is
    an ordered list: the first model that works is used, the rest are fallbacks."""

    tier: Literal["strong", "ultra"] = Field(
        "strong", description="strong | ultra (bigger, slower models); per command: --strong / --ultra"
    )
    # On the author's suite paid answers in ~15-20 s (median) for well under a cent a question; free
    # is ~5x slower (~90 s) at the same accuracy (README, Models).
    use_paid: bool = Field(
        True,
        description="paid models (~15-20 s, under a cent a question) or free (~90 s); per command: --paid / --free",
    )
    strong: list[str] = Field(
        default_factory=lambda: [
            "openrouter/free",  # OpenRouter picks an available free model
            "nvidia/nemotron-3.5-super-vl:free",
            "xingchen-agi/xing4.0-29b-a4b:free",
        ]
    )
    ultra: list[str] = Field(
        default_factory=lambda: [
            "nvidia/nemotron-3-ultra-550b-a55b:free",
            "thinkingmachines/inkling:free",
            "poolside/laguna-s-2.1:free",
            "openrouter/free",
        ]
    )
    strong_paid: list[str] = Field(
        default_factory=lambda: [
            "deepseek/deepseek-v4.1-flash-1017",
            "openai/gpt-oss-120b",
            "openrouter/auto",
        ]
    )
    ultra_paid: list[str] = Field(
        default_factory=lambda: ["deepseek/deepseek-v4-pro-0813", "z-ai/glm-5.2", "openrouter/auto"]
    )
    base_url: str = "https://openrouter.ai/api/v1"
    provider: str = "openai"  # langchain model_provider; any OpenAI-compatible endpoint works
    api_key_env: str = "OPENROUTER_API_KEY"
    # OpenRouter provider routing for paid models (the request's `provider` object). A paid model is
    # served by many providers, some of which stall for minutes. Providers that start answering within
    # 10 s (p90) and stream at least 50 tokens/s (median) over the last 5 minutes are preferred,
    # cheapest first; the rest are tried after them, so a request still runs. {} restores OpenRouter's
    # default (price-weighted load balancing).
    routing: dict[str, Any] = Field(
        default_factory=lambda: {
            "sort": "price",
            "preferred_max_latency": {"p90": 10},
            "preferred_min_throughput": {"p50": 50},
        }
    )
    requests_per_minute: int = 20  # client-side throttle for free models (OpenRouter allows ~20/min)
    max_retries: int = 1  # per model, by the OpenAI client (honors Retry-After), before falling back
    # Past a wall-clock limit, a model's attempt at a call is abandoned and the next model in the list
    # takes over for the rest of the question.
    timeout_seconds: int = Field(
        125, description="limit for an answer call (normally 5-60 s); also the client's per-read timeout"
    )
    step_timeout_seconds: int = Field(
        30, description="limit for every other call: router, grader, rewrite, expand, titles (normally 1-5 s)"
    )

    TIERS: ClassVar[tuple[str, ...]] = ("strong", "ultra", "strong_paid", "ultra_paid")

    @property
    def tier_name(self) -> str:
        return f"{self.tier}{'_paid' if self.use_paid else ''}"

    def chain(self) -> list[str]:
        return list(getattr(self, self.tier_name))


class LimitsConfig(BaseModel):
    max_llm_calls_per_question: int = Field(25, description="hard cap per question")
    max_grade_loops: int = 2
    max_answer_retries: int = 2
    daily_call_budget: int = Field(
        1000, description="haive warns at 90% and stops at 100% (--force overrides)"
    )
    max_claims: int = Field(
        10, description="claims per answer, asked of the model and shown (--all shows any extra)"
    )
    expand_with_tools: bool = Field(
        True, description="the model picks expand tools (false: deterministic heuristics)"
    )


class WebConfig(BaseModel):
    mode: Literal["off", "ask", "auto"] = Field(
        "ask", description="off | ask (before searching) | auto; per command: --web"
    )
    provider: Literal["auto", "tavily", "duckduckgo"] = Field(
        "auto", description="tavily if TAVILY_API_KEY is set, else duckduckgo"
    )
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
# Settings shown commented out (# key = value) are the defaults: uncomment a line to change it.
# Defaults left commented update with haive. Repos are added with `haive add`; groups are edited by
# hand:
#
# [groups]
# valuation = ["valuation-backend", "valuation-frontend"]

"""

# Sections written after repos/groups, most-used first. The embedding model and vector store are
# always written out: the index was built with them, and a new default would silently mismatch it.
SECTIONS = ("models", "limits", "web", "index", "embeddings", "vector_store")
PINNED = ("embeddings", "vector_store")


def toml_value(v: object) -> str:
    """A TOML value on one line (inline tables for dicts)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int | float):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)  # a TOML basic string for anything JSON escapes
    if isinstance(v, list):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {toml_value(x)}" for k, x in v.items()) + " }" if v else "{}"
    raise TypeError(f"can't write {type(v).__name__} to TOML")


def _section(name: str, current: BaseModel) -> str:
    default = type(current)()
    lines = [f"[{name}]"]
    for key, info in type(current).model_fields.items():
        value = getattr(current, key)
        if info.description:
            lines.append(f"# {info.description}")
        set_ = name in PINNED or value != getattr(default, key)
        lines.append(("" if set_ else "# ") + f"{key} = {toml_value(value)}")
    return "\n".join(lines)


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
    # Settings left at their defaults are written commented out, so they're visible but a changed
    # default (a model list, a limit) still reaches existing configs.
    head = tomli_w.dumps(
        {k: v for k, v in cfg.model_dump(include={"repos", "groups"}, exclude_none=True).items() if v}
    )
    body = "\n\n".join(_section(n, getattr(cfg, n)) for n in SECTIONS)
    path.write_text(TEMPLATE_HEADER + (head + "\n" if head else "") + body + "\n")
    return path


def load_secrets() -> None:
    """Load the .env next to config.toml into the environment (existing env vars win)."""
    from dotenv import load_dotenv

    env = paths.env_file()
    if env.exists():
        load_dotenv(env, override=False)


def secret(name: str) -> str | None:
    val = os.environ.get(name)
    return val or None
