"""Shared types and interfaces. This module is the frozen contract between components.

Everything that crosses a module boundary (index -> store, tools -> graph, graph -> render)
is defined here. Graph state stores these as plain dicts (``model_dump()``) so checkpoints
stay JSON-friendly; convert back with ``Model.model_validate``.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

ChunkKind = Literal["code", "config", "doc", "commit"]
Intent = Literal["lookup", "recall", "history", "debug", "compare", "architecture", "general"]
INTENTS: tuple[str, ...] = ("lookup", "recall", "history", "debug", "compare", "architecture", "general")
WebMode = Literal["off", "ask", "auto"]
Source = Literal["repo", "web", "model"]
Confidence = Literal["low", "med", "high"]
GradeStatus = Literal["sufficient", "missing_symbol", "need_context", "need_external", "nothing_relevant"]


def sha1_12(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:12]


# ---------------------------------------------------------------- indexing


class Chunk(BaseModel):
    """A unit of indexed text. ``text`` is the raw chunk; the embedding uses ``embed_text``."""

    id: str
    repo: str
    path: str
    kind: ChunkKind
    language: str = ""
    symbol: str = ""
    start_line: int = 1
    end_line: int = 1
    text: str
    content_hash: str = ""
    context: str = ""  # extra embedding-only header, e.g. the enclosing class signature

    @classmethod
    def make(cls, repo: str, path: str, kind: ChunkKind, text: str, **kw: Any) -> Chunk:
        h = sha1_12(text)
        return cls(id=f"{repo}:{path}:{h}", repo=repo, path=path, kind=kind, text=text, content_hash=h, **kw)

    @property
    def embed_text(self) -> str:
        header = f"{self.repo}/{self.path}"
        if self.symbol:
            header += f" :: {self.symbol}"
        if self.context:
            header += f"\n{self.context}"
        return f"{header}\n{self.text}"

    def metadata(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "path": self.path,
            "kind": self.kind,
            "language": self.language,
            "symbol": self.symbol,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "text": self.text,
        }


class Hit(BaseModel):
    """A vector-search result."""

    id: str
    score: float
    repo: str
    path: str
    kind: ChunkKind
    language: str = ""
    symbol: str = ""
    start_line: int = 1
    end_line: int = 1
    text: str = ""


class CommitInfo(BaseModel):
    repo: str
    sha: str
    date: str
    author: str
    subject: str
    body: str = ""
    files: list[str] = Field(default_factory=list)
    insertions: int = 0
    deletions: int = 0
    diff: str = ""


class RepoProfile(BaseModel):
    repo: str
    languages: dict[str, int] = Field(default_factory=dict)  # language -> line count
    frameworks: list[str] = Field(default_factory=list)
    dependencies: dict[str, list[str]] = Field(default_factory=dict)  # manifest path -> deps
    entrypoints: list[str] = Field(default_factory=list)
    compose_services: list[dict[str, Any]] = Field(default_factory=list)  # name, image, ports, env, file
    workflows: list[dict[str, Any]] = Field(default_factory=list)  # file, name, triggers
    terraform_resources: list[str] = Field(default_factory=list)
    env_vars: list[str] = Field(default_factory=list)
    routes_defined: list[dict[str, Any]] = Field(default_factory=list)  # method, path, file, line
    urls_called: list[dict[str, Any]] = Field(default_factory=list)  # url, file, line


# ---------------------------------------------------------------- tools


class Match(BaseModel):
    """A ripgrep (or fallback) match."""

    repo: str
    path: str
    line: int
    text: str


class WebResult(BaseModel):
    url: str
    title: str = ""
    snippet: str = ""


class Frame(BaseModel):
    path: str
    line: int | None = None
    col: int | None = None
    function: str = ""
    external: bool = False
    repo: str | None = None  # set by pathmap when the frame maps to an indexed file
    repo_path: str | None = None


class ParsedTrace(BaseModel):
    format: str  # python | node | java | go | rust | dotnet | native | compiler | generic | unknown
    error_type: str = ""
    message: str = ""
    frames: list[Frame] = Field(default_factory=list)
    raw: str = ""

    def internal_frames(self) -> list[Frame]:
        return [f for f in self.frames if not f.external and f.repo]


# ---------------------------------------------------------------- graph


class Evidence(BaseModel):
    id: str
    source: Literal["code", "commit", "tool", "web"]
    repo: str | None = None
    path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    text: str
    why: str = ""


class RepoRef(BaseModel):
    repo: str
    path: str
    start_line: int
    end_line: int
    symbol: str | None = None


class WebRef(BaseModel):
    url: str


class Claim(BaseModel):
    text: str
    source: Source
    refs: list[RepoRef | WebRef] = Field(default_factory=list)
    unverified: bool = False  # set by finalize when verification never passed


class Section(BaseModel):
    heading: str | None = None
    claims: list[Claim] = Field(default_factory=list)


class SuggestedFix(BaseModel):
    diff: str
    approximate: bool = False


class Hypothesis(BaseModel):
    title: str
    confidence: Confidence = "low"
    claims: list[Claim] = Field(default_factory=list)


class Answer(BaseModel):
    summary: str
    sections: list[Section] = Field(default_factory=list)
    suggested_fix: SuggestedFix | None = None
    hypotheses: list[Hypothesis] | None = None
    followups: list[str] = Field(default_factory=list)

    def all_claims(self) -> list[Claim]:
        out = [c for s in self.sections for c in s.claims]
        for h in self.hypotheses or []:
            out.extend(h.claims)
        return out


# ---------------------------------------------------------------- protocols


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Filters use a small Mongo-like dialect: ``{"repo": {"$in": [...]}, "kind": {"$in": [...]}}``.
    Multiple keys are ANDed. Adapters normalize this to their backend."""

    def upsert(self, chunks: list[Chunk]) -> None: ...

    def delete(self, ids: list[str]) -> None: ...

    def search(self, query: str, k: int, filter: dict[str, Any] | None = None) -> list[Hit]: ...

    def count(self, filter: dict[str, Any] | None = None) -> int: ...


@runtime_checkable
class ChatModel(Protocol):
    """The subset of a LangChain chat model that haive uses."""

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any: ...
