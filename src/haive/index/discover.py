"""File discovery: what gets indexed, and how each file is classified.

Uses ``git ls-files`` (tracked + untracked-but-not-ignored), so ``.gitignore`` is always respected.
"""

from __future__ import annotations

import fnmatch
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from haive.tools import git

LOCKFILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "uv.lock", "poetry.lock", "Cargo.lock",
    "Gemfile.lock", "composer.lock", "go.sum", "Pipfile.lock", "bun.lockb", "packages.lock.json",
    "mix.lock", "flake.lock", "pubspec.lock", "Podfile.lock",
}  # fmt: skip
GENERATED_GLOBS = ["*.min.js", "*.min.css", "*.map", "*.pyc", "*.bundle.js", "*_pb2.py", "*.pb.go", "*.snap"]
GENERATED_DIRS = {"dist", "build", "node_modules", ".next", "out", "target", "vendor", "__pycache__", ".venv"}
BINARY_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp", ".tiff", ".pdf", ".zip", ".gz", ".tgz", ".bz2",
    ".xz", ".7z", ".rar", ".jar", ".war", ".class", ".so", ".dylib", ".dll", ".exe", ".bin", ".o", ".a",
    ".woff", ".woff2", ".ttf", ".otf", ".eot", ".mp3", ".mp4", ".mov", ".wav", ".ogg", ".webm", ".sqlite",
    ".db", ".parquet", ".pkl", ".npy", ".npz", ".onnx", ".pt", ".h5", ".psd", ".xlsx", ".docx", ".pptx",
}  # fmt: skip

# Secrets denylist: never indexed, never read into an LLM prompt.
SECRET_ALLOW = {".env.example", ".env.sample", ".env.template"}
SECRET_GLOBS = ["*.pem", "*.key", "id_rsa*", "id_dsa*", "id_ecdsa*", "id_ed25519*", "*.p12", "*.pfx",
                "credentials*", "secrets*", "*.keystore", "*.jks", ".npmrc", ".pypirc", ".netrc"]  # fmt: skip

SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id
    re.compile(r"(?i)aws_secret_access_key\s*[=:]\s*['\"]?[A-Za-z0-9/+=]{40}"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"),  # OpenAI / OpenRouter / Anthropic style
    re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}"),  # GitHub tokens
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),  # Slack
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}"),  # Google API key
    re.compile(r"\bpcsk_[A-Za-z0-9_]{20,}"),  # Pinecone
    re.compile(r"\btvly-[A-Za-z0-9]{20,}"),  # Tavily
]

EXT_LANG = {
    ".py": "python", ".pyi": "python", ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".jsx": "javascript", ".ts": "typescript", ".mts": "typescript", ".cts": "typescript", ".tsx": "tsx",
    ".go": "go", ".rs": "rust", ".java": "java", ".kt": "kotlin", ".kts": "kotlin", ".c": "c", ".h": "c",
    ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp", ".cs": "csharp",
    ".rb": "ruby", ".php": "php", ".swift": "swift", ".scala": "scala", ".sh": "bash", ".bash": "bash",
    ".zsh": "bash", ".sql": "sql", ".md": "markdown", ".mdx": "markdown", ".rst": "rst", ".txt": "text",
    ".yaml": "yaml", ".yml": "yaml", ".json": "json", ".toml": "toml", ".tf": "hcl", ".hcl": "hcl",
    ".tfvars": "hcl", ".ini": "ini", ".cfg": "ini", ".conf": "ini", ".xml": "xml", ".html": "html",
    ".css": "css", ".scss": "css", ".vue": "vue", ".svelte": "svelte", ".lua": "lua", ".r": "r",
    ".dart": "dart", ".ex": "elixir", ".exs": "elixir", ".csproj": "xml", ".gradle": "groovy",
    ".properties": "ini", ".env": "dotenv", ".proto": "proto", ".graphql": "graphql", ".gql": "graphql",
}  # fmt: skip
CODE_LANGS = {
    "python", "javascript", "typescript", "tsx", "go", "rust", "java", "kotlin", "c", "cpp", "csharp", "ruby",
    "php", "swift", "scala", "bash", "sql", "lua", "r", "dart", "elixir", "vue", "svelte", "groovy",
    "html", "css", "proto", "graphql",
}  # fmt: skip
DOC_LANGS = {"markdown", "rst", "text"}
NAME_LANG = {
    "Dockerfile": "dockerfile", "Makefile": "make", "Jenkinsfile": "groovy", "Procfile": "text",
    "docker-compose.yml": "compose", "docker-compose.yaml": "compose", "compose.yml": "compose",
    "compose.yaml": "compose", "README": "text", "LICENSE": "text",
}  # fmt: skip


@dataclass
class FileInfo:
    path: str  # repo-relative, posix
    language: str
    kind: str  # code | config | doc
    size: int


def classify(path: str) -> tuple[str, str]:
    """Return ``(language, kind)`` for a repo-relative path."""
    p = PurePosixPath(path)
    name = p.name
    lang = NAME_LANG.get(name)
    if lang is None and (name.startswith("Dockerfile") or name.endswith(".Dockerfile")):
        lang = "dockerfile"
    if lang is None and name.startswith(".env"):
        lang = "dotenv"
    if lang is None:
        lang = EXT_LANG.get(p.suffix.lower(), "text" if not p.suffix else p.suffix.lower().lstrip("."))
    if lang in CODE_LANGS:
        kind = "code"
    elif lang in DOC_LANGS:
        kind = "doc"
    else:
        kind = "config"
    return lang, kind


def is_secret_path(path: str) -> bool:
    name = PurePosixPath(path).name
    if name in SECRET_ALLOW:
        return False
    if name.startswith(".env"):
        return True
    return any(fnmatch.fnmatch(name.lower(), g) for g in SECRET_GLOBS)


def has_secret(text: str) -> bool:
    return any(p.search(text) for p in SECRET_PATTERNS)


def skip_reason(path: str, size: int, max_file_kb: int, extra_ignore: list[str]) -> str | None:
    p = PurePosixPath(path)
    name = p.name
    if is_secret_path(path):
        return "secret"
    if name in LOCKFILES:
        return "lockfile"
    if p.suffix.lower() in BINARY_EXT:
        return "binary"
    if any(part in GENERATED_DIRS for part in p.parts[:-1]):
        return "generated"
    if any(fnmatch.fnmatch(name, g) for g in GENERATED_GLOBS):
        return "generated"
    if any(fnmatch.fnmatch(path, g) or fnmatch.fnmatch(name, g) for g in extra_ignore):
        return "ignored"
    if size > max_file_kb * 1024:
        return "too_large"
    return None


def looks_binary(data: bytes) -> bool:
    return b"\0" in data[:8192]


def discover(root: Path, max_file_kb: int = 512, extra_ignore: list[str] | None = None) -> list[FileInfo]:
    """List indexable files in a repo (relative posix paths), sorted."""
    extra_ignore = extra_ignore or []
    out: list[FileInfo] = []
    for rel in git.ls_files(root):
        full = root / rel
        try:
            if not full.is_file() or full.is_symlink():
                continue
            size = full.stat().st_size
        except OSError:
            continue
        if skip_reason(rel, size, max_file_kb, extra_ignore):
            continue
        lang, kind = classify(rel)
        out.append(FileInfo(rel, lang, kind, size))
    out.sort(key=lambda f: f.path)
    return out


def read_text(root: Path, rel: str) -> str | None:
    """Read a file as text, or None for binaries/unreadable files. Never reads denylisted secrets."""
    if is_secret_path(rel):
        return None
    try:
        data = (root / rel).read_bytes()
    except OSError:
        return None
    if looks_binary(data):
        return None
    return data.decode("utf-8", "replace")


def sha256_file(root: Path, rel: str) -> str | None:
    try:
        return hashlib.sha256((root / rel).read_bytes()).hexdigest()
    except OSError:
        return None
