"""Structural chunking for config and docs, plus the per-file dispatcher.

Markdown by heading; YAML/JSON/TOML by top-level key; docker-compose per service; Dockerfile
whole; Terraform/HCL per block; SQL per statement; shell per function. Fallback: 60-line windows
with 10 lines of overlap. Chunks containing obvious secrets are dropped.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from haive.index.chunk_code import CHARS_PER_TOKEN, MAX_TOKENS, SPECS, chunk_code, window_chunks
from haive.index.discover import has_secret
from haive.types import Chunk

Boundary = tuple[int, str]  # (0-based line index, symbol)


def _segments(
    repo: str, path: str, lines: list[str], kind: str, language: str, bounds: list[Boundary],
    preamble: str = "<preamble>",
) -> list[Chunk]:  # fmt: skip
    if not bounds:
        return _whole_or_windows(repo, path, lines, kind, language, "")
    bounds = sorted(bounds)
    out: list[Chunk] = []
    if bounds[0][0] > 0:
        out += _piece(repo, path, lines, kind, language, 0, bounds[0][0] - 1, preamble)
    for i, (start, sym) in enumerate(bounds):
        end = bounds[i + 1][0] - 1 if i + 1 < len(bounds) else len(lines) - 1
        out += _piece(repo, path, lines, kind, language, start, end, sym)
    return out


def _piece(repo, path, lines, kind, language, s, e, symbol) -> list[Chunk]:
    while e > s and not lines[e].strip():
        e -= 1
    seg = lines[s : e + 1]
    text = "\n".join(seg)
    if not text.strip():
        return []
    if len(text) // CHARS_PER_TOKEN > MAX_TOKENS:
        return window_chunks(repo, path, seg, kind, language, start=s + 1, symbol=symbol)
    return [
        Chunk.make(repo, path, kind, text, language=language, symbol=symbol, start_line=s + 1, end_line=e + 1)
    ]


def _whole_or_windows(repo, path, lines, kind, language, symbol) -> list[Chunk]:
    return _piece(repo, path, lines, kind, language, 0, len(lines) - 1, symbol)


# ---------------------------------------------------------------- per format


def markdown_bounds(lines: list[str]) -> list[Boundary]:
    bounds: list[Boundary] = []
    stack: list[tuple[int, str]] = []
    fence = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("```", "~~~")):
            fence = not fence
            continue
        m = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if m and not fence:
            level, title = len(m.group(1)), m.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            bounds.append((i, " > ".join(t for _, t in stack)))
    return bounds


_YAML_KEY = re.compile(r"^([A-Za-z0-9_\"'./${}-][^:#]*):(\s|$)")


def yaml_bounds(lines: list[str]) -> list[Boundary]:
    bounds: list[Boundary] = []
    for i, line in enumerate(lines):
        m = _YAML_KEY.match(line)
        if m:
            bounds.append((i, m.group(1).strip("\"'")))
        elif line.startswith("---") and i > 0:
            bounds.append((i, "---"))
    return bounds


def compose_bounds(lines: list[str]) -> list[Boundary]:
    """Top-level keys, but one boundary per service under ``services:``."""
    bounds: list[Boundary] = []
    in_services = False
    svc_indent: int | None = None
    for i, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _YAML_KEY.match(line)
        if m:
            key = m.group(1).strip("\"'")
            in_services = key == "services"
            svc_indent = None
            bounds.append((i, key))
            continue
        if in_services:
            indent = len(line) - len(line.lstrip())
            sm = re.match(r"^\s+([A-Za-z0-9_.-]+):\s*$", line)
            if sm and (svc_indent is None or indent == svc_indent):
                svc_indent = indent
                bounds.append((i, f"services.{sm.group(1)}"))
    return bounds


def json_bounds(lines: list[str]) -> list[Boundary]:
    keyed = [(i, m) for i, line in enumerate(lines) if (m := re.match(r'^(\s*)"([^"]+)"\s*:', line))]
    if not keyed:
        return []
    indent = min(len(m.group(1)) for _, m in keyed)
    return [(i, m.group(2)) for i, m in keyed if len(m.group(1)) == indent]


def toml_bounds(lines: list[str]) -> list[Boundary]:
    return [
        (i, m.group(1)) for i, line in enumerate(lines) if (m := re.match(r"^\[\[?([^\]]+)\]\]?\s*$", line))
    ]


_HCL_BLOCK = re.compile(
    r'^(resource|data|module|variable|output|provider|locals|terraform)\b\s*("[^"]*")?\s*("[^"]*")?'
)


def hcl_bounds(lines: list[str]) -> list[Boundary]:
    out = []
    for i, line in enumerate(lines):
        m = _HCL_BLOCK.match(line)
        if m:
            out.append((i, ".".join(p.strip('"') for p in m.groups() if p)))
    return out


def sql_bounds(lines: list[str]) -> list[Boundary]:
    out: list[Boundary] = []
    start = None
    for i, line in enumerate(lines):
        s = line.strip()
        if start is None and s and not s.startswith("--"):
            start = i
            words = s.split()
            out.append((i, " ".join(words[:3]).rstrip("(;").lower()))
        if start is not None and s.endswith(";"):
            start = None
    return out


def shell_bounds(lines: list[str]) -> list[Boundary]:
    pat = re.compile(r"^(?:function\s+)?([A-Za-z_][\w:-]*)\s*\(\)\s*\{?|^function\s+([A-Za-z_][\w:-]*)")
    out = []
    for i, line in enumerate(lines):
        m = pat.match(line)
        if m:
            out.append((i, m.group(1) or m.group(2)))
    return out


def chunk_text(repo: str, path: str, text: str, language: str, kind: str) -> list[Chunk]:
    lines = text.split("\n")
    name = PurePosixPath(path).name
    if language == "compose" or (language == "yaml" and "compose" in name):
        return _segments(repo, path, lines, kind, language, compose_bounds(lines))
    if language == "dockerfile":
        return _whole_or_windows(repo, path, lines, kind, language, name)
    finder = {
        "markdown": markdown_bounds,
        "yaml": yaml_bounds,
        "json": json_bounds,
        "toml": toml_bounds,
        "hcl": hcl_bounds,
        "sql": sql_bounds,
        "bash": shell_bounds,
    }.get(language)
    if finder is None:
        return window_chunks(repo, path, lines, kind, language)
    bounds = finder(lines)
    if language == "bash" and not bounds:
        return _whole_or_windows(repo, path, lines, kind, language, name)
    if not bounds:
        return window_chunks(repo, path, lines, kind, language)
    return _segments(repo, path, lines, kind, language, bounds)


def chunk_file(repo: str, path: str, text: str, language: str, kind: str) -> list[Chunk]:
    """Chunk any file, then drop chunks that contain obvious secrets and dedupe ids."""
    if kind == "code" and language in SPECS:
        chunks = chunk_code(repo, path, text, language)
    else:
        chunks = chunk_text(repo, path, text, language, kind)
    seen: set[str] = set()
    out = []
    for c in chunks:
        if c.id in seen or has_secret(c.text):
            continue
        seen.add(c.id)
        out.append(c)
    return out
