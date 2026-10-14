"""Live-tree file reads: line ranges, and windows expanded to the enclosing function."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from haive.index.chunk_code import SPECS, get_parser, node_name
from haive.index.discover import classify, is_secret_path, read_text

MAX_EXPAND_LINES = 200


class FileAccessError(Exception):
    pass


@dataclass
class Snippet:
    path: str
    start_line: int
    end_line: int
    text: str  # raw lines joined
    symbol: str = ""


def safe_path(root: Path, rel: str) -> Path:
    full = (root / rel).resolve()
    if not full.is_relative_to(root.resolve()):
        raise FileAccessError(f"{rel} is outside the repo")
    if is_secret_path(rel):
        raise FileAccessError(f"{rel} is on the secrets denylist")
    return full


def file_lines(root: Path, rel: str) -> list[str] | None:
    safe_path(root, rel)
    text = read_text(root, rel)
    return None if text is None else text.split("\n")


def read_range(root: Path, rel: str, start: int, end: int) -> Snippet | None:
    lines = file_lines(root, rel)
    if lines is None:
        return None
    start = max(1, start)
    end = min(len(lines), max(start, end))
    return Snippet(rel, start, end, "\n".join(lines[start - 1 : end]))


def numbered(text: str, start: int) -> str:
    return "\n".join(f"{start + i:>5} | {line}" for i, line in enumerate(text.split("\n")))


def _def_types(language: str) -> set[str]:
    spec = SPECS.get(language)
    if not spec:
        return set()
    return (
        spec.defs
        | spec.members
        | {
            "function_definition",
            "method_definition",
            "arrow_function",
            "function_expression",
            "function_item",
            "method_declaration",
            "constructor_declaration",
            "function_declaration",
        }
    )


def enclosing(root: Path, rel: str, line: int) -> tuple[int, int, str] | None:
    """(start, end, symbol) of the innermost function/class containing ``line`` (1-based)."""
    language, _ = classify(rel)
    parser = get_parser(language)
    text = read_text(root, rel)
    if parser is None or text is None:
        return None
    tree = parser.parse(text.encode("utf-8"))
    row = line - 1
    types = _def_types(language)
    node: Any = tree.root_node.named_descendant_for_point_range((row, 0), (row, 0))
    best = None
    while node is not None:
        if node.type in types:
            best = node
            break
        node = node.parent
    if best is None:
        return None
    return best.start_point.row + 1, best.end_point.row + 1, node_name(best)


def window(root: Path, rel: str, line: int, radius: int = 20) -> Snippet | None:
    """±radius lines around ``line``, expanded to the enclosing function when it is not huge."""
    lines = file_lines(root, rel)
    if lines is None:
        return None
    line = max(1, min(line, len(lines)))
    s, e = max(1, line - radius), min(len(lines), line + radius)
    symbol = ""
    enc = enclosing(root, rel, line)
    if enc:
        es, ee, symbol = enc
        if ee - es + 1 <= MAX_EXPAND_LINES:
            s, e = min(s, es), max(e, ee)
    return Snippet(rel, s, e, "\n".join(lines[s - 1 : e]), symbol)
