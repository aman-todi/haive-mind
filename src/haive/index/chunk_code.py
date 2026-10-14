"""Code chunking with tree-sitter.

One chunk per top-level function, class, or method. Classes over the token cap are split into
their methods, with the class signature carried in the chunk's ``context`` (so it reaches the
embedding but not the stored raw text). Leftover top-level statements become ``<module>``
chunks, one per contiguous run. Anything still over the cap is split into overlapping windows.
If a grammar can't be loaded, we fall back to plain line windows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from haive.types import Chunk

MAX_TOKENS = 1500
CHARS_PER_TOKEN = 4
WINDOW_LINES = 60
WINDOW_OVERLAP = 10

TS_NAME = {
    "python": "python", "javascript": "javascript", "typescript": "typescript", "tsx": "tsx", "go": "go",
    "rust": "rust", "java": "java", "c": "c", "cpp": "cpp", "csharp": "csharp",
}  # fmt: skip


@dataclass
class LangSpec:
    defs: set[str]
    containers: set[str] = field(default_factory=set)
    members: set[str] = field(default_factory=set)
    wrappers: set[str] = field(default_factory=set)  # descended into transparently


_JS_DEFS = {"function_declaration", "generator_function_declaration", "class_declaration", "lexical_declaration",
            "variable_declaration", "abstract_class_declaration", "interface_declaration",
            "type_alias_declaration", "enum_declaration"}  # fmt: skip

SPECS: dict[str, LangSpec] = {
    "python": LangSpec(
        {"function_definition", "class_definition", "decorated_definition"},
        {"class_definition"},
        {"function_definition", "decorated_definition"},
    ),
    "javascript": LangSpec(_JS_DEFS, {"class_declaration", "class"}, {"method_definition"}, {"export_statement"}),
    "typescript": LangSpec(
        _JS_DEFS, {"class_declaration", "abstract_class_declaration"}, {"method_definition"}, {"export_statement"}
    ),
    "go": LangSpec({"function_declaration", "method_declaration", "type_declaration"}),
    "rust": LangSpec(
        {"function_item", "impl_item", "struct_item", "enum_item", "trait_item", "mod_item", "macro_definition"},
        {"impl_item", "trait_item", "mod_item"},
        {"function_item", "function_signature_item"},
    ),
    "java": LangSpec(
        {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration"},
        {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration"},
        {"method_declaration", "constructor_declaration"},
    ),
    "c": LangSpec({"function_definition", "struct_specifier", "type_definition", "enum_specifier"}),
    "cpp": LangSpec(
        {"function_definition", "class_specifier", "struct_specifier", "template_declaration", "enum_specifier"},
        {"class_specifier", "struct_specifier"},
        {"function_definition", "template_declaration"},
        {"namespace_definition", "declaration_list", "linkage_specification"},
    ),
    "csharp": LangSpec(
        {"class_declaration", "interface_declaration", "struct_declaration", "record_declaration",
         "enum_declaration"},
        {"class_declaration", "interface_declaration", "struct_declaration", "record_declaration"},
        {"method_declaration", "constructor_declaration", "property_declaration"},
        {"namespace_declaration", "file_scoped_namespace_declaration", "declaration_list"},
    ),
}  # fmt: skip
SPECS["tsx"] = SPECS["typescript"]

_FUNC_VALUES = {"arrow_function", "function_expression", "function", "class", "generator_function"}


@lru_cache(maxsize=32)
def get_parser(language: str) -> Any | None:
    name = TS_NAME.get(language)
    if not name:
        return None
    try:
        from tree_sitter_language_pack import get_parser as _gp

        return _gp(name)
    except Exception:
        return None


def _text(node: Any) -> str:
    return node.text.decode("utf-8", "replace") if node is not None and node.text else ""


def node_name(node: Any) -> str:
    t = node.type
    if t == "decorated_definition":
        inner = node.child_by_field_name("definition")
        return node_name(inner) if inner is not None else ""
    if t == "template_declaration":
        for c in node.named_children:
            if c.type not in ("template_parameter_list",):
                n = node_name(c)
                if n:
                    return n
        return ""
    if t in ("lexical_declaration", "variable_declaration"):
        for c in node.named_children:
            if c.type == "variable_declarator":
                return _text(c.child_by_field_name("name"))
        return ""
    if t == "impl_item":
        ty = _text(node.child_by_field_name("type"))
        tr = _text(node.child_by_field_name("trait"))
        return f"impl {tr} for {ty}" if tr else f"impl {ty}"
    if t == "type_declaration":
        for c in node.named_children:
            if c.type in ("type_spec", "type_alias"):
                return _text(c.child_by_field_name("name"))
        return ""
    if t == "method_declaration" and node.child_by_field_name("receiver") is not None:  # go
        recv = _text(node.child_by_field_name("receiver"))
        m = re.findall(r"[A-Za-z_]\w*", recv)
        name = _text(node.child_by_field_name("name"))
        return f"{m[-1]}.{name}" if m else name
    n = node.child_by_field_name("name")
    if n is not None:
        return _text(n)
    decl = node.child_by_field_name("declarator")  # C/C++ functions and typedefs
    while decl is not None:
        if decl.type in ("identifier", "field_identifier", "type_identifier", "qualified_identifier",
                         "destructor_name", "operator_name"):  # fmt: skip
            return _text(decl)
        nxt = decl.child_by_field_name("declarator")
        if nxt is None:
            for c in decl.named_children:
                if c.type in ("identifier", "field_identifier", "qualified_identifier", "type_identifier"):
                    return _text(c)
            break
        decl = nxt
    return ""


def _is_def(node: Any, spec: LangSpec) -> bool:
    if node.type not in spec.defs:
        return False
    if node.type in ("lexical_declaration", "variable_declaration"):
        for c in node.named_children:
            if c.type == "variable_declarator":
                v = c.child_by_field_name("value")
                if v is not None and v.type in _FUNC_VALUES:
                    return True
        return False
    if node.type in ("struct_specifier", "enum_specifier") and node.child_by_field_name("body") is None:
        return False
    return True


def _container_target(node: Any) -> Any:
    """For decorated/template wrappers, the node whose body holds members."""
    if node.type == "decorated_definition":
        return node.child_by_field_name("definition")
    if node.type == "template_declaration":
        for c in node.named_children:
            if c.type in ("class_specifier", "struct_specifier"):
                return c
    return node


def _members(node: Any, spec: LangSpec) -> list[Any]:
    body = node.child_by_field_name("body")
    if body is None:
        return []
    out = []
    for c in body.named_children:
        if c.type in spec.members:
            out.append(c)
        elif c.type == "export_statement":
            out.extend(x for x in c.named_children if x.type in spec.members)
    return out


def _collect(node: Any, spec: LangSpec, out: list[Any]) -> None:
    for c in node.named_children:
        if _is_def(c, spec):
            out.append(c)
        elif c.type in spec.wrappers:
            body = c.child_by_field_name("body")
            _collect(body if body is not None and c.type != "export_statement" else c, spec, out)


def _tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def window_chunks(
    repo: str, path: str, lines: list[str], kind: str, language: str, *, start: int = 1,
    symbol: str = "", context: str = "", size: int = WINDOW_LINES, overlap: int = WINDOW_OVERLAP,
) -> list[Chunk]:  # fmt: skip
    """Overlapping line windows. ``start`` is the 1-based line number of ``lines[0]``."""
    chunks: list[Chunk] = []
    if not lines:
        return chunks
    step = max(1, size - overlap)
    i = 0
    while i < len(lines):
        seg = lines[i : i + size]
        text = "\n".join(seg)
        if text.strip():
            chunks.append(
                Chunk.make(
                    repo,
                    path,
                    kind,
                    text,
                    language=language,
                    symbol=symbol,
                    context=context,
                    start_line=start + i,
                    end_line=start + i + len(seg) - 1,
                )  # fmt: skip
            )
        if i + size >= len(lines):
            break
        i += step
    return chunks


def _emit(repo, path, lines, language, s, e, symbol, context="") -> list[Chunk]:
    """Emit lines[s..e] (0-based inclusive) as one chunk, or windows if over the cap."""
    seg = lines[s : e + 1]
    text = "\n".join(seg)
    if not text.strip():
        return []
    if _tokens(text) > MAX_TOKENS:
        size = max(20, MAX_TOKENS * CHARS_PER_TOKEN // max(1, (len(text) // max(1, len(seg)))))
        return window_chunks(repo, path, seg, "code", language, start=s + 1, symbol=symbol, context=context,
                             size=size, overlap=max(5, size // 8))  # fmt: skip
    return [Chunk.make(repo, path, "code", text, language=language, symbol=symbol, context=context,
                       start_line=s + 1, end_line=e + 1)]  # fmt: skip


COMMENT_PREFIXES = {
    "python": ("#",),
    "rust": ("//", "/*", "*", "#["),
    "csharp": ("//", "/*", "*", "["),
}
DEFAULT_COMMENT_PREFIXES = ("//", "/*", "*", "@")


def _leading_comment_start(node: Any, lines: list[str], language: str) -> int:
    """Extend a definition upward over directly-attached comments, attributes, and annotations."""
    prefixes = COMMENT_PREFIXES.get(language, DEFAULT_COMMENT_PREFIXES)
    row = node.start_point.row
    while row > 0:
        prev = lines[row - 1].strip()
        if prev.startswith(prefixes) and not prev.startswith("#!"):
            row -= 1
        else:
            break
    return row


def chunk_code(repo: str, path: str, text: str, language: str) -> list[Chunk]:
    lines = text.split("\n")
    spec = SPECS.get(language)
    parser = get_parser(language) if spec else None
    if parser is None or spec is None:
        return window_chunks(repo, path, lines, "code", language)
    try:
        tree = parser.parse(text.encode("utf-8"))
    except Exception:
        return window_chunks(repo, path, lines, "code", language)

    defs: list[Any] = []
    _collect(tree.root_node, spec, defs)
    chunks: list[Chunk] = []
    covered = [False] * len(lines)

    for node in defs:
        s = _leading_comment_start(node, lines, language)
        e = min(node.end_point.row, len(lines) - 1)
        name = node_name(node) or node.type
        for i in range(s, e + 1):
            covered[i] = True
        target = _container_target(node)
        members = _members(target, spec) if target is not None and target.type in spec.containers else []
        whole = "\n".join(lines[s : e + 1])
        if members and _tokens(whole) > MAX_TOKENS:
            first = members[0].start_point.row
            sig_end = max(s, min(first - 1, s + 15))
            signature = lines[node.start_point.row].strip()
            # Header chunk: signature, docstring, fields up to the first member.
            chunks += _emit(repo, path, lines, language, s, sig_end, name)
            for m in members:
                ms = _leading_comment_start(m, lines, language)
                me = m.end_point.row
                mname = node_name(m) or m.type
                chunks += _emit(repo, path, lines, language, ms, me, f"{name}.{mname}", context=signature)
        else:
            chunks += _emit(repo, path, lines, language, s, e, name)

    # Leftover top-level statements: one <module> chunk per contiguous non-trivial run.
    run: list[int] = []

    def flush() -> None:
        nonblank = [i for i in run if lines[i].strip()]
        if any(len(lines[i].strip()) > 3 for i in nonblank):
            chunks.extend(_emit(repo, path, lines, language, nonblank[0], nonblank[-1], "<module>"))
        run.clear()

    for i, done in enumerate(covered):
        if done:
            if run:
                flush()
        else:
            run.append(i)
    if run:
        flush()
    chunks.sort(key=lambda c: (c.start_line, c.end_line))
    return chunks
