"""Code search with ripgrep (``rg --json``), with a slow pure-Python fallback.

Respects .gitignore, includes dotfiles (e.g. .github/), never returns denylisted secret files.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from haive.index.discover import is_secret_path, looks_binary
from haive.tools import git
from haive.types import Match

MAX_LINE = 400


def have_rg() -> bool:
    return shutil.which("rg") is not None


def search(
    pattern: str,
    roots: dict[str, Path],
    *,
    fixed: bool = False,
    ignore_case: bool = False,
    word: bool = False,
    max_per_repo: int = 50,
    globs: list[str] | None = None,
    use_rg: bool | None = None,
) -> list[Match]:
    use_rg = have_rg() if use_rg is None else use_rg
    out: list[Match] = []
    for repo, root in roots.items():
        if use_rg:
            found = _rg(pattern, root, fixed, ignore_case, word, max_per_repo, globs)
        else:
            found = _py(pattern, root, fixed, ignore_case, word, max_per_repo, globs)
        out += [Match(repo=repo, path=p, line=n, text=t) for p, n, t in found if not is_secret_path(p)]
    return out


def _rg(pattern, root, fixed, ignore_case, word, max_per_repo, globs) -> list[tuple[str, int, str]]:
    args = ["rg", "--json", "--hidden", "-g", "!.git", "--max-columns", str(MAX_LINE), "--max-filesize", "1M"]
    if fixed:
        args.append("-F")
    if ignore_case:
        args.append("-i")
    if word:
        args.append("-w")
    for g in globs or []:
        args += ["-g", g]
    args += ["-e", pattern, "."]
    try:
        proc = subprocess.run(args, cwd=root, capture_output=True, timeout=30)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return _py(pattern, root, fixed, ignore_case, word, max_per_repo, globs)
    out: list[tuple[str, int, str]] = []
    for line in proc.stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") != "match":
            continue
        d = ev["data"]
        path = d["path"].get("text", "")
        if path.startswith("./"):
            path = path[2:]
        text = d["lines"].get("text", "").rstrip("\n")
        out.append((path, d["line_number"], text[:MAX_LINE]))
        if len(out) >= max_per_repo:
            break
    return out


def _py(pattern, root, fixed, ignore_case, word, max_per_repo, globs) -> list[tuple[str, int, str]]:
    import fnmatch

    pat = re.escape(pattern) if fixed else pattern
    if word:
        pat = rf"\b(?:{pat})\b"
    try:
        rx = re.compile(pat, re.I if ignore_case else 0)
    except re.error:
        rx = re.compile(re.escape(pattern), re.I if ignore_case else 0)
    include = [g for g in globs or [] if not g.startswith("!")]
    exclude = [g[1:] for g in globs or [] if g.startswith("!")]
    out: list[tuple[str, int, str]] = []
    try:
        files = git.ls_files(root)
    except git.GitError:
        files = [
            str(p.relative_to(root)) for p in Path(root).rglob("*") if p.is_file() and ".git" not in p.parts
        ]
    for rel in files:
        if include and not any(fnmatch.fnmatch(rel, g) for g in include):
            continue
        if any(fnmatch.fnmatch(rel, g) for g in exclude):
            continue
        try:
            data = (Path(root) / rel).read_bytes()
        except OSError:
            continue
        if len(data) > 1_000_000 or looks_binary(data):
            continue
        for n, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
            if rx.search(line):
                out.append((rel, n, line[:MAX_LINE]))
                if len(out) >= max_per_repo:
                    return out
    return out


DEF_TEMPLATES = [
    r"\bdef\s+{s}\b",
    r"\bclass\s+{s}\b",
    r"\bfunction\*?\s+{s}\b",
    r"\b(?:const|let|var)\s+{s}\s*=",
    r"\bfunc\s+(?:\([^)]*\)\s*)?{s}\b",
    r"\bfn\s+{s}\b",
    r"\b(?:struct|enum|trait|interface|type|record)\s+{s}\b",
    r"\b{s}\s*[:=]\s*(?:async\s+)?(?:function|\()",
    r"^\s*(?:export\s+)?(?:async\s+)?{s}\s*\(",
    r"[\w<>\[\],\s]+\s+{s}\s*\([^;]*\)\s*(?:\{{|$|throws)",
]


def find_definition(symbol: str, roots: dict[str, Path], max_per_repo: int = 10) -> list[Match]:
    """Search for likely definitions of ``symbol`` across languages."""
    name = symbol.split(".")[-1].split("::")[-1]
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        return []
    pat = "|".join(t.format(s=re.escape(name)) for t in DEF_TEMPLATES)
    return search(pat, roots, max_per_repo=max_per_repo)
