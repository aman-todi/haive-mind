"""LangChain tools over haive's deterministic code tools (ripgrep, file reads, git).

Each tool returns ``(content, artifact)``: ``content`` is what a model reads, ``artifact`` is the
list of evidence dicts the graph keeps. Tools are built per question so they are bound to the repos
in scope; they never touch SQLite (the indexed file list is captured up front), so ``ToolNode`` can
run them in parallel threads safely.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool

from haive.graph.state import ev_from_snippet
from haive.tools import files, git, ripgrep
from haive.types import Evidence

MAX_READ_LINES = 150


def _summarize(evs: list[dict[str, Any]], empty: str) -> str:
    if not evs:
        return empty
    return "\n".join(
        f"{e['repo']}/{e['path']}:{e['start_line']}-{e['end_line']} ({e['why']})"
        if e.get("path")
        else e["text"][:200]
        for e in evs
    )


def make_tools(roots: dict[str, Path], indexed: list[tuple[str, str]]) -> list[BaseTool]:
    def find_path(path: str) -> tuple[str, str] | None:
        path = path.strip("`'\" ").lstrip("./")
        exact = [c for c in indexed if c[1] == path]
        if exact:
            return exact[0]
        suffix = [c for c in indexed if c[1].endswith("/" + path) or c[1].endswith(path)]
        return min(suffix, key=lambda c: len(c[1])) if suffix else None

    @tool(response_format="content_and_artifact")
    def find_definition(symbol: str) -> tuple[str, list[dict[str, Any]]]:
        """Find where a function, class, method, type, or variable is defined. Pass the bare name."""
        name = re.split(r"[.:]+", symbol)[-1].strip("()` ")
        evs = []
        for d in ripgrep.find_definition(name, roots, 5)[:2]:
            snip = files.window(roots[d.repo], d.path, d.line, 15)
            if snip:
                evs.append(ev_from_snippet(d.repo, snip, "tool", f"definition of {name}"))
        return _summarize(evs, f"no definition of {name} found"), evs

    @tool(response_format="content_and_artifact")
    def find_usages(symbol: str) -> tuple[str, list[dict[str, Any]]]:
        """Find call sites and other uses of a symbol (callers), excluding its definition."""
        name = re.split(r"[.:]+", symbol)[-1].strip("()` ")
        defs = {(d.repo, d.path) for d in ripgrep.find_definition(name, roots, 5)}
        uses = [
            u
            for u in ripgrep.search(name, roots, fixed=True, word=True, max_per_repo=20)
            if (u.repo, u.path) not in defs
        ]
        evs = []
        for u in uses[:3]:
            snip = files.window(roots[u.repo], u.path, u.line, 10)
            if snip:
                evs.append(ev_from_snippet(u.repo, snip, "tool", f"use of {name}"))
        return _summarize(evs, f"no uses of {name} found"), evs

    @tool(response_format="content_and_artifact")
    def read_file(path: str, line: int | None = None) -> tuple[str, list[dict[str, Any]]]:
        """Read a file from the repos (config, docs, or code). With `line`, read the function or
        region around that line; otherwise the start of the file. Partial paths are fine."""
        hit = find_path(path)
        if hit is None:
            return f"{path} is not an indexed file", []
        repo, rel = hit
        try:
            snip = (
                files.window(roots[repo], rel, line, 25)
                if line
                else files.read_range(roots[repo], rel, 1, MAX_READ_LINES)
            )
        except files.FileAccessError as e:
            return str(e), []
        evs = [ev_from_snippet(repo, snip, "tool", f"read {rel}")] if snip else []
        return _summarize(evs, f"could not read {path}"), evs

    @tool(response_format="content_and_artifact")
    def search_code(text: str) -> tuple[str, list[dict[str, Any]]]:
        """Search the repos for an exact string (an error message, env var, config key, or route)."""
        by_file: dict[tuple[str, str], list[int]] = {}
        for m in ripgrep.search(text, roots, fixed=True, max_per_repo=20):
            by_file.setdefault((m.repo, m.path), []).append(m.line)
        evs = []
        for (repo, path), lines in sorted(by_file.items(), key=lambda kv: -len(kv[1]))[:3]:
            snip = files.window(roots[repo], path, lines[0], 8)
            if snip:
                evs.append(ev_from_snippet(repo, snip, "tool", f"search: {text[:40]}"))
        return _summarize(evs, f"no matches for {text!r}"), evs

    @tool(response_format="content_and_artifact")
    def recent_commits(path: str) -> tuple[str, list[dict[str, Any]]]:
        """List the most recent commits that touched a file (why it changed, and when)."""
        hit = find_path(path)
        if hit is None:
            return f"{path} is not an indexed file", []
        repo, rel = hit
        evs = []
        for c in git.log(roots[repo], n=5, path=rel, numstat=True):
            text = f"commit {c['sha'][:12]} {c['date']} {c['author']}\n{c['subject']}\n{c['body']}".strip()
            evs.append(Evidence(id=f"commit:{repo}:{c['sha'][:12]}", source="commit", repo=repo, text=text,
                                why=f"recent commit touching {rel}").model_dump())  # fmt: skip
        return _summarize(evs, f"no commits for {rel}"), evs

    return [find_definition, find_usages, read_file, search_code, recent_commits]


PATH_LINE = re.compile(r"^([\w./@~+-]+\.[A-Za-z0-9]+)(?::(\d+))?$")


def heuristic_calls(missing: list[str]) -> list[dict[str, Any]]:
    """Tool calls chosen without a model, from the grader's `missing` list."""
    calls: list[dict[str, Any]] = []
    for item in missing:
        item = item.strip()
        m = PATH_LINE.match(item)
        if (
            m
            and ("/" in item or "." in item.split(":")[0])
            and not re.fullmatch(r"[A-Za-z_]\w*\.[A-Za-z_]\w*", item)
        ):
            args: dict[str, Any] = {"path": m.group(1)}
            if m.group(2):
                args["line"] = int(m.group(2))
            calls.append({"name": "read_file", "args": args})
            continue
        name = re.split(r"[.:]+", item)[-1].strip("()` ")
        if re.fullmatch(r"[A-Za-z_]\w*", name):
            calls.append({"name": "find_definition", "args": {"symbol": name}})
            calls.append({"name": "find_usages", "args": {"symbol": name}})
            continue
        calls.append({"name": "search_code", "args": {"text": item}})
    return calls
