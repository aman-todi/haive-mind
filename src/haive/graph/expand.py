"""Expand (code only): fetch what the grader said was missing — definitions, callers, files, config."""

from __future__ import annotations

import re
from typing import Any

from haive.graph.retrieve import ripgrep_evidence
from haive.graph.state import Deps, GraphState, NodeCtx, ev_from_snippet
from haive.tools import files, ripgrep

PATH_LINE = re.compile(r"^([\w./@~+-]+\.[A-Za-z0-9]+)(?::(\d+))?$")


def _find_path(deps: Deps, path: str, scope: list[str]) -> tuple[str, str] | None:
    path = path.strip("`'\" ").lstrip("./")
    candidates = deps.db.indexed_paths(scope)
    exact = [c for c in candidates if c[1] == path]
    if exact:
        return exact[0]
    suffix = [c for c in candidates if c[1].endswith("/" + path) or c[1].endswith(path)]
    return min(suffix, key=lambda c: len(c[1])) if suffix else None


def make_expand(deps: Deps):
    def expand(state: GraphState, ctx: NodeCtx) -> dict[str, Any]:
        scope = state.get("scope_repos") or list(deps.cfg.repos)
        roots = deps.roots(scope)
        have = {(e.get("repo"), e.get("path")) for e in state.get("evidence") or []}
        missing = (state.get("grade") or {}).get("missing", [])[:4]
        evs: list[dict[str, Any]] = []
        for item in missing:
            item = item.strip()
            m = PATH_LINE.match(item)
            if (
                m
                and ("/" in item or "." in item.split(":")[0])
                and not re.fullmatch(r"[A-Za-z_]\w*\.[A-Za-z_]\w*", item)
            ):
                hit = _find_path(deps, m.group(1), scope)
                if hit:
                    repo, path = hit
                    line = int(m.group(2)) if m.group(2) else None
                    try:
                        snip = (
                            files.window(roots[repo], path, line, 25)
                            if line
                            else files.read_range(roots[repo], path, 1, 120)
                        )
                    except files.FileAccessError:
                        snip = None
                    if snip:
                        evs.append(ev_from_snippet(repo, snip, "tool", f"expand: {item}"))
                    continue
            name = re.split(r"[.:]+", item)[-1].strip("()` ")
            if re.fullmatch(r"[A-Za-z_]\w*", name):
                defs = ctx.timed("find_definition", name, lambda n=name: ripgrep.find_definition(n, roots, 5))
                for d in defs[:2]:
                    snip = files.window(roots[d.repo], d.path, d.line, 15)
                    if snip:
                        evs.append(ev_from_snippet(d.repo, snip, "tool", f"definition of {name}"))
                # callers / uses not already in evidence
                uses = ctx.timed(
                    "callers",
                    name,
                    lambda n=name: ripgrep.search(n, roots, fixed=True, word=True, max_per_repo=20),
                )
                for u in [
                    u
                    for u in uses
                    if (u.repo, u.path) not in have and not any(d.path == u.path for d in defs)
                ][:2]:
                    snip = files.window(roots[u.repo], u.path, u.line, 10)
                    if snip:
                        evs.append(ev_from_snippet(u.repo, snip, "tool", f"use of {name}"))
                continue
            evs += ripgrep_evidence(deps, [item], scope, ctx, word=False, max_files=3)
        ctx.detail = f"{len(evs)} new evidence for {missing}"
        return {"evidence": evs, "loops": state.get("loops", 0) + 1}

    return expand
