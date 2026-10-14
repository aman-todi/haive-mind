"""Debug: parse the trace (code), then gather evidence in parallel (code). Diagnosis happens in the
answer node with the diagnose template; the suggested diff is checked in verify."""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from haive.graph.retrieve import vector_evidence
from haive.graph.state import Deps, GraphState, NodeCtx, ev_from_snippet
from haive.tools import files, git, ripgrep
from haive.tools.pathmap import PathMapper, call_boundary
from haive.tools.trace_parse import message_fragment, parse
from haive.types import Evidence, Frame, ParsedTrace

MAX_FRAMES = 5
RECENT_DAYS = 30

LIB_PATTERNS = [
    r"site-packages/([\w.-]+)",
    r"dist-packages/([\w.-]+)",
    r"node_modules/((?:@[\w.-]+/)?[\w.-]+)",
    r"\.cargo/registry/src/[^/]+/([\w-]+)-\d",
    r"/go/pkg/mod/([\w./-]+?)@",
    r"\.nuget/packages/([\w.-]+)",
]


def library_of(frame: Frame) -> str | None:
    for pat in LIB_PATTERNS:
        m = re.search(pat, frame.path.replace("\\", "/"))
        if m:
            return m.group(1)
    if frame.function and "." in frame.function and frame.path == "":
        return frame.function.split(".")[0]
    return None


def trace_summary(pt: ParsedTrace, max_frames: int = 12) -> str:
    lines = [f"format: {pt.format}", f"error: {pt.error_type}: {pt.message}".strip()]
    for f in pt.frames[:max_frames]:
        where = f"{f.repo}/{f.repo_path}" if f.repo else f.path
        tag = "external" if f.external else "internal"
        lines.append(f"  [{tag}] {where}:{f.line or '?'} in {f.function or '?'}")
    if len(pt.frames) > max_frames:
        lines.append(f"  ... {len(pt.frames) - max_frames} more frames")
    return "\n".join(lines)


def make_parse_trace(deps: Deps):
    def parse_trace(state: GraphState, ctx: NodeCtx) -> dict[str, Any]:
        text = state.get("trace_text") or state["question"]
        pt = ctx.timed("trace_parse", "", lambda: parse(text))
        scope = state.get("scope_repos") or list(deps.cfg.repos)
        mapper = PathMapper(deps.db.indexed_paths(scope), deps.roots(scope), scope)
        pt.frames = mapper.map_frames(pt.frames)
        if not pt.internal_frames() and set(scope) != set(deps.cfg.repos):
            everyone = list(deps.cfg.repos)
            mapper = PathMapper(deps.db.indexed_paths(everyone), deps.roots(everyone), scope)
            pt.frames = mapper.map_frames(pt.frames)
            if pt.internal_frames():
                scope = sorted(set(scope) | {f.repo for f in pt.internal_frames() if f.repo})
        idents = list(state.get("identifiers") or [])
        for f in call_boundary(pt.frames, pt.format)[:MAX_FRAMES]:
            name = re.split(r"[.:]+", f.function)[-1] if f.function else ""
            if name and re.fullmatch(r"[A-Za-z_]\w*", name) and name not in idents:
                idents.append(name)
        update: dict[str, Any] = {
            "parsed_trace": pt.model_dump(),
            "identifiers": idents[:12],
            "scope_repos": scope,
        }
        if not state["question"].strip() or state["question"] == text:
            update["question"] = f"Why does this fail: {pt.error_type}: {pt.message}".strip()[:500]
        ctx.detail = (
            f"{pt.format} {pt.error_type}: {len(pt.frames)} frames, {len(pt.internal_frames())} internal"
        )
        return update

    return parse_trace


def make_gather_debug(deps: Deps):
    def gather_debug(state: GraphState, ctx: NodeCtx) -> dict[str, Any]:
        pt = ParsedTrace.model_validate(state["parsed_trace"])
        scope = state.get("scope_repos") or list(deps.cfg.repos)
        roots = deps.roots(scope)
        frames = call_boundary(pt.frames, pt.format)[:MAX_FRAMES]
        tasks: list[tuple[str, Any]] = []

        def frame_window(i: int, f: Frame) -> list[dict[str, Any]]:
            snip = files.window(roots[f.repo], f.repo_path, f.line or 1, 20)
            return (
                [ev_from_snippet(f.repo, snip, "tool", f"frame #{i}: {f.function or 'frame'} line {f.line}")]
                if snip
                else []
            )

        def raise_site() -> list[dict[str, Any]]:
            frag = message_fragment(pt.message)
            if not frag:
                return []
            hits = ripgrep.search(frag, roots, fixed=True, max_per_repo=5)
            out = []
            for h in hits[:3]:
                snip = files.window(roots[h.repo], h.path, h.line, 6)
                if snip:
                    out.append(ev_from_snippet(h.repo, snip, "tool", f"message fragment {frag!r} found here"))
            return out

        def blame(f: Frame) -> list[dict[str, Any]]:
            root = roots[f.repo]
            line = f.line or 1
            cutoff = time.time() - RECENT_DAYS * 86400
            bl = git.blame(root, f.repo_path, max(1, line - 2), line + 2)
            recent = [b for b in bl if b.time >= cutoff and not b.sha.startswith("0000000")]
            out = []
            if recent:
                text = "\n".join(
                    f"line {b.line}: changed {time.strftime('%Y-%m-%d', time.gmtime(b.time))} "
                    f"by {b.author} in {b.sha[:12]}: {b.text}"
                    for b in recent
                )
                out.append(
                    Evidence(
                        id=f"blame:{f.repo}:{f.repo_path}:{line}",
                        source="tool",
                        repo=f.repo,
                        path=f.repo_path,
                        start_line=recent[0].line,
                        end_line=recent[-1].line,
                        text="\n".join(b.text for b in bl if recent[0].line <= b.line <= recent[-1].line),
                        why=f"changed in the last {RECENT_DAYS} days:\n{text}",
                    ).model_dump()
                )
            for c in git.log(root, n=5, path=f.repo_path, numstat=True):
                out.append(
                    Evidence(
                        id=f"commit:{f.repo}:{c['sha'][:12]}",
                        source="commit",
                        repo=f.repo,
                        text=f"commit {c['sha'][:12]} {c['date']} {c['author']}\n{c['subject']}\n"
                        f"files: {', '.join(c['files'][:8])}",
                        why=f"recent commit touching {f.repo_path}",
                    ).model_dump()
                )
            return out

        def similar() -> list[dict[str, Any]]:
            q = " ".join([pt.error_type, *(f.function for f in frames if f.function)][:6]) or pt.message
            return vector_evidence(deps, q, scope, ["code"], 6, NodeCtx(ctx.meter), "similar code")

        for i, f in enumerate(frames):
            tasks.append((f"window {f.repo_path}:{f.line}", lambda i=i, f=f: frame_window(i, f)))
        tasks.append(("ripgrep message", raise_site))
        for f in frames[:3]:
            tasks.append((f"blame {f.repo_path}:{f.line}", lambda f=f: blame(f)))
        if deps.use_vector:
            tasks.append(("vector similar", similar))

        evs: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [(name, time.monotonic(), pool.submit(fn)) for name, fn in tasks]
            for name, t0, fut in futures:
                try:
                    evs += fut.result()
                except Exception as e:  # a failing tool must not sink the whole gather
                    ctx.tool(name, f"failed: {e}", (time.monotonic() - t0) * 1000)
                    continue
                ctx.tool(name, "", (time.monotonic() - t0) * 1000)

        update: dict[str, Any] = {"evidence": evs}
        innermost = (
            pt.frames[-1] if pt.format == "python" and pt.frames else (pt.frames[0] if pt.frames else None)
        )
        lib = library_of(innermost) if innermost and innermost.external else None
        if lib is None:
            for word in re.findall(r"\b([a-z][\w-]{2,})\.\w+", pt.error_type):
                if word not in ("java", "javax", "system", "builtins"):
                    lib = word
                    break
        if lib or (innermost is not None and innermost.external) or not frames:
            frag = message_fragment(pt.message) or pt.message[:80]
            update["debug_web_query"] = " ".join(x for x in (lib, pt.error_type, frag) if x)[:300]
        ctx.detail = f"{len(evs)} evidence from {len(tasks)} tasks; web={'yes' if update.get('debug_web_query') else 'no'}"
        return update

    return gather_debug


def route_after_gather(state: GraphState) -> str:
    return "web" if state.get("debug_web_query") and not state.get("web_done") else "grade"
