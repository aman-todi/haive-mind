"""Retrieval for lookup / recall / history: code first (symbols, ripgrep, git), vector after."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from haive.graph.state import Deps, GraphState, NodeCtx, ev_from_hit, ev_from_snippet
from haive.tools import files, git, ripgrep
from haive.types import Evidence

MAX_RG_FILES = 6
STOPWORDS = set(
    "a an the and or of to in on for with is are was were be how what where when why which who does do did i my "
    "me we our you your it its this that these those from by as at into about there their can could should would "
    "use used using work works implement implemented handle handled".split()
)


def keywords(q: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", q) if w.lower() not in STOPWORDS][:8]


def query_of(state: GraphState) -> str:
    return state.get("rewritten_query") or state["question"]


def symbol_evidence(deps: Deps, idents: list[str], scope: list[str], ctx: NodeCtx) -> list[dict[str, Any]]:
    out = []
    for ident in idents:
        name = re.split(r"[.:]+", ident)[-1]
        if not re.fullmatch(r"[A-Za-z_]\w*", name):
            continue
        rows = ctx.timed("symbols", name, lambda n=name: deps.db.find_symbol(n, scope, limit=4))
        for r in rows:
            root = deps.roots([r["repo"]]).get(r["repo"])
            if not root:
                continue
            try:
                snip = files.read_range(root, r["path"], r["start_line"], r["end_line"])
            except files.FileAccessError:
                continue
            if snip:
                snip.symbol = r["symbol"]
                out.append(ev_from_snippet(r["repo"], snip, "code", f"symbol match: {r['symbol']}"))
    return out


def ripgrep_evidence(
    deps: Deps,
    terms: list[str],
    scope: list[str],
    ctx: NodeCtx,
    *,
    word: bool = True,
    max_files: int = MAX_RG_FILES,
) -> list[dict[str, Any]]:
    roots = deps.roots(scope)
    by_file: dict[tuple[str, str], list[int]] = defaultdict(list)
    for term in terms:
        if len(term) < 3:
            continue
        hits = ctx.timed(
            "ripgrep",
            term,
            lambda t=term: ripgrep.search(
                t, roots, fixed=True, word=word and bool(re.fullmatch(r"\w+", t)), max_per_repo=30
            ),
        )
        for h in hits:
            by_file[(h.repo, h.path)].append(h.line)
    ranked = sorted(by_file.items(), key=lambda kv: -len(kv[1]))[:max_files]
    out = []
    for (repo, path), lines in ranked:
        try:
            snip = files.window(roots[repo], path, lines[0], radius=12)
        except files.FileAccessError:
            continue
        if snip:
            out.append(ev_from_snippet(repo, snip, "tool", f"ripgrep: {len(lines)} matches"))
    return out


def vector_evidence(
    deps: Deps, query: str, scope: list[str], kinds: list[str], k: int, ctx: NodeCtx, why: str = "vector"
) -> list[dict[str, Any]]:
    hits = ctx.timed(
        "vector", f"{kinds} k={k}", lambda: deps.store.search(query, k, {"repo": scope, "kind": kinds})
    )
    out = []
    for h in hits:
        ev = ev_from_hit(deps, h, f"{why} {h.score:.2f}")
        if ev:
            out.append(ev)
    return out


def _dedupe_overlaps(evs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop items whose line range is contained in an earlier item from the same file."""
    out: list[dict[str, Any]] = []
    for e in evs:
        if e.get("path") and any(
            o.get("path") == e["path"]
            and o["repo"] == e["repo"]
            and o["start_line"] <= e["start_line"]
            and o["end_line"] >= e["end_line"]
            for o in out
        ):
            continue
        out.append(e)
    return out


def lookup(deps: Deps, state: GraphState, ctx: NodeCtx, scope: list[str]) -> list[dict[str, Any]]:
    idents = state.get("identifiers") or []
    q = query_of(state)
    evs: list[dict[str, Any]] = []
    if deps.use_ripgrep:
        evs += symbol_evidence(deps, idents, scope, ctx)
        evs += ripgrep_evidence(deps, idents, scope, ctx)
    if deps.use_vector:
        evs += vector_evidence(deps, q, scope, ["code", "config"], 8, ctx)
    elif not idents:
        evs += ripgrep_evidence(deps, keywords(q), scope, ctx, word=False)
    return _dedupe_overlaps(evs)[:14]


def recall(deps: Deps, state: GraphState, ctx: NodeCtx, scope: list[str]) -> list[dict[str, Any]]:
    q = query_of(state)
    evs: list[dict[str, Any]] = []
    if deps.use_vector:
        hits = ctx.timed(
            "vector",
            "code/config/doc k=24",
            lambda: deps.store.search(q, 24, {"repo": scope, "kind": ["code", "config", "doc"]}),
        )
        per_repo: dict[str, int] = defaultdict(int)
        for h in hits:
            if per_repo[h.repo] >= 2:
                continue
            ev = ev_from_hit(deps, h, f"best in {h.repo} {h.score:.2f}")
            if ev:
                per_repo[h.repo] += 1
                evs.append(ev)
    if deps.use_ripgrep:
        evs += ripgrep_evidence(
            deps, state.get("identifiers") or keywords(q)[:4], scope, ctx, word=False, max_files=4
        )
    return _dedupe_overlaps(evs)[:14]


def _commit_ev(repo: str, c: dict[str, Any], why: str) -> dict[str, Any]:
    files_ = ", ".join(c.get("files", [])[:12])
    text = f"commit {c['sha'][:12]}  {c['date']}  {c['author']}\n{c['subject']}\n{c.get('body', '')}".strip()
    if files_:
        text += f"\nfiles: {files_}"
    if c.get("patch"):
        text += "\n" + c["patch"][:2500]
    return Evidence(
        id=f"commit:{repo}:{c['sha'][:12]}", source="commit", repo=repo, text=text, why=why
    ).model_dump()


def history(deps: Deps, state: GraphState, ctx: NodeCtx, scope: list[str]) -> list[dict[str, Any]]:
    q = query_of(state)
    idents = state.get("identifiers") or []
    roots = deps.roots(scope)
    evs: list[dict[str, Any]] = []
    if deps.use_vector:
        evs += vector_evidence(deps, q, scope, ["commit"], 8, ctx, "commit search")
    if deps.use_ripgrep:
        terms = idents or keywords(q)[:3]
        for repo, root in roots.items():
            for t in terms[:4]:
                for c in ctx.timed(
                    "git log -S", t, lambda r=root, t=t: git.log(r, n=6, pickaxe=t, numstat=True)
                ):
                    evs.append(_commit_ev(repo, c, f"git log -S{t}"))
                for c in ctx.timed(
                    "git log --grep", t, lambda r=root, t=t: git.log(r, n=4, grep=t, numstat=True)
                ):
                    evs.append(_commit_ev(repo, c, f"git log --grep={t}"))
        # git log -L on the functions those identifiers name, plus the current code itself
        for ev in symbol_evidence(deps, idents[:3], scope, ctx)[:3]:
            evs.append(ev)
            root = roots.get(ev["repo"])
            rng = (f"{ev['start_line']},{ev['end_line']}", ev["path"])
            for c in ctx.timed(
                "git log -L", f"{rng[1]}:{rng[0]}", lambda r=root, rng=rng: git.log(r, n=5, line_range=rng)
            ):
                evs.append(_commit_ev(ev["repo"], c, f"git log -L {rng[1]}:{rng[0]}"))
    # the same commit often comes back from several searches: keep one item, merge the reasons
    merged: dict[str, dict[str, Any]] = {}
    for e in evs:
        if e["id"] in merged:
            if e["why"] not in merged[e["id"]]["why"]:
                merged[e["id"]]["why"] += f"; {e['why']}"
        else:
            merged[e["id"]] = e
    return list(merged.values())[:20]


RETRIEVERS = {"lookup": lookup, "recall": recall, "history": history}


def make_retrieve(deps: Deps):
    def retrieve(state: GraphState, ctx: NodeCtx) -> dict[str, Any]:
        intent = state.get("intent", "lookup")
        fn = RETRIEVERS.get(intent, lookup)
        evs = fn(deps, state, ctx, state.get("scope_repos") or list(deps.cfg.repos))
        ctx.detail = f"{intent}: {len(evs)} evidence"
        return {"evidence": evs}

    return retrieve
