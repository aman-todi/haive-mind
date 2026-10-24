"""Retrieval for lookup / recall / history: code first (symbols, ripgrep, git), vector after."""

from __future__ import annotations

import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from haive.graph.router import code_identifiers
from haive.graph.state import Deps, GraphState, NodeCtx, ev_from_hit, ev_from_snippet
from haive.graph.timewindow import DEFAULT_MIN_COMMITS, default_window, parse_window
from haive.index.commits import utc_iso
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
    # Code names (snake_case, camelCase, dotted, ENV_VARS) are precise ripgrep targets; plain words the
    # router passes as identifiers ("scope", "repos") match all over and would push vector hits out.
    code = [i for i in idents if code_identifiers(i) == [i]]
    words = [i for i in idents if i not in code]
    q = query_of(state)
    evs: list[dict[str, Any]] = []
    if deps.use_ripgrep:
        evs += symbol_evidence(deps, idents, scope, ctx)
        evs += ripgrep_evidence(deps, code, scope, ctx)
    if deps.use_vector:
        evs += vector_evidence(deps, q, scope, ["code", "config"], 8, ctx)
    if deps.use_ripgrep:
        evs += ripgrep_evidence(
            deps, words or ([] if deps.use_vector else keywords(q)), scope, ctx, word=False
        )
    return _dedupe_overlaps(evs)[:14]


VECTOR_PER_REPO = 4  # chunks per repo in cross-repo (recall) vector search
RECALL_EVIDENCE = 24
# With an automatic scope (recall over all repos), keep repos whose best match scores at least this
# fraction of the best repo's. The trace lists every repo's best score, to tune it.
RELEVANT_FRACTION = 0.7


def vector_per_repo(
    deps: Deps, query: str, scope: list[str], ctx: NodeCtx, prune: bool = False
) -> tuple[list[dict[str, Any]], list[str]]:
    """One vector search per repo, in parallel, so every repo in scope gets its best chunks (a single
    top-k over all repos lets one or two repos fill every slot). Repos are ordered by their best
    match and their hits interleaved: each repo's best chunk comes before any repo's second. With
    ``prune``, repos well below the best one are dropped. Returns (evidence, repos kept)."""
    kinds = ["code", "config", "doc"]

    def search(repo: str) -> list[Any]:
        return deps.store.search(query, VECTOR_PER_REPO, {"repo": [repo], "kind": kinds})

    def run() -> list[list[Any]]:
        with ThreadPoolExecutor(max_workers=min(8, len(scope)) or 1) as pool:
            return list(pool.map(search, scope))

    per_repo = [
        hits for hits in ctx.timed("vector", f"{len(scope)} repos x k={VECTOR_PER_REPO}", run) if hits
    ]
    per_repo.sort(key=lambda hits: -hits[0].score)
    if prune and per_repo and per_repo[0][0].score > 0:
        floor = RELEVANT_FRACTION * per_repo[0][0].score
        kept = [hits for hits in per_repo if hits[0].score >= floor]
        dropped = [hits for hits in per_repo if hits[0].score < floor]
        ctx.tool(
            "relevance",
            f"kept {', '.join(f'{h[0].repo} {h[0].score:.2f}' for h in kept)}"
            + (f"; dropped {', '.join(f'{h[0].repo} {h[0].score:.2f}' for h in dropped)}" if dropped else ""),
            0,
        )
        per_repo = kept
    evs: list[dict[str, Any]] = []
    for rank in range(VECTOR_PER_REPO):
        for hits in per_repo:
            if rank < len(hits):
                h = hits[rank]
                ev = ev_from_hit(deps, h, f"best in {h.repo} {h.score:.2f}")
                if ev:
                    evs.append(ev)
    return evs, [hits[0].repo for hits in per_repo]


def recall(
    deps: Deps, state: GraphState, ctx: NodeCtx, scope: list[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    q = query_of(state)
    auto = bool(state.get("scope_auto"))
    evs: list[dict[str, Any]] = []
    kept = scope
    if deps.use_vector:
        vector, relevant = vector_per_repo(deps, q, scope, ctx, prune=auto)
        evs += vector
        if auto and relevant:
            kept = relevant
            ctx.notices.append(f"Searched all {len(scope)} repos; the most relevant: {', '.join(kept)}.")
    if deps.use_ripgrep:
        evs += ripgrep_evidence(
            deps, state.get("identifiers") or keywords(q)[:4], kept, ctx, word=False, max_files=4
        )
    return _dedupe_overlaps(evs)[:RECALL_EVIDENCE], ({"scope_repos": kept} if kept != scope else {})


RECENT_COMMITS = 12
HISTORY_WINDOW = "history window"  # evidence label for the commits of a requested time window
MAX_WINDOW_COMMITS = 30  # cap on commits listed for one window (and history evidence overall)


def _commit_ev(repo: str, c: dict[str, Any], why: str) -> dict[str, Any]:
    files_ = ", ".join(c.get("files", [])[:12])
    text = f"commit {c['sha'][:12]}  {utc_iso(c['date'])}  {c['author']}\n{c['subject']}\n{c.get('body', '')}"
    text = text.strip()
    if files_:
        text += f"\nfiles: {files_}"
    if c.get("patch"):
        text += "\n" + c["patch"][:2500]
    return Evidence(
        id=f"commit:{repo}:{c['sha'][:12]}", source="commit", repo=repo, text=text, why=why
    ).model_dump()


def _newest(ctx: NodeCtx, roots: dict[str, Any], n: int, since: str | None) -> list[tuple[str, dict]]:
    """The newest ``n`` commits across ``roots`` (optionally since a date), newest first."""
    found = [
        (utc_iso(c["date"]), repo, c)
        for repo, root in roots.items()
        for c in ctx.timed(
            "git log",
            f"{repo} -n{n}" + (f" --since={since}" if since else ""),
            lambda r=root: git.log(r, n=n, numstat=True, since=since),
        )
    ]
    return [(repo, c) for _, repo, c in sorted(found, key=lambda x: x[0], reverse=True)[:n]]


def history(deps: Deps, state: GraphState, ctx: NodeCtx, scope: list[str]) -> list[dict[str, Any]]:
    q = query_of(state)
    idents = state.get("identifiers") or []
    roots = deps.roots(scope)
    window = parse_window(q)
    since = window.git_since if window else None
    evs: list[dict[str, Any]] = []
    if deps.use_ripgrep and (window or not idents):
        # "what changed (recently | in the last 2 weeks | in the last 20 commits)?": list the window itself
        counted = True  # whether the label already says how many commits
        if window and window.commits:
            newest = _newest(ctx, roots, min(window.commits, MAX_WINDOW_COMMITS), None)
        else:
            counted = False
            w = window or default_window()
            newest = _newest(ctx, roots, MAX_WINDOW_COMMITS, w.git_since)
            if not window and len(newest) < DEFAULT_MIN_COMMITS:
                newest = _newest(ctx, roots, RECENT_COMMITS, None)
                w.label = f"the newest {len(newest)} commits, since fewer than {DEFAULT_MIN_COMMITS} landed in {w.label}"
                counted = True
            window = w
        evs += [_commit_ev(repo, c, HISTORY_WINDOW) for repo, c in newest]
        ctx.notices.append(
            f"History window: {window.label}"
            + ("." if counted else f" ({len(newest)} commits).")
            + (
                " Ask e.g. 'in the last 2 months' or 'last 20 commits' to change it."
                if not parse_window(q)
                else ""
            )
        )
    if deps.use_vector and not (window and evs):
        evs += vector_evidence(deps, q, scope, ["commit"], 8, ctx, "commit search")
    if deps.use_ripgrep:
        terms = idents or ([] if window else keywords(q)[:3])
        for repo, root in roots.items():
            for t in terms[:4]:
                if idents:  # pickaxe on plain words ("changed") matches unrelated code
                    for c in ctx.timed(
                        "git log -S",
                        t,
                        lambda r=root, t=t: git.log(r, n=6, pickaxe=t, numstat=True, since=since),
                    ):
                        evs.append(_commit_ev(repo, c, f"git log -S{t}"))
                for c in ctx.timed(
                    "git log --grep",
                    t,
                    lambda r=root, t=t: git.log(r, n=4, grep=t, numstat=True, since=since),
                ):
                    evs.append(_commit_ev(repo, c, f"git log --grep={t}"))
        # git log -L on the functions those identifiers name, plus the current code itself
        for ev in symbol_evidence(deps, idents[:3], scope, ctx)[:3]:
            evs.append(ev)
            root = roots.get(ev["repo"])
            rng = (f"{ev['start_line']},{ev['end_line']}", ev["path"])
            for c in ctx.timed(
                "git log -L",
                f"{rng[1]}:{rng[0]}",
                lambda r=root, rng=rng: git.log(r, n=5, line_range=rng, since=since),
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
    return list(merged.values())[:MAX_WINDOW_COMMITS]


RETRIEVERS = {"lookup": lookup, "recall": recall, "history": history}


def make_retrieve(deps: Deps):
    def retrieve(state: GraphState, ctx: NodeCtx) -> dict[str, Any]:
        intent = state.get("intent", "lookup")
        fn = RETRIEVERS.get(intent, lookup)
        out = fn(deps, state, ctx, state.get("scope_repos") or list(deps.cfg.repos))
        evs, extra = out if isinstance(out, tuple) else (out, {})
        ctx.detail = f"{intent}: {len(evs)} evidence"
        return {"evidence": evs, **extra}

    return retrieve
