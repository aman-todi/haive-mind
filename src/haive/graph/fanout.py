"""Compare and architecture: Send fan-out per target / per repo, then deterministic merging and
connection mapping between repos (shared env vars, compose service names, routes defined in one
repo and called in another, shared deploy workflows)."""

from __future__ import annotations

import re
from typing import Any

from langgraph.types import Send
from pydantic import BaseModel, Field

from haive import prompts
from haive.graph.retrieve import ripgrep_evidence, symbol_evidence, vector_evidence
from haive.graph.state import Deps, GraphState, NodeCtx, ev_from_snippet
from haive.index.profile import profile_summary
from haive.llm.budget import BudgetExceeded
from haive.llm.models import LLMError
from haive.llm.structured import structured
from haive.tools import files
from haive.types import Evidence, RepoProfile

ARCH_QUERY = "architecture overview services deploy configuration entrypoint how components connect"


# ---------------------------------------------------------------- compare


def compare_fanout(state: GraphState) -> list[Send]:
    return [
        Send(
            "compare_sub",
            {
                "question": state.get("rewritten_query") or state["question"],
                "target": t,
                "identifiers": state.get("identifiers") or [],
                "llm_calls": state.get("llm_calls", 0),
            },
        )
        for t in state.get("compare_targets") or []
    ]


def make_compare_sub(deps: Deps):
    def compare_sub(payload: dict[str, Any], ctx: NodeCtx) -> dict[str, Any]:
        t = payload["target"]
        repos = t["repos"]
        evs: list[dict[str, Any]] = []
        if deps.use_ripgrep:
            evs += symbol_evidence(deps, payload["identifiers"], repos, ctx)[:3]
            evs += ripgrep_evidence(deps, payload["identifiers"], repos, ctx, max_files=3)
        if deps.use_vector:
            evs += vector_evidence(deps, payload["question"], repos, ["code", "config", "doc"], 6, ctx)
        for e in evs:
            e["why"] = f"[{t['name']}] {e.get('why', '')}".strip()
        ctx.detail = f"{t['name']}: {len(evs)} evidence"
        return {"evidence": evs[:10]}

    return compare_sub


def make_merge(deps: Deps):
    def merge(state: GraphState, ctx: NodeCtx) -> dict[str, Any]:
        ctx.detail = f"{len(state.get('evidence') or [])} evidence across {len(state.get('compare_targets') or [])} targets"
        return {}

    return merge


# ---------------------------------------------------------------- architecture


class RepoSummary(BaseModel):
    repo: str
    purpose: str
    components: list[str] = Field(default_factory=list)
    interfaces: list[str] = Field(
        default_factory=list, description="APIs exposed/consumed, queues, DBs, env vars"
    )
    deploy: str = ""


def _profile(deps: Deps, repo: str) -> RepoProfile:
    data = deps.db.get_profile(repo)
    return RepoProfile.model_validate(data) if data else RepoProfile(repo=repo)


def make_load_profiles(deps: Deps):
    def load_profiles(state: GraphState, ctx: NodeCtx) -> dict[str, Any]:
        scope = state.get("scope_repos") or list(deps.cfg.repos)
        roots = deps.roots(scope)
        evs: list[dict[str, Any]] = []
        for repo in scope:
            prof = _profile(deps, repo)
            evs.append(
                Evidence(
                    id=f"profile:{repo}",
                    source="tool",
                    repo=repo,
                    text=profile_summary(prof),
                    why="repo profile (deterministic)",
                ).model_dump()
            )
            key_files = [s["file"] for s in prof.compose_services] + [w["file"] for w in prof.workflows]
            indexed = [p for _, p in deps.db.indexed_paths([repo])]
            readmes = [p for p in indexed if p.lower() in ("readme.md", "readme", "readme.rst")]
            tf = [p for p in indexed if p.endswith(".tf")][:2]
            for path in dict.fromkeys(readmes[:1] + key_files + tf):
                try:
                    snip = files.read_range(roots[repo], path, 1, 80)
                except files.FileAccessError:
                    continue
                if snip:
                    evs.append(ev_from_snippet(repo, snip, "tool", "architecture key file"))
            if deps.use_vector:
                evs += vector_evidence(
                    deps,
                    state.get("rewritten_query") or f"{state['question']} {ARCH_QUERY}",
                    [repo],
                    ["doc", "config", "code"],
                    4,
                    ctx,
                    "architecture",
                )
        ctx.detail = f"{len(scope)} profiles, {len(evs)} evidence"
        return {"evidence": evs}

    return load_profiles


def route_after_profiles(deps: Deps):
    def route(state: GraphState) -> list[Send] | str:
        scope = state.get("scope_repos") or []
        # Per-repo summaries only for multi-repo scopes, and only if the budget allows them.
        if len(scope) > 1 and deps.use_loop and deps.remaining(state) > len(scope) + 2:
            return [
                Send("summarize_repo", {"repo": r, "llm_calls": state.get("llm_calls", 0)}) for r in scope
            ]
        return "connect_map"

    return route


def make_summarize_repo(deps: Deps):
    def summarize_repo(payload: dict[str, Any], ctx: NodeCtx) -> dict[str, Any]:
        repo = payload["repo"]
        prof = _profile(deps, repo)
        hits = (
            deps.store.search(ARCH_QUERY, 4, {"repo": [repo], "kind": ["doc", "config"]})
            if deps.use_vector
            else []
        )
        excerpts = "\n\n".join(f"{h.path}:\n{h.text[:1200]}" for h in hits) or "(none)"
        prompt = prompts.messages("summarize_repo", profile=profile_summary(prof), excerpts=excerpts)
        try:
            s = structured(deps.llm, "fast", RepoSummary, prompt, ctx.meter)
            s.repo = repo
        except (LLMError, BudgetExceeded) as e:
            ctx.detail = f"{repo}: failed {type(e).__name__}"
            return {"notices": [f"Summary for {repo} failed; using its profile only."]}
        text = (
            f"purpose: {s.purpose}\ncomponents: {', '.join(s.components)}\n"
            f"interfaces: {', '.join(s.interfaces)}\ndeploy: {s.deploy}"
        )
        ev = Evidence(
            id=f"summary:{repo}", source="tool", repo=repo, text=text, why="per-repo summary (fast model)"
        )
        ctx.detail = repo
        return {"summaries": [s.model_dump()], "evidence": [ev.model_dump()]}

    return summarize_repo


def _route_regex(path: str) -> re.Pattern[str]:
    parts = []
    for seg in path.strip("/").split("/"):
        if re.fullmatch(r"\{[^}]*\}|:\w+|<[^>]*>|\[[^\]]*\]", seg):
            parts.append(r"[^/]+")
        else:
            parts.append(re.escape(seg))
    return re.compile("^/?" + "/".join(parts) + "/?$")


def _url_path(url: str) -> str:
    u = re.sub(r"^\w+://[^/]+", "", url)
    u = u.split("?")[0]
    return u if u.startswith("/") else "/" + u


def connections(profiles: dict[str, RepoProfile]) -> list[dict[str, Any]]:
    """Deterministic links between repos (and within one repo)."""
    out: list[dict[str, Any]] = []
    repos = list(profiles)
    # routes defined in A, called in B
    for a in repos:
        for r in profiles[a].routes_defined:
            if not r["path"] or r["path"] == "/":
                continue
            rx = _route_regex(r["path"])
            for b in repos:
                for u in profiles[b].urls_called:
                    if rx.match(_url_path(u["url"])):
                        out.append(
                            {
                                "kind": "http",
                                "from": b,
                                "to": a,
                                "detail": f"{b}/{u['file']}:{u['line']} calls {r['method']} {r['path']} "
                                f"defined at {a}/{r['file']}:{r['line']}",
                                "refs": [(b, u["file"], u["line"]), (a, r["file"], r["line"])],
                            }
                        )
    # shared env vars
    generic = {"PATH", "HOME", "NODE_ENV", "DEBUG", "PORT", "ENV", "LOG_LEVEL", "PYTHONPATH"}
    for i, a in enumerate(repos):
        for b in repos[i + 1 :]:
            shared = sorted((set(profiles[a].env_vars) & set(profiles[b].env_vars)) - generic)
            if shared:
                out.append(
                    {
                        "kind": "env",
                        "from": a,
                        "to": b,
                        "detail": f"{a} and {b} share env vars: {', '.join(shared)}",
                        "refs": [],
                    }
                )
    # compose service names referenced by other services / repos (e.g. http://api:8000)
    services = {s["name"]: (repo, s["file"]) for repo in repos for s in profiles[repo].compose_services}
    for repo in repos:
        for s in profiles[repo].compose_services:
            for dep in s.get("depends_on") or []:
                if dep in services:
                    out.append(
                        {
                            "kind": "compose",
                            "from": repo,
                            "to": services[dep][0],
                            "detail": f"service {s['name']} ({repo}/{s['file']}) depends on {dep}",
                            "refs": [],
                        }
                    )
            for var, url in (s.get("env_urls") or {}).items():
                host = re.match(r"^\w+://([\w-]+)(?::\d+)?", url)
                if host and host.group(1) in services and host.group(1) != s["name"]:
                    to_repo, f = services[host.group(1)]
                    out.append(
                        {
                            "kind": "compose",
                            "from": repo,
                            "to": to_repo,
                            "detail": f"service {s['name']} ({repo}/{s['file']}) sets {var}={url}, "
                            f"which points at compose service {host.group(1)} ({to_repo}/{f})",
                            "refs": [],
                        }
                    )
        for u in profiles[repo].urls_called:
            host = re.match(r"^\w+://([\w-]+)(?::\d+)?", u["url"])
            if host and host.group(1) in services:
                to_repo, f = services[host.group(1)]
                out.append(
                    {
                        "kind": "compose",
                        "from": repo,
                        "to": to_repo,
                        "detail": f"{repo}/{u['file']}:{u['line']} calls compose service {host.group(1)} ({to_repo}/{f})",
                        "refs": [(repo, u["file"], u["line"])],
                    }
                )
    # shared deploy workflows (same workflow name or same reusable workflow)
    by_name: dict[str, list[str]] = {}
    for repo in repos:
        for w in profiles[repo].workflows:
            by_name.setdefault(str(w["name"]).lower(), []).append(f"{repo}/{w['file']}")
    for name, where in by_name.items():
        if len({w.split("/")[0] for w in where}) > 1:
            out.append(
                {
                    "kind": "workflow",
                    "from": "",
                    "to": "",
                    "detail": f"workflow '{name}' in {', '.join(where)}",
                    "refs": [],
                }
            )
    return out


def make_connect_map(deps: Deps):
    def connect_map(state: GraphState, ctx: NodeCtx) -> dict[str, Any]:
        scope = state.get("scope_repos") or list(deps.cfg.repos)
        roots = deps.roots(scope)
        profiles = {r: _profile(deps, r) for r in scope}
        conns = connections(profiles)
        evs: list[dict[str, Any]] = []
        if conns:
            evs.append(
                Evidence(
                    id="connections",
                    source="tool",
                    text="\n".join(f"- [{c['kind']}] {c['detail']}" for c in conns),
                    why="connection map (deterministic)",
                ).model_dump()
            )
        seen = set()
        for c in conns:
            for repo, path, line in c["refs"]:
                if (repo, path, line) in seen or repo not in roots:
                    continue
                seen.add((repo, path, line))
                try:
                    snip = files.window(roots[repo], path, line, 4)
                except files.FileAccessError:
                    continue
                if snip:
                    evs.append(ev_from_snippet(repo, snip, "tool", f"connection: {c['kind']}"))
        ctx.detail = f"{len(conns)} connections"
        return {"evidence": evs[:20]}

    return connect_map
