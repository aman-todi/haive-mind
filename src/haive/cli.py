"""haive command-line interface."""

from __future__ import annotations

import json
import os
import re
import select
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm, Prompt
from rich.table import Table

from haive import __version__, paths, threads
from haive.config import Config, RepoConfig, load_config, save_config
from haive.llm.budget import daily_status

app = typer.Typer(
    help="haive: ask questions about, and debug, your own git repositories.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
threads_app = typer.Typer(help="List or delete chat threads.", invoke_without_command=True)
app.add_typer(threads_app, name="threads")

console = Console()
err = Console(stderr=True)

RepoOpt = Annotated[list[str] | None, typer.Option("--repo", "-r", help="Limit to a repo (repeatable).")]
GroupOpt = Annotated[list[str] | None, typer.Option("--group", "-g", help="Limit to a group (repeatable).")]
PaidOpt = Annotated[
    bool, typer.Option("--paid", help="Answer with models.strong_paid instead of the free strong model.")
]
WebOpt = Annotated[
    str | None, typer.Option("--web", help="Web search: off | ask | auto (default from config).")
]


def fail(msg: str, code: int = 1) -> None:
    err.print(f"[red]error:[/] {msg}")
    raise typer.Exit(code)


def runtime(require_init: bool = True):
    from haive.runtime import Runtime

    if require_init and not paths.config_file().exists():
        fail("haive is not initialized. Run `haive init` first.")
    return Runtime()


def rel_time(ts: float | None) -> str:
    return threads.relative(ts) if ts else "never"


@app.callback(invoke_without_command=True)
def _main(
    version: Annotated[bool, typer.Option("--version", help="Show version and exit.")] = False,
) -> None:
    if version:
        console.print(f"haive {__version__}")
        raise typer.Exit()


# ================================================================== setup


ENV_TEMPLATE = """# haive secrets. These are read from the environment first, then from this file.
OPENROUTER_API_KEY=
PINECONE_API_KEY=
# Optional: better web search results
# TAVILY_API_KEY=
"""


def check_openrouter(cfg: Config) -> tuple[bool, str]:
    key = os.environ.get(cfg.models.api_key_env)
    if not key:
        return False, f"{cfg.models.api_key_env} is not set"
    if "openrouter.ai" not in cfg.models.base_url:
        return True, f"key set; endpoint {cfg.models.base_url}"
    try:
        import httpx

        r = httpx.get(
            f"{cfg.models.base_url.rstrip('/')}/key", headers={"Authorization": f"Bearer {key}"}, timeout=15
        )
        if r.status_code != 200:
            return False, f"OpenRouter rejected the key (HTTP {r.status_code})"
        data = r.json().get("data", {})
        tier = "free tier" if data.get("is_free_tier") else "paid credits"
        return True, f"valid ({tier}; limit {data.get('limit')}, usage {data.get('usage')})"
    except Exception as e:  # network trouble shouldn't block init
        return False, f"could not reach OpenRouter: {e}"


@app.command()
def init(
    backend: Annotated[str | None, typer.Option(help="Vector store: pinecone | chroma.")] = None,
    check: Annotated[
        bool, typer.Option("--check/--no-check", help="Validate keys and create the index.")
    ] = True,
) -> None:
    """Create config + data dirs, validate keys, and create the vector index."""
    from haive.config import load_secrets

    paths.ensure_dirs()
    cfg_path = paths.config_file()
    cfg = load_config() if cfg_path.exists() else Config()
    if backend:
        if backend not in ("pinecone", "chroma"):
            fail("backend must be pinecone or chroma")
        cfg.vector_store.backend = backend  # type: ignore[assignment]
    save_config(cfg)
    console.print(f"[green]✓[/] config: {cfg_path}")
    env = paths.env_file()
    if not env.exists():
        env.write_text(ENV_TEMPLATE)
        env.chmod(0o600)
        console.print(f"[green]✓[/] secrets file: {env} (fill in your keys, or export them)")
    console.print(f"[green]✓[/] data dir: {paths.data_dir()}")
    from haive.db import DB

    DB(paths.db_file()).close()
    load_secrets()
    if not check:
        return

    ok, msg = check_openrouter(cfg)
    console.print(f"{'[green]✓' if ok else '[yellow]![/yellow]'}[/] OpenRouter: {escape(msg)}")
    if cfg.vector_store.backend == "pinecone":
        if not os.environ.get("PINECONE_API_KEY"):
            console.print("[yellow]![/] Pinecone: PINECONE_API_KEY is not set")
        else:
            try:
                from haive.index.embed import HashEmbedder
                from haive.store.pinecone_store import PineconeStore

                vs = cfg.vector_store
                store = PineconeStore(
                    HashEmbedder(cfg.embeddings.dim), vs.pinecone_index, vs.pinecone_cloud, vs.pinecone_region
                )
                created = store.ensure_index()
                console.print(
                    f"[green]✓[/] Pinecone index '{vs.pinecone_index}' {'created' if created else 'ok'}"
                )
            except Exception as e:
                console.print(f"[red]✗[/] Pinecone: {escape(str(e))}")
    else:
        try:
            import langchain_chroma  # noqa: F401

            console.print(f"[green]✓[/] Chroma at {paths.chroma_dir()}")
        except ImportError:
            console.print("[red]✗[/] Chroma backend needs the extra: uv tool install '.[chroma]'")
    console.print(
        f"{'[green]✓' if shutil.which('rg') else '[yellow]!'}[/] ripgrep "
        f"{'found' if shutil.which('rg') else 'not found (slow Python fallback will be used)'}"
    )
    from haive.index.chunk_code import TS_GRAMMARS, get_language

    missing = [lang for lang in TS_GRAMMARS if get_language(lang) is None]
    if missing:
        console.print(
            f"[yellow]![/] tree-sitter grammars missing for {', '.join(missing)} (line windows used)"
        )
    else:
        console.print("[green]✓[/] tree-sitter grammars available")
    console.print(
        "\nNext: [bold]haive add <path|git-url>[/], then [bold]haive index[/]. The first index downloads the "
        f"embedding model ({cfg.embeddings.model}).\nCheck that the default free models in {cfg_path} still exist "
        "on openrouter.ai/models, and review OpenRouter's privacy settings for free models."
    )


def _is_url(s: str) -> bool:
    return bool(re.match(r"^(https?://|git@|ssh://|git://)", s)) or s.endswith(".git")


def _name_from(src: str) -> str:
    base = re.sub(r"\.git$", "", src.rstrip("/")).split("/")[-1].split(":")[-1]
    return re.sub(r"[^A-Za-z0-9_.-]", "-", base) or "repo"


class AddError(Exception):
    pass


class AddSkip(Exception):
    pass


def _register(rt: Any, source: str, name: str | None, groups: list[str]) -> str:
    """Register one repo in ``rt.cfg`` (not saved). Returns its name.

    Raises AddSkip when the repo is already registered at the same path, AddError otherwise."""
    from haive.tools import git

    cfg = rt.cfg
    name = name or _name_from(source)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise AddError(f"{name!r}: repo names may only contain letters, digits, '.', '_' and '-'")
    if not _is_url(source):
        root = Path(source).expanduser().resolve()
        if any(rc.root == root for rc in cfg.repos.values()):
            raise AddSkip(f"{root} is already registered")
    if name in cfg.repos:
        raise AddError(f"a repo named {name!r} is already registered (use --name to pick another)")
    if name in cfg.groups:
        raise AddError(f"{name!r} is already a group name")
    if _is_url(source):
        console.print(f"Cloning {source} …")
        try:
            root = git.clone(source, name)
        except git.GitError as e:
            raise AddError(f"clone failed: {e}") from e
        remote = source
    else:
        if not git.is_repo(root):
            raise AddError(f"{root} is not a git repository")
        remote = git.remote_url(root)
    cfg.repos[name] = RepoConfig(path=str(root), remote_url=remote)
    for g in groups:
        members = cfg.groups.setdefault(g, [])
        if name not in members:
            members.append(name)
    rt.db.upsert_repo(name, str(root), remote)
    return name


def _scan(directory: str) -> tuple[list[str], int]:
    """Git repos directly inside ``directory`` (sorted), and how many other folders were passed over."""
    from haive.tools import git

    base = Path(directory).expanduser().resolve()
    if not base.is_dir():
        fail(f"{base} is not a directory")
    repos, others = [], 0
    for child in sorted(base.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if (child / ".git").exists() and git.is_repo(child):
            repos.append(str(child))
        else:
            others += 1
    return repos, others


@app.command()
def add(
    sources: Annotated[
        list[str],
        typer.Argument(
            help="Local paths or git URLs (URLs are cloned into the data dir). With --scan: directories."
        ),
    ],
    name: Annotated[
        str | None, typer.Option("--name", help="Repo name (default: directory name). Single repo only.")
    ] = None,
    group: Annotated[
        list[str] | None, typer.Option("--group", "-g", help="Add to a group (repeatable).")
    ] = None,
    scan: Annotated[
        bool, typer.Option("--scan", help="Register every git repo directly inside the given directories.")
    ] = False,
) -> None:
    """Register one or more repos (indexing is separate: `haive index`)."""
    rt = runtime()
    groups = group or []
    targets: list[str] = []
    passed_over = 0
    if scan:
        for d in sources:
            found, others = _scan(d)
            targets += found
            passed_over += others
        if not targets:
            fail(f"no git repositories found directly inside {', '.join(sources)}")
    else:
        targets = list(sources)
    if name and len(targets) > 1:
        fail("--name only works when adding a single repo")

    if len(targets) == 1 and not scan:  # single repo: any problem is an error
        try:
            added = _register(rt, targets[0], name, groups)
        except (AddError, AddSkip) as e:
            fail(str(e))
        save_config(rt.cfg)
        root = rt.cfg.repos[added].path
        console.print(
            f"[green]✓[/] added {added} → {root}" + (f" (groups: {', '.join(groups)})" if groups else "")
        )
        console.print(f"Run [bold]haive index {added}[/] to index it.")
        return

    added_names: list[str] = []
    skipped = errors = 0
    for src in targets:
        try:
            added_names.append(_register(rt, src, None, groups))
            console.print(f"[green]✓[/] {added_names[-1]} → {rt.cfg.repos[added_names[-1]].path}")
        except AddSkip as e:
            skipped += 1
            console.print(f"[dim]- skipped: {escape(str(e))}[/]")
        except AddError as e:
            errors += 1
            err.print(f"[yellow]![/] {escape(src)}: {escape(str(e))}")
    if added_names:
        save_config(rt.cfg)
    summary = f"{len(added_names)} added, {skipped} already registered, {errors} failed"
    if scan and passed_over:
        summary += f", {passed_over} non-repo folders ignored"
    if groups and added_names:
        summary += f" (groups: {', '.join(groups)})"
    console.print(summary)
    if added_names:
        console.print("Run [bold]haive index[/] to index everything registered.")
    if errors:
        raise typer.Exit(1)


@app.command()
def remove(
    name: str,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Don't ask for confirmation.")] = False,
) -> None:
    """Unregister a repo and delete its vectors and rows."""
    rt = runtime()
    cfg = rt.cfg
    if name not in cfg.repos:
        fail(f"unknown repo {name!r}")
    if not yes and not Confirm.ask(f"Remove {name} and delete its index?"):
        raise typer.Exit()
    from haive.index.pipeline import Indexer

    root = cfg.repos[name].root
    n = Indexer(cfg, rt.db, rt.store).remove_repo(name)
    del cfg.repos[name]
    for g in list(cfg.groups):
        cfg.groups[g] = [m for m in cfg.groups[g] if m != name]
    save_config(cfg)
    console.print(f"[green]✓[/] removed {name} ({n} vectors deleted)")
    if root.is_relative_to(paths.clones_dir()):
        console.print(f"The clone at {root} was left in place; delete it if you no longer need it.")


@app.command()
def repos() -> None:
    """List repos: path, last indexed, chunk/commit counts, stale?"""
    from haive.index import freshness

    rt = runtime()
    if not rt.cfg.repos:
        console.print("No repos registered. Use [bold]haive add <path>[/].")
        return
    t = Table(header_style="bold")
    for col in ("repo", "path", "last indexed", "code", "config", "doc", "commits", "stale?"):
        t.add_column(col)
    for name in rt.cfg.repos:
        row = rt.db.get_repo(name)
        counts = rt.db.chunk_counts(name)
        rf = freshness.check_repo(rt.cfg, rt.db, name)
        if rf.never_indexed:
            stale = "[yellow]not indexed[/]"
        elif rf.error:
            stale = f"[red]{escape(rf.error[:40])}[/]"
        elif rf.changed:
            stale = f"[yellow]{len(rf.changed)} files[/]"
        elif rf.head_moved:
            stale = "[yellow]new commits[/]"
        else:
            stale = "[green]fresh[/]"
        t.add_row(
            name,
            str(rt.cfg.repos[name].path),
            rel_time(row["last_indexed_at"] if row else None),
            *(str(counts.get(k, 0)) for k in ("code", "config", "doc", "commit")),
            stale,
        )
    console.print(t)


@app.command()
def groups() -> None:
    """List groups and their repos (groups are edited in config.toml)."""
    cfg = runtime().cfg
    if not cfg.groups:
        console.print(f"No groups. Add them under [groups] in {paths.config_file()}, or use `haive add -g`.")
        return
    t = Table(header_style="bold")
    t.add_column("group")
    t.add_column("repos")
    for g, members in cfg.groups.items():
        shown = [m if m in cfg.repos else f"[red]{m} (unknown)[/]" for m in members]
        t.add_row(g, ", ".join(shown))
    console.print(t)


class IndexProgress:
    """Rich progress for the indexer (files, chunks, embedding throughput)."""

    def __init__(self) -> None:
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.fields[repo]}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[info]}"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        )
        self.task = None
        self.chunks = 0
        self.embed_s = 0.0

    def __call__(self, event: str, **kw: Any) -> None:
        if event == "start":
            self.chunks, self.embed_s = 0, 0.0
            self.task = self.progress.add_task("index", total=max(kw["total"], 1), repo=kw["repo"], info="")
        elif event == "file" and self.task is not None:
            self.progress.advance(self.task)
        elif event == "embedded" and self.task is not None:
            self.chunks += kw["n"]
            self.embed_s += kw["seconds"]
            rate = self.chunks / self.embed_s if self.embed_s else 0
            self.progress.update(self.task, info=f"{self.chunks} chunks · {rate:.0f}/s")
        elif event == "done" and self.task is not None:
            self.progress.update(self.task, completed=self.progress.tasks[-1].total)


def _index(rt: Any, names: list[str], full: bool) -> None:
    from haive.index.pipeline import Indexer
    from haive.tools.git import GitError

    prog = IndexProgress()
    idx = Indexer(rt.cfg, rt.db, rt.store, progress=prog)
    with prog.progress:
        results = []
        for n in names:
            try:
                results.append(idx.index_repo(n, full=full))
            except GitError as e:
                err.print(f"[red]✗[/] {n}: {e}")
    for st in results:
        console.print(f"[green]✓[/] {st.summary()}")


@app.command()
def index(
    names: Annotated[list[str] | None, typer.Argument(help="Repos or groups (default: all).")] = None,
    full: Annotated[bool, typer.Option("--full", help="Rebuild from scratch.")] = False,
) -> None:
    """Index repos (incremental by default)."""
    rt = runtime()
    targets, unknown = rt.cfg.resolve_names(names) if names else (list(rt.cfg.repos), [])
    if unknown:
        fail(f"unknown repos/groups: {', '.join(unknown)}")
    if not targets:
        fail("no repos to index. Use `haive add <path>` first.")
    _index(rt, targets, full)


@app.command()
def sync() -> None:
    """git pull --ff-only on every repo, then incremental index."""
    from haive.tools import git

    rt = runtime()
    ok = []
    for name, rc in rt.cfg.repos.items():
        try:
            out = git.pull_ff_only(rc.root)
            console.print(f"[green]✓[/] {name}: {escape(out.splitlines()[-1] if out else 'up to date')}")
            ok.append(name)
        except git.GitError as e:
            err.print(
                f"[yellow]![/] {name}: pull skipped ({escape(str(e).splitlines()[-1] if str(e) else 'error')})"
            )
            ok.append(name)  # still index local changes
    if ok:
        _index(rt, ok, full=False)


@app.command()
def status() -> None:
    """Vector store, models per role, today's LLM usage, index health."""
    rt = runtime()
    cfg = rt.cfg
    console.print("[bold]Vector store[/]")
    store_counts: dict[str, int | str] = {}
    try:
        store = rt.store
        console.print(f"  backend: {cfg.vector_store.backend}  total vectors: {store.count()}")
        for name in cfg.repos:
            try:
                store_counts[name] = store.count({"repo": [name]})
            except Exception as e:
                store_counts[name] = f"? ({type(e).__name__})"
    except Exception as e:
        console.print(f"  [red]unavailable:[/] {escape(str(e))}")
    console.print("\n[bold]Models[/]")
    m = cfg.models
    console.print(
        f"  fast: {m.fast}\n  strong: {m.strong}{'' if m.use_paid_strong else '  (in use)'}\n"
        f"  strong_paid: {m.strong_paid}{'  (in use)' if m.use_paid_strong else '  (off; --paid or use_paid_strong)'}\n"
        f"  fallback: {', '.join(m.fallback)}\n  endpoint: {m.base_url}"
    )
    console.print("\n[bold]LLM usage today[/]")
    used = rt.db.calls_today()
    st = daily_status(used, cfg.limits.daily_call_budget)
    color = {"ok": "green", "warn": "yellow", "over": "red"}[st]
    console.print(f"  [{color}]{used} / {cfg.limits.daily_call_budget} calls[/]")
    for r in rt.db.usage_today():
        console.print(
            f"  {r['role']:>6} {r['model']}: {r['calls']} calls, {r['prompt_tokens']}+{r['completion_tokens']} "
            f"tokens, {r['errors']} errors"
        )
    console.print("\n[bold]Index health[/]")
    for name in cfg.repos:
        row = rt.db.get_repo(name)
        n_db = sum(rt.db.chunk_counts(name).values())
        n_store = store_counts.get(name, "?")
        mark = "[green]✓[/]" if n_store == n_db else "[yellow]![/]"
        console.print(
            f"  {mark} {name}: {n_db} chunks in SQLite, {n_store} vectors; "
            f"last indexed {rel_time(row['last_indexed_at'] if row else None)}"
        )


# ================================================================== questions


def _scope(repo: list[str] | None, group: list[str] | None) -> list[str]:
    return [*(repo or []), *(group or [])]


def _budget_gate(rt: Any, force: bool) -> None:
    used = rt.db.calls_today()
    st = daily_status(used, rt.cfg.limits.daily_call_budget)
    if st == "over" and not force:
        fail(
            f"daily LLM call budget reached ({used}/{rt.cfg.limits.daily_call_budget}). Use --force to continue."
        )
    if st == "warn":
        err.print(f"[yellow]Warning:[/] {used}/{rt.cfg.limits.daily_call_budget} daily LLM calls used.")
    if not os.environ.get(rt.cfg.models.api_key_env):
        fail(f"{rt.cfg.models.api_key_env} is not set (see `haive init`).")


def _on_interrupt(payload: dict[str, Any]) -> Any:
    if payload.get("kind") == "clarify":
        q = payload.get("question") or "Which repos or groups?"
        err.print(f"[bold]?[/] {escape(q)}  [dim]options: {', '.join(payload.get('options', []))}, all[/]")
        return Prompt.ask("scope", default="all", console=err)
    if payload.get("kind") == "web":
        return Confirm.ask(
            f"Search the web for [cyan]{escape(payload.get('query', ''))}[/]?", default=True, console=err
        )
    return None


def _web_mode(web: str | None) -> str | None:
    if web is not None and web not in ("off", "ask", "auto"):
        fail("--web must be off, ask, or auto")
    return web


def _show(result: Any, as_json: bool, trace: bool) -> None:
    from haive.render import render_answer, render_trace, to_json

    if as_json:
        meta = result.meta()
        if trace:
            meta["node_log"] = result.state.get("node_log")
        print(to_json(result.answer, meta))
        return
    if result.answer:
        render_answer(console, result.answer, result.meta())
    if trace:
        render_trace(console, result.state.get("node_log") or [])


def _one_shot(
    question: str,
    trace_text: str | None,
    scope: list[str],
    web: str | None,
    as_json: bool,
    trace: bool,
    force: bool,
    interactive: bool,
    paid: bool = False,
) -> None:
    rt = runtime()
    _budget_gate(rt, force)
    if paid:
        rt.cfg.models.use_paid_strong = True
    runner = rt.runner()
    tid = f"oneshot-{uuid.uuid4().hex}"
    try:
        result = runner.run(
            question,
            thread_id=tid,
            trace_text=trace_text,
            scope=scope or None,
            web_mode=_web_mode(web),
            on_interrupt=_on_interrupt,
            interactive=interactive,
        )
    finally:
        try:
            rt.checkpointer.delete_thread(tid)
        except Exception:
            pass
    _show(result, as_json, trace)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Your question.")],
    repo: RepoOpt = None,
    group: GroupOpt = None,
    web: WebOpt = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the structured answer as JSON.")] = False,
    trace: Annotated[bool, typer.Option("--trace", help="Show the path through the graph.")] = False,
    force: Annotated[bool, typer.Option("--force", help="Ignore the daily call budget.")] = False,
    paid: PaidOpt = False,
) -> None:
    """Ask a question about your repos."""
    interactive = sys.stdin.isatty() and sys.stderr.isatty() and not as_json
    _one_shot(question, None, _scope(repo, group), web, as_json, trace, force, interactive, paid)


@app.command()
def debug(
    file: Annotated[
        str | None, typer.Argument(help="File with the trace/error output, or '-' for stdin.")
    ] = None,
    question: Annotated[str, typer.Option("-q", "--question", help="Extra question or context.")] = "",
    repo: RepoOpt = None,
    group: GroupOpt = None,
    web: WebOpt = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
    trace: Annotated[bool, typer.Option("--trace")] = False,
    force: Annotated[bool, typer.Option("--force")] = False,
    paid: PaidOpt = False,
) -> None:
    """Diagnose an error trace and recommend a fix (never applied)."""
    from_stdin = file in (None, "-")
    if from_stdin:
        if sys.stdin.isatty():
            err.print("Paste the error output, then press Ctrl-D:")
        text = sys.stdin.read()
    else:
        p = Path(file)
        if not p.exists():
            fail(f"{file} does not exist")
        text = p.read_text(errors="replace")
    if not text.strip():
        fail("no trace text given")
    interactive = not from_stdin and sys.stdin.isatty() and sys.stderr.isatty() and not as_json
    _one_shot(question, text, _scope(repo, group), web, as_json, trace, force, interactive, paid)


# ================================================================== chat

CHAT_HELP = (
    "[bold]/scope[/] <repos or groups|all>  [bold]/web[/] off|ask|auto  [bold]/new[/]  [bold]/threads[/]  "
    "[bold]/exit[/]\nPaste an error trace to debug it."
)


TRACE_STARTS = re.compile(
    r"^(Traceback \(most recent call last\)|panic: |thread '.*' panicked|Exception in thread|Unhandled exception|"
    r"==\d+==ERROR|\w*(Error|Exception)\b.*:|\S+\.\w+[:(]\d+)"
)


_pending: list[str] = []


def _read_input(prompt: str) -> str | None:
    """One message. Multi-line pastes are collected three ways: lines that arrive in the same burst
    (terminal paste), a ``` fence, or (when stdin is piped) a trace-like first line read up to a blank line."""
    if _pending:
        first = _pending.pop()
    else:
        try:
            first = input(prompt)
        except EOFError:
            return None
    lines = [first]

    def more() -> str | None:
        try:
            return input()
        except EOFError:
            return None

    if first.strip().startswith("```"):
        while (line := more()) is not None and not line.strip().startswith("```"):
            lines.append(line)
        return "\n".join(lines[1:])
    try:
        selectable = sys.stdin.isatty() and select.select([sys.stdin], [], [], 0) is not None
    except (OSError, ValueError):
        selectable = False
    if selectable:
        while select.select([sys.stdin], [], [], 0.05)[0]:
            line = sys.stdin.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))
    elif TRACE_STARTS.match(first.strip()):
        # piped input: traces contain single blank lines, so stop at two, at EOF, or at a slash command
        while (line := more()) is not None:
            if line.startswith("/"):
                _pending.append(line)
                break
            if not line.strip() and lines and not lines[-1].strip():
                break
            lines.append(line)
    return "\n".join(lines).strip("\n")


@app.command()
def chat(
    resume: Annotated[str | None, typer.Option("--resume", help="Thread ID (or prefix) to resume.")] = None,
    repo: RepoOpt = None,
    group: GroupOpt = None,
    web: WebOpt = None,
    force: Annotated[bool, typer.Option("--force")] = False,
    paid: PaidOpt = False,
) -> None:
    """Interactive chat with saved, titled, resumable threads."""
    from haive.render import render_answer
    from haive.tools.trace_parse import looks_like_trace

    rt = runtime()
    _budget_gate(rt, force)
    if paid:
        rt.cfg.models.use_paid_strong = True
    runner = rt.runner(persist_thread=True)
    scope = _scope(repo, group)
    web_mode = _web_mode(web) or rt.cfg.web.mode
    if resume:
        tid = threads.resolve(rt.db, resume)
        if tid in (None, "ambiguous"):
            fail(f"no unique thread matches {resume!r}")
        row = threads.get(rt.db, tid)
        if not scope and row and row.get("scope_json"):
            scope = (
                json.loads(row["scope_json"])
                if len(json.loads(row["scope_json"])) < len(rt.cfg.repos)
                else []
            )
        console.print(f"Resuming [bold]{escape(row.get('title') or tid[:8])}[/] ({row.get('turns')} turns)")
    else:
        tid = threads.new_id()
    console.print(f"[dim]thread {tid[:8]} · scope: {', '.join(scope) or 'all'} · web: {web_mode}[/]")
    console.print(f"[dim]{CHAT_HELP}[/]")
    while True:
        text = _read_input("\n› ")
        if text is None:
            break
        text = text.strip()
        if not text:
            continue
        if text.startswith("/"):
            cmd, _, arg = text.partition(" ")
            arg = arg.strip()
            if cmd in ("/exit", "/quit"):
                break
            if cmd == "/help":
                console.print(CHAT_HELP)
            elif cmd == "/scope":
                if not arg or arg == "all":
                    scope = []
                else:
                    names = re.split(r"[,\s]+", arg)
                    resolved, unknown = rt.cfg.resolve_names(names)
                    if unknown:
                        console.print(f"[yellow]unknown: {', '.join(unknown)}[/]")
                    scope = names if resolved else scope
                console.print(f"scope: {', '.join(scope) or 'all'}")
            elif cmd == "/web":
                if arg in ("off", "ask", "auto"):
                    web_mode = arg
                console.print(f"web: {web_mode}")
            elif cmd == "/new":
                tid = threads.new_id()
                console.print(f"[dim]new thread {tid[:8]}[/]")
            elif cmd == "/threads":
                _print_threads(rt, 10)
            else:
                console.print(f"unknown command {cmd}. {CHAT_HELP}")
            continue
        is_trace = "\n" in text and looks_like_trace(text)
        try:
            result = runner.run(
                "" if is_trace else text,
                thread_id=tid,
                trace_text=text if is_trace else None,
                scope=scope or None,
                web_mode=web_mode,
                on_interrupt=_on_interrupt,
                interactive=True,
            )
        except KeyboardInterrupt:
            console.print("[dim]cancelled[/]")
            continue
        if result.answer:
            render_answer(console, result.answer, result.meta())
        if result.state.get("title"):
            console.print(f"[dim]thread: {escape(result.state['title'])}[/]")


def _print_threads(rt: Any, limit: int) -> None:
    rows = threads.list_threads(rt.db, limit)
    if not rows:
        console.print("No threads yet. Start one with [bold]haive chat[/].")
        return
    t = Table(header_style="bold")
    for col in ("id", "title", "scope", "turns", "updated"):
        t.add_column(col)
    for r in rows:
        scope = json.loads(r["scope_json"] or "[]")
        t.add_row(
            r["id"][:8],
            escape(r["title"] or "(untitled)"),
            ", ".join(scope) or "all",
            str(r["turns"]),
            threads.relative(r["updated_at"]),
        )
    console.print(t)


@threads_app.callback()
def threads_list(
    ctx: typer.Context,
    limit: Annotated[int, typer.Option("--limit", help="How many to show.")] = 20,
) -> None:
    """List threads: ID, title, scope, turns, last updated (newest first)."""
    if ctx.invoked_subcommand is None:
        _print_threads(runtime(), limit)


@threads_app.command("delete")
def threads_delete(thread_id: Annotated[str, typer.Argument(help="Thread ID or prefix.")]) -> None:
    """Delete a thread and its saved state."""
    rt = runtime()
    tid = threads.resolve(rt.db, thread_id)
    if tid in (None, "ambiguous"):
        fail(f"no unique thread matches {thread_id!r}")
    threads.delete(rt.db, tid, rt.checkpointer)
    console.print(f"[green]✓[/] deleted thread {tid[:8]}")


# ================================================================== eval


@app.command("eval")
def eval_cmd(
    suite: Annotated[
        str | None, typer.Option("--suite", help="Questions YAML (default eval/questions.yaml).")
    ] = None,
    ablation: Annotated[
        str | None,
        typer.Option(
            "--ablation", help="vector_only | ripgrep_only | hybrid | hybrid+loop | pinecone | chroma | all"
        ),
    ] = None,
    models: Annotated[
        list[str] | None, typer.Option("--models", help="ROLE=MODEL overrides (repeatable).")
    ] = None,
    limit: Annotated[int | None, typer.Option("--limit", help="Only the first N questions.")] = None,
    setup_sample: Annotated[
        bool, typer.Option("--setup-sample", help="Build and register the bundled sample repos + bug cases.")
    ] = False,
) -> None:
    """Run the eval suite (dev command; needs a source checkout)."""
    root = Path(__file__).resolve().parents[2]
    if not (root / "eval" / "run.py").exists():
        fail("haive eval runs from a source checkout (eval/ not found next to the package)")
    sys.path.insert(0, str(root))
    from eval.run import main as eval_main

    overrides = {}
    for m in models or []:
        role, _, mid = m.partition("=")
        if role not in ("fast", "strong") or not mid:
            fail(f"--models expects fast=MODEL or strong=MODEL, got {m!r}")
        overrides[role] = mid
    t0 = time.monotonic()
    try:
        out = eval_main(
            suite=suite,
            ablation=ablation,
            model_overrides=overrides,
            limit=limit,
            console=console,
            runtime=runtime(),
            setup_sample=setup_sample,
        )
    except ValueError as e:
        fail(str(e))
    console.print(f"[green]✓[/] results in {out} ({time.monotonic() - t0:.0f}s)")


if __name__ == "__main__":
    app()
