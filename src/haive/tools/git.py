"""Read-only git access via subprocess.

haive never writes to an indexed repo. Every call goes through ``_git``, which only allows
read-only subcommands. The two state-changing exceptions are ``clone`` (into haive's own data
dir) and ``pull --ff-only`` (``haive sync``), each with its own function and guard.
``GIT_OPTIONAL_LOCKS=0`` stops ``git status`` from opportunistically rewriting the index.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from haive import paths

READ_ONLY = {
    "ls-files",
    "log",
    "blame",
    "rev-list",
    "rev-parse",
    "diff-tree",
    "status",
    "merge-base",
    "cat-file",
    "remote",
    "config",
    "for-each-ref",
}


class GitError(RuntimeError):
    pass


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env.setdefault("LC_ALL", "C")
    return env


def _git(root: Path | str, *args: str, check: bool = True, timeout: float = 60) -> str:
    if not args or args[0] not in READ_ONLY:
        raise GitError(f"refusing non-read-only git command: {args[:1]}")
    if args[0] == "config" and not any(a in ("--get", "--get-regexp", "--list", "-l") for a in args):
        raise GitError("git config is only allowed for reads")
    if args[0] == "remote" and len(args) > 1 and args[1] not in ("get-url", "-v"):
        raise GitError("git remote is only allowed for reads")
    proc = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        env=_env(),
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        raise GitError(proc.stderr.decode("utf-8", "replace").strip() or f"git {args[0]} failed")
    return proc.stdout.decode("utf-8", "replace")


def is_repo(root: Path | str) -> bool:
    try:
        return _git(root, "rev-parse", "--is-inside-work-tree").strip() == "true"
    except (GitError, FileNotFoundError, NotADirectoryError):
        return False


def head(root: Path | str) -> str | None:
    try:
        return _git(root, "rev-parse", "HEAD").strip()
    except GitError:
        return None  # empty repo


def remote_url(root: Path | str) -> str | None:
    try:
        return _git(root, "remote", "get-url", "origin").strip() or None
    except GitError:
        return None


def ls_files(root: Path | str) -> list[str]:
    tracked = _git(root, "ls-files", "-z", "--cached").split("\0")
    untracked = _git(root, "ls-files", "-z", "--others", "--exclude-standard").split("\0")
    seen: dict[str, None] = {}
    for p in tracked + untracked:
        if p:
            seen[p] = None
    return list(seen)


def changed_since(root: Path | str, rev: str) -> list[str]:
    """Paths changed between ``rev`` and the working tree, plus untracked files.

    Uses ``diff-tree`` (commit to commit) plus ``status``, never porcelain ``git diff``: that one
    refreshes stale stat info and rewrites ``.git/index``, which would break the read-only guarantee.
    """
    out = _git(root, "diff-tree", "-r", "--name-only", "-z", "--no-renames", rev, "HEAD").split("\0")
    return sorted({p for p in out if p} | set(status_paths(root)))


def status_paths(root: Path | str) -> list[str]:
    """Paths with uncommitted changes (modified, added, deleted, untracked)."""
    out = _git(root, "status", "--porcelain", "-z", "--untracked-files=all", "--no-renames")
    paths_: list[str] = []
    for entry in out.split("\0"):
        if len(entry) > 3:
            paths_.append(entry[3:])
    return paths_


def status_porcelain(root: Path | str) -> str:
    return _git(root, "status", "--porcelain", "--untracked-files=all")


def is_ancestor(root: Path | str, ancestor: str, rev: str = "HEAD") -> bool:
    try:
        base = _git(root, "merge-base", ancestor, rev).strip()
    except GitError:
        return False
    return base == _git(root, "rev-parse", ancestor).strip()


def rev_list(root: Path | str, spec: str, max_count: int | None = None) -> list[str]:
    args = ["rev-list", spec]
    if max_count:
        args.insert(1, f"--max-count={max_count}")
    try:
        return [s for s in _git(root, *args).split() if s]
    except GitError:
        return []


@dataclass
class BlameLine:
    sha: str
    author: str
    time: int
    line: int
    text: str


def blame(root: Path | str, path: str, start: int, end: int) -> list[BlameLine]:
    try:
        out = _git(root, "blame", "--line-porcelain", "-L", f"{start},{end}", "--", path)
    except GitError:
        return []
    lines: list[BlameLine] = []
    cur: dict[str, str] = {}
    sha, lineno = "", 0
    for raw in out.splitlines():
        if raw.startswith("\t"):
            lines.append(
                BlameLine(sha, cur.get("author", ""), int(cur.get("author-time", "0") or 0), lineno, raw[1:])
            )
            cur = {}
        elif not cur and len(raw) >= 40 and " " in raw and all(c in "0123456789abcdef" for c in raw[:40]):
            parts = raw.split()
            sha, lineno = parts[0], int(parts[2])
            cur = {"_": "1"}
        else:
            k, _, v = raw.partition(" ")
            cur[k] = v
    return lines


LOG_FORMAT = "%x1e%H%x1f%aI%x1f%an%x1f%s%x1f%b%x1f"


def log(
    root: Path | str,
    *,
    n: int | None = None,
    path: str | None = None,
    pickaxe: str | None = None,
    grep: str | None = None,
    line_range: tuple[str, str] | None = None,
    rev: str | None = None,
    numstat: bool = False,
) -> list[dict]:
    """``git log`` returning dicts: sha, date, author, subject, body, files, insertions, deletions."""
    args = ["log", f"--format={LOG_FORMAT}", "--no-color"]
    if n:
        args.append(f"-n{n}")
    if numstat:
        args.append("--numstat")
    if pickaxe:
        args.append(f"-S{pickaxe}")
    if grep:
        args += [f"--grep={grep}", "-i"]
    if line_range:
        func_or_range, file = line_range  # e.g. (":fetch_user", "api/client.py") or ("10,40", path)
        args += ["-L", f"{func_or_range}:{file}"]
    if rev:
        args.append(rev)
    if path and not line_range:
        args += ["--", path]
    try:
        out = _git(root, *args)
    except GitError:
        return []
    return _parse_log(out, with_patch=bool(line_range))


def _parse_log(out: str, with_patch: bool = False) -> list[dict]:
    commits: list[dict] = []
    for rec in out.split("\x1e"):
        if not rec.strip():
            continue
        parts = rec.split("\x1f")
        if len(parts) < 6:
            continue
        sha, date_, author, subject, body, rest = (
            parts[0].strip(),
            parts[1],
            parts[2],
            parts[3],
            parts[4],
            parts[5],
        )
        files: list[str] = []
        ins = dels = 0
        patch = ""
        if with_patch:
            patch = rest.strip()
        else:
            for line in rest.strip().splitlines():
                cols = line.split("\t")
                if len(cols) == 3:
                    a, d, f = cols
                    ins += int(a) if a.isdigit() else 0
                    dels += int(d) if d.isdigit() else 0
                    files.append(f)
        commits.append(
            {
                "sha": sha,
                "date": date_,
                "author": author,
                "subject": subject,
                "body": body.strip(),
                "files": files,
                "insertions": ins,
                "deletions": dels,
                "patch": patch,
            }
        )
    return commits


# ---------------------------------------------------------------- state-changing (guarded)


def clone(url: str, name: str) -> Path:
    """Clone ``url`` into haive's clones dir. The only place haive runs ``git clone``."""
    dest = paths.clones_dir() / name
    if dest.exists():
        raise GitError(f"{dest} already exists")
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["git", "clone", "--", url, str(dest)], capture_output=True, env=_env(), timeout=600
    )
    if proc.returncode != 0:
        raise GitError(proc.stderr.decode("utf-8", "replace").strip())
    return dest


def pull_ff_only(root: Path | str) -> str:
    """``git pull --ff-only``; the only place haive updates a registered repo (``haive sync``)."""
    proc = subprocess.run(
        ["git", "-C", str(root), "pull", "--ff-only"], capture_output=True, env=_env(), timeout=300
    )
    if proc.returncode != 0:
        raise GitError(proc.stderr.decode("utf-8", "replace").strip())
    return proc.stdout.decode("utf-8", "replace").strip()
