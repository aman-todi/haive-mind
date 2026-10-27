"""Commit indexing: the most recent ``max_commits`` per repo, incrementally.

Incremental runs use ``git rev-list {last_commit_sha}..HEAD``. If the last indexed commit is no
longer an ancestor of HEAD (rebase, force-push, branch switch), the repo's commit chunks are
rebuilt. Commits past ``max_commits`` are pruned.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from haive.db import DB
from haive.tools import git
from haive.types import Chunk, CommitInfo, VectorStore

MAX_FILES_LISTED = 25


@dataclass
class CommitStats:
    added: int = 0
    removed: int = 0
    rebuilt: bool = False
    head: str | None = None


def commit_id(repo: str, sha: str) -> str:
    return f"{repo}:commit:{sha}"


def utc_iso(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).astimezone(UTC).isoformat()
    except ValueError:
        return iso


def read_commits(root: Path, repo: str, n: int, rev: str | None = None) -> list[CommitInfo]:
    out = []
    for c in git.log(root, n=n, rev=rev, numstat=True):
        out.append(
            CommitInfo(
                repo=repo,
                sha=c["sha"],
                date=utc_iso(c["date"]),
                author=c["author"],
                subject=c["subject"],
                body=c["body"],
                files=c["files"],
                insertions=c["insertions"],
                deletions=c["deletions"],
            )
        )
    return out


def commit_text(ci: CommitInfo) -> str:
    files = ci.files[:MAX_FILES_LISTED]
    more = f" (+{len(ci.files) - len(files)} more)" if len(ci.files) > len(files) else ""
    parts = [
        f"commit {ci.sha[:12]}  {ci.date}  {ci.author}",
        ci.subject,
    ]
    if ci.body:
        parts.append(ci.body)
    parts.append(f"files: {', '.join(files)}{more}")
    parts.append(f"+{ci.insertions} -{ci.deletions}")
    return "\n".join(parts)


def commit_chunk(ci: CommitInfo) -> Chunk:
    text = commit_text(ci)
    return Chunk(
        id=commit_id(ci.repo, ci.sha),
        repo=ci.repo,
        path="",
        kind="commit",
        language="",
        symbol=ci.sha[:12],
        start_line=0,
        end_line=0,
        text=text,
        content_hash=ci.sha[:12],
    )


def index_commits(db: DB, store: VectorStore, repo: str, root: Path, max_commits: int = 300) -> CommitStats:
    stats = CommitStats(head=git.head(root))
    if stats.head is None:
        return stats
    row = db.get_repo(repo)
    last = row["last_commit_sha"] if row else None

    if last and last == stats.head:
        return stats
    if last and git.is_ancestor(root, last, stats.head):
        new = read_commits(root, repo, max_commits, rev=f"{last}..{stats.head}")
    else:
        if last:
            stats.rebuilt = True
        old = [r["sha"] for r in db.q("SELECT sha FROM commits WHERE repo=?", (repo,))]
        if old:
            store.delete([commit_id(repo, s) for s in old])
            stats.removed += len(old)
            with db.transaction():
                db.conn.execute("DELETE FROM commits WHERE repo=?", (repo,))
                db.conn.execute("DELETE FROM chunks WHERE repo=? AND kind='commit'", (repo,))
        new = read_commits(root, repo, max_commits, rev=stats.head)

    chunks = [commit_chunk(ci) for ci in new]
    if chunks:
        store.upsert(chunks)
    with db.transaction():
        for ci, ch in zip(new, chunks, strict=True):
            db.conn.execute(
                "INSERT OR REPLACE INTO commits VALUES(?,?,?,?,?,?,?,?)",
                (
                    repo,
                    ci.sha,
                    ci.date,
                    ci.author,
                    ci.subject,
                    json.dumps(ci.files),
                    ci.insertions,
                    ci.deletions,
                ),
            )
            db.conn.execute(
                "INSERT OR REPLACE INTO chunks VALUES(?,?,?,?,?,?,?,?,?,?)",
                (ch.id, repo, "", "commit", "", ch.symbol, 0, 0, ch.content_hash, ch.text),
            )
    stats.added = len(new)
    stats.removed += prune(db, store, repo, max_commits)
    return stats


def prune(db: DB, store: VectorStore, repo: str, max_commits: int) -> int:
    rows = db.q(
        "SELECT sha FROM commits WHERE repo=? ORDER BY date DESC LIMIT -1 OFFSET ?", (repo, max_commits)
    )
    shas = [r["sha"] for r in rows]
    if not shas:
        return 0
    store.delete([commit_id(repo, s) for s in shas])
    with db.transaction():
        for s in shas:
            db.conn.execute("DELETE FROM commits WHERE repo=? AND sha=?", (repo, s))
            db.conn.execute("DELETE FROM chunks WHERE id=?", (commit_id(repo, s),))
    return len(shas)
