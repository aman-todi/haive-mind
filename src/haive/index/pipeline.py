"""Index pipeline: discover -> hash -> chunk -> upsert/delete, plus commits and the repo profile.

Incremental by default: candidates come from ``git diff --name-only <last_indexed_head>`` plus
untracked files plus anything added to or dropped from discovery. Unchanged chunks keep their id
(content-hashed), so only changed chunks are re-embedded. Renames are a delete plus an add.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from haive.config import Config
from haive.db import DB
from haive.index import commits as commit_index
from haive.index.chunk_text import chunk_file
from haive.index.discover import FileInfo, discover, read_text, sha256_file
from haive.index.profile import build_profile
from haive.tools import git
from haive.types import Chunk, VectorStore

FLUSH_CHUNKS = 128

Progress = Callable[..., None]


@dataclass
class IndexStats:
    repo: str
    files_seen: int = 0
    files_new: int = 0
    files_changed: int = 0
    files_deleted: int = 0
    chunks_added: int = 0
    chunks_deleted: int = 0
    commits_added: int = 0
    commits_removed: int = 0
    commits_rebuilt: bool = False
    seconds: float = 0.0

    @property
    def files_touched(self) -> int:
        return self.files_new + self.files_changed + self.files_deleted

    def summary(self) -> str:
        parts = [
            f"{self.files_new} new",
            f"{self.files_changed} changed",
            f"{self.files_deleted} deleted files",
            f"+{self.chunks_added}/-{self.chunks_deleted} chunks",
            f"+{self.commits_added}/-{self.commits_removed} commits",
        ]
        if self.commits_rebuilt:
            parts.append("commit history rebuilt")
        return f"{self.repo}: " + ", ".join(parts) + f" ({self.seconds:.1f}s)"


def _noop(*_: Any, **__: Any) -> None:
    pass


class Indexer:
    def __init__(self, cfg: Config, db: DB, store: VectorStore, progress: Progress | None = None):
        self.cfg = cfg
        self.db = db
        self.store = store
        self.progress = progress or _noop

    def root(self, name: str) -> Path:
        return self.cfg.repo_root(name)

    def _discover(self, root: Path) -> dict[str, FileInfo]:
        ic = self.cfg.index
        return {f.path: f for f in discover(root, ic.max_file_kb, ic.extra_ignore)}

    # ------------------------------------------------------------ public

    def index_repo(self, name: str, full: bool = False) -> IndexStats:
        t0 = time.monotonic()
        root = self.root(name)
        if not git.is_repo(root):
            raise git.GitError(f"{root} is not a git repository")
        rc = self.cfg.repos[name]
        if self.db.get_repo(name) is None:
            self.db.upsert_repo(name, str(root), rc.remote_url)
        stats = IndexStats(repo=name)
        files = self._discover(root)
        stats.files_seen = len(files)

        if full:
            self._wipe(name, stats)
        known = self.db.file_hashes(name)
        row = self.db.get_repo(name)
        last_head = row["last_indexed_head"] if row else None
        if not full and last_head and git.is_ancestor(root, last_head, "HEAD"):
            try:
                cands = set(git.changed_since(root, last_head))
            except git.GitError:
                cands = set(files)
            cands |= set(files) - set(known)
            cands |= set(known) - set(files)
        else:
            cands = set(files) | set(known)

        self.progress("start", repo=name, total=len(cands))
        self._apply(name, root, sorted(cands), files, known, stats)

        commit_sha = self._commits(name, root, stats)
        if stats.files_touched or full or self.db.get_profile(name) is None:
            self.db.save_profile(name, build_profile(name, root, list(files.values())).model_dump())
        self.db.set_repo_indexed(name, git.head(root), commit_sha)
        stats.seconds = time.monotonic() - t0
        self.progress("done", repo=name, stats=stats)
        return stats

    def index_paths(self, name: str, paths: list[str], update_head: bool = True) -> IndexStats:
        """Re-index specific paths (used by the inline freshness check)."""
        t0 = time.monotonic()
        root = self.root(name)
        stats = IndexStats(repo=name)
        files = self._discover(root)
        known = self.db.file_hashes(name)
        self._apply(name, root, sorted(set(paths)), files, known, stats)
        if update_head:
            commit_sha = self._commits(name, root, stats)
            self.db.set_repo_indexed(name, git.head(root), commit_sha)
        if stats.files_touched:
            self.db.save_profile(name, build_profile(name, root, list(files.values())).model_dump())
        stats.seconds = time.monotonic() - t0
        return stats

    def remove_repo(self, name: str) -> int:
        ids = self.db.delete_repo(name)
        if ids:
            self.store.delete(ids)
        return len(ids)

    # ------------------------------------------------------------ internals

    def _wipe(self, name: str, stats: IndexStats) -> None:
        ids = [r["id"] for r in self.db.q("SELECT id FROM chunks WHERE repo=?", (name,))]
        if ids:
            self.store.delete(ids)
        stats.chunks_deleted += len([i for i in ids if ":commit:" not in i])
        with self.db.transaction():
            self.db.conn.execute("DELETE FROM chunks WHERE repo=?", (name,))
            self.db.conn.execute("DELETE FROM files WHERE repo=?", (name,))
            self.db.conn.execute("DELETE FROM commits WHERE repo=?", (name,))
            self.db.conn.execute(
                "UPDATE repos SET last_commit_sha=NULL, last_indexed_head=NULL WHERE name=?", (name,)
            )

    def _commits(self, name: str, root: Path, stats: IndexStats) -> str | None:
        ic = self.cfg.index
        cs = commit_index.index_commits(self.db, self.store, name, root, ic.max_commits)
        stats.commits_added += cs.added
        stats.commits_removed += cs.removed
        stats.commits_rebuilt = stats.commits_rebuilt or cs.rebuilt
        return cs.head

    def _apply(
        self,
        name: str,
        root: Path,
        cands: list[str],
        files: dict[str, FileInfo],
        known: dict[str, str],
        stats: IndexStats,
    ) -> None:
        pending_chunks: list[Chunk] = []
        pending_rows: list[tuple[str, FileInfo | None, str | None, list[Chunk], set[str]]] = []

        def flush() -> None:
            if pending_chunks:
                t = time.monotonic()
                self.store.upsert(pending_chunks)
                self.progress("embedded", repo=name, n=len(pending_chunks), seconds=time.monotonic() - t)
            dels = [i for _, _, _, _, old in pending_rows for i in old]
            if dels:
                self.store.delete(dels)
            with self.db.transaction():
                for path, info, sha, chunks, _old in pending_rows:
                    self.db.conn.execute(
                        "DELETE FROM chunks WHERE repo=? AND path=? AND kind!='commit'", (name, path)
                    )
                    if info is None:
                        self.db.conn.execute("DELETE FROM files WHERE repo=? AND path=?", (name, path))
                        continue
                    self.db.conn.executemany(
                        "INSERT OR REPLACE INTO chunks VALUES(?,?,?,?,?,?,?,?,?,?)",
                        [
                            (
                                c.id,
                                name,
                                path,
                                c.kind,
                                c.language,
                                c.symbol,
                                c.start_line,
                                c.end_line,
                                c.content_hash,
                                c.text,
                            )
                            for c in chunks
                        ],
                    )
                    self.db.conn.execute(
                        "INSERT OR REPLACE INTO files VALUES(?,?,?,?,?,?,?)",
                        (name, path, sha, info.language, info.kind, info.size, time.time()),
                    )
            pending_chunks.clear()
            pending_rows.clear()

        for path in cands:
            self.progress("file", repo=name, path=path)
            info = files.get(path)
            if info is None:
                if path in known:
                    old = self.db.chunk_ids_for(name, path)
                    stats.files_deleted += 1
                    stats.chunks_deleted += len(old)
                    pending_rows.append((path, None, None, [], old))
                continue
            sha = sha256_file(root, path)
            if sha is None or known.get(path) == sha:
                continue
            text = read_text(root, path)
            chunks = chunk_file(name, path, text, info.language, info.kind) if text is not None else []
            old = self.db.chunk_ids_for(name, path)
            new_ids = {c.id for c in chunks}
            to_add = [c for c in chunks if c.id not in old]
            to_del = old - new_ids
            if path in known:
                stats.files_changed += 1
            else:
                stats.files_new += 1
            stats.chunks_added += len(to_add)
            stats.chunks_deleted += len(to_del)
            pending_chunks.extend(to_add)
            pending_rows.append((path, info, sha, chunks, to_del))
            if len(pending_chunks) >= FLUSH_CHUNKS:
                flush()
        flush()
