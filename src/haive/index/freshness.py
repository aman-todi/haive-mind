"""Freshness check before every question (repos in scope only, ~200 ms budget).

Compares the live tree against the last indexed HEAD and stored file hashes. Up to
``INLINE_LIMIT`` changed files are re-indexed inline; more than that produces a warning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from haive.config import Config
from haive.db import DB
from haive.index.discover import sha256_file, skip_reason
from haive.tools import git

INLINE_LIMIT = 50


@dataclass
class RepoFreshness:
    repo: str
    changed: list[str] = field(default_factory=list)
    head_moved: bool = False
    never_indexed: bool = False
    error: str | None = None

    @property
    def stale(self) -> bool:
        return bool(self.changed) or self.head_moved or self.never_indexed


@dataclass
class FreshnessReport:
    repos: list[RepoFreshness] = field(default_factory=list)
    ms: float = 0.0


def check_repo(cfg: Config, db: DB, name: str) -> RepoFreshness:
    rf = RepoFreshness(repo=name)
    root = cfg.repo_root(name)
    row = db.get_repo(name)
    if row is None or not row["last_indexed_at"]:
        rf.never_indexed = True
        return rf
    last_head = row["last_indexed_head"]
    try:
        head = git.head(root)
        rf.head_moved = bool(head and head != last_head)
        cands = git.changed_since(root, last_head) if last_head else git.status_paths(root)
    except (git.GitError, OSError) as e:
        rf.error = str(e)
        return rf
    known = db.file_hashes(name)
    ic = cfg.index
    for path in cands:
        full = root / path
        if not full.exists():
            if path in known:
                rf.changed.append(path)
            continue
        try:
            size = full.stat().st_size
        except OSError:
            continue
        if skip_reason(path, size, ic.max_file_kb, ic.extra_ignore):
            if path in known:
                rf.changed.append(path)  # now ignored: drop it from the index
            continue
        if known.get(path) != sha256_file(root, path):
            rf.changed.append(path)
    return rf


def check(cfg: Config, db: DB, repos: list[str]) -> FreshnessReport:
    t0 = time.monotonic()
    rep = FreshnessReport([check_repo(cfg, db, r) for r in repos if r in cfg.repos])
    rep.ms = (time.monotonic() - t0) * 1000
    return rep
