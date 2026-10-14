"""SQLite index database (haive.db): schema, migrations, and small query helpers.

The working tree is the source of truth; this database only indexes it.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import date
from pathlib import Path
from typing import Any

from haive import paths

MIGRATIONS: list[str] = [
    # v1
    """
    CREATE TABLE repos(
        name TEXT PRIMARY KEY, path TEXT NOT NULL, remote_url TEXT,
        last_indexed_head TEXT, last_indexed_at REAL, last_commit_sha TEXT);
    CREATE TABLE files(
        repo TEXT, path TEXT, sha256 TEXT, language TEXT, kind TEXT, size INTEGER, indexed_at REAL,
        PRIMARY KEY(repo, path));
    CREATE TABLE chunks(
        id TEXT PRIMARY KEY, repo TEXT, path TEXT, kind TEXT, language TEXT, symbol TEXT,
        start_line INTEGER, end_line INTEGER, content_hash TEXT, text TEXT);
    CREATE INDEX chunks_repo_path ON chunks(repo, path);
    CREATE INDEX chunks_symbol ON chunks(symbol);
    CREATE TABLE commits(
        repo TEXT, sha TEXT, date TEXT, author TEXT, subject TEXT, files_json TEXT,
        insertions INTEGER, deletions INTEGER, PRIMARY KEY(repo, sha));
    CREATE TABLE repo_profiles(repo TEXT PRIMARY KEY, json TEXT, updated_at REAL);
    CREATE TABLE threads(
        id TEXT PRIMARY KEY, title TEXT, scope_json TEXT, turns INTEGER DEFAULT 0,
        created_at REAL, updated_at REAL);
    CREATE TABLE usage(
        date TEXT, role TEXT, model TEXT, calls INTEGER DEFAULT 0, prompt_tokens INTEGER DEFAULT 0,
        completion_tokens INTEGER DEFAULT 0, errors INTEGER DEFAULT 0, PRIMARY KEY(date, role, model));
    """,
]


class DB:
    def __init__(self, path: Path | str | None = None):
        if path is None:
            path = paths.db_file()
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.migrate()

    # ------------------------------------------------------------ schema

    def migrate(self) -> None:
        c = self.conn
        c.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, applied_at REAL)"
        )
        row = c.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
        current = row[0] or 0
        for v, sql in enumerate(MIGRATIONS, start=1):
            if v <= current:
                continue
            with c:
                c.executescript(sql)
                c.execute("INSERT INTO schema_migrations VALUES(?, ?)", (v, time.time()))

    @property
    def version(self) -> int:
        return self.conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] or 0

    def close(self) -> None:
        self.conn.close()

    def q(self, sql: str, args: tuple | list = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, args).fetchall()

    def x(self, sql: str, args: tuple | list = ()) -> None:
        with self.conn:
            self.conn.execute(sql, args)

    # ------------------------------------------------------------ repos

    def upsert_repo(self, name: str, path: str, remote_url: str | None = None) -> None:
        self.x(
            "INSERT INTO repos(name, path, remote_url) VALUES(?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET path=excluded.path, remote_url=excluded.remote_url",
            (name, path, remote_url),
        )

    def get_repo(self, name: str) -> sqlite3.Row | None:
        rows = self.q("SELECT * FROM repos WHERE name=?", (name,))
        return rows[0] if rows else None

    def set_repo_indexed(self, name: str, head: str | None, commit_sha: str | None) -> None:
        self.x(
            "UPDATE repos SET last_indexed_head=?, last_indexed_at=?, last_commit_sha=? WHERE name=?",
            (head, time.time(), commit_sha, name),
        )

    def delete_repo(self, name: str) -> list[str]:
        """Delete all rows for a repo. Returns the chunk ids that were removed."""
        ids = [r["id"] for r in self.q("SELECT id FROM chunks WHERE repo=?", (name,))]
        with self.conn:
            for table in ("files", "chunks", "commits"):
                self.conn.execute(f"DELETE FROM {table} WHERE repo=?", (name,))
            self.conn.execute("DELETE FROM repo_profiles WHERE repo=?", (name,))
            self.conn.execute("DELETE FROM repos WHERE name=?", (name,))
        return ids

    # ------------------------------------------------------------ files / chunks

    def file_hashes(self, repo: str) -> dict[str, str]:
        return {
            r["path"]: r["sha256"] for r in self.q("SELECT path, sha256 FROM files WHERE repo=?", (repo,))
        }

    def chunk_ids_for(self, repo: str, path: str) -> set[str]:
        return {r["id"] for r in self.q("SELECT id FROM chunks WHERE repo=? AND path=?", (repo, path))}

    def chunk_counts(self, repo: str) -> dict[str, int]:
        rows = self.q("SELECT kind, COUNT(*) n FROM chunks WHERE repo=? GROUP BY kind", (repo,))
        return {r["kind"]: r["n"] for r in rows}

    def find_symbol(self, symbol: str, repos: list[str], limit: int = 10) -> list[sqlite3.Row]:
        marks = ",".join("?" * len(repos))
        return self.q(
            f"SELECT * FROM chunks WHERE repo IN ({marks}) AND kind != 'commit' AND "
            f"(symbol = ? OR symbol LIKE ? OR symbol LIKE ?) LIMIT ?",
            [*repos, symbol, f"%.{symbol}", f"%{symbol}%", limit],
        )

    def indexed_paths(self, repos: list[str]) -> list[tuple[str, str]]:
        marks = ",".join("?" * len(repos))
        return [
            (r["repo"], r["path"])
            for r in self.q(f"SELECT repo, path FROM files WHERE repo IN ({marks})", repos)
        ]

    # ------------------------------------------------------------ profiles

    def save_profile(self, repo: str, profile: dict[str, Any]) -> None:
        self.x(
            "INSERT INTO repo_profiles VALUES(?,?,?) ON CONFLICT(repo) DO UPDATE SET json=excluded.json, "
            "updated_at=excluded.updated_at",
            (repo, json.dumps(profile), time.time()),
        )

    def get_profile(self, repo: str) -> dict[str, Any] | None:
        rows = self.q("SELECT json FROM repo_profiles WHERE repo=?", (repo,))
        return json.loads(rows[0]["json"]) if rows else None

    # ------------------------------------------------------------ usage

    def record_usage(
        self, role: str, model: str, prompt_tokens: int = 0, completion_tokens: int = 0, error: bool = False
    ) -> None:
        self.x(
            "INSERT INTO usage VALUES(?,?,?,1,?,?,?) ON CONFLICT(date, role, model) DO UPDATE SET "
            "calls=calls+1, prompt_tokens=prompt_tokens+excluded.prompt_tokens, "
            "completion_tokens=completion_tokens+excluded.completion_tokens, errors=errors+excluded.errors",
            (date.today().isoformat(), role, model, prompt_tokens, completion_tokens, int(error)),
        )

    def calls_today(self) -> int:
        row = self.q("SELECT COALESCE(SUM(calls), 0) n FROM usage WHERE date=?", (date.today().isoformat(),))
        return row[0]["n"]

    def usage_today(self) -> list[sqlite3.Row]:
        return self.q("SELECT * FROM usage WHERE date=? ORDER BY role, model", (date.today().isoformat(),))
