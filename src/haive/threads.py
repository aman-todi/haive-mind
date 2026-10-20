"""Chat thread metadata (the ``threads`` table). Graph state itself lives in the SQLite checkpointer."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from haive import prompts
from haive.db import DB
from haive.llm.budget import BudgetExceeded, Meter
from haive.llm.models import LLM, LLMError


def new_id() -> str:
    return uuid.uuid4().hex


def ensure(db: DB, thread_id: str, scope: list[str]) -> None:
    now = time.time()
    db.x(
        "INSERT OR IGNORE INTO threads(id, title, scope_json, turns, created_at, updated_at) VALUES(?,?,?,?,?,?)",
        (thread_id, None, json.dumps(scope), 0, now, now),
    )


def record_turn(db: DB, thread_id: str, scope: list[str], title: str | None = None) -> None:
    ensure(db, thread_id, scope)
    db.x(
        "UPDATE threads SET turns=turns+1, updated_at=?, scope_json=?, title=COALESCE(title, ?) WHERE id=?",
        (time.time(), json.dumps(scope), title, thread_id),
    )


def get(db: DB, thread_id: str) -> dict[str, Any] | None:
    rows = db.q("SELECT * FROM threads WHERE id=?", (thread_id,))
    return dict(rows[0]) if rows else None


def resolve(db: DB, prefix: str) -> str | None:
    rows = db.q("SELECT id FROM threads WHERE id LIKE ? ORDER BY updated_at DESC", (prefix + "%",))
    if len(rows) == 1:
        return rows[0]["id"]
    return None if not rows else ("ambiguous" if len(rows) > 1 else None)


def list_threads(db: DB, limit: int = 20) -> list[dict[str, Any]]:
    return [dict(r) for r in db.q("SELECT * FROM threads ORDER BY updated_at DESC LIMIT ?", (limit,))]


def delete(db: DB, thread_id: str, checkpointer: Any = None) -> None:
    db.x("DELETE FROM threads WHERE id=?", (thread_id,))
    if checkpointer is not None:
        checkpointer.delete_thread(thread_id)


def make_title(llm: LLM, question: str, meter: Meter) -> str:
    """At most 8 words from the model; falls back to the first 60 characters of the question."""
    fallback = " ".join(question.split())[:60]
    try:
        text = llm.text("title", prompts.messages("title", question=question[:1000]), meter)
    except (LLMError, BudgetExceeded):
        return fallback
    title = (
        " ".join(text.strip().strip("\"'").splitlines()[0].split()[:8]).rstrip(".") if text.strip() else ""
    )
    return title or fallback


def relative(ts: float, now: float | None = None) -> str:
    d = (now or time.time()) - ts
    for unit, secs in (("d", 86400), ("h", 3600), ("m", 60)):
        if d >= secs:
            return f"{int(d // secs)}{unit} ago"
    return "just now"
