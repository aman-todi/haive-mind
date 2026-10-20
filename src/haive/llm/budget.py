"""Call budgets and per-run metering.

``Meter`` is a LangChain callback handler: every chat-model call made with it in the run config is
counted, checked against the question's remaining budget (before the call is sent), timed, logged
for ``--trace``, and recorded in the ``usage`` table.
"""

from __future__ import annotations

import logging
import time
from typing import Any
from uuid import UUID

import openai
from langchain_core.callbacks import BaseCallbackHandler


class BudgetExceeded(RuntimeError):
    pass


class _HideBudgetStops(logging.Filter):
    """LangChain logs a warning whenever a callback raises; a budget stop is expected, not noise."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "BudgetExceeded" not in record.getMessage()


logging.getLogger("langchain_core.callbacks.manager").addFilter(_HideBudgetStops())


class _ClientAttempts(logging.Handler):
    """Failed attempts inside one model call: the OpenAI client retries timeouts, 429s and 5xx on
    its own, so a slow call is otherwise indistinguishable from a slow model. Read from the
    client's debug log; with parallel calls (fan-out) an attempt may be credited to a sibling."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.events: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if msg.startswith("Encountered a timeout"):
            self.events.append("timeout")
        elif msg.startswith("Encountered an HTTP status error"):
            self.events.append(f"HTTP {msg.rsplit(' ', 1)[-1]}")
        elif msg.startswith("Encountered exception"):
            self.events.append("connection error")


_attempts = _ClientAttempts()
_client_log = logging.getLogger("openai._base_client")
_client_log.addHandler(_attempts)
_client_log.setLevel(logging.DEBUG)


class Meter(BaseCallbackHandler):
    """Counts LLM calls made by one graph node (or one CLI action)."""

    raise_error = True  # so BudgetExceeded raised in on_chat_model_start stops the call

    def __init__(self, limit: int | None = None, db: Any = None):
        self.limit = limit  # calls still allowed for this question (None = unlimited)
        self.db = db
        self.calls = 0
        self.log: list[dict[str, Any]] = []
        self._runs: dict[UUID, tuple[float, str, str]] = {}
        self._attempt_mark: dict[UUID, int] = {}

    def check(self) -> None:
        if self.limit is not None and self.calls >= self.limit:
            raise BudgetExceeded(f"LLM call budget reached ({self.limit} remaining calls used)")

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list,
        *,
        run_id: UUID,
        metadata: dict | None = None,
        **kw: Any,
    ) -> None:
        self.check()
        meta = metadata or {}
        model = meta.get("ls_model_name") or (serialized or {}).get("kwargs", {}).get("model_name") or "?"
        self.calls += 1
        self._runs[run_id] = (time.monotonic(), meta.get("haive_step", "?"), str(model))
        self._attempt_mark[run_id] = len(_attempts.events)

    def _finish(self, run_id: UUID, ok: bool, usage: dict[str, Any] | None = None, error: str = "") -> None:
        t0, role, model = self._runs.pop(run_id, (time.monotonic(), "?", "?"))
        entry: dict[str, Any] = {
            "type": "llm", "role": role, "model": model, "ms": round((time.monotonic() - t0) * 1000), "ok": ok,
        }  # fmt: skip
        if error:
            entry["error"] = error[:200]
        mark = self._attempt_mark.pop(run_id, None)
        if mark is not None and len(_attempts.events) > mark:
            entry["failed_attempts"] = _attempts.events[mark:]
        self.log.append(entry)
        if self.db is not None:
            usage = usage or {}
            self.db.record_usage(
                role,
                model,
                int(usage.get("input_tokens", 0) or 0),
                int(usage.get("output_tokens", 0) or 0),
                not ok,
            )

    def on_llm_end(self, response: Any, *, run_id: UUID, **kw: Any) -> None:
        usage = None
        try:
            usage = response.generations[0][0].message.usage_metadata
        except (AttributeError, IndexError):
            pass
        self._finish(run_id, True, usage)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kw: Any) -> None:
        self._finish(run_id, False, error=f"{type(error).__name__}: {error}")
        if isinstance(error, openai.NotFoundError):  # model or its only provider is gone, not busy
            self.log[-1]["skip"] = "process"
        elif isinstance(
            error, (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)
        ):
            self.log[-1]["skip"] = "question"  # timeouts included; a 400 (e.g. no tool support) is not

    # Tool runs (the expand agent's ToolNode) show up in --trace too; they cost no LLM calls.
    def on_tool_start(self, serialized: dict[str, Any], input_str: str, *, run_id: UUID, **kw: Any) -> None:
        self._runs[run_id] = (
            time.monotonic(),
            "tool",
            f"{(serialized or {}).get('name', 'tool')} {input_str[:80]}",
        )

    def on_tool_end(self, output: Any, *, run_id: UUID, **kw: Any) -> None:
        t0, _, label = self._runs.pop(run_id, (time.monotonic(), "tool", "tool"))
        name, _, detail = label.partition(" ")
        self.log.append(
            {"type": "tool", "name": name, "detail": detail, "ms": round((time.monotonic() - t0) * 1000)}
        )

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kw: Any) -> None:
        t0, _, label = self._runs.pop(run_id, (time.monotonic(), "tool", "tool"))
        self.log.append({"type": "tool", "name": label.split(" ")[0], "detail": f"failed: {error}"[:160],
                         "ms": round((time.monotonic() - t0) * 1000)})  # fmt: skip


def daily_status(calls_today: int, budget: int) -> str:
    """'ok', 'warn' (>= 90%), or 'over' (>= 100%)."""
    if budget <= 0:
        return "ok"
    if calls_today >= budget:
        return "over"
    if calls_today >= 0.9 * budget:
        return "warn"
    return "ok"
