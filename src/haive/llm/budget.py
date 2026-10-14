"""Call budgets, rate limiting, and per-run metering."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class BudgetExceeded(RuntimeError):
    pass


class DailyBudgetExceeded(RuntimeError):
    pass


@dataclass
class Meter:
    """Counts LLM calls made by one graph node (or one CLI action) and logs them for --trace."""

    limit: int | None = None  # remaining calls allowed for this question
    calls: int = 0
    log: list[dict[str, Any]] = field(default_factory=list)

    def check(self) -> None:
        if self.limit is not None and self.calls >= self.limit:
            raise BudgetExceeded(f"LLM call budget reached ({self.limit} remaining calls used)")

    def record(self, role: str, model: str, ms: float, ok: bool, error: str = "") -> None:
        self.calls += 1
        entry = {"type": "llm", "role": role, "model": model, "ms": round(ms), "ok": ok}
        if error:
            entry["error"] = error[:200]
        self.log.append(entry)


class RateLimiter:
    """Sliding-window limiter (OpenRouter free models allow ~20 requests/minute)."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic, sleep=time.sleep):
        self.per_minute = per_minute
        self.clock = clock
        self.sleep = sleep
        self.stamps: deque[float] = deque()
        self.lock = threading.Lock()

    def acquire(self) -> float:
        """Block until a call is allowed. Returns seconds waited."""
        if self.per_minute <= 0:
            return 0.0
        waited = 0.0
        with self.lock:
            while True:
                now = self.clock()
                while self.stamps and now - self.stamps[0] >= 60:
                    self.stamps.popleft()
                if len(self.stamps) < self.per_minute:
                    self.stamps.append(now)
                    return waited
                delay = 60 - (now - self.stamps[0]) + 0.01
                self.sleep(delay)
                waited += delay


def daily_status(calls_today: int, budget: int) -> str:
    """'ok', 'warn' (>= 90%), or 'over' (>= 100%)."""
    if budget <= 0:
        return "ok"
    if calls_today >= budget:
        return "over"
    if calls_today >= 0.9 * budget:
        return "warn"
    return "ok"
