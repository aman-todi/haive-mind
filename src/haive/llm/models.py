"""Role-based chat models via OpenRouter (or any OpenAI-compatible endpoint), with fallback,
429 backoff, throttling, and usage accounting."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

from haive.config import Config
from haive.db import DB
from haive.llm.budget import BudgetExceeded, Meter, RateLimiter
from haive.types import ChatModel

MAX_BACKOFF = 20.0
RATE_LIMIT_RETRIES = 2


class LLMError(RuntimeError):
    pass


def get_model(model_id: str, cfg: Config) -> ChatModel:
    """Build a LangChain chat model for ``model_id`` from config."""
    from langchain.chat_models import init_chat_model

    m = cfg.models
    kwargs: dict[str, Any] = {
        "model_provider": m.provider,
        "temperature": 0,
        "max_retries": 0,
        "timeout": 120,
    }
    if m.base_url:
        kwargs["base_url"] = m.base_url
    key = os.environ.get(m.api_key_env)
    if key:
        kwargs["api_key"] = key
    if "openrouter.ai" in (m.base_url or ""):
        kwargs["default_headers"] = {
            "HTTP-Referer": "https://github.com/aman-todi/haive-mind",
            "X-Title": "haive-mind",
        }
    return init_chat_model(model_id, **kwargs)


def status_code(err: BaseException) -> int | None:
    code = getattr(err, "status_code", None)
    if code is None:
        resp = getattr(err, "response", None)
        code = getattr(resp, "status_code", None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


def retry_after(err: BaseException) -> float | None:
    resp = getattr(err, "response", None)
    headers = getattr(resp, "headers", None) or {}
    try:
        val = headers.get("retry-after") or headers.get("Retry-After")
        return float(val) if val is not None else None
    except (TypeError, ValueError, AttributeError):
        return None


def is_rate_limit(err: BaseException) -> bool:
    return status_code(err) == 429 or "rate limit" in str(err).lower() or "429" in str(err)[:40]


def is_server_error(err: BaseException) -> bool:
    code = status_code(err)
    return code is not None and code >= 500


class LLM:
    """Entry point for all LLM calls. ``factory`` builds a model from an id (tests inject fakes)."""

    def __init__(
        self,
        cfg: Config,
        db: DB | None = None,
        factory: Callable[[str], ChatModel] | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        limiter: RateLimiter | None = None,
    ):
        self.cfg = cfg
        self.db = db
        self.factory = factory or (lambda mid: get_model(mid, cfg))
        self.sleep = sleep
        self.limiter = limiter or RateLimiter(cfg.models.requests_per_minute, sleep=sleep)
        self._models: dict[str, ChatModel] = {}
        self.no_native: set[str] = set()  # models whose native structured output failed

    def model(self, model_id: str) -> ChatModel:
        if model_id not in self._models:
            self._models[model_id] = self.factory(model_id)
        return self._models[model_id]

    def chain(self, role: str) -> list[str]:
        primary = getattr(self.cfg.models, role)
        out = [primary]
        for f in self.cfg.models.fallback:
            if f not in out:
                out.append(f)
        return out

    def _record(self, role: str, model_id: str, msg: Any = None, error: bool = False) -> None:
        if self.db is None:
            return
        usage = getattr(msg, "usage_metadata", None) or {}
        self.db.record_usage(
            role,
            model_id,
            int(usage.get("input_tokens", 0) or 0),
            int(usage.get("output_tokens", 0) or 0),
            error,
        )

    def invoke_once(self, role: str, model_id: str, runnable: Any, messages: Any, meter: Meter) -> Any:
        """One metered model call with 429 backoff (raises after RATE_LIMIT_RETRIES)."""
        attempt = 0
        while True:
            meter.check()
            self.limiter.acquire()
            t0 = time.monotonic()
            try:
                out = runnable.invoke(messages)
            except Exception as e:
                meter.record(
                    role, model_id, (time.monotonic() - t0) * 1000, False, f"{type(e).__name__}: {e}"
                )
                self._record(role, model_id, error=True)
                if is_rate_limit(e) and attempt < RATE_LIMIT_RETRIES:
                    wait = retry_after(e)
                    wait = min(MAX_BACKOFF, wait if wait is not None else 2.0 * (2**attempt))
                    self.sleep(wait)
                    attempt += 1
                    continue
                raise
            raw = out.get("raw") if isinstance(out, dict) and "raw" in out else out
            meter.record(role, model_id, (time.monotonic() - t0) * 1000, True)
            self._record(role, model_id, raw)
            return out

    def text(self, role: str, messages: Any, meter: Meter) -> str:
        """Plain-text completion with fallback down the chain."""
        last: Exception | None = None
        for model_id in self.chain(role):
            try:
                msg = self.invoke_once(role, model_id, self.model(model_id), messages, meter)
                return msg.content if isinstance(msg.content, str) else str(msg.content)
            except BudgetExceeded:
                raise
            except Exception as e:
                last = e
        raise LLMError(f"all models failed for role {role}: {last}")
