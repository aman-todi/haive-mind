"""Chat models via OpenRouter (or any OpenAI-compatible endpoint): one tier's model list per question.

Built on LangChain primitives:
- ``init_chat_model`` builds each model, with a shared ``InMemoryRateLimiter`` (client-side
  throttle) and the OpenAI client's own retries, which honor ``Retry-After`` on 429s;
- ``.with_fallbacks()`` moves down the tier's list when a model keeps failing;
- ``deadline()`` caps each model attempt's wall-clock time (the client's ``timeout`` only bounds the
  wait between bytes, so a stalled provider that keeps the connection alive could hold a call for
  minutes);
- the ``Meter`` callback handler counts calls, enforces the per-question budget, and records usage.
"""

from __future__ import annotations

import os
import queue
import threading
from collections.abc import Callable
from typing import Any

import openai
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from pydantic import ValidationError

from haive.config import Config
from haive.db import DB
from haive.llm.budget import BudgetExceeded, Meter


# Errors that mean "this model didn't work out, try the next one". BudgetExceeded is deliberately
# not here: it must stop the whole chain.
class DeadlineExceeded(TimeoutError):
    """A model attempt ran past ``models.timeout_seconds`` of wall-clock time."""


RECOVERABLE: tuple[type[BaseException], ...] = (
    DeadlineExceeded,
    openai.APIError,  # rate limits after the client's retries, 5xx, 4xx, timeouts, connection errors
    OutputParserException,
    ValidationError,
    ValueError,
    NotImplementedError,
)


class LLMError(RuntimeError):
    pass


def _meter_of(config: RunnableConfig) -> Meter | None:
    cbs = config.get("callbacks")
    handlers = cbs if isinstance(cbs, list) else getattr(cbs, "handlers", None) or []
    return next((h for h in handlers if isinstance(h, Meter)), None)


def deadline(runnable: Runnable, model_id: str, seconds_for: Callable[[str], float]) -> Runnable:
    """``runnable`` with a wall-clock limit, ``seconds_for(step)``: past it, the attempt is abandoned
    (its daemon thread finishes or dies on its own) and ``DeadlineExceeded`` sends the chain to the
    next model, which then serves the rest of the question like after any other failure."""

    def call(inp: Any, config: RunnableConfig) -> Any:
        step = (config.get("metadata") or {}).get("haive_step", "?")
        seconds = seconds_for(step)
        box: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def work() -> None:
            try:
                box.put((True, runnable.invoke(inp, config)))
            except BaseException as e:  # handed back to the caller's thread
                box.put((False, e))

        threading.Thread(target=work, daemon=True, name=f"llm-{model_id}").start()
        try:
            ok, value = box.get(timeout=seconds)
        except queue.Empty:
            meter = _meter_of(config)
            if meter is not None:
                meter.log.append(
                    {
                        "type": "llm",
                        "step": step,
                        "model": model_id,
                        "ms": round(seconds * 1000),
                        "ok": False,
                        "error": f"no answer within {seconds:.0f} s (abandoned)",
                        "skip": "question",
                    }
                )
            raise DeadlineExceeded(f"{model_id}: no answer within {seconds:.0f} s") from None
        if ok:
            return value
        raise value

    return RunnableLambda(call, name=f"deadline:{model_id}")


def get_model(model_id: str, cfg: Config, rate_limiter: InMemoryRateLimiter | None = None) -> BaseChatModel:
    """Build a LangChain chat model for ``model_id`` from config."""
    from langchain.chat_models import init_chat_model

    m = cfg.models
    kwargs: dict[str, Any] = {
        "model_provider": m.provider,
        "temperature": 0,
        "max_retries": m.max_retries,
        "timeout": m.timeout_seconds,
    }
    if rate_limiter is not None and is_free(model_id):  # paid models aren't held to the free tier's limit
        kwargs["rate_limiter"] = rate_limiter
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
        if m.routing and not is_free(model_id):  # a free model has one provider: nothing to route
            kwargs["extra_body"] = {"provider": m.routing}
    return init_chat_model(model_id, **kwargs)


def is_free(model_id: str) -> bool:
    return model_id.endswith(":free") or model_id == "openrouter/free"


class Throttle(InMemoryRateLimiter):
    """LangChain's token bucket starts empty, so the first call of every command waited 1/rate
    seconds (3 s at 20/min). This one starts full: short bursts go through, sustained load is paced."""

    def __init__(self, requests_per_minute: int):
        super().__init__(
            requests_per_second=requests_per_minute / 60, check_every_n_seconds=0.1, max_bucket_size=3
        )
        self.available_tokens = self.max_bucket_size


def make_rate_limiter(requests_per_minute: int) -> InMemoryRateLimiter | None:
    return Throttle(requests_per_minute) if requests_per_minute > 0 else None


class LLM:
    """Entry point for all LLM calls. ``factory`` builds a model from an id (tests inject fakes).

    Every call in a question uses the same chain: the configured tier's list (strong or ultra, free or
    paid). ``step`` (router, grade, answer, ...) only labels the call in --trace and usage stats."""

    def __init__(
        self, cfg: Config, db: DB | None = None, factory: Callable[[str], BaseChatModel] | None = None
    ):
        self.cfg = cfg
        self.db = db
        self.rate_limiter = make_rate_limiter(cfg.models.requests_per_minute)
        self.factory = factory or (lambda mid: get_model(mid, cfg, self.rate_limiter))
        self._models: dict[str, BaseChatModel] = {}
        self._chains: dict[Any, Runnable] = {}
        # Models that answered 404 (removed, or their only provider is down): skipped for the rest of
        # this process, so one dead model doesn't cost an attempt at every step of every question.
        self.unavailable: set[str] = set()
        # Models that failed a call this question (timeouts, rate limits, 5xx after the client's
        # retries): skipped for the rest of it, so one model answers every step of a question.
        self.failed: set[str] = set()

    # ------------------------------------------------------------ models

    def model(self, model_id: str) -> BaseChatModel:
        if model_id not in self._models:
            self._models[model_id] = self.factory(model_id)
        return self._models[model_id]

    def deadline_for(self, step: str) -> float:
        """The answer may take a while; every other step returns a few hundred tokens in seconds."""
        m = self.cfg.models
        return m.timeout_seconds if step == "answer" else min(m.step_timeout_seconds, m.timeout_seconds)

    def guard(self, runnable: Runnable, model_id: str) -> Runnable:
        return deadline(runnable, model_id, self.deadline_for)

    def model_ids(self) -> list[str]:
        chain = self.cfg.models.chain()
        return [m for m in chain if m not in self.unavailable | self.failed] or chain

    def begin_question(self) -> None:
        self.failed = set()

    def note_failures(self, meter: Meter) -> None:
        for e in meter.log:
            if e.get("skip") == "process":
                self.unavailable.add(e["model"])
            elif e.get("skip") == "question":
                self.failed.add(e["model"])

    def with_fallbacks(self, build: Callable[[BaseChatModel], Runnable], key: Any = None) -> Runnable:
        """``build(model)`` for each model in the chain, joined with ``.with_fallbacks()``.
        Cached per (models, key) so repeated calls reuse the same runnable."""
        ids = tuple(self.model_ids())
        cache_key = (ids, key)
        if cache_key not in self._chains:
            runnables = [self.guard(build(self.model(mid)), mid) for mid in ids]
            chain = runnables[0]
            if len(runnables) > 1:
                chain = chain.with_fallbacks(runnables[1:], exceptions_to_handle=RECOVERABLE)
            self._chains[cache_key] = chain
        return self._chains[cache_key]

    def run_config(self, step: str, meter: Meter) -> RunnableConfig:
        meter.db = self.db
        return {"callbacks": [meter], "metadata": {"haive_step": step}, "run_name": f"haive.{step}"}

    # ------------------------------------------------------------ calls

    def text(self, step: str, messages: Any, meter: Meter) -> str:
        """Plain-text completion with fallback down the chain."""
        chain = self.with_fallbacks(lambda m: m, key="text")
        try:
            msg = chain.invoke(messages, config=self.run_config(step, meter))
        except BudgetExceeded:
            raise
        except Exception as e:  # any provider failure degrades the caller, never crashes the graph
            raise LLMError(f"all models failed ({step}): {type(e).__name__}: {e}") from e
        finally:
            self.note_failures(meter)
        return msg.content if isinstance(msg.content, str) else str(msg.content)
