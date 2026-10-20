"""Fakes: a scripted LangChain chat model and helpers. Tests never touch the network."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ConfigDict, Field


def schema_name(messages: Any) -> str | None:
    for m in reversed(list(messages)):
        content = getattr(m, "content", "")
        if isinstance(content, str):
            match = re.search(r"^Schema: (\w+)", content, re.M)
            if match:
                return match.group(1)
    return None


class FakeChatModel(BaseChatModel):
    """A real ``BaseChatModel`` that returns scripted outputs (so callbacks, fallbacks, rate
    limiters, and tool binding all run through LangChain).

    ``script`` maps a key to a list of outputs; items are consumed in order and the last repeats.
    Keys: a schema name (``"RouterOutput"``) for structured calls, ``"tools"`` for tool-calling
    turns of the expand agent, ``"text"`` for plain completions. An item can be a dict / pydantic
    model (sent as JSON, or as tool-call args on the native path), a raw string, a list of
    ``{"name", "args"}`` tool calls (``"tools"`` key), an Exception (raised), or a callable that
    receives the messages and returns any of those.

    ``supports_tools=False`` (default) makes ``bind_tools`` raise, so structured output takes the
    JSON-prompt path; ``True`` exercises native tool calling.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    script: dict[str, list[Any]] = Field(default_factory=dict)
    supports_tools: bool = False
    model_name: str = "fake-model"
    calls: list[str] = Field(default_factory=list)
    seen: list[list[BaseMessage]] = Field(default_factory=list)

    def __init__(self, script: dict[str, list[Any]] | None = None, **kw: Any):
        super().__init__(script={k: list(v) for k, v in (script or {}).items()}, **kw)

    @property
    def messages(self) -> list[list[BaseMessage]]:
        return self.seen

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    def bind_tools(self, tools: Sequence[Any], *, tool_choice: Any = None, **kw: Any):
        if not self.supports_tools:
            raise NotImplementedError("fake model without tool calling")
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], tool_choice=tool_choice, **kw)

    def _next(self, key: str, messages: list[BaseMessage]) -> Any:
        queue = self.script.get(key) or self.script.get("*")
        if not queue:
            raise AssertionError(f"FakeChatModel has no script for {key!r}")
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if callable(item) and not isinstance(item, type):
            item = item(messages)
        if isinstance(item, BaseException):
            raise item
        return item

    def _generate(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any
    ) -> ChatResult:
        tools = kwargs.get("tools") or []
        tool_names = [t["function"]["name"] for t in tools]
        usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        self.seen.append(list(messages))
        if len(tool_names) == 1 and tool_names[0] in self.script:  # native structured output
            key = tool_names[0]
            self.calls.append(key)
            item = self._next(key, messages)
            if isinstance(item, BaseModel):
                item = item.model_dump()
            if isinstance(item, str):
                msg = AIMessage(content=item, usage_metadata=usage)
            else:
                msg = AIMessage(
                    content="", tool_calls=[{"name": key, "args": item, "id": "call_0"}], usage_metadata=usage
                )
            return ChatResult(generations=[ChatGeneration(message=msg)])
        if tool_names:  # agent turn
            self.calls.append("tools")
            item = self._next("tools", messages)
            calls = [
                {"name": c["name"], "args": c.get("args", {}), "id": f"call_{i}"} for i, c in enumerate(item)
            ]
            return ChatResult(
                generations=[
                    ChatGeneration(message=AIMessage(content="", tool_calls=calls, usage_metadata=usage))
                ]
            )
        key = schema_name(messages) or "text"
        self.calls.append(key)
        item = self._next(key, messages)
        if isinstance(item, BaseModel):
            item = item.model_dump_json()
        if isinstance(item, (dict, list)):
            item = json.dumps(item)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=item, usage_metadata=usage))])


def factory_for(model: FakeChatModel | dict[str, FakeChatModel]) -> Callable[[str], FakeChatModel]:
    if isinstance(model, dict):
        return lambda mid: model[mid]
    return lambda mid: model


def api_error(status: int) -> openai.APIStatusError:
    """A real OpenAI-client error (as the client raises after its own retries)."""
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(status, request=request)
    cls = {429: openai.RateLimitError, 400: openai.BadRequestError, 404: openai.NotFoundError}.get(
        status, openai.InternalServerError if status >= 500 else openai.APIStatusError
    )
    return cls(f"HTTP {status}", response=response, body=None)
