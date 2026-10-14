"""Fakes: a scripted chat model and helpers. Tests never touch the network."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage
from pydantic import BaseModel


def schema_name(messages: Any) -> str:
    for m in reversed(list(messages)):
        content = getattr(m, "content", "")
        if isinstance(content, str):
            match = re.search(r"^Schema: (\w+)", content, re.M)
            if match:
                return match.group(1)
    return "text"


class FakeChatModel:
    """Returns scripted outputs keyed by the requested schema name (``"text"`` for plain calls).

    Each script entry is a list; items are consumed in order and the last one repeats. An item can be
    a dict / pydantic model (sent as JSON), a raw string, an Exception (raised), or a callable that
    receives the messages and returns any of those.
    """

    def __init__(self, script: dict[str, list[Any]] | None = None):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.calls: list[str] = []
        self.messages: list[Any] = []

    def invoke(self, messages: Any, config: Any = None, **kw: Any) -> AIMessage:
        name = schema_name(messages)
        self.calls.append(name)
        self.messages.append(messages)
        queue = self.script.get(name) or self.script.get("*")
        if not queue:
            raise AssertionError(f"FakeChatModel has no script for {name!r}")
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if callable(item) and not isinstance(item, type):
            item = item(messages)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, BaseModel):
            item = item.model_dump_json()
        if isinstance(item, (dict, list)):
            item = json.dumps(item)
        return AIMessage(
            content=item, usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        )


def factory_for(model: FakeChatModel | dict[str, FakeChatModel]) -> Callable[[str], FakeChatModel]:
    if isinstance(model, dict):
        return lambda mid: model[mid]
    return lambda mid: model


class FakeHTTPError(Exception):
    def __init__(self, status: int, retry_after: str | None = None):
        super().__init__(f"HTTP {status}")
        self.status_code = status

        class _R:
            headers = {"retry-after": retry_after} if retry_after else {}
            status_code = status

        self.response = _R()
