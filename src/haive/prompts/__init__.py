"""Prompt templates (``*.md``), versioned with the code, loaded as LangChain ``ChatPromptTemplate``s.

Variables are written ``$name`` in the files (so JSON and code examples need no escaping). A file
may be split into messages with marker lines:

    <!-- system -->    following text is a system message
    <!-- history -->   a MessagesPlaceholder("history") for recent chat turns (optional)
    <!-- human -->     following text is a human message

A file without markers is a single human message. ``text()`` renders a file as a plain string,
for snippets such as the per-intent answer templates.
"""

from __future__ import annotations

import re
from functools import lru_cache
from importlib import resources

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

MARKER = re.compile(r"^<!-- (system|history|human) -->\s*$", re.M)


@lru_cache(maxsize=64)
def _raw(name: str) -> str:
    return resources.files(__package__).joinpath(f"{name}.md").read_text(encoding="utf-8")


def _body(name: str) -> str:
    """File content without the leading version comment."""
    return re.sub(r"\A<!-- version:[^>]*-->\s*\n", "", _raw(name))


def _fstring(text: str) -> str:
    """$var -> {var}, with literal braces escaped for f-string templates."""
    text = text.replace("{", "{{").replace("}", "}}")
    return re.sub(r"\$([A-Za-z_]\w*)", r"{\1}", text)


@lru_cache(maxsize=64)
def template(name: str) -> ChatPromptTemplate:
    body = _body(name)
    parts = MARKER.split(body)
    if len(parts) == 1:
        return ChatPromptTemplate.from_messages([("human", _fstring(body.strip()))])
    messages: list = []
    for role, content in zip(parts[1::2], parts[2::2], strict=True):
        if role == "history":
            messages.append(MessagesPlaceholder("history", optional=True))
            content = content.strip()
            if content:
                raise ValueError(f"{name}.md: text after <!-- history --> needs its own role marker")
            continue
        messages.append((role, _fstring(content.strip())))
    return ChatPromptTemplate.from_messages(messages)


def messages(name: str, **vars: object) -> list[BaseMessage]:
    t = template(name)
    values = {k: (v if k == "history" else str(v)) for k, v in vars.items()}
    return t.format_messages(**values)


def text(name: str, **vars: object) -> str:
    return re.sub(r"\$([A-Za-z_]\w*)", lambda m: str(vars.get(m.group(1), m.group(0))), _body(name)).strip()


def version(name: str) -> str:
    first = _raw(name).splitlines()[0]
    return first.split("version:", 1)[1].strip(" ->") if "version:" in first else "0"
