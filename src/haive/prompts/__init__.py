"""Prompt templates (``*.md``), versioned with the code. Variables use ``$name`` (string.Template)."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from string import Template


@lru_cache(maxsize=64)
def _raw(name: str) -> str:
    return resources.files(__package__).joinpath(f"{name}.md").read_text(encoding="utf-8")


def render(name: str, **vars: object) -> str:
    return Template(_raw(name)).safe_substitute({k: str(v) for k, v in vars.items()})


def version(name: str) -> str:
    first = _raw(name).splitlines()[0]
    return first.split("version:", 1)[1].strip(" ->") if "version:" in first else "0"
