"""Web search: DuckDuckGo (``ddgs``, no key) by default, Tavily when a key is configured."""

from __future__ import annotations

import os

from haive.types import WebResult


class WebSearchError(RuntimeError):
    pass


def pick_provider(provider: str = "auto") -> str:
    if provider == "auto":
        return "tavily" if os.environ.get("TAVILY_API_KEY") else "duckduckgo"
    return provider


def search(query: str, provider: str = "auto", max_results: int = 5) -> list[WebResult]:
    provider = pick_provider(provider)
    try:
        if provider == "tavily":
            return _tavily(query, max_results)
        return _ddg(query, max_results)
    except WebSearchError:
        raise
    except Exception as e:  # network errors, rate limits, missing optional deps
        raise WebSearchError(f"{provider} search failed: {e}") from e


def _ddg(query: str, max_results: int) -> list[WebResult]:
    from ddgs import DDGS

    rows = DDGS().text(query, max_results=max_results) or []
    return dedupe(
        [
            WebResult(
                url=r.get("href") or r.get("url", ""), title=r.get("title", ""), snippet=r.get("body", "")
            )
            for r in rows
        ]
    )


def _tavily(query: str, max_results: int) -> list[WebResult]:
    try:
        from langchain_tavily import TavilySearch
    except ImportError as e:
        raise WebSearchError(
            "TAVILY_API_KEY is set but langchain-tavily is not installed (pip extra 'tavily')"
        ) from e
    res = TavilySearch(max_results=max_results).invoke({"query": query})
    rows = res.get("results", []) if isinstance(res, dict) else []
    return dedupe(
        [
            WebResult(url=r.get("url", ""), title=r.get("title", ""), snippet=(r.get("content") or "")[:800])
            for r in rows
        ]
    )


def dedupe(results: list[WebResult]) -> list[WebResult]:
    seen: set[str] = set()
    out = []
    for r in results:
        if r.url and r.url not in seen:
            seen.add(r.url)
            out.append(r)
    return out
