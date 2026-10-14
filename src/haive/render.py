"""Rendering answers: rich terminal output with provenance tags, or JSON."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from haive.types import Answer, Claim, RepoRef, WebRef

TAG_STYLE = {"repo": "bold green", "web": "bold blue", "model": "bold magenta", "unverified": "bold red"}
CONF_STYLE = {"high": "green", "med": "yellow", "low": "red"}


def fmt_ref(r: RepoRef | WebRef) -> str:
    if isinstance(r, WebRef):
        return r.url
    rng = f"{r.start_line}" if r.start_line == r.end_line else f"{r.start_line}-{r.end_line}"
    return f"{r.repo}/{r.path}:{rng}"


def tag_for(c: Claim) -> str:
    return "unverified" if c.unverified else c.source


def claim_markup(c: Claim) -> str:
    tag = tag_for(c)
    out = f"[{TAG_STYLE[tag]}]\\[{tag}][/] {escape(c.text)}"
    if c.unverified and c.source != "unverified":
        out += f" [dim](claimed {c.source})[/]"
    if c.refs:
        out += "\n      " + "  ".join(f"[cyan]{escape(fmt_ref(r))}[/]" for r in c.refs)
    return out


def render_answer(console: Console, answer: Answer, meta: dict[str, Any] | None = None) -> None:
    meta = meta or {}
    console.print()
    console.print(
        Panel(escape(answer.summary), title=f"[bold]{meta.get('intent', 'answer')}[/]", expand=True)
    )
    for h_i, h in enumerate(answer.hypotheses or [], 1):
        style = CONF_STYLE.get(h.confidence, "white")
        console.print(
            f"\n[bold]Hypothesis {h_i}:[/] {escape(h.title)}  [{style}]({h.confidence} confidence)[/]"
        )
        for c in h.claims:
            console.print("  • " + claim_markup(c))
    for s in answer.sections:
        if s.heading:
            console.print(f"\n[bold underline]{escape(s.heading)}[/]")
        for c in s.claims:
            console.print("  • " + claim_markup(c))
    if answer.suggested_fix and answer.suggested_fix.diff.strip():
        label = "Suggested fix (approximate: context lines did not match the file)" if (
            answer.suggested_fix.approximate) else "Suggested fix (not applied)"  # fmt: skip
        console.print(f"\n[bold]{label}[/]")
        console.print(Syntax(answer.suggested_fix.diff, "diff", theme="ansi_dark", word_wrap=True))
    if answer.followups:
        console.print("\n[bold]Follow-ups[/]")
        for f in answer.followups:
            console.print(f"  ? {escape(f)}")
    notices = meta.get("notices") or []
    footer = []
    if meta.get("scope"):
        footer.append(f"scope: {', '.join(meta['scope'])}")
    if "llm_calls" in meta:
        footer.append(f"LLM calls: {meta['llm_calls']}")
    if "latency_s" in meta:
        footer.append(f"{meta['latency_s']:.1f}s")
    console.print()
    for n in notices:
        console.print(f"[dim]• {escape(n)}[/]")
    if footer:
        console.print(f"[dim]{' · '.join(footer)}[/]")


def to_json(answer: Answer | None, meta: dict[str, Any]) -> str:
    return json.dumps({"answer": answer.model_dump() if answer else None, **meta}, indent=2, default=str)


def render_trace(console: Console, node_log: list[dict[str, Any]]) -> None:
    t = Table(title="graph trace", show_lines=False, header_style="bold")
    t.add_column("#", justify="right", style="dim")
    t.add_column("step")
    t.add_column("detail")
    t.add_column("ms", justify="right")
    for i, e in enumerate(node_log, 1):
        kind = e.get("type")
        if kind == "node":
            step, detail = f"[bold]{e['node']}[/]", escape(e.get("detail", ""))
        elif kind == "llm":
            ok = "" if e.get("ok") else f" [red]{escape(e.get('error', 'error'))}[/]"
            step, detail = "  llm", f"{e['role']} → {escape(e['model'])}{ok}"
        else:
            step, detail = f"  {kind}", escape(e.get("name", "") + " " + e.get("detail", ""))
        t.add_row(str(i), step, detail, str(e.get("ms", "")))
    console.print(t)
