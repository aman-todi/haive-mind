"""Eval runner: full-graph runs per ablation, retrieval-only store comparisons, and reports.

Results go to eval/results/<timestamp>/ (per-question CSVs and summary.md). The README results
table is regenerated from the latest run, so it only ever shows measured numbers.
"""

from __future__ import annotations

import csv
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from eval import metrics
from haive.graph.router import code_identifiers
from haive.tools import ripgrep
from haive.tools.pathmap import PathMapper
from haive.tools.trace_parse import parse

EVAL_DIR = Path(__file__).parent
ROOT = EVAL_DIR.parent
FULL_ABLATIONS = ["vector_only", "ripgrep_only", "hybrid", "hybrid+loop"]
STORE_ABLATIONS = ["pinecone", "chroma"]
README_START, README_END = "<!-- eval-results:start -->", "<!-- eval-results:end -->"
COLUMNS = [
    "intent_accuracy",
    "file_hit_rate",
    "recall@10",
    "citation_validity",
    "debug_top1",
    "debug_any",
    "llm_calls_mean",
    "latency_p50_s",
    "latency_p95_s",
    "questions",
    "errors",
]


def load_suite(path: Path) -> list[dict[str, Any]]:
    qs = yaml.safe_load(path.read_text()) or []
    for q in qs:
        q.setdefault("question", "")
        q.setdefault("expected_files", [])
    return qs


def ablation_list(ablation: str | None) -> list[str]:
    if ablation in (None, ""):
        return ["hybrid+loop"]
    if ablation == "all":
        return FULL_ABLATIONS + STORE_ABLATIONS
    names = [a.strip() for a in ablation.split(",")]
    for n in names:
        if n not in FULL_ABLATIONS + STORE_ABLATIONS:
            raise ValueError(f"unknown ablation {n!r}")
    return names


def scope_of(rt, q: dict[str, Any]) -> list[str]:
    if q.get("repos"):
        repos, _ = rt.cfg.resolve_names(q["repos"])
        return repos
    return list(rt.cfg.repos)


def trace_of(q: dict[str, Any]) -> str | None:
    tf = q.get("trace_file")
    if not tf:
        return None
    p = Path(tf)
    return (p if p.is_absolute() else ROOT / p).read_text()


def retrieved_files(rt, q: dict[str, Any], ablation: str, store=None) -> list[tuple[str, str]]:
    """Deterministic retrieval before any LLM step (used for recall@10)."""
    scope = scope_of(rt, q)
    roots = {r: rt.cfg.repo_root(r) for r in scope}
    files: list[tuple[str, str]] = []
    trace = trace_of(q)
    if trace:
        pt = parse(trace)
        mapper = PathMapper(rt.db.indexed_paths(scope), roots, scope)
        files += [(f.repo, f.repo_path) for f in mapper.map_frames(pt.frames) if f.repo and not f.external]
    text = q["question"] + (" " + parse(trace).message if trace else "")
    if ablation != "vector_only":
        counts: dict[tuple[str, str], int] = {}
        for ident in code_identifiers(text) or [w for w in re.findall(r"[A-Za-z_]\w{3,}", text)][:4]:
            for m in ripgrep.search(ident, roots, fixed=True, max_per_repo=30):
                counts[(m.repo, m.path)] = counts.get((m.repo, m.path), 0) + 1
        files += [f for f, _ in sorted(counts.items(), key=lambda kv: -kv[1])]
    if ablation != "ripgrep_only":
        store = store or rt.store
        hits = store.search(text or "code", 10, {"repo": scope, "kind": ["code", "config", "doc"]})
        vec = [(h.repo, h.path) for h in hits]
        files = files + vec if ablation in ("hybrid", "hybrid+loop") else vec + files
    return list(dict.fromkeys(files))


def run_full(rt, questions: list[dict[str, Any]], ablation: str, console) -> list[dict[str, Any]]:
    runner = rt.runner(ablation=ablation)
    rows = []
    for q in questions:
        row: dict[str, Any] = {"id": q["id"], "ablation": ablation}
        try:
            row["recall10"] = metrics.recall_at_k(retrieved_files(rt, q, ablation), q["expected_files"])
            tid = f"eval-{ablation}-{q['id']}-{time.time_ns()}"
            res = runner.run(
                q["question"],
                thread_id=tid,
                trace_text=trace_of(q),
                scope=q.get("repos"),
                web_mode=q.get("web", "off"),
                interactive=False,
            )
            try:
                rt.checkpointer.delete_thread(tid)
            except Exception:
                pass
            ans = res.answer.model_dump() if res.answer else None
            row["intent"] = res.state.get("intent")
            row["intent_ok"] = (row["intent"] == q["expected_intent"]) if q.get("expected_intent") else None
            row["file_hit"] = metrics.file_hit(ans, q["expected_files"])
            fv = res.state.get("first_verify") or {}
            row["citations"], row["citations_ok"] = fv.get("repo_claims", 0), fv.get("repo_claims_ok", 0)
            if q.get("expected_line") and q["expected_files"]:
                row["debug_top"], row["debug_any"] = metrics.debug_hits(
                    ans, q["expected_files"][0], q["expected_line"]
                )
            row["llm_calls"] = res.state.get("llm_calls", 0)
            row["latency_s"] = round(res.latency_s, 2)
        except Exception as e:  # keep going; errors are counted in the summary
            row["error"] = f"{type(e).__name__}: {e}"[:300]
        rows.append(row)
        console.print(
            f"  [{ablation}] {q['id']}: "
            + (
                "[red]error[/]"
                if row.get("error")
                else f"intent={row.get('intent')} hit={row.get('file_hit')} recall10={row.get('recall10')}"
            )
        )
    return rows


def store_for(rt, backend: str):
    """A store holding the same chunks as the configured one, for retrieval-only comparisons."""
    from haive.types import Chunk

    if backend == rt.cfg.vector_store.backend:
        return rt.store
    if backend == "pinecone":
        return None  # only comparable when it's the configured (populated) backend
    from haive.store.chroma_store import ChromaStore

    store = ChromaStore(rt.embedder, collection=f"eval{time.time_ns()}")
    rows = rt.db.q("SELECT * FROM chunks")
    chunks = [
        Chunk(
            id=r["id"],
            repo=r["repo"],
            path=r["path"],
            kind=r["kind"],
            language=r["language"] or "",
            symbol=r["symbol"] or "",
            start_line=r["start_line"],
            end_line=r["end_line"],
            text=r["text"],
            content_hash=r["content_hash"] or "",
        )
        for r in rows
    ]
    store.upsert(chunks)
    return store


def run_retrieval_only(rt, questions, backend: str, console) -> list[dict[str, Any]]:
    store = store_for(rt, backend)
    if store is None:
        console.print(f"  [{backend}] skipped: not the configured backend, so its index isn't populated")
        return []
    rows = []
    for q in questions:
        t0 = time.monotonic()
        files = retrieved_files(rt, q, "vector_only", store=store)
        rows.append(
            {
                "id": q["id"],
                "ablation": backend,
                "recall10": metrics.recall_at_k(files, q["expected_files"]),
                "latency_s": round(time.monotonic() - t0, 3),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("id", "ablation"), k))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summary_table(summaries: dict[str, dict[str, Any]]) -> str:
    lines = ["| ablation | " + " | ".join(COLUMNS) + " |", "|---" * (len(COLUMNS) + 1) + "|"]
    for ab, s in summaries.items():
        lines.append(f"| {ab} | " + " | ".join(metrics.fmt(c, s.get(c)) for c in COLUMNS) + " |")
    return "\n".join(lines)


def update_readme(table: str, header: str, readme: Path | None = None) -> bool:
    readme = readme or ROOT / "README.md"
    text = readme.read_text()
    if README_START not in text or README_END not in text:
        return False
    before, rest = text.split(README_START, 1)
    _, after = rest.split(README_END, 1)
    readme.write_text(f"{before}{README_START}\n{header}\n\n{table}\n{README_END}{after}")
    return True


def main(
    suite=None,
    ablation=None,
    model=None,
    limit=None,
    console=None,
    runtime=None,
    setup_sample: bool = False,
    results_dir: Path | None = None,
    readme: Path | None = None,
) -> Path:
    from rich.console import Console

    from haive.runtime import Runtime

    console = console or Console()
    rt = runtime or Runtime()
    if setup_sample:
        from eval.sample import setup_sample as _setup

        return _setup(rt, console)
    if model:
        setattr(rt.cfg.models, rt.cfg.models.tier_name, [model])
    path = Path(suite) if suite else EVAL_DIR / "questions.yaml"
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    questions = load_suite(path)[: limit or None]
    if not questions:
        raise SystemExit(f"{path} has no questions. Draft them from your repos (see README › Eval).")
    abls = ablation_list(ablation)
    full = [a for a in abls if a in FULL_ABLATIONS]
    est = len(questions) * len(full) * rt.cfg.limits.max_llm_calls_per_question
    left = rt.cfg.limits.daily_call_budget - rt.db.calls_today()
    console.print(
        f"{len(questions)} questions × {len(abls)} ablation(s); up to {est} LLM calls ({left} left today)"
    )
    out = (results_dir or EVAL_DIR / "results") / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}
    for ab in abls:
        rows = (
            run_full(rt, questions, ab, console)
            if ab in FULL_ABLATIONS
            else run_retrieval_only(rt, questions, ab, console)
        )
        if not rows:
            continue
        write_csv(out / f"{ab.replace('+', '_')}.csv", rows)
        summaries[ab] = metrics.summarize(rows)
    models = f"model={rt.cfg.models.chain()[0]}"
    header = (
        f"Measured {datetime.now():%Y-%m-%d} on `{path.name}` ({len(questions)} questions; {models}; "
        f"store={rt.cfg.vector_store.backend}; embeddings={rt.cfg.embeddings.model})."
    )
    table = summary_table(summaries)
    (out / "summary.md").write_text(f"# Eval results\n\n{header}\n\n{table}\n")
    write_csv(out / "summary.csv", [{"ablation": k, **v} for k, v in summaries.items()])
    console.print(table)
    if update_readme(table, header, readme):
        console.print("README results table updated.")
    return out
