"""Eval metrics. Deterministic first: intent accuracy, file hit rate, retrieval recall@10,
first-try citation validity, debug line accuracy, LLM calls, latency percentiles."""

from __future__ import annotations

import statistics
from typing import Any

LINE_SLACK = 3


def norm_file(f: str | dict[str, str]) -> tuple[str | None, str]:
    """'repo/path' or {'repo','path'} or 'path' -> (repo|None, path)."""
    if isinstance(f, dict):
        return f.get("repo"), f["path"]
    return (None, f)


def file_matches(expected: tuple[str | None, str], repo: str, path: str) -> bool:
    erepo, epath = expected
    if erepo is not None and erepo != repo:
        return False
    return path == epath or path.endswith("/" + epath) or f"{repo}/{path}" == epath


def verified_refs(answer: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not answer:
        return []
    claims = [c for s in answer.get("sections", []) for c in s.get("claims", [])]
    for h in answer.get("hypotheses") or []:
        claims += h.get("claims", [])
    return [
        r
        for c in claims
        if c.get("source") == "repo" and not c.get("unverified")
        for r in c.get("refs", [])
        if "path" in r
    ]


def file_hit(answer: dict[str, Any] | None, expected_files: list) -> bool | None:
    if not expected_files:
        return None
    refs = verified_refs(answer)
    exp = [norm_file(f) for f in expected_files]
    return any(file_matches(e, r["repo"], r["path"]) for e in exp for r in refs)


def recall_at_k(retrieved: list[tuple[str, str]], expected_files: list, k: int = 10) -> bool | None:
    if not expected_files:
        return None
    top = list(dict.fromkeys(retrieved))[:k]
    exp = [norm_file(f) for f in expected_files]
    return any(file_matches(e, repo, path) for e in exp for repo, path in top)


def debug_hits(answer: dict[str, Any] | None, expected_file: str, expected_line: int) -> tuple[bool, bool]:
    """(correct file+line in the top hypothesis, in any hypothesis)."""
    hyps = (answer or {}).get("hypotheses") or []
    exp = norm_file(expected_file)

    def hit(h: dict[str, Any]) -> bool:
        for c in h.get("claims", []):
            if c.get("unverified"):
                continue
            for r in c.get("refs", []):
                if (
                    "path" in r
                    and file_matches(exp, r["repo"], r["path"])
                    and (r["start_line"] - LINE_SLACK <= expected_line <= r["end_line"] + LINE_SLACK)
                ):
                    return True
        return False

    return (bool(hyps) and hit(hyps[0]), any(hit(h) for h in hyps))


def percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    i = (len(xs) - 1) * p
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def rate(values: list[bool | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cites = [(r["citations_ok"], r["citations"]) for r in rows if r.get("citations")]
    lat = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    calls = [r["llm_calls"] for r in rows if r.get("llm_calls") is not None]
    return {
        "questions": len(rows),
        "intent_accuracy": rate([r.get("intent_ok") for r in rows]),
        "file_hit_rate": rate([r.get("file_hit") for r in rows]),
        "recall@10": rate([r.get("recall10") for r in rows]),
        "citation_validity": (sum(a for a, _ in cites) / sum(b for _, b in cites)) if cites else None,
        "debug_top1": rate([r.get("debug_top") for r in rows]),
        "debug_any": rate([r.get("debug_any") for r in rows]),
        "llm_calls_mean": statistics.mean(calls) if calls else None,
        "latency_p50_s": percentile(lat, 0.5),
        "latency_p95_s": percentile(lat, 0.95),
        "errors": sum(1 for r in rows if r.get("error")),
    }


RATE_KEYS = {"intent_accuracy", "file_hit_rate", "recall@10", "citation_validity", "debug_top1", "debug_any"}


def fmt(key: str, v: Any) -> str:
    if v is None:
        return "–"
    if key in RATE_KEYS:
        return f"{v:.0%}"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


# First-try verification errors (graph/verify.py messages) grouped into kinds, to see what to fix.
ERROR_KINDS = [
    ("line range invalid", "line range outside the file"),
    ("is neither in the cited lines", "symbol not in the cited lines"),
    ("repo claim needs at least one", "answer cites nothing"),
    ("path was not among the retrieved evidence", "file not in the evidence"),
    ("does not exist in the working tree", "file does not exist"),
    ("unknown repo", "unknown repo"),
    ("commit was not among", "commit not in the evidence"),
    ("not a commit sha", "malformed commit sha"),
    ("the answer has no claims", "answer with no claims"),
    ("url not in this run", "web URL not in the results"),
]


def error_kind(message: str) -> str:
    for needle, kind in ERROR_KINDS:
        if needle in message:
            return kind
    return "wrong citation type"


def error_kinds(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        for msg in r.get("first_errors_list") or []:
            k = error_kind(msg)
            counts[k] = counts.get(k, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
