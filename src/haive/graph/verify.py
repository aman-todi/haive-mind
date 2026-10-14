"""Answer verification (pure code). Checks every claim's provenance against the live tree,
this run's evidence, and this run's web results, plus the suggested diff in debug answers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from haive.tools.files import FileAccessError, file_lines
from haive.types import Answer, Claim, Evidence, RepoRef, WebRef, WebResult

SYMBOL_SLACK = 3


@dataclass
class VerifyResult:
    errors: list[str] = field(default_factory=list)
    bad_claims: dict[str, list[str]] = field(default_factory=dict)  # claim key -> errors
    approximate_fix: bool = False

    @property
    def ok(self) -> bool:
        return not self.bad_claims


def claim_keys(answer: Answer) -> list[tuple[str, Claim]]:
    out = []
    for si, s in enumerate(answer.sections):
        for ci, c in enumerate(s.claims):
            out.append((f"s{si}.c{ci}", c))
    for hi, h in enumerate(answer.hypotheses or []):
        for ci, c in enumerate(h.claims):
            out.append((f"h{hi}.c{ci}", c))
    return out


class _Lines:
    def __init__(self, roots: dict[str, Path]):
        self.roots = roots
        self.cache: dict[tuple[str, str], list[str] | None] = {}

    def get(self, repo: str, path: str) -> list[str] | None:
        key = (repo, path)
        if key not in self.cache:
            root = self.roots.get(repo)
            try:
                self.cache[key] = file_lines(root, path) if root else None
            except FileAccessError:
                self.cache[key] = None
        return self.cache[key]


def check_repo_ref(ref: RepoRef, lines_: _Lines, evidence_paths: set[tuple[str, str]]) -> list[str]:
    loc = f"{ref.repo}/{ref.path}:{ref.start_line}-{ref.end_line}"
    if ref.repo not in lines_.roots:
        return [f"{loc}: unknown repo '{ref.repo}'"]
    lines = lines_.get(ref.repo, ref.path)
    if lines is None:
        return [f"{loc}: file does not exist in the working tree"]
    errs = []
    if ref.start_line < 1 or ref.end_line < ref.start_line or ref.end_line > len(lines):
        errs.append(f"{loc}: line range invalid (file has {len(lines)} lines)")
    elif ref.symbol:
        lo = max(0, ref.start_line - 1 - SYMBOL_SLACK)
        hi = min(len(lines), ref.end_line + SYMBOL_SLACK)
        name = re.split(r"[.:]+", ref.symbol)[-1].strip("()") or ref.symbol
        if not any(name in ln for ln in lines[lo:hi]):
            errs.append(f"{loc}: symbol '{ref.symbol}' not found within the cited lines")
    if (ref.repo, ref.path) not in evidence_paths:
        errs.append(f"{loc}: path was not among the retrieved evidence")
    return errs


def check_claim(
    claim: Claim, lines_: _Lines, evidence_paths: set[tuple[str, str]], web_urls: set[str]
) -> list[str]:
    repo_refs = [r for r in claim.refs if isinstance(r, RepoRef)]
    web_refs = [r for r in claim.refs if isinstance(r, WebRef)]
    if claim.source == "model":
        return ["model claim must not have refs"] if claim.refs else []
    if claim.source == "web":
        if not web_refs or repo_refs:
            return ["web claim needs web refs only"]
        return [f"url not in this run's web results: {r.url}" for r in web_refs if r.url not in web_urls]
    # repo
    if not repo_refs:
        return ["repo claim needs at least one repo ref"]
    errs = []
    if web_refs:
        errs.append("repo claim must cite repo refs only")
    for r in repo_refs:
        errs += check_repo_ref(r, lines_, evidence_paths)
    return errs


def parse_diff(diff: str) -> dict[str, list[str]]:
    """Map target path -> '-' and context lines (without the leading marker)."""
    files: dict[str, list[str]] = {}
    current: str | None = None
    for line in diff.splitlines():
        if line.startswith("--- "):
            p = line[4:].strip().split("\t")[0]
            current = None if p == "/dev/null" else re.sub(r"^[ab]/", "", p)
            if current:
                files.setdefault(current, [])
            continue
        if line.startswith("+++ "):
            p = line[4:].strip().split("\t")[0]
            if current is None and p != "/dev/null":
                current = re.sub(r"^[ab]/", "", p)
                files.setdefault(current, [])
            continue
        if line.startswith(("@@", "diff ", "index ", "\\")):
            continue
        if current is None:
            current = ""
            files.setdefault(current, [])
        if line.startswith("-") or line.startswith(" "):
            files[current].append(line[1:])
    return files


def check_diff(diff: str, lines_: _Lines, default: tuple[str, str] | None) -> bool:
    """True if every '-' and context line exists in the cited file (whitespace-insensitive)."""
    parsed = parse_diff(diff)
    if not parsed:
        return False
    for path, old_lines in parsed.items():
        target = None
        for repo in lines_.roots:
            if path and lines_.get(repo, path) is not None:
                target = (repo, path)
                break
            if path and path.split("/", 1)[0] == repo and lines_.get(repo, path.split("/", 1)[1]) is not None:
                target = (repo, path.split("/", 1)[1])
                break
        if target is None:
            target = default
        if target is None:
            return False
        content = lines_.get(*target)
        if content is None:
            return False
        have = {ln.strip() for ln in content}
        for ol in old_lines:
            if ol.strip() and ol.strip() not in have:
                return False
    return True


def verify(
    answer: Answer, evidence: list[Evidence], web_results: list[WebResult], roots: dict[str, Path]
) -> VerifyResult:
    res = VerifyResult()
    lines_ = _Lines(roots)
    ev_paths = {(e.repo, e.path) for e in evidence if e.repo and e.path}
    urls = {w.url for w in web_results}
    for key, claim in claim_keys(answer):
        errs = check_claim(claim, lines_, ev_paths, urls)
        if errs:
            res.bad_claims[key] = errs
            snippet = claim.text[:80].replace("\n", " ")
            res.errors += [f'[{key}] "{snippet}": {e}' for e in errs]
    if answer.suggested_fix and answer.suggested_fix.diff.strip():
        default = None
        for _, c in claim_keys(answer):
            for r in c.refs:
                if isinstance(r, RepoRef):
                    default = (r.repo, r.path)
                    break
            if default:
                break
        res.approximate_fix = not check_diff(answer.suggested_fix.diff, lines_, default)
    return res
