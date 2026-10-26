"""Answer verification (pure code). Checks every claim's provenance against the live tree,
this run's evidence (files and commits), and this run's web results, plus the suggested diff in
debug answers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from haive.tools.files import FileAccessError, file_lines
from haive.types import MODULE_SYMBOL, Answer, Claim, CommitRef, Evidence, RepoRef, WebRef, WebResult

SYMBOL_SLACK = 3
DOC_REACH = 40  # lines a doc comment may run past the cited range before the definition it documents
COMMENT_STARTS = ("#", "//", "/*", "*", '"""', "'''")
EMPTY_ANSWER = "answer"  # bad_claims key for an answer with no claims despite evidence
MIN_SHA = 7


@dataclass
class VerifyResult:
    errors: list[str] = field(default_factory=list)
    bad_claims: dict[str, list[str]] = field(default_factory=dict)  # claim key -> errors
    approximate_fix: bool = False
    relabeled: int = 0  # claims whose source was corrected to match their refs

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


SYMBOL_WORDS = {"class", "function", "method", "def", "fn", "func", "struct", "module"}


def symbol_name(symbol: str) -> str:
    """The bare identifier in a cited symbol: `parse_metric()` -> parse_metric, HnswIndex<T>::search ->
    search, "BruteForceIndex class" -> BruteForceIndex."""
    words = [w for w in re.findall(r"[A-Za-z_]\w*", symbol) if w not in SYMBOL_WORDS]
    return words[-1] if words else symbol


def enclosing_headers(lines: list[str], start_line: int, max_up: int = 2000) -> list[str]:
    """Header lines of the scopes around ``start_line`` (1-based), innermost first: walking up, each
    non-blank, non-comment line indented less than the scope below it (``def``/``class``/a function
    signature/``TEST_P(...) {``). Lets a claim about lines inside a function name that function."""
    first = start_line - 1
    while first < len(lines) and not lines[first].strip():
        first += 1
    if first >= len(lines):
        return []
    indent = len(lines[first]) - len(lines[first].lstrip())
    out: list[str] = []
    for k in range(start_line - 2, max(-1, start_line - 2 - max_up), -1):
        if indent == 0:
            break
        ln = lines[k]
        stripped = ln.lstrip()
        # comments and decorators aren't headers; a line opening with a closing bracket ends a multi-line
        # signature (``) -> X:``) or an earlier block (``}``), so the header is further up; a lone ``{``
        # opens the block of the signature above it (C#/Java brace style)
        if not stripped or stripped.startswith(("#", "//", "/*", "*", "@", ")", "]", "}")) or stripped == "{":
            continue
        k_indent = len(ln) - len(stripped)
        if k_indent < indent:
            out.append(ln)
            indent = k_indent
    return out


def documented_definition(lines: list[str], end_line: int, max_down: int = DOC_REACH) -> str | None:
    """The first code line after a doc comment that the cited range ends inside or just before, so a
    claim citing a docstring (``/** ... */``, ``///``) may name the function it documents."""
    for k in range(end_line, min(len(lines), end_line + max_down)):
        stripped = lines[k].strip()
        if not stripped or stripped.startswith((*COMMENT_STARTS, "@", "[")) or stripped.endswith("*/"):
            continue
        return lines[k]
    return None


def check_repo_ref(ref: RepoRef, lines_: _Lines, evidence_paths: set[tuple[str, str]]) -> list[str]:
    loc = f"{ref.repo}/{ref.path}:{ref.start_line}-{ref.end_line}"
    if ref.repo not in lines_.roots:
        return [f"{loc}: unknown repo '{ref.repo}'"]
    lines = lines_.get(ref.repo, ref.path)
    errs = []
    if (ref.repo, ref.path) not in evidence_paths:
        errs.append(f"{loc}: path was not among the retrieved evidence")
    if lines is None:
        return [f"{loc}: file does not exist in the working tree", *errs]
    if ref.start_line < 1 or ref.end_line < ref.start_line or ref.end_line > len(lines):
        errs.append(f"{loc}: line range invalid (file has {len(lines)} lines)")
    elif ref.symbol and ref.symbol != MODULE_SYMBOL:  # <module>: top-level code, no definition to find
        lo = max(0, ref.start_line - 1 - SYMBOL_SLACK)
        hi = min(len(lines), ref.end_line + SYMBOL_SLACK)
        name = symbol_name(ref.symbol)
        near = any(name in ln for ln in lines[lo:hi])
        word = re.compile(rf"\b{re.escape(name)}\b")
        documented = documented_definition(lines, ref.end_line)
        if (
            not near
            and not any(word.search(h) for h in enclosing_headers(lines, ref.start_line))
            and not (documented and word.search(documented))
        ):
            errs.append(
                f"{loc}: symbol '{ref.symbol}' is neither in the cited lines nor a definition around them"
            )
    return errs


def repair_sha(ref: CommitRef, evidence_commits: set[tuple[str, str]]) -> None:
    """Fix a garbled sha in place when it shares a run of ``MIN_SHA`` hex characters with exactly one
    of this run's commits in that repo (models copying many shas drop or swap characters:
    ``b3086e546c4`` for ``b1d3086e546c``). Anything less certain is left for verification to reject."""
    sha = ref.sha.strip().lower()
    if any(r == ref.repo and (s.startswith(sha) or sha.startswith(s)) for r, s in evidence_commits):
        return
    runs = {sha[i : i + MIN_SHA] for i in range(len(sha) - MIN_SHA + 1)}
    runs = {r for r in runs if all(ch in "0123456789abcdef" for ch in r)}
    hits = {s for r, s in evidence_commits if r == ref.repo and any(run in s for run in runs)}
    if len(hits) == 1:
        ref.sha = hits.pop()


def check_commit_ref(ref: CommitRef, evidence_commits: set[tuple[str, str]]) -> list[str]:
    repair_sha(ref, evidence_commits)
    loc = f"{ref.repo}@{ref.sha}"
    sha = ref.sha.strip().lower()
    if len(sha) < MIN_SHA or not all(ch in "0123456789abcdef" for ch in sha):
        return [f"{loc}: not a commit sha (use at least {MIN_SHA} hex characters from the evidence)"]
    if not any(r == ref.repo and (s.startswith(sha) or sha.startswith(s)) for r, s in evidence_commits):
        return [f"{loc}: commit was not among the retrieved evidence"]
    return []


def check_claim(
    claim: Claim,
    lines_: _Lines,
    evidence_paths: set[tuple[str, str]],
    web_urls: set[str],
    evidence_commits: set[tuple[str, str]],
) -> list[str]:
    repo_refs = [r for r in claim.refs if isinstance(r, RepoRef)]
    commit_refs = [r for r in claim.refs if isinstance(r, CommitRef)]
    web_refs = [r for r in claim.refs if isinstance(r, WebRef)]
    if claim.source == "model":
        return ["model claim must not have refs"] if claim.refs else []
    if claim.source == "web":
        if not web_refs or repo_refs or commit_refs:
            return ["web claim needs web refs only"]
        return [f"url not in this run's web results: {r.url}" for r in web_refs if r.url not in web_urls]
    # repo
    if not repo_refs and not commit_refs:
        return ["repo claim needs at least one file or commit ref"]
    errs = []
    if web_refs:
        errs.append("repo claim must cite repo refs only")
    for r in repo_refs:
        errs += check_repo_ref(r, lines_, evidence_paths)
    for c in commit_refs:
        errs += check_commit_ref(c, evidence_commits)
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


def settle_sources(answer: Answer, check) -> int:
    """Relabel claims whose source contradicts their refs, in place, instead of sending them back: a
    repo claim with no refs is the model's own statement, and a model claim whose refs check out is a
    repo claim (if they don't, the refs are dropped). An answer that cites nothing at all is left as is,
    so verification still asks for citations. Returns how many claims changed."""
    keys = claim_keys(answer)
    if not any(c.refs for _, c in keys):
        return 0
    changed = 0
    for _, c in keys:
        if c.source == "repo" and not c.refs:
            c.source, changed = "model", changed + 1
        elif c.source == "model" and c.refs:
            if all(isinstance(r, (RepoRef, CommitRef)) for r in c.refs) and not check(
                c.model_copy(update={"source": "repo"})
            ):
                c.source = "repo"
            else:
                c.refs = []
            changed += 1
    return changed


def verify(
    answer: Answer, evidence: list[Evidence], web_results: list[WebResult], roots: dict[str, Path]
) -> VerifyResult:
    """Check every claim; claims whose source contradicts their refs are relabeled first (in place,
    see ``settle_sources``)."""
    res = VerifyResult()
    lines_ = _Lines(roots)
    ev_paths = {(e.repo, e.path) for e in evidence if e.repo and e.path}
    ev_commits = {(e.repo, e.sha) for e in evidence if e.repo and e.sha}
    urls = {w.url for w in web_results}
    res.relabeled = settle_sources(answer, lambda c: check_claim(c, lines_, ev_paths, urls, ev_commits))
    keys = claim_keys(answer)
    if not keys and (evidence or web_results):
        msg = (
            f"the answer has no claims, but {len(evidence) + len(web_results)} evidence items were retrieved: "
            "put the findings in sections as claims that cite them"
        )
        res.bad_claims[EMPTY_ANSWER] = [msg]
        res.errors.append(msg)
    for key, claim in keys:
        errs = check_claim(claim, lines_, ev_paths, urls, ev_commits)
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
