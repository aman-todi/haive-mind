"""Debug eval cases by bug injection, never in the user's repos.

Each bug is applied to a fresh clone in a temp/eval directory, committed there (so blame shows a
recent change), and the bug's command is run in that copy to capture a real error. The case
records the trace plus the expected file and line and the known cause. The copy is then
registered as a separate eval repo.

Bug spec (YAML list):
  - id: missing-key
    file: app/valuation/dcf.py
    find: 'company.get("dividends")'
    replace: 'company["dividends"]'
    command: python -c "from app.valuation.dcf import choose_method; choose_method({'fcf': 1})"
    cause: dividends is optional but is now indexed directly
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Bug:
    id: str
    file: str
    find: str
    replace: str
    command: str
    cause: str
    question: str = ""
    requires: str = ""  # executable the command needs (python3, go, node, javac); skipped if missing


@dataclass
class Case:
    id: str
    repo: str
    trace_file: str
    expected_file: str
    expected_line: int
    cause: str
    question: str = ""


def _git(root: Path, *args: str) -> str:
    env = dict(
        os.environ,
        GIT_AUTHOR_NAME="haive-eval",
        GIT_AUTHOR_EMAIL="eval@haive",
        GIT_COMMITTER_NAME="haive-eval",
        GIT_COMMITTER_EMAIL="eval@haive",
    )
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True, env=env
    ).stdout


def load_bugs(path: Path) -> list[Bug]:
    return [Bug(**b) for b in yaml.safe_load(path.read_text()) or []]


def inject(source: Path, dest: Path, bugs: list[Bug], repo_name: str, timeout: int = 60) -> list[Case]:
    """Clone ``source`` to ``dest`` (reads only from source), then make one case per bug.

    Each bug's trace is captured in isolation (apply, run, record, restore). Afterwards every bug
    that produced an error is committed to the copy, one commit each, so the indexed eval repo
    contains all of them at their recorded lines. Keep each replacement line-count neutral.
    """
    if dest.exists():
        raise FileExistsError(f"{dest} exists; remove it first")
    subprocess.run(["git", "clone", "-q", "--no-hardlinks", str(source), str(dest)], check=True)
    traces = dest.parent / f"{repo_name}-traces"
    traces.mkdir(parents=True, exist_ok=True)
    cases: list[Case] = []
    good: list[Bug] = []
    for bug in bugs:
        if bug.requires and shutil.which(bug.requires) is None:
            print(f"skip {bug.id}: {bug.requires} is not installed")
            continue
        target = dest / bug.file
        text = target.read_text()
        if bug.find not in text:
            print(f"skip {bug.id}: pattern not found in {bug.file}")
            continue
        if bug.find.count("\n") != bug.replace.count("\n"):
            print(f"warning {bug.id}: replacement changes the line count; later line numbers may drift")
        line = text.split(bug.find, 1)[0].count("\n") + 1
        target.write_text(text.replace(bug.find, bug.replace, 1))
        try:
            proc = subprocess.run(
                bug.command, shell=True, cwd=dest, capture_output=True, text=True, timeout=timeout
            )
        finally:
            target.write_text(text)
        output = (proc.stderr or "") + (proc.stdout or "")
        if proc.returncode == 0 or not output.strip():
            print(f"skip {bug.id}: command did not fail")
            continue
        tf = traces / f"{bug.id}.txt"
        tf.write_text(output)
        cases.append(Case(bug.id, repo_name, str(tf), bug.file, line, bug.cause, bug.question))
        good.append(bug)
    for bug in good:
        target = dest / bug.file
        target.write_text(target.read_text().replace(bug.find, bug.replace, 1))
        _git(dest, "add", bug.file)
        _git(dest, "commit", "-q", "-m", f"eval: inject {bug.id}")
    return cases


def cases_to_questions(cases: list[Case]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"debug-{c.id}",
            "question": c.question,
            "trace_file": c.trace_file,
            "repos": [c.repo],
            "expected_intent": "debug",
            "expected_files": [f"{c.repo}/{c.expected_file}"],
            "expected_line": c.expected_line,
            "notes": c.cause,
        }
        for c in cases
    ]
