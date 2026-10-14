"""Set up the bundled sample eval: a sample repo plus a bug-injected copy, registered in config."""

from __future__ import annotations

from pathlib import Path

import yaml

from eval.inject_bugs import cases_to_questions, inject, load_bugs
from haive import paths
from haive.config import RepoConfig, save_config

EVAL_DIR = Path(__file__).parent


def setup_sample(rt, console) -> Path:
    from tests.fixtures.repo_builder import build_repo

    base = paths.data_dir() / "eval"
    sample = base / "eval-sample"
    bugs_repo = base / "eval-sample-bugs"
    if not sample.exists():
        build_repo(sample)
        console.print(f"built sample repo at {sample}")
    cases = []
    if not bugs_repo.exists():
        cases = inject(sample, bugs_repo, load_bugs(EVAL_DIR / "bugs.sample.yaml"), "eval-sample-bugs")
        console.print(f"injected {len(cases)} bugs into {bugs_repo}")
    for name, root in (("eval-sample", sample), ("eval-sample-bugs", bugs_repo)):
        rt.cfg.repos[name] = RepoConfig(path=str(root))
        rt.db.upsert_repo(name, str(root))
    save_config(rt.cfg)
    template = EVAL_DIR / "questions.sample.yaml"
    suite = base / "questions.sample.yaml"
    questions = yaml.safe_load(template.read_text()) or []
    if cases:
        questions += cases_to_questions(cases)
        suite.write_text(yaml.safe_dump(questions, sort_keys=False))
    elif not suite.exists():
        console.print(f"[yellow]{bugs_repo} exists but {suite} does not; delete the bug repo and rerun[/]")
        suite.write_text(yaml.safe_dump(questions, sort_keys=False))
    console.print(
        f"Suite: {suite}\nNext: [bold]haive index eval-sample eval-sample-bugs[/], then "
        f"[bold]haive eval --suite {suite}[/]"
    )
    return suite
