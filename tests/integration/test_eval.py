"""Eval harness end to end on the bundled sample, with a scripted fake LLM (no network)."""

from __future__ import annotations

import subprocess

import pytest
import yaml
from rich.console import Console

from eval import metrics
from eval.run import main as eval_main
from haive.config import load_config, save_config
from haive.llm.models import LLM
from haive.runtime import Runtime
from tests.fakes import FakeChatModel, factory_for
from tests.graph.helpers import grade, router

pytest.importorskip("chromadb")


def test_metrics_units():
    ans = {
        "sections": [
            {
                "claims": [
                    {
                        "source": "repo",
                        "refs": [{"repo": "r", "path": "a/b.py", "start_line": 3, "end_line": 9}],
                    },
                    {
                        "source": "repo",
                        "unverified": True,
                        "refs": [{"repo": "r", "path": "c.py", "start_line": 1, "end_line": 1}],
                    },
                ]
            }
        ],
        "hypotheses": [
            {
                "claims": [
                    {
                        "source": "repo",
                        "refs": [{"repo": "r", "path": "x.py", "start_line": 1, "end_line": 2}],
                    }
                ]
            },
            {
                "claims": [
                    {
                        "source": "repo",
                        "refs": [{"repo": "r", "path": "a/b.py", "start_line": 10, "end_line": 12}],
                    }
                ]
            },
        ],
    }
    assert metrics.file_hit(ans, ["r/a/b.py"]) is True
    assert metrics.file_hit(ans, ["r/c.py"]) is False  # unverified refs don't count
    assert metrics.file_hit(ans, []) is None
    assert metrics.debug_hits(ans, "r/a/b.py", 14) == (False, True)
    assert metrics.recall_at_k([("r", "z.py")] * 3 + [("r", "a/b.py")], ["r/a/b.py"], k=2) is True
    assert metrics.percentile([1, 2, 3, 4], 0.5) == 2.5
    s = metrics.summarize(
        [
            {
                "intent_ok": True,
                "file_hit": True,
                "citations": 2,
                "citations_ok": 1,
                "latency_s": 1.0,
                "llm_calls": 3,
            },
            {
                "intent_ok": False,
                "file_hit": None,
                "citations": 2,
                "citations_ok": 2,
                "latency_s": 3.0,
                "llm_calls": 5,
            },
        ]
    )
    assert s["intent_accuracy"] == 0.5 and s["file_hit_rate"] == 1.0 and s["citation_validity"] == 0.75
    assert s["llm_calls_mean"] == 4 and s["latency_p50_s"] == 2.0


def answer_citing_expected(messages):
    """Fake model: cite the first evidence header in the prompt (a real file + range)."""
    import re

    text = messages[0].content if hasattr(messages[0], "content") else str(messages)
    m = re.search(r"\[E\d+\] repo=(\S+) path=(\S+) lines=(\d+)-(\d+)", text)
    claims = []
    if m:
        claims = [
            {
                "text": "here",
                "source": "repo",
                "refs": [
                    {
                        "repo": m.group(1),
                        "path": m.group(2),
                        "start_line": int(m.group(3)),
                        "end_line": int(m.group(4)),
                    }
                ],
            }
        ]
    return {
        "summary": "s",
        "sections": [{"claims": claims}],
        "hypotheses": [{"title": "t", "confidence": "med", "claims": claims}],
    }


def test_eval_on_sample(monkeypatch, tmp_path):
    fake = FakeChatModel(
        {
            "RouterOutput": [router("lookup")],
            "GradeOutput": [grade()],
            "Answer": [answer_citing_expected],
            "RepoSummary": [{"repo": "x", "purpose": "p"}],
        }
    )
    monkeypatch.setattr(Runtime, "llm", property(lambda self: LLM(self.cfg, self.db, factory_for(fake))))
    from typer.testing import CliRunner

    from haive.cli import app

    CliRunner().invoke(app, ["init", "--no-check", "--backend", "chroma"], catch_exceptions=False)
    cfg = load_config()
    cfg.embeddings.model = "hash"
    cfg.models.requests_per_minute = 0
    save_config(cfg)
    con = Console(record=True, width=200)
    suite = eval_main(setup_sample=True, console=con)
    questions = yaml.safe_load(suite.read_text())
    debug_cases = [q for q in questions if q["id"].startswith("debug-")]
    assert len(debug_cases) >= 5  # python cases always run; go/java when installed
    rt = Runtime()
    sample = rt.cfg.repo_root("eval-sample")
    from haive.index.pipeline import Indexer

    for name in ("eval-sample", "eval-sample-bugs"):
        Indexer(rt.cfg, rt.db, rt.store).index_repo(name)
    readme = tmp_path / "README.md"
    readme.write_text("x\n<!-- eval-results:start -->\nold\n<!-- eval-results:end -->\ny\n")
    out = eval_main(
        suite=str(suite),
        ablation="hybrid+loop,ripgrep_only,chroma",
        console=con,
        results_dir=tmp_path / "results",
        readme=readme,
    )
    summary = (out / "summary.md").read_text()
    assert "| hybrid+loop |" in summary and "| ripgrep_only |" in summary and "| chroma |" in summary
    assert "old" not in readme.read_text() and "| hybrid+loop |" in readme.read_text()
    import csv

    rows = list(csv.DictReader((out / "hybrid_loop.csv").open()))
    assert {r["id"] for r in rows} >= {"lookup-choose-method", "debug-missing-key"}
    assert not any(r.get("error") for r in rows), [r["error"] for r in rows if r.get("error")]
    by_id = {r["id"]: r for r in rows}
    assert "first_errors" in rows[0]  # first-try verification errors, per question
    assert by_id["lookup-choose-method"]["recall10"] == "True"
    assert by_id["debug-missing-key"]["recall10"] == "True"  # trace frames map to the bug repo
    from eval.run import retrieved_files

    for q in questions[:4]:  # the two designs share retrieval, so recall@10 must not differ
        assert retrieved_files(rt, q, "combined") == retrieved_files(rt, q, "hybrid+loop")
    print(summary)
    # the sample source repo is never modified by bug injection or eval runs
    assert (
        subprocess.run(
            ["git", "-C", str(sample), "status", "--porcelain"], capture_output=True, text=True
        ).stdout
        == ""
    )


def test_first_try_errors_are_grouped_by_kind():
    rows = [
        {"first_errors_list": ["[s0.c0] \"x\": r/a.py:9-99: line range invalid (file has 20 lines)",
                               "[s0.c1] \"y\": r/b.py:1-2: path was not among the retrieved evidence"]},
        {"first_errors_list": ["[s0.c0] \"z\": r/a.py:3-4: line range invalid (file has 20 lines)"]},
        {},
    ]  # fmt: skip
    assert metrics.error_kinds(rows) == {"line range outside the file": 2, "file not in the evidence": 1}
    assert metrics.error_kind("model claim must not have refs") == "wrong citation type"
