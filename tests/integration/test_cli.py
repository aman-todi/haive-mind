"""End-to-end CLI runs with a persistent Chroma store, hash embeddings, and a scripted fake LLM."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from haive import paths
from haive.cli import app
from haive.config import load_config, save_config
from haive.llm.models import LLM
from haive.runtime import Runtime
from tests.fakes import FakeChatModel
from tests.graph.helpers import DCF_CLAIM, answer, grade, repo_claim, router

pytest.importorskip("chromadb")
TRACES = Path(__file__).parent.parent / "fixtures" / "traces"


def snapshot(root: Path) -> dict:
    """Everything the read-only guarantee covers: git status, and mtimes of every file incl. .git."""
    # The probe itself must not write: plain `git status` may refresh and rewrite .git/index
    # (depending on the user's git config, e.g. core.checkstat), which would look like haive did it.
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--ignored"],
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    ).stdout
    mtimes = {str(p.relative_to(root)): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
    return {"status": status, "mtimes": mtimes}


@pytest.fixture
def cli(monkeypatch, haive_home):
    fake = FakeChatModel(
        {
            "RouterOutput": [router("lookup", identifiers=["choose_method"])],
            "GradeOutput": [grade()],
            "Answer": [answer(DCF_CLAIM)],
            "text": ["Where the DCF picks its method"],
        }
    )
    built: list[str] = []

    def factory(model_id):
        built.append(model_id)
        return fake

    monkeypatch.setattr(Runtime, "llm", property(lambda self: LLM(self.cfg, self.db, factory)))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("COLUMNS", "200")
    runner = CliRunner()

    def run(*args: str, input: str | None = None, ok: bool = True):
        res = runner.invoke(app, list(args), input=input, catch_exceptions=False)
        if ok:
            assert res.exit_code == 0, res.output
        return res

    run.fake = fake  # type: ignore[attr-defined]
    run.built = built  # type: ignore[attr-defined]
    return run


def setup_repo_at(cli, root: Path):
    return setup_repo(cli, root)


def setup_repo(cli, sample_repo, *extra: str):
    cli("init", "--no-check", "--backend", "chroma")
    cfg = load_config()
    cfg.embeddings.model = "hash"
    cfg.models.requests_per_minute = 0
    save_config(cfg)
    cli("add", str(sample_repo), "--name", "valuation", "-g", "valuation-project", *extra)
    return cli("index")


def test_full_flow_is_read_only(cli, sample_repo, other_repo):
    before = snapshot(sample_repo)
    out = setup_repo(cli, sample_repo).output
    assert "valuation:" in out and " new, 0 changed, 0 deleted files" in out
    cli("add", str(other_repo), "--name", "frontend", "-g", "valuation-project")
    cli("index", "frontend")

    res = cli("repos")
    assert "valuation" in res.output and "fresh" in res.output
    assert "valuation-project" in cli("groups").output
    status = cli("status").output
    assert "chroma" in status and "Index health" in status

    res = cli("ask", "Where does the DCF choose a valuation method?", "--json", "-r", "valuation")
    data = json.loads(res.output)
    assert data["intent"] == "lookup" and data["scope"] == ["valuation"]
    assert data["answer"]["sections"][0]["claims"][0]["refs"][0]["path"] == "app/valuation/dcf.py"
    assert data["llm_calls"] == 3

    res = cli("ask", "Where does the DCF choose a valuation method?", "--trace")
    assert "[repo]" in res.output and "valuation/app/valuation/dcf.py:1-7" in res.output
    assert "graph trace" in res.output and "router" in res.output

    cli.fake.script["Answer"] = [
        {
            "summary": "fcf missing",
            "sections": [],
            "hypotheses": [
                {
                    "title": "missing key",
                    "confidence": "high",
                    "claims": [repo_claim("app/valuation/dcf.py", 5, 5, text="indexes fcf")],
                }
            ],
        }
    ]
    res = cli("debug", str(TRACES / "python_chained.txt"), "--json", "--web", "off")
    data = json.loads(res.output)
    assert data["intent"] == "debug" and data["answer"]["hypotheses"][0]["confidence"] == "high"
    res = cli("debug", "-", "--json", "--web", "off", input=(TRACES / "python_chained.txt").read_text())
    assert json.loads(res.output)["intent"] == "debug"

    cli.fake.script["Answer"] = [answer(DCF_CLAIM)]
    res = cli("chat", "--web", "off", input="Where does the DCF choose a method?\n/threads\n/exit\n")
    assert "[repo]" in res.output and "Where the DCF picks its method" in res.output
    th = cli("threads").output
    assert "Where the DCF picks its method" in th and "1" in th

    after = snapshot(sample_repo)
    changed = sorted(
        k
        for k in set(after["mtimes"]) | set(before["mtimes"])
        if after["mtimes"].get(k) != before["mtimes"].get(k)
    )
    assert not changed, f"files in an indexed repo were modified: {changed}"
    assert after["status"] == before["status"]


def test_sync_pulls_then_indexes(cli, sample_repo, tmp_path):
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(sample_repo), str(clone)], check=True)
    setup_repo_at(cli, clone)
    (sample_repo / "NEW.md").write_text("# new upstream doc\n")
    subprocess.run(["git", "-C", str(sample_repo), "add", "NEW.md"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(sample_repo),
            "-c",
            "user.name=x",
            "-c",
            "user.email=x@x",
            "commit",
            "-qm",
            "upstream change",
        ],
        check=True,
    )
    out = cli("sync").output
    assert "valuation:" in out and "+1/-0 commits" in out
    assert (clone / "NEW.md").exists()


def test_threads_resume_and_delete(cli, sample_repo):
    setup_repo(cli, sample_repo)
    cli("chat", "--web", "off", input="Where does the DCF choose a method?\n/exit\n")
    rt = Runtime()
    rows = rt.db.q("SELECT id, turns FROM threads")
    assert len(rows) == 1 and rows[0]["turns"] == 1
    tid = rows[0]["id"]
    out = cli("chat", "--resume", tid[:8], "--web", "off", input="and the discount function?\n/exit\n").output
    assert "Resuming" in out
    assert Runtime().db.q("SELECT turns FROM threads")[0]["turns"] == 2
    state = Runtime().checkpointer.get_tuple({"configurable": {"thread_id": tid}})
    assert len(state.checkpoint["channel_values"]["messages"]) == 4  # two user + two assistant turns
    cli("threads", "delete", tid[:8])
    assert not Runtime().db.q("SELECT * FROM threads")


def test_chat_routes_pasted_trace_to_debug(cli, sample_repo):
    setup_repo(cli, sample_repo)
    cli.fake.script["Answer"] = [{"summary": "diag", "sections": []}]
    trace = (TRACES / "python_chained.txt").read_text()
    cli("chat", "--web", "off", input=trace + "\n/exit\n")
    assert "RouterOutput" not in cli.fake.calls  # traces skip the router LLM


def test_remove(cli, sample_repo):
    setup_repo(cli, sample_repo)
    rt = Runtime()
    assert rt.store.count({"repo": ["valuation"]}) > 0
    cli("remove", "valuation", "--yes")
    rt = Runtime()
    assert "valuation" not in rt.cfg.repos and "valuation" not in rt.cfg.groups.get("valuation-project", [])
    assert rt.store.count({"repo": ["valuation"]}) == 0
    assert not rt.db.q("SELECT * FROM chunks WHERE repo='valuation'")


def test_incremental_index_and_full(cli, sample_repo):
    setup_repo(cli, sample_repo)
    (sample_repo / "app" / "valuation" / "dcf.py").write_text("def choose_method(c):\n    return 'fcf'\n")
    (sample_repo / "web" / "src" / "api.ts").unlink()
    (sample_repo / "NOTES.md").write_text("# notes\n")
    out = cli("index").output
    assert "1 new, 1 changed, 1 deleted files" in out
    out = cli("index", "--full").output
    assert "commit history rebuilt" not in out and "+3/-0 commits" in out


def test_budget_gate_and_missing_key(cli, sample_repo, monkeypatch):
    setup_repo(cli, sample_repo)
    cfg = load_config()
    cfg.limits.daily_call_budget = 1
    save_config(cfg)
    Runtime().db.record_usage("fast", "m")
    res = cli("ask", "anything", ok=False)
    assert res.exit_code == 1 and "daily LLM call budget" in res.output
    cli("ask", "anything", "--force", "--json")
    monkeypatch.delenv("OPENROUTER_API_KEY")
    res = cli("ask", "anything", "--force", ok=False)
    assert "OPENROUTER_API_KEY is not set" in res.output


def test_not_initialized(cli):
    res = cli("repos", ok=False)
    assert res.exit_code == 1 and "haive init" in res.output
    assert not paths.config_file().exists()


def test_add_validation(cli, sample_repo, tmp_path):
    setup_repo(cli, sample_repo)
    assert cli("add", str(tmp_path), ok=False).exit_code == 1
    assert cli("add", str(sample_repo), "--name", "valuation", ok=False).exit_code == 1
    assert cli("add", str(sample_repo), "--name", "bad:name", ok=False).exit_code == 1


def test_paid_flag_uses_strong_paid_model(cli, sample_repo):
    setup_repo(cli, sample_repo)
    cli("ask", "Where does the DCF choose a valuation method?", "--json")
    assert "deepseek/deepseek-v4-pro-0813" not in cli.built
    cli("ask", "Where does the DCF choose a valuation method?", "--json", "--paid")
    assert "deepseek/deepseek-v4-pro-0813" in cli.built
    assert "strong_paid: deepseek/deepseek-v4-pro-0813  (off" in cli("status").output


def test_batch_add_scan_and_multiple_paths(cli, sample_repo, other_repo, tmp_path):
    cli("init", "--no-check", "--backend", "chroma")
    (tmp_path / "not-a-repo").mkdir()
    out = cli("add", "--scan", str(tmp_path), "-g", "everything").output
    assert "2 added, 0 already registered, 0 failed" in out and "non-repo folders ignored" in out
    cfg = load_config()
    assert set(cfg.repos) == {"valuation", "frontend"}
    assert cfg.groups["everything"] == ["frontend", "valuation"]
    # re-running is safe: already-registered repos are skipped, not errors
    out = cli("add", "--scan", str(tmp_path)).output
    assert "0 added, 2 already registered, 0 failed" in out
    # several explicit paths; a bad one is reported but doesn't stop the rest
    cli("remove", "frontend", "--yes")
    res = cli("add", str(other_repo), str(tmp_path / "not-a-repo"), ok=False)
    assert res.exit_code == 1 and "1 added, 0 already registered, 1 failed" in res.output
    assert "frontend" in load_config().repos
    assert "only works when adding a single repo" in cli("add", str(sample_repo), str(other_repo), "--name", "x",
                                                         ok=False).output  # fmt: skip
    cli("index")  # no names: indexes every registered repo
    assert "fresh" in cli("repos").output
