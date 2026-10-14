import subprocess

from haive.index import commits as cm
from haive.tools import git
from tests.fixtures.repo_builder import _git, commit, write


def test_index_commits_without_diffs(sample_repo, db, store):
    db.upsert_repo("v", str(sample_repo))
    st = cm.index_commits(db, store, "v", sample_repo, max_commits=300)
    assert st.added == 3 and not st.rebuilt
    rows = db.q("SELECT subject FROM commits WHERE repo='v' ORDER BY date")
    assert [r["subject"] for r in rows] == [
        "Initial valuation app",
        "Add retries to the API client",
        "Bump web api version",
    ]
    retries = next(c for c in db.q("SELECT text FROM chunks WHERE kind='commit'") if "retries" in c["text"])
    assert "flaky" in retries["text"] and "app/api/client.py" in retries["text"]
    # commit chunks carry the message and file list only, never code
    texts = [c["text"] for c in db.q("SELECT text FROM chunks WHERE kind='commit'")]
    assert all("diff" not in t and "@@" not in t for t in texts)
    assert store.count({"kind": "commit"}) == 3


def test_incremental_and_prune(sample_repo, db, store):
    db.upsert_repo("v", str(sample_repo))
    st = cm.index_commits(db, store, "v", sample_repo, max_commits=2)
    assert st.added == 2
    db.set_repo_indexed("v", st.head, st.head)
    write(sample_repo, {"NEW.md": "# new\n"})
    commit(sample_repo, "Add new doc", "2024-04-01T10:00:00+00:00")
    st2 = cm.index_commits(db, store, "v", sample_repo, max_commits=2)
    assert st2.added == 1 and st2.removed == 1
    subjects = {r["subject"] for r in db.q("SELECT subject FROM commits")}
    assert subjects == {"Add new doc", "Bump web api version"}
    assert store.count({"kind": "commit"}) == 2


def test_rebase_triggers_rebuild(sample_repo, db, store):
    db.upsert_repo("v", str(sample_repo))
    st = cm.index_commits(db, store, "v", sample_repo)
    db.set_repo_indexed("v", st.head, st.head)
    _git(sample_repo, "reset", "-q", "--hard", "HEAD~1")
    write(sample_repo, {"x.md": "x\n"})
    commit(sample_repo, "Diverged", "2024-05-01T10:00:00+00:00")
    st2 = cm.index_commits(db, store, "v", sample_repo)
    assert st2.rebuilt
    subjects = [r["subject"] for r in db.q("SELECT subject FROM commits ORDER BY date")]
    assert subjects == ["Initial valuation app", "Add retries to the API client", "Diverged"]
    assert store.count({"kind": "commit"}) == 3


def test_git_rejects_writes(sample_repo):
    import pytest

    with pytest.raises(git.GitError):
        git._git(sample_repo, "commit", "-m", "x")
    with pytest.raises(git.GitError):
        git._git(sample_repo, "config", "user.name", "x")
    status = subprocess.run(
        ["git", "-C", str(sample_repo), "status", "--porcelain"], capture_output=True, text=True
    )
    assert status.stdout == ""


def test_commit_with_secret_in_message_gets_no_chunk(sample_repo, db, store):
    write(sample_repo, {"NOTE.md": "x\n"})
    commit(sample_repo, "Rotate key sk-abcdefghijklmnopqrstuvwxyz0123", "2024-06-01T10:00:00+00:00")
    db.upsert_repo("v", str(sample_repo))
    cm.index_commits(db, store, "v", sample_repo)
    assert len(db.q("SELECT sha FROM commits")) == 4  # the row is kept
    assert not any("sk-abc" in c["text"] for c in db.q("SELECT text FROM chunks WHERE kind='commit'"))
    assert store.count({"kind": "commit"}) == 3
