from haive import paths
from haive.config import Config, RepoConfig, load_config, save_config
from haive.db import DB


def test_paths_respect_haive_home(haive_home):
    assert paths.config_file() == haive_home / "config" / "config.toml"
    assert paths.db_file() == haive_home / "data" / "haive.db"


def test_config_roundtrip(tmp_path):
    c = Config()
    c.repos["a"] = RepoConfig(path="~/code/a")
    c.groups["g"] = ["a"]
    p = save_config(c, tmp_path / "c.toml")
    loaded = load_config(p)
    assert loaded.repos["a"].path == "~/code/a"
    assert loaded.groups == {"g": ["a"]}
    assert loaded.limits.max_llm_calls_per_question == 25
    assert "api_key =" not in p.read_text()  # secrets never land in config.toml
    text = p.read_text()
    # defaults are listed commented out: visible, but not frozen into the file
    assert "\n# max_claims = 10\n" in text and "\n# use_paid = true\n" in text
    assert '\nmodel = "jinaai/jina-embeddings-v2-base-code"\n' in text  # the index's model is pinned
    assert "\n[groups]\ng = [" in text  # groups the user defined are kept
    c.limits.max_claims = 6
    c.models.strong = ["m"]
    loaded = load_config(save_config(c, p))
    assert loaded.limits.max_claims == 6 and loaded.models.strong == ["m"]
    assert loaded.limits.max_llm_calls_per_question == 25 and loaded.models.use_paid


def test_resolve_names_fuzzy_and_groups():
    c = Config()
    for n in ("valuation-backend", "valuation-frontend", "weatherpruf"):
        c.repos[n] = RepoConfig(path=f"/x/{n}")
    c.groups["valuation"] = ["valuation-backend", "valuation-frontend"]
    assert c.resolve_names(["Weatherpruf"]) == (["weatherpruf"], [])
    assert c.resolve_names(["weatherproof"]) == (["weatherpruf"], [])
    assert c.resolve_names(["valuation"]) == (["valuation-backend", "valuation-frontend"], [])
    assert c.resolve_names(["nope-xyz"]) == ([], ["nope-xyz"])
    assert c.resolve_names(["all"])[0] == list(c.repos)


def test_db_migrations_idempotent(tmp_path):
    d = DB(tmp_path / "h.db")
    assert d.version == 1
    d.close()
    d2 = DB(tmp_path / "h.db")
    assert d2.version == 1


def test_usage_accounting(db):
    db.record_usage("router", "m1", 10, 5)
    db.record_usage("router", "m1", 10, 5, error=True)
    db.record_usage("answer", "m2")
    assert db.calls_today() == 3
    rows = {(r["step"], r["model"]): r for r in db.usage_today()}
    assert rows[("router", "m1")]["prompt_tokens"] == 20
    assert rows[("router", "m1")]["errors"] == 1


def test_db_is_safe_to_share_across_threads(tmp_path):
    # The graph's parallel branches and the LLM-call threads share one connection.
    import threading

    db = DB(tmp_path / "h.db")
    for r in ("a", "b"):
        db.save_profile(r, {"repo": r, "n": 0})
    errors: list[BaseException] = []

    def work(repo: str) -> None:
        try:
            for i in range(200):
                db.record_usage("summary", "m")
                prof = db.get_profile(repo)
                assert prof is not None and prof["repo"] == repo
                db.save_profile(repo, {"repo": repo, "n": i})
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=work, args=(r,)) for r in ("a", "b") * 4]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[0]
    assert db.calls_today() == 1600
