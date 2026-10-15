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
    assert loaded.limits.max_llm_calls_per_question == 16
    assert "api_key =" not in p.read_text()  # secrets never land in config.toml


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
    db.record_usage("fast", "m1", 10, 5)
    db.record_usage("fast", "m1", 10, 5, error=True)
    db.record_usage("strong", "m2")
    assert db.calls_today() == 3
    rows = {(r["role"], r["model"]): r for r in db.usage_today()}
    assert rows[("fast", "m1")]["prompt_tokens"] == 20
    assert rows[("fast", "m1")]["errors"] == 1
