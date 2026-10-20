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
    assert d.version == 2
    d.close()
    d2 = DB(tmp_path / "h.db")
    assert d2.version == 2


def test_usage_accounting(db):
    db.record_usage("router", "m1", 10, 5)
    db.record_usage("router", "m1", 10, 5, error=True)
    db.record_usage("answer", "m2")
    assert db.calls_today() == 3
    rows = {(r["step"], r["model"]): r for r in db.usage_today()}
    assert rows[("router", "m1")]["prompt_tokens"] == 20
    assert rows[("router", "m1")]["errors"] == 1


def test_v1_usage_table_migrates_role_to_step(tmp_path):
    import sqlite3

    from haive.db import MIGRATIONS

    path = tmp_path / "old.db"
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at REAL)")
    c.executescript(MIGRATIONS[0])
    c.execute("INSERT INTO schema_migrations VALUES(1, 0)")
    c.execute("INSERT INTO usage VALUES(date('now'), 'fast', 'm', 4, 1, 1, 0)")
    c.commit()
    c.close()
    db = DB(path)
    assert db.version == 2 and db.usage_today()[0]["step"] == "fast" and db.calls_today() == 4
