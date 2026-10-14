from haive.index.discover import classify, discover, has_secret, is_secret_path, skip_reason


def test_discover_respects_gitignore_and_skips(sample_repo):
    paths = {f.path for f in discover(sample_repo)}
    assert "app/api/client.py" in paths
    assert ".github/workflows/ci.yml" in paths
    assert ".env.example" in paths
    assert ".env" not in paths  # secret
    assert "ignored/skip.py" not in paths  # gitignored
    assert "debug.log" not in paths  # gitignored
    assert "web/package-lock.json" not in paths  # lockfile
    assert "assets/logo.png" not in paths  # binary


def test_skip_reasons():
    assert skip_reason("a/.env.local", 1, 512, []) == "secret"
    assert skip_reason("id_rsa", 1, 512, []) == "secret"
    assert skip_reason("certs/server.pem", 1, 512, []) == "secret"
    assert skip_reason("uv.lock", 1, 512, []) == "lockfile"
    assert skip_reason("dist/app.js", 1, 512, []) == "generated"
    assert skip_reason("a.min.js", 1, 512, []) == "generated"
    assert skip_reason("big.py", 600 * 1024, 512, []) == "too_large"
    assert skip_reason("nb.ipynb", 1, 512, ["*.ipynb"]) == "ignored"
    assert skip_reason("ok.py", 1, 512, []) is None


def test_secret_denylist():
    assert is_secret_path(".env")
    assert is_secret_path("config/.env.production")
    assert not is_secret_path(".env.example")
    assert not is_secret_path(".env.sample")
    assert is_secret_path("credentials.json")
    assert is_secret_path("secrets.yaml")


def test_secret_patterns():
    assert has_secret("key = 'AKIAABCDEFGHIJKLMNOP'")
    assert has_secret("token: sk-abcdefghijklmnopqrstuvwxyz")
    assert has_secret("-----BEGIN RSA PRIVATE KEY-----")
    assert has_secret("ghp_" + "a" * 36)
    assert not has_secret("def sk(): return 'skip'")


def test_classify():
    assert classify("a/b.py") == ("python", "code")
    assert classify("README.md") == ("markdown", "doc")
    assert classify("docker-compose.yml") == ("compose", "config")
    assert classify("Dockerfile") == ("dockerfile", "config")
    assert classify("x.tsx") == ("tsx", "code")
    assert classify("main.tf") == ("hcl", "config")
