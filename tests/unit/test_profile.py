from haive.index.discover import discover
from haive.index.profile import build_profile, profile_summary


def test_profile(sample_repo):
    p = build_profile("valuation", sample_repo, discover(sample_repo))
    assert "python" in p.languages and "typescript" in p.languages
    assert "FastAPI" in p.frameworks and "React" in p.frameworks
    assert any("valuation = app.server:main" in e for e in p.entrypoints)
    assert any(e.startswith("npm run dev") for e in p.entrypoints)
    assert {s["name"] for s in p.compose_services} == {"api", "web"}
    web = next(s for s in p.compose_services if s["name"] == "web")
    assert web["env"] == ["API_URL"] and web["ports"] == ["3000:3000"]
    assert p.workflows[0]["triggers"] == ["push", "pull_request"]
    assert p.terraform_resources == ["aws_s3_bucket"]
    assert {"API_URL", "DATABASE_URL"} <= set(p.env_vars)
    routes = {(r["method"], r["path"]) for r in p.routes_defined}
    assert ("GET", "/users/{user_id}") in routes and ("POST", "/valuations") in routes
    assert any(u["url"].startswith("/users/") for u in p.urls_called)
    text = profile_summary(p)
    assert "compose service web" in text and "routes defined" in text
