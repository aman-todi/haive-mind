"""Deterministic repo profile (no LLM): languages, dependencies, entrypoints, compose services,
workflows, terraform resources, env vars, HTTP routes defined and URLs called."""

from __future__ import annotations

import json
import re
import tomllib
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from haive.index.discover import FileInfo, read_text
from haive.types import RepoProfile

KNOWN_FRAMEWORKS = {
    "fastapi": "FastAPI", "flask": "Flask", "django": "Django", "starlette": "Starlette", "streamlit": "Streamlit",
    "sqlalchemy": "SQLAlchemy", "pydantic": "Pydantic", "celery": "Celery", "langchain": "LangChain",
    "langgraph": "LangGraph", "pytest": "pytest", "typer": "Typer", "click": "Click", "pandas": "pandas",
    "express": "Express", "next": "Next.js", "react": "React", "vue": "Vue", "svelte": "Svelte",
    "@sveltejs/kit": "SvelteKit", "@nestjs/core": "NestJS", "koa": "Koa", "fastify": "Fastify", "hono": "Hono",
    "prisma": "Prisma", "@prisma/client": "Prisma", "jest": "Jest", "vitest": "Vitest", "vite": "Vite",
    "axios": "axios", "tailwindcss": "Tailwind", "electron": "Electron", "react-native": "React Native",
    "github.com/gin-gonic/gin": "Gin", "github.com/labstack/echo": "Echo", "github.com/gofiber/fiber": "Fiber",
    "actix-web": "Actix Web", "axum": "Axum", "rocket": "Rocket", "tokio": "Tokio", "serde": "Serde",
    "Microsoft.AspNetCore": "ASP.NET Core", "Microsoft.EntityFrameworkCore": "Entity Framework Core",
    "org.springframework.boot": "Spring Boot",
}  # fmt: skip

ENTRY_NAMES = {"main.py", "__main__.py", "app.py", "manage.py", "wsgi.py", "asgi.py", "main.go", "main.rs",
               "index.js", "index.ts", "server.js", "server.ts", "app.js", "app.ts", "main.ts", "main.js",
               "Program.cs", "Main.java", "Application.java", "main.c", "main.cpp"}  # fmt: skip

ENV_PATTERNS = [
    r"os\.environ\[\s*['\"]([A-Z][A-Z0-9_]+)['\"]",
    r"os\.environ\.get\(\s*['\"]([A-Z][A-Z0-9_]+)['\"]",
    r"os\.getenv\(\s*['\"]([A-Z][A-Z0-9_]+)['\"]",
    r"process\.env\.([A-Z][A-Z0-9_]+)",
    r"process\.env\[\s*['\"]([A-Z][A-Z0-9_]+)['\"]",
    r"import\.meta\.env\.([A-Z][A-Z0-9_]+)",
    r"GetEnvironmentVariable\(\s*\"([A-Z][A-Z0-9_]+)\"",
    r"env::var\(\s*\"([A-Z][A-Z0-9_]+)\"",
    r"os\.Getenv\(\s*\"([A-Z][A-Z0-9_]+)\"",
    r"System\.getenv\(\s*\"([A-Z][A-Z0-9_]+)\"",
    r"\$\{([A-Z][A-Z0-9_]+)(?::-[^}]*)?\}",
]
ENV_RE = re.compile("|".join(ENV_PATTERNS))

ROUTE_PATTERNS = [
    # FastAPI / Flask decorators: @app.get("/x"), @router.post('/x'), @bp.route("/x", methods=["POST"])
    re.compile(r"@\w+\.(get|post|put|patch|delete|route|api_route|websocket)\(\s*['\"]([^'\"]*)['\"]"),
    # Express / Koa / Hono: app.get('/x', ...), router.post("/x", ...)
    re.compile(
        r"\b(?:app|router|server|api)\.(get|post|put|patch|delete|all|use)\(\s*['\"`](/[^'\"`]*)['\"`]"
    ),
    # ASP.NET attributes and minimal APIs
    re.compile(r"\[(Http(?:Get|Post|Put|Patch|Delete)|Route)\(\s*\"([^\"]*)\""),
    re.compile(r"\.Map(Get|Post|Put|Patch|Delete)\(\s*\"([^\"]*)\""),
    # Spring
    re.compile(r"@(Get|Post|Put|Patch|Delete|Request)Mapping\(\s*(?:value\s*=\s*)?\"([^\"]*)\""),
]
CALL_RE = re.compile(
    r"\b(fetch|axios(?:\.\w+)?|requests\.\w+|httpx\.\w+|ky(?:\.\w+)?|\$http\.\w+|http\.(?:Get|Post)|"
    r"HttpClient\.\w+|client\.(?:get|post|put|patch|delete))\(\s*[`'\"]([^`'\"]+)[`'\"]"
)


def _deps_pyproject(text: str) -> list[str]:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return []
    deps = list(data.get("project", {}).get("dependencies", []))
    for group in data.get("project", {}).get("optional-dependencies", {}).values():
        deps += group
    deps += list(data.get("tool", {}).get("poetry", {}).get("dependencies", {}))
    for group in data.get("dependency-groups", {}).values():
        deps += [d for d in group if isinstance(d, str)]
    return [_pkg_name(d) for d in deps if isinstance(d, str) and d.lower() != "python"]


def _pkg_name(spec: str) -> str:
    return re.split(r"[\s<>=!~;\[@]", spec.strip(), maxsplit=1)[0].lower()


def _deps_requirements(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            out.append(_pkg_name(line))
    return out


def _deps_package_json(text: str) -> tuple[list[str], dict[str, str]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [], {}
    deps = list(data.get("dependencies", {})) + list(data.get("devDependencies", {}))
    scripts = data.get("scripts", {}) if isinstance(data.get("scripts"), dict) else {}
    return deps, scripts


def _deps_go_mod(text: str) -> list[str]:
    out = []
    in_block = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("require ("):
            in_block = True
            continue
        if in_block and s == ")":
            in_block = False
            continue
        if in_block and s:
            out.append(s.split()[0])
        elif s.startswith("require "):
            out.append(s.split()[1])
    return out


def _deps_cargo(text: str) -> list[str]:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return []
    return list(data.get("dependencies", {})) + list(data.get("dev-dependencies", {}))


def _deps_csproj(text: str) -> list[str]:
    return re.findall(r'PackageReference\s+Include="([^"]+)"', text)


def _frameworks(deps: list[str]) -> list[str]:
    found = []
    for d in deps:
        for key, name in KNOWN_FRAMEWORKS.items():
            if (d == key or d.startswith(key + "/") or d.startswith(key + ".")) and name not in found:
                found.append(name)
    return found


def _yaml(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return None


def _compose_services(path: str, text: str) -> list[dict[str, Any]]:
    data = _yaml(text)
    if not isinstance(data, dict) or not isinstance(data.get("services"), dict):
        return []
    out = []
    for name, svc in data["services"].items():
        svc = svc or {}
        env = svc.get("environment") or {}
        env_names = list(env) if isinstance(env, dict) else [str(e).split("=", 1)[0] for e in env]
        depends = svc.get("depends_on") or []
        out.append(
            {"name": name, "file": path, "image": svc.get("image"),
             "build": svc.get("build") if isinstance(svc.get("build"), str) else (svc.get("build") or {}).get("context"),
             "ports": [str(p) for p in svc.get("ports", []) or []], "env": env_names,
             "depends_on": list(depends) if isinstance(depends, (list, dict)) else []}
        )  # fmt: skip
    return out


def _workflow(path: str, text: str) -> dict[str, Any] | None:
    data = _yaml(text)
    if not isinstance(data, dict):
        return None
    on = data.get("on", data.get(True))  # YAML 1.1 parses a bare `on` key as True
    if isinstance(on, dict):
        triggers = list(on)
    elif isinstance(on, list):
        triggers = on
    elif isinstance(on, str):
        triggers = [on]
    else:
        triggers = []
    jobs = list((data.get("jobs") or {}).keys()) if isinstance(data.get("jobs"), dict) else []
    return {
        "file": path,
        "name": data.get("name", PurePosixPath(path).stem),
        "triggers": triggers,
        "jobs": jobs,
    }


def _clean_url(u: str) -> str:
    u = re.sub(r"^\$\{[^}]*\}", "", u)  # strip template base like ${API_URL}
    u = re.sub(r"\$\{[^}]*\}", "{param}", u)
    return u


def build_profile(repo: str, root: Path, files: list[FileInfo]) -> RepoProfile:
    prof = RepoProfile(repo=repo)
    langs: Counter[str] = Counter()
    all_deps: list[str] = []
    env: set[str] = set()

    for f in files:
        name = PurePosixPath(f.path).name
        text = read_text(root, f.path)
        if text is None:
            continue
        if f.kind == "code":
            langs[f.language] += text.count("\n") + 1
        deps: list[str] = []
        if name == "pyproject.toml":
            deps = _deps_pyproject(text)
            try:
                scripts = tomllib.loads(text).get("project", {}).get("scripts", {})
            except tomllib.TOMLDecodeError:
                scripts = {}
            prof.entrypoints += [f"{k} = {v} ({f.path})" for k, v in scripts.items()]
        elif re.match(r"requirements.*\.txt$", name):
            deps = _deps_requirements(text)
        elif name == "package.json":
            deps, scripts = _deps_package_json(text)
            prof.entrypoints += [f"npm run {k}: {v} ({f.path})" for k, v in scripts.items()]
        elif name == "go.mod":
            deps = _deps_go_mod(text)
        elif name == "Cargo.toml":
            deps = _deps_cargo(text)
        elif name.endswith(".csproj"):
            deps = _deps_csproj(text)
        if deps:
            prof.dependencies[f.path] = deps
            all_deps += deps

        if name in ENTRY_NAMES or (name == "main.go" and "/cmd/" in f"/{f.path}"):
            prof.entrypoints.append(f.path)
        if f.language == "python" and 'if __name__ == "__main__"' in text and f.path not in prof.entrypoints:
            prof.entrypoints.append(f.path)

        if f.language in ("compose",) or (f.language == "yaml" and "compose" in name):
            prof.compose_services += _compose_services(f.path, text)
        if f.path.startswith(".github/workflows/") and f.language == "yaml":
            wf = _workflow(f.path, text)
            if wf:
                prof.workflows.append(wf)
        if f.language == "hcl":
            for m in re.finditer(r'^\s*resource\s+"([^"]+)"', text, re.M):
                if m.group(1) not in prof.terraform_resources:
                    prof.terraform_resources.append(m.group(1))
        if name in (".env.example", ".env.sample", ".env.template"):
            env.update(
                m.group(1) for m in re.finditer(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]+)\s*=", text, re.M)
            )

        if f.kind in ("code", "config"):
            for m in ENV_RE.finditer(text):
                env.update(g for g in m.groups() if g)
        if f.kind == "code":
            for lineno, line in enumerate(text.splitlines(), 1):
                for pat in ROUTE_PATTERNS:
                    for m in pat.finditer(line):
                        method, route = m.group(1), m.group(2)
                        method = method.upper().replace("HTTP", "").replace("MAPPING", "") or "ANY"
                        if method in ("ROUTE", "USE", "ALL", "API_ROUTE", "REQUEST"):
                            method = "ANY"
                        prof.routes_defined.append(
                            {"method": method, "path": route, "file": f.path, "line": lineno}
                        )
                for m in CALL_RE.finditer(line):
                    prof.urls_called.append({"url": _clean_url(m.group(2)), "file": f.path, "line": lineno})

    for svc in prof.compose_services:
        env.update(svc["env"])
    prof.languages = dict(langs.most_common())
    prof.frameworks = _frameworks(all_deps)
    prof.env_vars = sorted(env)
    return prof


def profile_summary(p: RepoProfile, max_items: int = 15) -> str:
    """Compact, human-readable profile text for prompts."""

    def cap(xs: list) -> str:
        xs = list(xs)
        return ", ".join(map(str, xs[:max_items])) + (
            f" (+{len(xs) - max_items} more)" if len(xs) > max_items else ""
        )

    lines = [f"repo: {p.repo}"]
    if p.languages:
        lines.append("languages (lines): " + cap(f"{k}={v}" for k, v in p.languages.items()))
    if p.frameworks:
        lines.append("frameworks: " + cap(p.frameworks))
    if p.entrypoints:
        lines.append("entrypoints: " + cap(p.entrypoints))
    for s in p.compose_services[:max_items]:
        lines.append(
            f"compose service {s['name']} ({s['file']}): image={s.get('image')} build={s.get('build')} "
            f"ports={s.get('ports')} env={s.get('env')} depends_on={s.get('depends_on')}"
        )
    for w in p.workflows[:max_items]:
        lines.append(f"workflow {w['file']}: name={w['name']} triggers={w['triggers']} jobs={w['jobs']}")
    if p.terraform_resources:
        lines.append("terraform resources: " + cap(p.terraform_resources))
    if p.env_vars:
        lines.append("env vars: " + cap(p.env_vars))
    if p.routes_defined:
        lines.append(
            "routes defined: "
            + cap(f"{r['method']} {r['path']} ({r['file']}:{r['line']})" for r in p.routes_defined)
        )
    if p.urls_called:
        lines.append("urls called: " + cap(f"{u['url']} ({u['file']}:{u['line']})" for u in p.urls_called))
    return "\n".join(lines)
