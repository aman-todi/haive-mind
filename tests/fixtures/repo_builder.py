"""Builds a tiny multi-language git repo with history (used by tests and the eval sample)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

FILES_V1: dict[str, str] = {
    "README.md": """# Valuation app

Small demo service that values companies.

## Architecture

The `api` service (FastAPI) serves valuations. The `web` frontend calls it.

## Deploy

CI builds images and `infra/main.tf` provisions the bucket.
""",
    "pyproject.toml": """[project]
name = "valuation"
version = "0.1.0"
dependencies = ["fastapi>=0.100", "requests>=2"]

[project.scripts]
valuation = "app.server:main"
""",
    "app/__init__.py": "",
    "app/api/__init__.py": "",
    "app/api/client.py": """import os

import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000")


class ApiClient:
    \"\"\"HTTP client for the valuation API.\"\"\"

    def __init__(self, base_url: str = API_URL):
        self.base_url = base_url

    def fetch_user(self, user_id: int) -> dict:
        resp = requests.get(f"{self.base_url}/users/{user_id}")
        resp.raise_for_status()
        return resp.json()
""",
    "app/valuation/__init__.py": "",
    "app/valuation/dcf.py": """def choose_method(company: dict) -> str:
    \"\"\"Pick a valuation method for the company.\"\"\"
    if company.get("dividends"):
        return "ddm"
    if company["fcf"] > 0:
        return "fcf"
    return "multiples"


def discount(cashflows: list[float], rate: float) -> float:
    total = 0.0
    for i, cf in enumerate(cashflows, start=1):
        total += cf / (1 + rate) ** i
    return total
""",
    "app/server.py": """from fastapi import FastAPI

from app.valuation.dcf import choose_method

app = FastAPI()


@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {"id": user_id}


@app.post("/valuations")
def create_valuation(company: dict):
    return {"method": choose_method(company)}


def main():
    import uvicorn

    uvicorn.run(app, port=8000)
""",
    "web/package.json": """{
  "name": "web",
  "scripts": {"dev": "vite", "build": "tsc && vite build"},
  "dependencies": {"react": "^18.0.0", "axios": "^1.0.0"},
  "devDependencies": {"vite": "^5.0.0", "typescript": "^5.0.0"}
}
""",
    "web/package-lock.json": '{"lockfileVersion": 3}\n',
    "web/src/api.ts": """export async function loadUser(id: number) {
  const res = await fetch(`${process.env.API_URL}/users/${id}`);
  return res.json();
}

export class Cache {
  private store = new Map<string, unknown>();
  get(key: string) {
    return this.store.get(key);
  }
}
""",
    "cmd/worker/main.go": """package main

import "fmt"

type Job struct {
\tID int
}

func (j *Job) Run() error {
\tfmt.Println("running", j.ID)
\treturn nil
}

func main() {
\tj := &Job{ID: 1}
\tj.Run()
}
""",
    "src/main/java/com/acme/Billing.java": """package com.acme;

public class Billing {
    public int total(int[] items) {
        int sum = 0;
        for (int i = 0; i < items.length; i++) {
            sum += items[i];
        }
        return sum;
    }
}
""",
    "docker-compose.yml": """services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgres://db/app
  web:
    image: node:20
    ports:
      - "3000:3000"
    environment:
      API_URL: http://api:8000
    depends_on:
      - api
""",
    ".github/workflows/ci.yml": """name: CI
on:
  push:
    branches: [main]
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
""",
    "infra/main.tf": """resource "aws_s3_bucket" "assets" {
  bucket = "valuation-assets"
}

variable "region" {
  default = "us-east-1"
}
""",
    ".env": "OPENROUTER_API_KEY=sk-or-v1-0123456789abcdefghijklmnop\n",
    ".env.example": "API_URL=http://localhost:8000\nDATABASE_URL=\n",
    "db/schema.sql": """CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  name TEXT
);

CREATE INDEX users_name ON users(name);
""",
    "scripts/deploy.sh": """#!/bin/bash
build() {
  docker build -t app .
}

push() {
  docker push app
}
""",
    "assets/logo.png": "\x89PNG\r\n\x1a\n\x00\x00binary",
    ".gitignore": "ignored/\n*.log\n.env\n",
    "ignored/skip.py": "print('ignored')\n",
}

CLIENT_V2 = """import os
import time

import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000")


class ApiClient:
    \"\"\"HTTP client for the valuation API.\"\"\"

    def __init__(self, base_url: str = API_URL, retries: int = 3):
        self.base_url = base_url
        self.retries = retries

    def fetch_user(self, user_id: int) -> dict:
        for attempt in range(self.retries):
            resp = requests.get(f"{self.base_url}/users/{user_id}")
            if resp.status_code == 200:
                return resp.json()
            time.sleep(2 ** attempt)
        raise RuntimeError("user fetch failed after all retries")
"""


def _git(root: Path, *args: str, date: str | None = None) -> str:
    env = dict(os.environ)
    env.update(
        GIT_AUTHOR_NAME="Dev",
        GIT_AUTHOR_EMAIL="dev@example.com",
        GIT_COMMITTER_NAME="Dev",
        GIT_COMMITTER_EMAIL="dev@example.com",
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_CONFIG_NOSYSTEM="1",
    )
    if date:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = date
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, env=env, text=True
    ).stdout


def write(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".png"):
            p.write_bytes(content.encode("latin-1"))
        else:
            p.write_text(content)


def commit(root: Path, message: str, date: str) -> str:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message, date=date)
    return _git(root, "rev-parse", "HEAD").strip()


def build_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q", "-b", "main")
    files = dict(FILES_V1)
    secret = files.pop(".env")
    write(root, files)
    commit(root, "Initial valuation app", "2024-01-01T10:00:00+00:00")
    write(root, {"app/api/client.py": CLIENT_V2})
    commit(
        root,
        "Add retries to the API client\n\nThe upstream API is flaky; retry with backoff.",
        "2024-02-01T10:00:00+00:00",
    )
    write(root, {"web/src/api.ts": FILES_V1["web/src/api.ts"] + "\nexport const VERSION = 2;\n"})
    commit(root, "Bump web api version", "2024-03-01T10:00:00+00:00")
    # untracked-but-ignored and untracked secret files
    (root / ".env").write_text(secret)
    (root / "debug.log").write_text("noise\n")
    return root


def build_small_repo(root: Path, name: str = "other") -> Path:
    """A second repo for group/compare tests: calls the first repo's /valuations route."""
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q", "-b", "main")
    write(
        root,
        {
            "README.md": f"# {name}\n\nFrontend that requests valuations.\n",
            "src/valuations.js": """const axios = require("axios");

async function requestValuation(company) {
  const res = await axios.post(`${process.env.API_URL}/valuations`, company);
  return res.data;
}

module.exports = { requestValuation };
""",
            "docker-compose.yml": "services:\n  frontend:\n    image: node:20\n    environment:\n      API_URL: http://api:8000\n",
        },
    )
    commit(root, "Initial frontend", "2024-01-05T10:00:00+00:00")
    return root
