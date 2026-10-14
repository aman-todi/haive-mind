from haive.index.chunk_text import chunk_file, chunk_text


def syms(chunks):
    return [c.symbol for c in chunks]


def test_markdown_by_heading():
    md = "intro\n\n# Title\ntext\n## Sub\nmore\n```\n# not a heading\n```\n# Other\nx\n"
    out = chunk_text("r", "README.md", md, "markdown", "doc")
    assert syms(out) == ["<preamble>", "Title", "Title > Sub", "Other"]


def test_yaml_top_level_keys():
    y = "name: CI\non:\n  push:\njobs:\n  test:\n    runs-on: x\n"
    assert syms(chunk_text("r", "ci.yml", y, "yaml", "config")) == ["name", "on", "jobs"]


def test_compose_per_service():
    y = "version: '3'\nservices:\n  api:\n    image: a\n  web:\n    image: b\nvolumes:\n  data:\n"
    assert syms(chunk_text("r", "docker-compose.yml", y, "compose", "config")) == [
        "version",
        "services",
        "services.api",
        "services.web",
        "volumes",
    ]


def test_json_top_level_keys():
    j = '{\n  "name": "web",\n  "scripts": {\n    "dev": "vite"\n  },\n  "deps": {}\n}\n'
    assert syms(chunk_text("r", "package.json", j, "json", "config")) == [
        "<preamble>",
        "name",
        "scripts",
        "deps",
    ]


def test_toml_tables():
    t = 'a = 1\n[project]\nname = "x"\n[[tool.items]]\nk = 2\n'
    assert syms(chunk_text("r", "p.toml", t, "toml", "config")) == ["<preamble>", "project", "tool.items"]


def test_hcl_blocks():
    t = 'resource "aws_s3_bucket" "assets" {\n  bucket = "x"\n}\n\nvariable "region" {\n}\n'
    assert syms(chunk_text("r", "main.tf", t, "hcl", "config")) == [
        "resource.aws_s3_bucket.assets",
        "variable.region",
    ]


def test_sql_statements():
    s = "CREATE TABLE a (\n id int\n);\n\nCREATE INDEX i ON a(id);\n"
    assert syms(chunk_text("r", "s.sql", s, "sql", "code")) == ["create table a", "create index i"]


def test_shell_functions_and_dockerfile():
    sh = "#!/bin/bash\nbuild() {\n  x\n}\nfunction push {\n  y\n}\n"
    assert syms(chunk_text("r", "d.sh", sh, "bash", "code")) == ["<preamble>", "build", "push"]
    df = "FROM python:3.12\nRUN pip install x\n"
    out = chunk_text("r", "Dockerfile", df, "dockerfile", "config")
    assert len(out) == 1 and out[0].end_line == 2


def test_fallback_windows():
    text = "\n".join(str(i) for i in range(70))
    out = chunk_text("r", "notes.ini", text, "ini", "config")
    assert [(c.start_line, c.end_line) for c in out] == [(1, 60), (51, 70)]


def test_chunk_file_drops_secret_chunks():
    md = "# Setup\nuse key sk-abcdefghijklmnopqrstuvwxyz123\n# Other\nfine\n"
    out = chunk_file("r", "README.md", md, "markdown", "doc")
    assert syms(out) == ["Other"]


def test_embed_text_has_context_header():
    out = chunk_file("r", "README.md", "# Deploy\nsteps\n", "markdown", "doc")
    assert out[0].embed_text.startswith("r/README.md :: Deploy\n")
    assert out[0].text == "# Deploy\nsteps"
