import pytest

import haive.index.chunk_code as cc
from haive.index.chunk_code import chunk_code, get_parser

needs_grammars = pytest.mark.skipif(get_parser("python") is None, reason="tree-sitter grammars unavailable")

PY = '''import os
X = 1

@dec
def foo(a):
    return a

class Big:
    """doc"""
    y = 2

    def m1(self):
        return 1

    # second
    def m2(self):
        return 2
'''


def syms(chunks):
    return [(c.symbol, c.start_line, c.end_line) for c in chunks]


@needs_grammars
def test_python_functions_classes_module():
    out = syms(chunk_code("r", "a.py", PY, "python"))
    assert ("<module>", 1, 2) in out
    assert ("foo", 4, 6) in out
    assert ("Big", 8, 17) in out


@needs_grammars
def test_large_class_split_into_methods(monkeypatch):
    monkeypatch.setattr(cc, "MAX_TOKENS", 12)
    chunks = chunk_code("r", "a.py", PY, "python")
    by = {c.symbol: c for c in chunks}
    assert "Big.m1" in by and "Big.m2" in by
    assert by["Big.m2"].start_line == 15  # includes the attached comment
    assert by["Big.m1"].context == "class Big:"
    assert "class Big:" in by["Big.m1"].embed_text
    assert "class Big:" not in by["Big.m1"].text


@needs_grammars
@pytest.mark.parametrize(
    "lang,path,src,expected",
    [
        (
            "typescript",
            "a.ts",
            "import x from 'y';\nexport function foo(a: number) { return a }\n"
            "export const bar = (b) => b * 2;\nexport class C { m() { return 1 } }\n",
            {"foo", "bar", "C", "<module>"},
        ),
        ("javascript", "a.js", "function a() {}\nconst b = function() {};\nclass K {}\n", {"a", "b", "K"}),
        (
            "go",
            "m.go",
            "package main\n\nfunc (s *S) Do() error { return nil }\n\nfunc main() {}\n",
            {"S.Do", "main", "<module>"},
        ),
        (
            "rust",
            "m.rs",
            "use std::io;\nstruct P { x: i32 }\nimpl P { fn new() -> Self { P { x: 1 } } }\nfn main() {}\n",
            {"P", "impl P", "main", "<module>"},
        ),
        (
            "java",
            "F.java",
            "package a;\npublic class Foo {\n  int bar() { return 1; }\n}\n",
            {"Foo", "<module>"},
        ),
        (
            "c",
            "a.c",
            "#include <stdio.h>\n\nstatic int add(int a, int b) { return a+b; }\n",
            {"add", "<module>"},
        ),
        (
            "cpp",
            "a.cpp",
            "namespace n {\nint f(int x) { return x; }\nclass K { void g() {} };\n}\n",
            {"f", "K"},
        ),
        ("csharp", "a.cs", "namespace App {\n  public class Svc { public void Run() {} }\n}\n", {"Svc"}),
    ],
)
def test_languages(lang, path, src, expected):
    got = {c.symbol for c in chunk_code("r", path, src, lang)}
    assert expected <= got, got


def test_unknown_language_falls_back_to_windows():
    text = "\n".join(f"line {i}" for i in range(130))
    chunks = chunk_code("r", "a.zig", text, "zig")
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 60), (51, 110), (101, 130)]


@needs_grammars
def test_huge_function_split_into_windows(monkeypatch):
    monkeypatch.setattr(cc, "MAX_TOKENS", 50)
    body = "\n".join(f"    x{i} = {i} + {i}" for i in range(200))
    chunks = chunk_code("r", "a.py", f"def big():\n{body}\n", "python")
    assert len(chunks) > 3
    assert all(c.symbol == "big" for c in chunks)


def test_chunk_ids_stable():
    a = chunk_code("r", "a.py", PY, "python")
    b = chunk_code("r", "a.py", PY, "python")
    assert [c.id for c in a] == [c.id for c in b]
    assert all(c.id.startswith("r:a.py:") for c in a)
