from pathlib import Path

import pytest

from haive.tools.trace_parse import looks_like_trace, message_fragment, parse

T = Path(__file__).parent.parent / "fixtures" / "traces"


def load(name):
    return parse((T / name).read_text())


def test_python_chained():
    p = load("python_chained.txt")
    assert p.format == "python"
    assert p.error_type == "RuntimeError"
    assert "valuation failed" in p.message
    assert [(f.path.split("/")[-1], f.line, f.function) for f in p.frames] == [
        ("dcf.py", 5, "choose_method"),
        ("server.py", 15, "create_valuation"),
        ("routing.py", 74, "app"),
    ]
    assert [f.external for f in p.frames] == [False, False, True]


def test_node():
    p = load("node.txt")
    assert p.format == "node" and p.error_type == "TypeError"
    assert p.message.startswith("Cannot read properties of undefined")
    f = p.frames[0]
    assert (f.path, f.line, f.col, f.function, f.external) == (
        "/app/web/src/api.ts",
        3,
        14,
        "loadUser",
        False,
    )
    assert p.frames[1].external and p.frames[2].external


def test_java_caused_by_is_root():
    p = load("java.txt")
    assert p.format == "java"
    assert p.error_type == "java.lang.ArrayIndexOutOfBoundsException"
    assert p.frames[0].path == "com/acme/Billing.java" and p.frames[0].line == 7
    assert p.frames[0].function == "Billing.total"
    assert any(f.external and "ArrayList" in f.path for f in p.frames)


def test_kotlin():
    p = load("kotlin.txt")
    assert p.error_type == "kotlin.KotlinNullPointerException"
    assert p.frames[0].path == "com/acme/app/Service.kt" and not p.frames[0].external
    assert p.frames[1].external


def test_go_panic():
    p = load("go.txt")
    assert p.format == "go" and p.error_type == "runtime error"
    assert p.message.startswith("index out of range")
    assert [(f.function, f.line, f.external) for f in p.frames] == [
        ("main.(*Job).Run", 10, False),
        ("main.main", 16, False),
        ("runtime.main", 250, True),
    ]


def test_rust_panic_and_backtrace():
    p = load("rust.txt")
    assert p.format == "rust" and p.error_type == "panic"
    assert p.message.startswith("index out of bounds")
    internal = [(f.path, f.line, f.function) for f in p.frames if not f.external]
    assert internal[0] == ("src/main.rs", 5, "panic site")
    assert ("./src/main.rs", 5, "myapp::compute") in internal
    old = load("rust_old.txt")
    assert old.message.startswith("called `Option::unwrap()`") and old.frames[0].path == "src/lib.rs"


def test_dotnet():
    p = load("dotnet.txt")
    assert p.format == "dotnet" and p.error_type == "System.NullReferenceException"
    assert (p.frames[0].path, p.frames[0].line, p.frames[0].function) == (
        "/src/App/Services/Svc.cs",
        12,
        "App.Services.Svc.Run",
    )
    assert p.frames[-1].external


def test_asan_and_gdb():
    p = load("asan.txt")
    assert p.format == "native" and p.error_type == "AddressSanitizer"
    assert p.message.startswith("heap-buffer-overflow")
    assert (p.frames[0].path, p.frames[0].line, p.frames[0].col, p.frames[0].function) == (
        "/home/dev/proj/src/items.c",
        14,
        12,
        "sum_items",
    )
    g = load("gdb.txt")
    assert g.error_type == "SIGSEGV" and g.frames[0].path == "src/parser.c" and g.frames[0].line == 22


@pytest.mark.parametrize(
    "name,etype,path,line,col",
    [
        ("gcc.txt", "error", "src/items.c", 14, 12),
        ("tsc.txt", "TS2339", "web/src/api.ts", 3, 14),
        ("rustc.txt", "E0308", "src/main.rs", 4, 18),
        ("msbuild.txt", "CS0103", "Program.cs", 10, 5),
        ("mypy.txt", "error", "app/valuation/dcf.py", 5, None),
    ],
)
def test_compiler_lines(name, etype, path, line, col):
    p = load(name)
    assert p.format == "compiler"
    assert p.error_type == etype
    assert (p.frames[0].path, p.frames[0].line, p.frames[0].col) == (path, line, col)


def test_generic_fallback():
    p = load("generic.txt")
    assert p.format == "generic"
    assert [(f.path, f.line, f.col) for f in p.frames] == [
        ("app/api/client.py", 18, None),
        ("web/src/api.ts", 3, 14),
    ]


def test_unknown():
    p = parse("something broke")
    assert p.format == "unknown" and not p.frames and p.message == "something broke"


def test_looks_like_trace():
    assert looks_like_trace((T / "python_chained.txt").read_text())
    assert looks_like_trace((T / "go.txt").read_text())
    assert not looks_like_trace("how does auth work?")
    assert not looks_like_trace("a short\nmultiline question")


def test_message_fragment():
    assert (
        message_fragment("valuation failed for company 'acme' with code 42") == "valuation failed for company"
    )
    assert message_fragment("Index 3 out of bounds for length 3") == "out of bounds for length"
    assert message_fragment("'x'") is None
