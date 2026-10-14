from haive.tools.pathmap import PathMapper, call_boundary
from haive.tools.trace_parse import parse
from haive.types import Frame


def mapper(roots, scope=None):
    files = [
        ("v", "app/valuation/dcf.py"), ("v", "app/server.py"), ("v", "web/src/api.ts"), ("v", "web/src/index.ts"),
        ("v", "src/main/java/com/acme/Billing.java"), ("o", "app/server.py"), ("v", "cmd/worker/main.go"),
    ]  # fmt: skip
    return PathMapper(files, roots, scope)


def test_longest_suffix(tmp_path):
    m = mapper({"v": tmp_path, "o": tmp_path})
    assert m.map_path("/home/dev/valuation/app/valuation/dcf.py") == ("v", "app/valuation/dcf.py")
    assert m.map_path("com/acme/Billing.java") == ("v", "src/main/java/com/acme/Billing.java")
    assert m.map_path("C:\\work\\web\\src\\api.ts") == ("v", "web/src/api.ts")
    assert m.map_path("/x/unknown.py") is None


def test_tie_prefers_scope(tmp_path):
    assert mapper({"v": tmp_path, "o": tmp_path}, scope=["o"]).map_path("/srv/app/server.py") == (
        "o",
        "app/server.py",
    )
    assert mapper({"v": tmp_path, "o": tmp_path}, scope=["v"]).map_path("/srv/app/server.py") == (
        "v",
        "app/server.py",
    )


def test_dist_js_maps_to_ts_by_stem_and_function(tmp_path):
    (tmp_path / "web/src").mkdir(parents=True)
    (tmp_path / "web/src/api.ts").write_text("export function loadUser() {}\n")
    (tmp_path / "web/src/index.ts").write_text("main()\n")
    m = mapper({"v": tmp_path})
    assert m.map_path("/srv/web/dist/api.js", "loadUser") == ("v", "web/src/api.ts")


def test_map_frames_marks_unmapped_external(tmp_path):
    m = mapper({"v": tmp_path})
    frames = m.map_frames([Frame(path="/a/app/server.py", line=3), Frame(path="/elsewhere/lib.py", line=1)])
    assert frames[0].repo == "v" and frames[0].repo_path == "app/server.py" and not frames[0].external
    assert frames[1].external and frames[1].repo is None


def test_call_boundary_orders(tmp_path):
    m = mapper({"v": tmp_path})
    py = parse(
        'Traceback (most recent call last):\n  File "/a/app/server.py", line 15, in create\n    x\n'
        '  File "/a/site-packages/lib.py", line 2, in f\n    y\nValueError: bad\n'
    )
    internal = call_boundary(m.map_frames(py.frames), py.format)
    assert internal[0].repo_path == "app/server.py"
    go = parse(
        "panic: boom\n\ngoroutine 1 [running]:\nmain.(*Job).Run()\n\t/x/cmd/worker/main.go:10 +0x1\n"
        "main.main()\n\t/x/cmd/worker/main.go:16 +0x1\n"
    )
    internal = call_boundary(m.map_frames(go.frames), go.format)
    assert [f.line for f in internal] == [10, 16]  # innermost first
