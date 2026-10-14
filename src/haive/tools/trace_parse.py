"""Parse error output into ``ParsedTrace`` (pure code, no LLM).

Dedicated parsers: Python (incl. chained exceptions), Node/V8, Java/Kotlin, Go panics, Rust panics
and backtraces, .NET, C/C++ sanitizer/gdb, and compiler/linter lines. Anything else falls back to
a generic ``path:line[:col]`` / ``path(line[,col])`` token scan. Library frames are marked external.
"""

from __future__ import annotations

import re

from haive.types import Frame, ParsedTrace

EXTERNAL_MARKERS = (
    "site-packages/", "dist-packages/", "node_modules/", "/lib/python", "<frozen", "node:internal",
    "internal/", ".cargo/registry", "/rustc/", "/usr/lib/", "/usr/local/lib/", "/usr/include/", "/go/pkg/mod/",
    "/usr/local/go/src/", "/usr/lib/go", "go/src/runtime/", "/library/std/", "/library/core/", "<anonymous>",
    "/System/Library/", "/opt/homebrew/", ".nuget/packages", "/Library/Frameworks/", "sanitizer_common",
    "/libc-", "libc.so", "<string>", "<stdin>",
)  # fmt: skip
EXTERNAL_JAVA_PREFIXES = ("java.", "javax.", "jdk.", "sun.", "kotlin.", "kotlinx.", "scala.", "org.junit.",
                          "org.springframework.", "com.sun.", "org.apache.", "io.netty.", "android.")  # fmt: skip
EXTERNAL_DOTNET_PREFIXES = ("System.", "Microsoft.")

FILE_EXT = (r"(?:py|pyi|js|mjs|cjs|jsx|ts|mts|cts|tsx|go|rs|java|kt|kts|scala|c|h|cc|cpp|cxx|hpp|hh|cs|rb|php|"
            r"swift|ex|exs|dart|lua|sql|sh|vue|svelte|yaml|yml|json|toml|tf)")  # fmt: skip


def is_external(path: str, function: str = "") -> bool:
    p = path.replace("\\", "/")
    if any(m in p for m in EXTERNAL_MARKERS):
        return True
    if p.startswith("node:") or p in ("native", "<unknown>"):
        return True
    return False


def _frame(
    path: str, line: str | int | None, col: str | int | None = None, function: str = "", **kw
) -> Frame:
    path = path.strip().strip("\"'")
    if path.startswith("file://"):
        path = path[7:]
    return Frame(
        path=path,
        line=int(line) if line not in (None, "") else None,
        col=int(col) if col not in (None, "") else None,
        function=(function or "").strip(),
        external=kw.pop("external", None) if "external" in kw else is_external(path, function),
    )


# ---------------------------------------------------------------- python

PY_FRAME = re.compile(r'^\s*File "([^"]+)", line (\d+)(?:, in (.+))?')
PY_EXC = re.compile(r"^([A-Za-z_][\w.]*)(?::\s?(.*))?$")
PY_CHAIN = ("During handling of the above exception", "The above exception was the direct cause")


def parse_python(text: str) -> ParsedTrace | None:
    if "Traceback (most recent call last)" not in text:
        return None
    frames: list[Frame] = []
    error_type = message = ""
    lines = text.splitlines()
    in_tb = False
    for i, line in enumerate(lines):
        if line.startswith("Traceback (most recent call last)") or line.strip().startswith("Traceback (most"):
            in_tb = True
            continue
        m = PY_FRAME.match(line)
        if m:
            frames.append(_frame(m.group(1), m.group(2), None, m.group(3) or ""))
            continue
        if in_tb and line and not line.startswith((" ", "\t")):
            em = PY_EXC.match(line.strip())
            if em and not any(line.startswith(c) for c in PY_CHAIN):
                error_type = em.group(1)
                message = (em.group(2) or "").strip()
                # multi-line messages: following unindented lines until blank/chain marker
                for nxt in lines[i + 1 :]:
                    if (
                        not nxt.strip()
                        or any(nxt.startswith(c) for c in PY_CHAIN)
                        or nxt.startswith("Traceback")
                    ):
                        break
                    message += "\n" + nxt
                in_tb = False
    return ParsedTrace(
        format="python", error_type=error_type, message=message.strip(), frames=frames, raw=text
    )


# ---------------------------------------------------------------- node

NODE_AT = re.compile(
    r"^\s*at (?:(?:async )?(.+?) \()?((?:node:|file://)?(?:[A-Za-z]:)?[^():\s][^():]*?):(\d+):(\d+)\)?\s*$"
)
NODE_ERR = re.compile(r"^(?:Uncaught\s+)?([A-Z]\w*(?:Error|Exception)|Error)(?: \[(\w+)\])?:\s?(.*)$")


def parse_node(text: str) -> ParsedTrace | None:
    lines = text.splitlines()
    frames = []
    for line in lines:
        m = NODE_AT.match(line)
        if m:
            frames.append(_frame(m.group(2), m.group(3), m.group(4), m.group(1) or ""))
    if not frames:
        return None
    error_type = message = ""
    for line in lines:
        m = NODE_ERR.match(line.strip())
        if m:
            error_type = m.group(1) + (f" [{m.group(2)}]" if m.group(2) else "")
            message = m.group(3)
            break
    return ParsedTrace(format="node", error_type=error_type, message=message, frames=frames, raw=text)


# ---------------------------------------------------------------- java / kotlin

JAVA_AT = re.compile(
    r"^\s*at ([\w$.<>/]+)\.([\w$<>]+)\(([\w$-]+\.(?:java|kt|kts|scala|groovy)|Native Method|Unknown Source)(?::(\d+))?\)"
)
JAVA_EXC = re.compile(
    r"^(?:Exception in thread \"[^\"]*\" |Caused by: )?([a-zA-Z_$][\w$]*(?:\.[\w$]+)+)(?::\s?(.*))?$"
)


def parse_java(text: str) -> ParsedTrace | None:
    lines = text.splitlines()
    if not any(JAVA_AT.match(line) for line in lines):
        return None
    frames = []
    error_type = message = ""
    for line in lines:
        s = line.strip()
        m = JAVA_AT.match(line)
        if m:
            cls, meth, file, ln = m.groups()
            pkg_parts = cls.split(".")
            # Map to package path + file name: com.foo.Bar -> com/foo/Bar.java
            pkg = [p for p in pkg_parts[:-1] if p and p[0].islower()]
            path = "/".join([*pkg, file]) if file not in ("Native Method", "Unknown Source") else file
            ext = cls.startswith(EXTERNAL_JAVA_PREFIXES) or file in ("Native Method", "Unknown Source")
            frames.append(_frame(path, ln, None, f"{pkg_parts[-1]}.{meth}", external=ext))
            continue
        em = JAVA_EXC.match(s)
        if em and (s.startswith(("Exception in thread", "Caused by:")) or not error_type):
            error_type, message = em.group(1), (em.group(2) or "").strip()
    return ParsedTrace(format="java", error_type=error_type, message=message, frames=frames, raw=text)


# ---------------------------------------------------------------- go

GO_FUNC = re.compile(r"^([\w./*()\-]+(?:\.[\w*()\-]+)+)\(.*\)$")
GO_LOC = re.compile(r"^\s+(\S+\.go):(\d+)(?: \+0x[0-9a-f]+)?$")


def parse_go(text: str) -> ParsedTrace | None:
    if not re.search(r"^(panic: |fatal error: )", text, re.M) and "goroutine " not in text:
        return None
    lines = text.splitlines()
    frames = []
    error_type, message = "panic", ""
    for line in lines:
        if line.startswith("panic: "):
            message = line[len("panic: ") :].strip()
            if message.startswith("runtime error:"):
                error_type = "runtime error"
                message = message[len("runtime error:") :].strip()
            break
        if line.startswith("fatal error: "):
            error_type, message = "fatal error", line[len("fatal error: ") :].strip()
            break
    pending_fn = ""
    for line in lines:
        fm = GO_FUNC.match(line.strip()) if not line.startswith(("\t", " ")) else None
        if fm:
            pending_fn = fm.group(1)
            continue
        lm = GO_LOC.match(line)
        if lm:
            fn = pending_fn
            ext = is_external(lm.group(1)) or fn.startswith(("runtime.", "testing.", "net/http.", "reflect."))
            frames.append(_frame(lm.group(1), lm.group(2), None, fn, external=ext))
            pending_fn = ""
    if not frames and not message:
        return None
    return ParsedTrace(format="go", error_type=error_type, message=message, frames=frames, raw=text)


# ---------------------------------------------------------------- rust

RUST_PANIC_NEW = re.compile(r"thread '([^']*)' panicked at ([^\s:]+\.rs):(\d+):(\d+):?\s*$")
RUST_PANIC_OLD = re.compile(r"thread '([^']*)' panicked at '(.*)', ([^\s:]+\.rs):(\d+):(\d+)")
RUST_BT_FN = re.compile(r"^\s*\d+:\s+(?:0x[0-9a-f]+ - )?(.+)$")
RUST_BT_AT = re.compile(r"^\s+at (.+\.rs):(\d+)(?::(\d+))?$")


def parse_rust(text: str) -> ParsedTrace | None:
    lines = text.splitlines()
    frames: list[Frame] = []
    message = ""
    found = False
    for i, line in enumerate(lines):
        m = RUST_PANIC_NEW.search(line)
        if m:
            found = True
            frames.append(_frame(m.group(2), m.group(3), m.group(4), "panic site"))
            message = lines[i + 1].strip() if i + 1 < len(lines) else ""
            break
        m = RUST_PANIC_OLD.search(line)
        if m:
            found = True
            message = m.group(2)
            frames.append(_frame(m.group(3), m.group(4), m.group(5), "panic site"))
            break
    if not found:
        return None
    fn = ""
    for line in lines:
        am = RUST_BT_AT.match(line)
        if am:
            ext = is_external(am.group(1)) or fn.startswith(
                ("std::", "core::", "alloc::", "rust_begin_unwind")
            )
            frames.append(_frame(am.group(1), am.group(2), am.group(3), fn, external=ext))
            continue
        fm = RUST_BT_FN.match(line)
        if fm and "stack backtrace" not in line:
            fn = re.sub(r"::h[0-9a-f]{16}$", "", fm.group(1).strip())
    return ParsedTrace(format="rust", error_type="panic", message=message, frames=frames, raw=text)


# ---------------------------------------------------------------- .NET

DOTNET_AT = re.compile(r"^\s*at ([\w.`<>\[\]+]+)\(([^)]*)\)(?: in (.+?):line (\d+))?\s*$")
DOTNET_EXC = re.compile(r"^(?:Unhandled exception\.\s*)?((?:[A-Z]\w*\.)+\w*Exception)(?::\s?(.*))?$")


def parse_dotnet(text: str) -> ParsedTrace | None:
    lines = text.splitlines()
    if not any((m := DOTNET_AT.match(line)) and m.group(3) for line in lines) and not any(
        DOTNET_EXC.match(line.strip()) for line in lines
    ):
        return None
    frames = []
    error_type = message = ""
    for line in lines:
        m = DOTNET_AT.match(line)
        if m:
            fn, _args, path, ln = m.groups()
            if path:
                frames.append(_frame(path, ln, None, fn, external=fn.startswith(EXTERNAL_DOTNET_PREFIXES)))
            else:
                frames.append(Frame(path="", function=fn, external=True))
            continue
        em = DOTNET_EXC.match(line.strip().split(" ---> ")[-1])
        if em and not error_type:
            error_type, message = em.group(1), (em.group(2) or "").strip()
    return ParsedTrace(format="dotnet", error_type=error_type, message=message, frames=frames, raw=text)


# ---------------------------------------------------------------- native (sanitizers, gdb)

SAN_FRAME = re.compile(
    r"^\s*#(\d+)\s+0x[0-9a-fA-F]+\s+in\s+(\S+)\s+(?:\([^)]*\)\s+)?(\S+?):(\d+)(?::(\d+))?\s*$"
)
GDB_FRAME = re.compile(r"^#(\d+)\s+(?:0x[0-9a-fA-F]+\s+in\s+)?(\S+)\s*\(.*\)\s+at\s+(\S+?):(\d+)\s*$")
SAN_ERR = re.compile(r"ERROR: (\w+Sanitizer): ([\w-]+)(.*)$", re.M)


def parse_native(text: str) -> ParsedTrace | None:
    frames = []
    for line in text.splitlines():
        m = SAN_FRAME.match(line)
        if m:
            frames.append(_frame(m.group(3), m.group(4), m.group(5), m.group(2)))
            continue
        g = GDB_FRAME.match(line)
        if g:
            frames.append(_frame(g.group(3), g.group(4), None, g.group(2)))
    if not frames:
        return None
    error_type = message = ""
    em = SAN_ERR.search(text)
    if em:
        error_type, message = em.group(1), (em.group(2) + em.group(3)).strip()
    else:
        sig = re.search(r"(Program received signal|signal) (SIG\w+)[,:]?\s*(.*)", text)
        if sig:
            error_type, message = sig.group(2), sig.group(3).strip()
    return ParsedTrace(format="native", error_type=error_type, message=message, frames=frames, raw=text)


# ---------------------------------------------------------------- compiler / linter

COMPILER_COLON = re.compile(
    rf"^\s*(?:-->\s*)?((?:[A-Za-z]:)?[^\s:()]+\.{FILE_EXT}):(\d+)(?::(\d+))?(?::\s*|\s+)"
    r"((?:fatal )?error|warning|note)?(?:\[?([A-Z]+\d+)\]?)?:?\s*(.*)$"
)
COMPILER_PAREN = re.compile(
    rf"^\s*((?:[A-Za-z]:)?[^\s:()]+\.{FILE_EXT})\((\d+)(?:,(\d+))?\):\s*(error|warning)\s+([A-Z]+\d+)?:?\s*(.*)$"
)
RUSTC_ERR = re.compile(r"^(error|warning)(?:\[(E\d+)\])?: (.+)$")
RUSTC_LOC = re.compile(r"^\s*--> (.+?):(\d+):(\d+)$")


def parse_compiler(text: str) -> ParsedTrace | None:
    lines = text.splitlines()
    frames: list[Frame] = []
    error_type = message = ""
    pending: tuple[str, str] | None = None
    for line in lines:
        rm = RUSTC_ERR.match(line)
        if rm:
            pending = (rm.group(2) or rm.group(1), rm.group(3))
            continue
        lm = RUSTC_LOC.match(line)
        if lm:
            frames.append(_frame(lm.group(1), lm.group(2), lm.group(3), ""))
            if pending and not error_type:
                error_type, message = pending
            pending = None
            continue
        m = COMPILER_PAREN.match(line)
        if m:
            frames.append(_frame(m.group(1), m.group(2), m.group(3), ""))
            if not error_type or (error_type.startswith("warning") and m.group(4) == "error"):
                error_type = m.group(5) or m.group(4)
                message = m.group(6)
            continue
        m = COMPILER_COLON.match(line)
        if m:
            frames.append(_frame(m.group(1), m.group(2), m.group(3), ""))
            sev, code, msg = m.group(4), m.group(5), m.group(6)
            if not error_type or (sev and "error" in sev and not error_type.endswith("error")):
                error_type = code or sev or "error"
                message = msg
    if not frames:
        return None
    return ParsedTrace(
        format="compiler", error_type=error_type, message=message.strip(), frames=frames, raw=text
    )


# ---------------------------------------------------------------- generic fallback

GENERIC = re.compile(
    rf"((?:[A-Za-z]:)?[\w./\\@~+-]*\w\.{FILE_EXT})(?::(\d+)(?::(\d+))?|\((\d+)(?:,\s*(\d+))?\)|, line (\d+))"
)


def parse_generic(text: str) -> ParsedTrace:
    frames = []
    seen = set()
    for m in GENERIC.finditer(text):
        path = m.group(1)
        line = m.group(2) or m.group(4) or m.group(6)
        col = m.group(3) or m.group(5)
        key = (path, line)
        if key in seen:
            continue
        seen.add(key)
        frames.append(_frame(path, line, col, ""))
    error_type = message = ""
    for line in text.splitlines():
        m = re.match(r"^\s*(?:\w+:\s*)?([A-Z]\w*(?:Error|Exception|Failure|Fault))[:\s]\s*(.*)$", line)
        if m:
            error_type, message = m.group(1), m.group(2)
            break
    if not message:
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        message = first[:300]
    return ParsedTrace(format="generic" if frames else "unknown", error_type=error_type, message=message,
                       frames=frames, raw=text)  # fmt: skip


PARSERS = [
    parse_python,
    parse_java,
    parse_dotnet,
    parse_go,
    parse_rust,
    parse_native,
    parse_node,
    parse_compiler,
]


def parse(text: str) -> ParsedTrace:
    text = text.replace("\r\n", "\n").strip("\n")
    for p in PARSERS:
        try:
            res = p(text)
        except Exception:
            res = None
        if res and (res.frames or res.error_type):
            return res
    return parse_generic(text)


def looks_like_trace(text: str) -> bool:
    """Heuristic for chat: is this pasted text an error trace / compiler output?"""
    if text.count("\n") < 1:
        return False
    signals = [
        "Traceback (most recent call last)", "panic: ", "panicked at", "Exception in thread", "Caused by:",
        "Unhandled exception", "ERROR: AddressSanitizer", "goroutine ", "stack backtrace:",
    ]  # fmt: skip
    if any(s in text for s in signals):
        return True
    p = parse(text)
    return p.format not in ("unknown", "generic") or len(p.frames) >= 2


def message_fragment(message: str, min_len: int = 12) -> str | None:
    """Longest literal fragment of an error message, with quoted values and numbers stripped."""
    s = re.sub(r"(['\"`]).*?\1", "\0", message)
    s = re.sub(r"\b0x[0-9a-fA-F]+\b|\b\d+(?:\.\d+)?\b", "\0", s)
    s = re.sub(r"[\[\](){}<>]", "\0", s)
    parts = [p.strip(" :,;.-") for p in s.split("\0")]
    parts = [p for p in parts if len(p) >= min_len]
    return max(parts, key=len) if parts else None
