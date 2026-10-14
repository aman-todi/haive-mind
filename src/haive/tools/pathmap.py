"""Map paths from traces to indexed repo files.

Longest matching path suffix (by path components) against indexed files in scope. Ties go to the
in-scope repo, then to the most recently modified file. Java frames arrive as package path + file
name, which is just a longer suffix. TypeScript traces pointing at compiled ``dist/*.js`` are
matched to sources by file stem and, when given, by the function name appearing in the source.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path, PurePosixPath

from haive.types import Frame

SOURCE_EXT_FOR = {
    ".js": (".ts", ".tsx", ".mts"),
    ".mjs": (".mts", ".ts"),
    ".cjs": (".cts", ".ts"),
    ".jsx": (".tsx",),
}


def _parts(path: str) -> list[str]:
    p = path.replace("\\", "/")
    if len(p) > 1 and p[1] == ":":  # windows drive
        p = p[2:]
    return [x for x in p.split("/") if x not in ("", ".")]


class PathMapper:
    def __init__(self, files: list[tuple[str, str]], roots: dict[str, Path], scope: list[str] | None = None):
        """``files``: (repo, repo-relative path) for every indexed file that may be matched."""
        self.roots = roots
        self.scope = set(scope or roots)
        self.by_name: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.by_stem: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for repo, path in files:
            name = PurePosixPath(path).name
            self.by_name[name].append((repo, path))
            self.by_stem[PurePosixPath(path).stem].append((repo, path))

    def _mtime(self, repo: str, path: str) -> float:
        try:
            return (self.roots[repo] / path).stat().st_mtime
        except (OSError, KeyError):
            return 0.0

    def _rank(self, cands: list[tuple[str, str]], score) -> tuple[str, str] | None:
        if not cands:
            return None
        return max(cands, key=lambda c: (score(c), c[0] in self.scope, self._mtime(*c)))

    def map_path(self, path: str, function: str = "") -> tuple[str, str] | None:
        parts = _parts(path)
        if not parts:
            return None
        name = parts[-1]
        cands = self.by_name.get(name, [])

        def suffix_len(c: tuple[str, str]) -> int:
            cp = _parts(c[1])
            n = 0
            while n < min(len(cp), len(parts)) and cp[-1 - n] == parts[-1 - n]:
                n += 1
            return n

        best = self._rank(cands, suffix_len)
        if best:
            return best
        # Compiled JS -> TS sources by stem (+ function name if it appears in the file).
        suffix = PurePosixPath(name).suffix
        if suffix in SOURCE_EXT_FOR:
            stem = PurePosixPath(name).stem
            srcs = [
                c for c in self.by_stem.get(stem, []) if PurePosixPath(c[1]).suffix in SOURCE_EXT_FOR[suffix]
            ]

            def fn_score(c: tuple[str, str]) -> int:
                if not function:
                    return 0
                fn = function.split(".")[-1].replace("async ", "").strip()
                try:
                    return int(fn in (self.roots[c[0]] / c[1]).read_text(errors="replace"))
                except (OSError, KeyError):
                    return 0

            return self._rank(srcs, fn_score)
        return None

    def map_frames(self, frames: list[Frame]) -> list[Frame]:
        out = []
        for f in frames:
            f = f.model_copy()
            if f.path:
                hit = self.map_path(f.path, f.function)
                if hit:
                    f.repo, f.repo_path = hit
                    # A mapped frame inside a vendored dir stays external.
                elif not f.external:
                    f.external = True  # not in any indexed repo: treat as library/system code
            out.append(f)
        return out


OUTERMOST_FIRST = {"python"}


def call_boundary(frames: list[Frame], fmt: str = "") -> list[Frame]:
    """Internal (mapped) frames ordered innermost first, with the call-boundary frame (the last
    internal frame before execution entered external code) moved to the front."""
    ordered = list(frames) if fmt in OUTERMOST_FIRST else list(reversed(frames))  # outermost -> innermost
    boundary = None
    for i, f in enumerate(ordered[:-1]):
        if not f.external and f.repo and ordered[i + 1].external:
            boundary = f
    internal = [f for f in reversed(ordered) if not f.external and f.repo]
    if boundary is not None:
        internal.remove(boundary)
        internal.insert(0, boundary)
    return internal
