# Decisions

Choices made where the spec was ambiguous or a dependency had moved. Newest last.

1. **Worked alone, in wave order.** No parallel sessions were used; tickets were built T0 → T14
   sequentially, with the shared contracts in `types.py` written first.

2. **Embeddings and stores use their libraries directly** (`fastembed`, `pinecone`, `chromadb`) instead of
   `langchain-community` / `langchain-pinecone` / `langchain-chroma`. `langchain-community` now warns that
   it is being sunset, and the LangChain store wrappers embed the stored text, while the spec wants the
   location header in the embedding only and the raw chunk as stored text. The Chroma adapter is still one
   file, and the store interface is the spec's `upsert/delete/search/count`.

3. **Tree-sitter: per-language grammar wheels, `tree-sitter>=0.25,<0.26`.** `tree-sitter-language-pack`
   1.x downloads grammars at runtime, and the combination with `tree-sitter` 0.26.0 segfaulted (heap
   corruption) when chunking real-sized files. It reproduced with the official `tree-sitter-python` wheel
   too, and went away on 0.25.2. The per-language wheels (`tree-sitter-python`, `-javascript`,
   `-typescript`, `-go`, `-rust`, `-java`, `-c`, `-cpp`, `-c-sharp`) also need no network at runtime. A
   stress test chunks this repo's sources in a subprocess to catch regressions. Revisit the pin when a
   0.26.x fix ships. Parsers are created per call (they aren't thread-safe); languages are cached.

4. **`git log -L :funcname` is unreliable** for indented functions (methods) without diff attributes, so
   history retrieval uses line ranges from the symbol index (`-L start,end:path`).

5. **No porcelain `git diff`.** It refreshes stale stat info and rewrites `.git/index` (found by the
   read-only test on a copied repo). Changes are computed with `git diff-tree` (commit to commit) plus
   `git status` under `GIT_OPTIONAL_LOCKS=0`. `diff` isn't on the git wrapper's allowlist at all.

6. **Graph state is plain JSON-friendly data.** Pydantic models are stored via `model_dump()`, so the
   SQLite checkpointer can persist chat threads. Per-turn accumulators use reducers where `None` means
   "reset", passed in each turn's input.

7. **Additions to the layout:** `runtime.py` (wires config, DB, store, LLM, checkpointer for commands),
   `Chunk.context` (embedding-only header such as the enclosing class signature for method chunks),
   `store.base.MemoryStore` and `index.embed.HashEmbedder` (tests and offline experiments; config accepts
   `backend = "memory"` and `model = "hash"`), and `eval/sample.py` + `eval/bugs.sample.yaml`.

8. **Leftover top-level statements** become one `<module>` chunk per contiguous run rather than one
   concatenated chunk, so every chunk's line range is exact and citable.

9. **Commit chunks have no path or lines** (`path=""`, lines 0). They inform history answers, but repo
   claims must cite file lines (from symbol evidence or `git log -L` context). The prompt says so.

10. **Budget reserve.** Grade, rewrite, and per-repo summaries leave one call for the answer. With nothing
    left, the answer node builds an evidence-only answer in code (every claim cites real evidence).

11. **Structured output order:** native (function calling) → JSON-mode prompt → JSON repair, two tries
    per model (the second with the validation error), then the fallback chain. Models whose native mode
    fails are remembered for the process. 429s honor `Retry-After` with a 20 s cap, two retries, then
    fallback; 5xx falls back immediately. A sliding-window limiter keeps under 20 requests/minute.

12. **Ephemeral threads.** `ask`/`debug` run on a throwaway thread id in the same SQLite saver and delete
    it afterwards; only `chat` threads get titles and rows in `threads`.

13. **Non-interactive defaults.** `--json`, piped stdin (`haive debug -`), or no TTY: clarify → all repos,
    web → declined, each with a notice.

14. **Chat paste detection.** On a TTY, lines that arrive in the same burst are one message. With piped
    stdin, a trace-like first line reads until two blank lines or a slash command. ``` fences work
    everywhere.

15. **Pinecone `count(filter)`** only supports a repo filter, via id-prefix listing (serverless indexes
    can't count by metadata). Repo names are restricted to `[A-Za-z0-9_.-]` so `repo:` prefixes are
    unambiguous.

16. **`haive remove`** deletes vectors and rows but leaves a cloned directory in place (it prints the
    path) rather than deleting files unasked.

17. **`haive sync`** still runs the incremental index when a pull fails (no upstream, diverged), with a
    warning per repo.

18. **Debug intent skips the router LLM** when a trace is given (the intent is known). When nothing in
    the repos matches, debug still goes to the answer (diagnosis from the trace alone, low confidence)
    instead of the general path.

19. **Eval sample.** The bundled sample repo (shared with tests) plus seven bug specs cover Python, Go,
    and Java. `--setup-sample` writes the generated suite into the data dir, never over the checked-in
    template. The README results table is regenerated only by real `haive eval` runs. The user's suite
    (`eval/questions.yaml`) is drafted during T15 from their registered repos.

20. **Embedding memory.** The default jina code model accepts 8k-token inputs. With 32 long chunks per
    batch, a CPU index was OOM-killed. Batch size defaults to 8 (`embeddings.batch_size`), and embedding
    input is capped at 6,000 characters (chunks are capped near that anyway).

21. **The spec lives at `docs/spec.md`**, not the repo root.
