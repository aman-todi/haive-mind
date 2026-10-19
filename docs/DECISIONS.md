# Decisions

Choices made where the spec was ambiguous or a dependency had moved. Newest last.

1. **Worked alone, in wave order.** No parallel sessions were used; tickets were built T0 → T14
   sequentially, with the shared contracts in `types.py` written first.

2. **LangChain where it's as good as or better than hand-written code** (revised; originally the stores
   and LLM plumbing were hand-written). Now: `langchain-pinecone` / `langchain-chroma` /
   `InMemoryVectorStore` behind a thin adapter, `Embeddings` subclasses over fastembed, `init_chat_model`
   with `InMemoryRateLimiter`, `.with_fallbacks()`, LCEL structured output, `ChatPromptTemplate`s, a
   callback handler for metering, and `@tool`s + `ToolNode` for expand. fastembed is still called
   directly, not via LangChain's `FastEmbedEmbeddings`: that lives in `langchain-community`, which is
   being sunset. To keep the location header in the embedding but return raw chunk text,
   `page_content` holds the header + chunk and `metadata.header_lines` tells the adapter what to strip.

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

9. **Commit chunks: last 300, messages only.** No diff text (revised from the spec's small-diff rule):
   code-level history comes from `git log -S`, `--grep`, and `-L` at question time, so diffs mostly
   cost embedding time. Chunks hold subject, body, author, date, changed files, and line counts; they
   have no path or lines (`path=""`), so repo claims cite file lines instead.

10. **Budget reserve.** Grade, rewrite, and per-repo summaries leave one call for the answer. With nothing
    left, the answer node builds an evidence-only answer in code (every claim cites real evidence).

11. **Structured output and retries (revised).** Per model: native `with_structured_output`
    (function calling) → JSON prompt with `PydanticOutputParser` format instructions and a cheap local
    cleanup (strip prose/fences, trailing commas) → one repair retry that feeds the parse error back
    (`with_fallbacks(exception_key="error")`), then `.with_fallbacks()` to the next model.
    `OutputFixingParser` was not used: in LangChain 1.x it only lives in the legacy `langchain-classic`
    package. 429s are retried by the OpenAI client (`max_retries=2`), which honors `Retry-After`; this
    replaces the spec's hand-rolled 20 s-capped backoff. A shared `InMemoryRateLimiter` (20/min) throttles
    client-side. Any failure outside the budget degrades the node (`LLMError`) instead of crashing.

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

22. **Model defaults (Oct 2026, from OpenRouter's live list).** fast = `nvidia/nemotron-3-super-120b-a12b:free`
    (native JSON-schema output + tools), strong = `nvidia/nemotron-3-ultra-550b-a55b:free`; alternatives
    `google/gemma-4-26b-a4b-it:free` and `poolside/laguna-s-2.1:free`. Paid strong option
    `deepseek/deepseek-v4-pro-0813`, off by default (`use_paid_strong`, or `--paid` per command); it only
    changes the strong role.

23. **Expand is a tool-calling subgraph** (`plan` → `ToolNode`), one planning call, at most 4 tool calls.
    It only uses the model when the budget still covers plan + grade + answer; otherwise (or with
    `limits.expand_with_tools = false`, or a model without tool calling) heuristics build the same tool
    calls. Tools capture the indexed file list up front so they never share the SQLite connection
    across `ToolNode`'s threads.

24. **Chat history goes to the router and answer as messages** (`MessagesPlaceholder("history")`, last 6),
    not as pasted text.

25. **Per-question LLM budget raised from 12 to 16, then to 25.** Counting the worst cases with the expand planner
    added: lookup with two expand loops is router 1 + grade 3 + plan 2 + answer 1 + 2 verify retries = 9,
    and each structured call can double when a free model's JSON needs the repair retry. An architecture
    question over a 3–4 repo group adds one summary per repo, and chat adds a title. 12 left little room,
    so summaries or answer retries were the first things cut on those paths. 16 covered them; 25 leaves
    headroom for further development and unforeseen paths. Typical questions still use 3–5 calls, and
    cost control is each installer's call via config (`max_llm_calls_per_question`, `daily_call_budget`).

26. **Chat follow-ups are rewritten to stand alone.** The router returns `standalone_question` when the
    latest message depends on earlier turns; it becomes `rewritten_query`, which retrieval, grading,
    expand, and web search use, and identifiers are pulled from it. The answer model still sees the
    original message plus history. No extra call: the router runs every turn anyway. A notice shows
    how the follow-up was read.

27. **Commits are citable.** Claims cite `RepoRef` (file lines), `CommitRef` (`{repo, sha}`), or
    `WebRef`. History answers were built from commit evidence but could only cite file lines, so the
    model cited changed files it never saw and every claim failed verification. A commit ref passes when
    its sha (7+ hex) matches a commit retrieved in this run. "What changed recently?" questions (no
    identifiers) now fetch the newest commits in scope directly; `git log -S` runs only on real
    identifiers, since pickaxe on plain words matches unrelated code.

28. **Answer retries stop when they stop helping.** A verify retry that fixes less than half of the
    remaining bad claims ends the loop: the same model with the same evidence rarely does better on a
    third try, and each try on a free strong model costs 30–60 s. The request timeout dropped from 120 s
    to 90 s (`models.timeout_seconds`), and `--trace` shows the OpenAI client's own failed attempts
    (timeouts, 429s) inside a call, which were previously invisible.

29. **Vague questions: assume and say so; ask only when no reading stands out.** The router (already
    one fast call per question) returns either an `assumption`, shown as a notice, or a clarifying
    question with 2-4 options when completely different answers fit. It is told never to ask about
    time ranges, missing scope, or detail level, since those have defaults; asking too often costs
    more than an occasional wrong guess the notice makes visible. The answer is folded into the
    question and the router runs again (intent and identifiers may change). Non-interactive runs
    answer as asked and say what they would have asked. No separate grader node: it would add a call
    to every question to catch a rare case.

30. **History time windows are parsed in code.** "last 2 weeks", "past three months", "since August",
    "since 2026-09-01", "last 20 commits" become `git log --since` / `-n`, applied to every git search
    in the question. With no window and no identifiers, the default is the last two weeks, falling
    back to the newest 12 commits when fewer than 3 landed in that window; a notice states the window
    used. Regexes rather than the model: the phrasings are few, and dates must be exact.

31. **The strong role falls back to the fast model first.** `fallback` defaults to
    `[nemotron-3-super, openrouter/free]`; a role skips its own model, so a stalled or rate-limited
    Ultra drops to Super (same family, faster) before the generic free router.

32. **One paid switch for both roles.** `use_paid` (or `--paid`) swaps fast to `fast_paid`
    (`deepseek/deepseek-v4.1-flash-1017`, $0.15 / $0.60 per 1M tokens) and strong to `strong_paid`. A
    paid answer behind a free router and grader still waited on free-tier rate limits and stalls for
    half its calls. The fast pick is a dated snapshot, like `strong_paid`, so its behavior and price
    don't move under a floating alias; one provider for both roles also means one data policy to review.

33. **Response time is shown above every answer**, with or without `--trace`. It excludes time spent at
    clarify and web prompts (the user's time, not haive's), so it reads the same in chat and one-shot.

34. **An answer with no claims fails verification when there was evidence.** A paid DeepSeek run
    returned a summary and empty sections; with nothing to check, it passed. Now it is retried once
    with an explicit instruction, and if it is still empty, finalize keeps the model's summary and lists
    the retrieved evidence (up to 12 items, commits cited by sha) under it, with a notice saying so.
