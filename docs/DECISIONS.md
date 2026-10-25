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

35. **The answer schema requires its content.** DeepSeek V4 Pro, called with the `Answer` tool schema,
    returned in 3 s with only `summary`, the one required field. `sections` (and each section's and
    hypothesis's `claims`) are now required in the JSON schema models see, via `json_schema_extra`,
    with descriptions saying the answer goes there; Python code can still build an empty `Answer`.
    The empty-answer retry (#34) stays as the backstop.

36. **Freshness runs after the router.** It ran first, before the scope was known, so it checked every
    registered repo; pulling haive-mind cost a weatherpruf question 6 s of re-indexing. Now the router
    resolves the scope and freshness checks only those repos, as the spec intended. General questions
    skip it.

37. **A model that answers 404 is skipped for the rest of the process.** The free Nemotron Super's only
    provider went down and every call returned 404; haive retried it at each step (router, grader,
    answer) before falling back. A 404 means the model or its sole provider is gone, not busy, so the
    LLM layer drops it from its fallback chains for the rest of the question (or chat session). 429s
    and 5xx stay transient and are retried as before.

38. **One model per question, chosen by tier.** The fast/strong role split (fast model for routing and
    grading, strong for answers) is replaced by tiers: `fast`, `strong`, `fast_paid`, `strong_paid`, each
    an ordered list, with `tier` (default fast) and `use_paid` picking one per command (`--strong`/
    `--fast`, `--paid`). Every step uses that list. A model that fails a call (connection error or
    timeout, 429, 5xx after the client's retries) is skipped for the rest of the question, so steps
    don't silently alternate between models; a 404 skips it for the process (#37). Free lists end in
    `openrouter/free`, paid lists in `openrouter/auto`; each fallback is served by a different provider
    than the one before it, chosen from OpenRouter's live list and provider uptime (Oct 2026). The
    separate `fallback` list is gone. Trade-off: a fast tier answers with the fast model too; in
    practice the free Super produced fully verified answers, and `--strong` is one flag away. `step`
    (router, grade, answer, ...) still labels each call in --trace and the usage table. `haive eval
    --model X` runs on X alone. Supersedes the role parts of #22, #31 and #32.

39. **Cleanup after the tier change.** Removed code the design no longer uses: `git.status_porcelain`,
    `git.rev_list`, the `ChatModel` protocol, the `turn` and `summaries` state fields, and unused
    parameters (`_commits(rebuild=)`, `is_external(function=)`, `_newest(deps)`). One duration
    formatter instead of two. "Role" is now "step" everywhere it labels an LLM call, including the
    usage table (`haive.db` migration v2 renames the column in place, so no re-index). Docs and
    comments no longer describe fast/strong roles, Linux-only paths, or freshness running first;
    PROGRESS.md is a short current status. Cross-references for earlier entries: #27 revises #9 (commit
    chunks are now citable via `CommitRef`), and #37's "rest of the question" became "rest of the
    process" for 404s in #38.

40. **Cross-repo recall searches each repo separately.** One top-24 vector search over all repos in
    scope let one or two repos fill every slot (a "vector search across my projects" question got
    hnsw-engine and haive-mind only). Recall now runs one search per repo (4 chunks each) in parallel,
    orders repos by their best match, and interleaves their hits so each repo's best chunk is in the
    evidence before any repo's second; evidence is capped at 24. The question is embedded once (the
    embedder caches query vectors), so the extra cost is one Pinecone query per repo.

41. **Vector scores, not repo names, pick the repos for cross-repo questions.** The router only sees
    repo and group names, so for "how have I done X across my projects?" it guessed scope from names
    (it picked gre-prep-tool for vector search). For recall, the router now keeps only repos the user
    names; otherwise scope is every repo, each gets its own vector search (#40), and repos whose best
    match is below 0.8x the best repo's are dropped. The kept repos become the scope for the rest of
    the question; a notice names them and --trace lists every repo's best score (kept and dropped) to
    tune the 0.8.

42. **Speed pass.** (1) LangChain's `InMemoryRateLimiter` starts with an empty bucket, so the first
    LLM call of every command waited 1/rate = 3 s, and back-to-back calls 3 s each; haive's `Throttle`
    starts full, and only free models are throttled (paid ones have no 20/min limit). (2) Loading the
    embedding model from disk takes 2-3 s per command; it now loads in a background thread while the
    router's LLM call runs (only if already downloaded). (3) History questions answered from a commit
    window skip the grader's LLM call: the evidence is exactly the commits asked for. Checked and left:
    CLI import (<1 s), freshness (~100 ms for 4 repos), ripgrep (~30 ms per search), the grader's
    evidence (already abbreviated to 18 lines per item). The rest is LLM time.

43. **Longer request timeout, one client retry; relevance cutoff 0.7.** A 90 s timeout could cut off
    a slow free-model answer that needed a few more seconds, then start it over. The timeout is now
    180 s and the client retries a failed request once (was twice), so a hung model costs at most
    ~6 min before the next model in the tier's list, instead of up to 9 min at 180 s with two retries.
    The cross-repo relevance cutoff (#41) is 0.7x the best repo's score, keeping more borderline repos.

44. **Request timeout 125 s.** 180 s (#43) was more than needed: slow free-model answers in real runs
    took up to ~60 s, and a hung request still costs ~2x the timeout. 125 s leaves room for a slow
    answer to finish while bounding a hung model at ~4 min before the next one in the list.

45. **Combined grade-and-answer, behind a setting.** With one model per question (#38), a separate
    grader no longer buys a cheaper model's opinion, and it only sees abbreviated evidence. With
    `limits.combined_answer` (or the eval ablation `combined`), the grader makes no LLM call (its
    code checks remain), and the answer call may return `need_more` (missing symbols, more context, or
    a web query) instead of an answer while expansion rounds and budget allow; haive then runs the same
    expand / web paths and asks again. The option is not offered on verify retries or after the loop
    is exhausted, so it can't loop. The nothing-relevant -> rewrite path is unchanged. Off by default
    until `haive eval --ablation hybrid+loop,combined` on the author's suite shows it is at least as
    good: it saves one call per question, but models lean toward answering with what they have.

46. **First eval run (5 lookup questions, free models): combined stays off.** Combined used more LLM
    calls (7.4 vs 6.6: its answers asked for more evidence, and each request costs an expand step plus
    another full answer call), was slower (p50 47 s vs 26 s), and had lower first-try citation
    validity (82% vs 95%). Intent and file hits tied (80%); both misses were the router misreading
    two ambiguously worded questions (reworded). The run also exposed an eval bug: recall@10 ranked
    ripgrep before vector hits for hybrid but not combined, so identical retrieval scored 20% vs 100%.

47. **Plain words rank after vector hits in lookup; an explicit scope never routes to general.** The
    router often passes plain words as identifiers ("scope", "repos"); ripgrep on those matches all
    over and pushed the vector hits out of the evidence (recall@10 20% with ripgrep first vs 100% with
    vector first on the same retrieval). Lookup now runs ripgrep on code-like names (snake_case,
    camelCase, dotted, ENV_VARS) before the vector search and on plain words after it; the eval's
    recall@10 mirrors that order. When the user names repos (-r/-g, or a clarify answer), the router
    no longer answers from general knowledge. Eval ties are broken by path so recall@10 is stable.

48. **Eval reports why first-try citations failed.** A paid 5-question run scored 100% intent, file
    hits, and recall@10, but 81% first-try citation validity; each failure costs a retry answer call.
    The verify node now keeps the first try's error messages in `first_verify`, eval writes them to a
    `first_errors` CSV column, and `summary.md` counts them by kind (line range outside the file,
    symbol not in the cited lines, file not in the evidence, ...) so the full run shows what to fix.

49. **A cited symbol may be the definition enclosing the cited lines.** The full eval run's first-try
    failures were 17x "symbol not found within the cited lines" (and 1 repo claim with no refs). Every
    one was a correct citation: lines inside a function or class, labeled with that function or class,
    whose `def`/`class` line was more than 3 lines above (e.g. `verify.py:180-187`, symbol `verify`;
    `budget.py:63-69`, symbol `Meter`). Each cost a retry answer call. Verification now also accepts
    the symbol if it names one of the enclosing scope headers, found by walking up through lines of
    smaller indentation (skipping comments, decorators, and lines that open with a closing bracket,
    i.e. the tail of a multi-line signature or an earlier block's `}`). Symbols are reduced to their
    bare name first (`parse_metric()`, `HnswIndex<T>::search`, "BruteForceIndex class"). A symbol
    that is neither near nor enclosing still fails, so invented names are still caught.

50. **haive doesn't index its own source by default.** With haive-mind registered, answers cited its
    eval suite (the expected files for the very question being asked) and its original spec as if it
    described current behavior. Global `extra_ignore` patterns would also hide matching files in
    every other repo. Most users clone haive-mind to run it, not to ask about it, so
    `haive add --scan` skips any repo whose pyproject declares the `haive-mind` project (whatever its
    folder is called), with a line saying so; adding it by path still works for people developing
    haive. The author unregisters it (`haive remove haive-mind`), and the eval suite no longer has
    questions about it.

51. **Combined grade-and-answer is removed.** The full 19-question paid run settled #45: combined
    made 2.4 LLM calls a question against 3.4, but the calls it saves are grader calls (short
    evidence, short reply, about a tenth of a cent), median latency didn't move (14.5 s vs 14.7 s),
    p95 more than doubled (111 s vs 49 s: each "need more" round redoes a full answer call), one
    expected file was missed, and one answer came back with no claims. With the free 5-question run
    (#46) also worse, the setting, its answer schema, the graph route, and the eval ablation are gone.

52. **Claims are relabeled to match their refs instead of retried.** In the same run, most "wrong
    citation type" failures were repo claims with no refs (statements from the repo profile, or "no
    CI files appear in the evidence") or the reverse. Verification now relabels an uncited repo claim
    as `model` (which it is), and a cited model claim as `repo` when its refs pass (else its refs are
    dropped), saving the retry call without overstating anything. An answer that cites nothing at all
    is still sent back, since a lookup answer with no citations is the failure verification exists
    to catch. Two symbol-check gaps from the same run are closed: a lone `{` line (C#/Java brace
    style) no longer hides the method or class above it, and a claim citing a doc comment may name
    the definition that comment documents. The eval had also been counting every symbol failure as
    "wrong citation type" since #49 changed the message's wording; fixed.

53. **Paid is the default; tiers are strong and ultra.** Free and paid were about equally accurate on
    the suite, but paid answered in 13.5 s at the median against 63 s free (p95 29 s vs 176 s) for
    about 1-2 cents for all 19 questions. `use_paid` now defaults to true, and `--free` opts out per
    command. The tiers are renamed: `fast` becomes `strong` (the default) and `strong` becomes `ultra`,
    each with a `_paid` list; flags are `--strong/--ultra` and `--paid/--free`, on `ask`, `debug`,
    `chat`, and now `eval`. In the free strong list, Nemotron 3 Super (0% uptime over a day on
    OpenRouter in late October) is replaced by its successor Nemotron 3.5 Super VL, and MiniMax M3.1
    Flash (its free variant was delisted) by Gemma 4 26B A4B, a faster MoE than the dense 31B with
    similar scores. The ultra and paid lists showed ~100% provider uptime and are unchanged.
