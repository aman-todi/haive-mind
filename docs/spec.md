# SPEC: haive-mind (`haive`)

A personal, local-first CLI that answers questions about, and helps debug, a developer's own git repositories. It indexes the repos (code, config, docs, and recent commits) into a vector store, then uses a LangGraph agent that mixes vector search, deterministic tools (ripgrep, file reads, git), optional web search, and several LLMs to answer with verified `file:line` citations. Every claim in an answer is labeled by where it came from: **[repo]**, **[web]**, or **[model]**.

Distributed as a GitHub repo. Each user brings their own keys (OpenRouter, Pinecone, optionally Tavily). Not a hosted app. No MCP server, no web UI.

---

## 0. Instructions for the orchestrating Claude Code session

- You are the **orchestrator**. Build the foundation ticket (T0) yourself, then hand out tickets from section 15 to parallel sessions (one git worktree/branch per ticket), review, and merge. If you are working alone, do the tickets in wave order.
- **Contracts first.** T0 freezes the shared types and interfaces in `src/haive/types.py` and the module skeleton. Parallel tickets must not change a shared contract. If a contract needs changing, the orchestrator does it and tells the other sessions.
- Each ticket ships with its own tests and must pass `uv run pytest` and `uv run ruff check` before merge. **Tests never touch the network**: LLMs, Pinecone, and web search are faked.
- Steps marked **[MANUAL]** need the human (accounts, keys, installs, choices). Stop and ask for them; list them clearly. See section 16.
- When something is ambiguous, choose the simplest option that fits this spec, record it in `docs/DECISIONS.md`, and continue. Ask the human only for things that are expensive to undo.
- Keep `PROGRESS.md` current: tickets done / in progress / blocked, decisions, known issues.
- The tool is **read-only toward user repos**. No code path may write to, stage, commit, or execute anything inside an indexed repo. The only git commands allowed to change state are `git clone` (into haive's own data dir) and `git pull --ff-only` (in `haive sync`).

---

## 1. Scope

**In scope**
- Registering repos (local paths or GitHub URLs, which are cloned into haive's data dir) and user-defined **groups** of repos that make up one project.
- Incremental indexing of code, config/docs, and the **last 1,000 commits** per repo.
- Seven question intents (section 7), including multi-language **debug** that diagnoses and **recommends** fixes. It never applies them.
- One-shot `ask`, `debug`, and interactive `chat` with saved, titled, resumable threads.
- Web search as a setting: `off` | `ask` | `auto`.
- Per-role models via OpenRouter, with a call budget per question.
- An eval harness with ablations.

**Out of scope (v1):** applying or writing patches, executing user code or tests, MCP server, web UI, native Windows, multi-user or shared indexes, call-graph analysis.

---

## 2. Tech stack

| Concern | Choice |
|---|---|
| Language / packaging | Python 3.11+, `uv`; installed with `uv tool install .` to expose the `haive` command |
| CLI / output | `typer`, `rich` |
| Config / models | `pydantic` v2, TOML config (`tomllib` read, `tomli-w` write), `platformdirs` for paths |
| Orchestration | `langgraph` (StateGraph, `Send`, `interrupt`/`Command`), `langgraph-checkpoint-sqlite` (`SqliteSaver`) |
| LLM interface | `langchain` `init_chat_model` → OpenAI-compatible client pointed at OpenRouter |
| Embeddings | `fastembed` (via `langchain_community.embeddings.FastEmbedEmbeddings`), default model `jinaai/jina-embeddings-v2-base-code` (768-d, runs locally) |
| Vector store | **Pinecone (default)** via `langchain-pinecone`; **Chroma (optional)** via `langchain-chroma`, as one adapter file |
| Parsing | `tree-sitter` + `tree-sitter-language-pack` |
| Search tools | `ripgrep` (`rg --json`) via subprocess, with a slow pure-Python fallback; `git` CLI via subprocess |
| Web search | `ddgs` (DuckDuckGo, no key) by default; Tavily (`langchain-tavily`) if a key is configured |
| Local DB | SQLite (stdlib `sqlite3`) |
| Tests / lint | `pytest`, `ruff`; LangChain fake chat models for graph tests |

Verify package names and APIs at install time. If one has moved, use its current equivalent and note it in `DECISIONS.md`.

---

## 3. CLI

```
haive init                                  # create config + data dirs, validate keys, create the vector index
haive add <path|git-url> [--name N] [--group G]   # register a repo (URLs are cloned into the data dir)
haive remove <name>                         # unregister a repo and delete its vectors and rows
haive repos                                 # list repos: path, last indexed, chunk/commit counts, stale?
haive groups                                # list groups and their repos (groups are edited in config.toml)
haive index [names...] [--full]             # incremental by default; --full rebuilds
haive sync                                  # git pull --ff-only on every repo, then incremental index
haive status                                # vector store, models per role, today's LLM call usage, index health

haive ask "<question>"  [-r REPO]... [-g GROUP]... [--web off|ask|auto] [--json] [--trace]
haive debug [FILE|-]    [-q "extra question"] [-r/-g ...] [--web ...] [--json] [--trace]
haive chat              [--resume ID] [-r/-g ...] [--web ...]
haive threads           [--limit N]         # ID, title, scope, turns, last updated
haive threads delete <ID>
haive eval              [--suite FILE] [--ablation NAME] [--models ROLE=MODEL ...]   # dev command
```

- `--trace` prints the path through the graph: nodes visited, tools called with timing, LLM calls by role and model, loop counts. This is meant for debugging and demos.
- `--json` prints the structured answer (section 10) instead of rich output. The eval harness uses it.
- Chat slash commands: `/scope <repos or groups|all>`, `/web off|ask|auto`, `/new`, `/threads`, `/exit`.
- `debug` reads the trace or error from a file or from stdin. In `chat`, a pasted multi-line trace is detected and routed to the debug intent.

---

## 4. Configuration and paths

- Config: `~/.config/haive/config.toml` (via `platformdirs`), written by `haive init`.
- Secrets: environment variables, or `~/.config/haive/.env`. Never stored in config.toml. Keys: `OPENROUTER_API_KEY`, `PINECONE_API_KEY`, optional `TAVILY_API_KEY`.
- Data: `~/.local/share/haive/` holds `haive.db`, `checkpoints.db`, `clones/`, and `chroma/`.

```toml
[repos.weatherpruf]
path = "~/code/weatherpruf"
[repos.valuation-backend]
path = "~/code/intrinsic-valuation"     # repos can be monorepos; groups are optional

[groups]
valuation = ["valuation-backend", "valuation-frontend", "valuation-infra"]

[index]
max_commits = 1000
small_diff_lines = 200        # include diff text only for commits with fewer changed lines
max_file_kb = 512
extra_ignore = ["*.ipynb"]

[vector_store]
backend = "pinecone"          # or "chroma"
pinecone_index = "haive"
pinecone_cloud = "aws"
pinecone_region = "us-east-1"

[embeddings]
model = "jinaai/jina-embeddings-v2-base-code"

[models]                      # OpenRouter model IDs; verify availability [MANUAL]
fast   = "google/gemma-4-26b-a4b-it:free"            # router, grader, rewrite, titles, per-repo summaries
strong = "nvidia/nemotron-3-ultra-550b-a55b:free"    # answer, diagnose, synthesis
fallback = ["openrouter/free"]                        # tried in order on errors or 429s
base_url = "https://openrouter.ai/api/v1"

[limits]
max_llm_calls_per_question = 12
max_grade_loops = 2
max_answer_retries = 2
daily_call_budget = 1000      # warn at 90%; refuse new questions at 100% unless --force

[web]
mode = "ask"                  # off | ask | auto
provider = "auto"             # auto = tavily if key present, else duckduckgo
max_results = 5
```

---

## 5. Storage

**The working tree is the source of truth.** File reads, ripgrep, git, and citation checks always run against the live repo. SQLite is an index of the repos, not a copy of them.

SQLite `haive.db` (versioned schema with a simple migrations table):
- `repos(name PK, path, remote_url, last_indexed_head, last_indexed_at, last_commit_sha)`
- `files(repo, path, sha256, language, kind, size, indexed_at, PK(repo, path))`
- `chunks(id PK, repo, path, kind, language, symbol, start_line, end_line, content_hash, text)`. `kind` is one of `code | config | doc | commit`.
- `commits(repo, sha, date, author, subject, files_json, insertions, deletions, PK(repo, sha))`
- `repo_profiles(repo PK, json, updated_at)` (see 6.5)
- `threads(id PK, title, scope_json, turns, created_at, updated_at)`
- `usage(date, role, model, calls, prompt_tokens, completion_tokens, errors)`

`checkpoints.db` is used by LangGraph's `SqliteSaver` only.

**Vector store:** one index/collection, 768-d, cosine distance.
- Vector id = chunk id.
- Metadata: `repo, path, kind, language, symbol, start_line, end_line`, plus page text, which stays well under Pinecone's metadata size limit because chunks are capped.
- Scoping is a metadata filter (`repo $in [...]`, `kind $in [...]`).
- Adapter note: Pinecone accepts several filter keys implicitly ANDed; Chroma needs an explicit `$and`. The adapter normalizes this.
- `store/base.py` exposes `upsert(chunks)`, `delete(ids)`, `search(query, k, filter) -> list[Hit]`, and `count(filter)`. `pinecone_store.py` and `chroma_store.py` implement it. The Chroma backend must stay within that one file.

**Chunk ids:**
- Content chunks: `{repo}:{path}:{sha1(content)[:12]}`. Unchanged chunks keep their id, so only changed chunks are re-embedded.
- Commit chunks: `{repo}:commit:{sha}`.

---

## 6. Indexing

### 6.1 Discovery
- `git ls-files` plus untracked, non-ignored files (`git ls-files --others --exclude-standard`). `.gitignore` is always respected.
- Skip binaries, files larger than `max_file_kb`, lockfiles (`package-lock.json`, `yarn.lock`, `pnpm-lock.yaml`, `uv.lock`, `poetry.lock`, `Cargo.lock`, …), minified/generated files (`*.min.js`, `dist/`, `build/`), and `extra_ignore`.
- **Secrets denylist, never indexed or sent to an LLM:** `.env*` (except `.env.example` / `.env.sample`), `*.pem`, `*.key`, `id_rsa*`, `*.p12`, `credentials*`, `secrets*`. Also apply a regex scan that drops chunks containing obvious tokens (AWS keys, `sk-…`, private key headers).

### 6.2 Chunking
- **Code (tree-sitter):** Python, JavaScript, TypeScript/TSX, Go, Rust, Java, C, C++, C#. One chunk per top-level function, class, or method. Large classes are split into methods, with the class signature repeated as a header. Leftover top-level statements are grouped into a "module" chunk. Chunks over ~1,500 tokens are split into overlapping line windows.
- **Config and docs, chunked by structure:** Markdown by heading; YAML/JSON/TOML by top-level key; Dockerfile and docker-compose whole (or one chunk per service); Terraform/HCL per block; SQL per statement; shell per function or whole file.
- **Fallback:** 60-line windows with 10 lines of overlap.
- The embedding text gets a context header (`{repo}/{path} :: {symbol}`) so the vector carries location context. The stored `text` is the raw chunk.

### 6.3 Commits
- Index the most recent `max_commits` per repo.
- Chunk text: subject + body, author, date, changed files (truncated list), insertions/deletions. Include the diff text (truncated to about 4k characters) only when total changed lines are under `small_diff_lines`.
- **Incremental:** `git rev-list {last_commit_sha}..HEAD`. If `last_commit_sha` is no longer an ancestor (rebase or force-push), rebuild that repo's commit chunks. Prune past `max_commits`.

### 6.4 Incremental updates and freshness
- `haive index`: for each repo, diff `last_indexed_head` against the current HEAD and the working tree (`git diff --name-only <last_indexed_head>` plus untracked files). Re-hash candidate files, re-chunk the changed ones, upsert new chunk ids, and delete chunk ids that disappeared. Handle deleted and renamed files.
- **Freshness check before every question**, for repos in scope only. It must finish in about 200 ms or less for typical repos. If 50 or fewer files changed, re-index them inline with a one-line notice. If more changed, warn and suggest `haive index` but continue.
- The first full index shows a progress bar (files, chunks, embedding throughput).

### 6.5 Repo profile (deterministic, no LLM)
Built at index time and stored in `repo_profiles`. It contains:
- languages by line count;
- frameworks and dependencies detected from manifests (`pyproject.toml`, `requirements*.txt`, `package.json`, `go.mod`, `Cargo.toml`, `*.csproj`);
- entrypoints (main files, `scripts` in package.json, CLI entry points);
- docker-compose services with their ports, images, and env var names;
- GitHub Actions workflows and their triggers;
- Terraform resource types;
- env var names referenced in code and `.env.example`;
- HTTP routes defined (common FastAPI/Flask/Express/ASP.NET patterns) and URLs/paths called (fetch/axios/requests).

The architecture intent and group connection-mapping use these profiles.

---

## 7. Intents

| Intent | Example | Retrieval plan (code first, LLM after) | Answer template |
|---|---|---|---|
| `lookup` | "Where does the DCF choose a valuation method?" / "How does X work?" | Exact identifiers → ripgrep and symbol match in `chunks`, then vector (kind=code) → read the enclosing function | Location(s) + walkthrough |
| `recall` | "How did I implement OAuth before?" | Vector across scope (kind code/config/doc), grouped by repo, top 2 per repo | Best examples per repo + "to adapt this here…" |
| `history` | "When and why did I add retries to the API client?" | Vector (kind=commit) + `git log -S<identifier>` / `--grep` + `git log -L` on matched functions | Timeline of commits with reasons |
| `debug` | Pasted traceback, compiler or linter output | Section 8 | Ranked hypotheses, evidence, suggested fix (text only), how to verify |
| `compare` | "How do Weatherpruf and the valuation app differ on deploy?" | `Send` fan-out: the same sub-plan per repo or group → merge | Side-by-side table + narrative |
| `architecture` | "Break down the valuation project" (repo or group) | Profiles + README/docs/compose/terraform/workflow chunks; for a group, fan out a per-repo summary (fast model), then deterministic **connection mapping** (shared env vars, compose service names, routes defined in one repo and called in another, shared deploy workflows) | Components, how they connect, data flow, deploy path |
| `general` | "How do I paginate in FastAPI?" (not about the repos) | No repo retrieval; web if the mode allows | Answer labeled [model]/[web], with a note that it's not from the repos |

The router validates scope in code. Repo and group names are matched against config, case-insensitively and with fuzzy matching. Unknown names or a truly ambiguous scope trigger a **clarify interrupt**. Group names expand to their repos everywhere.

---

## 8. Debug (diagnose and recommend only)

1. **Parse (pure code, `tools/trace_parse.py`).** Detect the format and extract `error_type`, `message`, and `frames[{path, line, col, function}]`.
   - Dedicated parsers: Python tracebacks (including chained exceptions), Node/V8 (`at fn (path:line:col)`), Java/Kotlin (`at pkg.Cls.m(File.java:N)`), Go panics, Rust panics and backtraces, .NET (`in path:line N`), C/C++ sanitizer/gdb (`#0 0x… in fn path:line:col`), and compiler/linter lines (`path:line:col: error: …`, `path(line,col): error …`).
   - **Generic fallback:** any `path:line[:col]` or `path(line[,col])` token.
   - **Map paths to repo files** by longest matching path suffix against indexed files in scope. Ties go to the in-scope repo, then the most recently modified file. Java frames match by package path plus file name.
   - Library frames (`site-packages`, `node_modules`, stdlib or system paths, `.cargo/registry`, …) are marked external. Keep the last internal frame before an external one, since that's the call boundary.
   - TypeScript traces pointing at `dist/*.js` are matched to sources by file stem and function name. Note this limitation in the docs.
2. **Gather evidence (code, run in parallel):**
   - For each internal frame, read a ±20-line window expanded to the enclosing function.
   - Ripgrep the longest literal fragment of the message (≥12 characters, with quoted values and numbers stripped) to find where it's raised.
   - `git blame -L` on frame lines, flagging lines changed in the last 30 days, plus `git log -n 5 -- <path>`.
   - Vector search on frame symbols and the error type, to find callers and similar handling elsewhere.
3. **Grade and expand** (shared loop, section 9). Typical expansions: read the caller, find a symbol's definition, open a config file or env var the message mentions.
4. **Web** (subject to the mode). If the failing frame is external or the message names a library, search `"<library> <error_type> <message fragment>"`.
5. **Diagnose (strong model).** Up to 3 hypotheses, each with a confidence level (low/med/high) and cited evidence. Then a **suggested fix shown as a unified diff in the answer** (never written to disk), and "how to verify" steps.
6. **Verify.** Normal citation checks, plus a diff check: the `-` and context lines of the suggested diff must exist in the cited file. If they don't, the fix is labeled "approximate".

---

## 9. Graph

### 9.1 State (pydantic / TypedDict in `graph/state.py`)
`messages` (chat history), `question`, `trace_text?`, `intent`, `scope_repos`, `identifiers`, `rewritten_query?`, `parsed_trace?`, `evidence: list[Evidence]`, `web_results: list[WebResult]`, `grade: {status, missing}`, `loops`, `llm_calls`, `answer: Answer?`, `verify_errors`, `answer_retries`, `web_mode`, `notices: list[str]`, `node_log` (for `--trace`).

`Evidence = {id, source: code|commit|tool|web, repo?, path?, start_line?, end_line?, text, why}`

### 9.2 Nodes and edges

```
START → freshness (code) → router (fast LLM, structured)
router ──needs clarification──► clarify (interrupt) → router
router ──conditional on intent──►
   lookup | recall | history → retrieve_<intent> (code + vector)
   debug        → parse_trace (code) → gather_debug (code, parallel)
   compare      → Send(per repo/group sub-retrieval) → merge (code)
   architecture → load_profiles (code) → [group: Send(per-repo summary, fast LLM)] → connect_map (code)
   general      → maybe_web
all retrieval paths → grade (fast LLM): sufficient | missing_symbol | need_context | need_external | nothing_relevant
   missing_symbol   → expand: ripgrep/definition lookup        ─┐
   need_context     → expand: read file / caller / config       ├─► grade   (max_grade_loops)
   need_external    → maybe_web                                 ─┘
   nothing_relevant → rewrite_query (fast LLM, once) → retrieve again; if still nothing → general path with a notice
   loops exhausted  → answer (flagged low-confidence)
maybe_web: mode off → skip with notice | ask → interrupt for approval | auto → web_search (code)
answer (strong LLM, intent template, structured output) → verify (code)
verify ──errors and retries left──► answer (with the specific errors)
verify ──ok or retries exhausted──► finalize (code: demote unverifiable claims, record usage, set thread title on first turn)
```

- **Budget guard:** every LLM-calling node checks `llm_calls` against `max_llm_calls_per_question`. Over budget, the graph goes straight to `answer`, or to `finalize` if the answer budget is also spent, with a notice.
- Interrupts (clarify, web approval) need a checkpointer. `ask`/`debug` use an ephemeral thread in the same SQLite saver. In a non-TTY context (piped), interrupts resolve to defaults: scope = all, web declined. A notice records this.
- `chat` reuses the thread's `messages`. The router sees recent turns, so follow-ups like "and in the other repo?" resolve.

### 9.3 LLM roles
| Role | Model | Calls |
|---|---|---|
| fast | router, grade, rewrite_query, per-repo summaries, thread title | structured output |
| strong | answer / diagnose / synthesis | structured output |
| fallback chain | used by both on 5xx, 429 after backoff, or repeated invalid output | |

---

## 10. Answers and provenance

The answer node must return this structure, through the structured-output helper:

```
Answer { summary: str,
         sections: [ { heading?: str, claims: [Claim] } ],
         suggested_fix?: { diff: str, approximate: bool },   # debug only
         hypotheses?: [ { title, confidence, claims: [Claim] } ],  # debug only
         followups: [str] }
Claim  { text: str, source: "repo" | "web" | "model", refs: [RepoRef | WebRef] }
RepoRef{ repo, path, start_line, end_line, symbol? }
WebRef { url }
```

**Verification (`graph/verify.py`, pure code):**
- A `repo` claim needs at least one ref. The file must exist in the live tree, the lines must be in range, the `symbol` (if given) must appear within the cited lines ±3, and the path must be in `evidence`.
- A `web` claim's URLs must be in this run's `web_results`.
- A `model` claim must have no refs.
- Failures go back to `answer` with an error list. After the retries are used up, failing claims are shown with an **[unverified]** tag. They are never silently dropped.

**Rendering (`render.py`):** each claim is prefixed with a colored tag (`[repo]`, `[web]`, `[model]`, `[unverified]`), and refs are printed as `repo/path:12-30` and URLs. Notices (stale index, web skipped, budget hit, scope defaulted) appear in a dim footer. `--json` dumps the raw `Answer` plus metadata (intent, scope, llm_calls, latency).

---

## 11. LLM layer (`llm/`)

- `get_model(role)` uses `init_chat_model(model_id, model_provider="openai", base_url=…, api_key=OPENROUTER_API_KEY)`, with optional `HTTP-Referer` / `X-Title` headers. Any OpenAI-compatible endpoint, or a different provider through config, must also work.
- `structured(role, schema, messages)`: try native structured output or tool calling, then a JSON-mode prompt, then a JSON repair step (extract the largest JSON block, fix common issues, validate with pydantic). Retry once on the same model, then move down the fallback chain.
- **429 handling:** respect `Retry-After`, exponential backoff with a 20 s cap, then fallback. OpenRouter free models are limited to roughly 20 requests per minute, so throttle to stay under that.
- **Usage accounting** goes to the `usage` table on every call. `haive status` shows today's calls against `daily_call_budget`.
- Prompts live in `src/haive/prompts/*.md` and are versioned, with no prompt strings scattered through the code.

---

## 12. Threads and chat

- `SqliteSaver` persists graph state per thread. The `threads` table stores metadata.
- After the first turn, the fast model generates a title of 8 words or fewer. Fall back to the first 60 characters of the question if that call fails or the budget is spent.
- `haive threads` shows a short ID (first 8 characters), title, scope, turns, and updated time (relative), newest first.
- `chat` starts a new thread unless `--resume ID` is given. Pasting a trace inside chat routes it to `debug`.

---

## 13. Eval (`eval/`)

**Suite (`eval/questions.yaml`, drafted by Claude Code from the user's registered repos):**
- About 30 questions, 4–5 per intent. Each has an `expected_files` list (paths that must appear among verified repo refs), optional `expected_intent`, and short notes.
- **Debug cases are made by bug injection, never in the user's repos.** Copy a repo to a temp directory, introduce 5–8 realistic bugs (e.g., a wrong key, an off-by-one error, a renamed env var, a missing None check, a bad import), run the relevant entrypoint or test to capture a real error, and store the trace with the expected file and line plus the known cause. Index the temp copy as a separate eval repo.

**Metrics (deterministic first):**
- intent accuracy;
- **file hit rate** (an expected file appears in verified repo refs);
- retrieval recall@10 for expected files before any LLM step;
- citation validity (share of repo refs passing verification on the first try);
- debug: correct file+line in the top hypothesis, and in any hypothesis;
- LLM calls per question; p50/p95 latency.

**Ablations** (`--ablation`):
- `vector_only`, `ripgrep_only`, `hybrid`, `hybrid+loop` (the default).
- `pinecone` vs `chroma` (retrieval recall only).
- A model sweep per role via `--models fast=… strong=…`.

**Output:** `eval/results/<timestamp>/` holds CSVs plus `summary.md`. The README results table is generated from the latest run, so it contains only measured numbers.

---

## 14. Repository layout

```
pyproject.toml  README.md  spec.md  PROGRESS.md  LICENSE  .github/workflows/ci.yml
docs/  DECISIONS.md  ARCHITECTURE.md (graph diagram + data flow)  PRIVACY.md
src/haive/
  cli.py  config.py  db.py  types.py  render.py  threads.py  paths.py
  index/   discover.py  chunk_code.py  chunk_text.py  commits.py  profile.py  embed.py  pipeline.py  freshness.py
  store/   base.py  pinecone_store.py  chroma_store.py
  tools/   ripgrep.py  files.py  git.py  trace_parse.py  web.py  pathmap.py
  llm/     models.py  structured.py  budget.py
  prompts/ router.md  grade.md  rewrite.md  answer_*.md  diagnose.md  summarize_repo.md  title.md
  graph/   state.py  build.py  router.py  retrieve.py  debug.py  fanout.py  grade.py  expand.py  web_node.py  answer.py  verify.py  finalize.py
eval/      questions.yaml  run.py  metrics.py  inject_bugs.py  results/
tests/     fixtures/ (tiny multi-language git repos built by a pytest fixture; sample traces per language)  unit/  graph/  integration/
```

CI (GitHub Actions): ruff + pytest on ubuntu and macOS, Python 3.11 and 3.12, no secrets required.

---

## 15. Tickets

Every ticket owns the files listed for it, ships its tests, and works only against `types.py` contracts and fakes for anything not built yet.

**Wave 0 (orchestrator, sequential)**
- **T0 Foundation:** `pyproject.toml` with deps, package skeleton, `paths.py`, `config.py` (pydantic models + TOML load/save + `haive init` template), `db.py` (schema + migrations), `types.py` (Chunk, Hit, Evidence, WebResult, ParsedTrace, Frame, Answer/Claim/Refs, RepoProfile, and the VectorStore/Embedder/ChatModel protocols), a stub `cli.py` with every command registered, test fixtures (a temp multi-language git repo with history, plus a fake LLM that returns scripted structured outputs), CI, `PROGRESS.md`, `DECISIONS.md`.

**Wave 1 (parallel after T0)**
- **T1 Discovery + chunking + profile:** `index/discover.py`, `chunk_code.py`, `chunk_text.py`, `profile.py`. Tests per language and per config type, the secrets denylist, and size/lockfile skips.
- **T2 Commit indexing:** `index/commits.py`. Last N commits, small-diff rule, incremental `rev-list`, rebase detection, pruning.
- **T3 Embeddings + vector stores:** `index/embed.py`, `store/*`. Pinecone index creation and checks, the Chroma adapter, filter normalization, batching. Tests use the Chroma in-memory client and a mocked Pinecone client.
- **T4 Deterministic tools:** `tools/ripgrep.py` (with fallback), `files.py` (range reads, enclosing-function expansion via tree-sitter), `git.py` (blame -L, log -n/-S/--grep/-L, rev-list, ls-files, clone, pull --ff-only), `web.py` (ddgs + Tavily, normalized `WebResult`).
- **T5 Trace parsing + path mapping:** `tools/trace_parse.py`, `tools/pathmap.py`. Fixture traces for every format in section 8, the generic fallback, and external-frame detection.
- **T6 LLM layer:** `llm/*`. Role models, OpenRouter wiring, structured output + repair, fallback chain, 429 backoff, budget and usage accounting.
- **T7 Provenance:** `graph/verify.py` (claim, ref, and diff checks) and `render.py` (rich output, `--json`).

**Wave 2**
- **T8 Index pipeline + repo commands** (needs T1–T3): `index/pipeline.py`, `freshness.py`, and the `init/add/remove/repos/groups/index/sync/status` CLI. Incremental upsert/delete, rename handling, progress output.
- **T9 Graph core + `ask`** (needs T3, T4, T6, T7): `graph/state, build, router, retrieve (lookup/recall/history), grade, expand, answer, finalize`, the budget guard, `--trace`, and the `ask` command. Graph tests use the fake LLM to cover each conditional edge.

**Wave 3 (parallel)**
- **T10 Debug** (T5, T9): `graph/debug.py`, the diagnose prompt and template, the `debug` command, stdin handling.
- **T11 Compare + architecture + groups** (T1 profile, T9): `graph/fanout.py`, connection mapping, per-repo summaries.
- **T12 Chat, threads, interrupts** (T9): `threads.py`, the `chat` REPL and slash commands, thread titles, the clarify interrupt, resume, non-TTY defaults.
- **T13 Web node + general intent** (T4, T9): `graph/web_node.py` with the off/ask/auto modes and approval interrupt, the general path, `need_external` handling.

**Wave 4**
- **T14 Eval + docs** (all): `eval/*` including bug injection on temp copies, ablations, and the results table. README (pitch, install, quickstart, the [MANUAL] setup steps, privacy notes, results), `ARCHITECTURE.md` with a graph diagram, `PRIVACY.md`.
- **T15 Hardening (orchestrator):** an end-to-end run on the user's real repos, fixes from the user's feedback, and a performance check (freshness check ≤200 ms, `ask` p50 reported).

**Merge order:** T0 → (T1–T7) → T8, T9 → (T10–T13) → T14 → T15. Run T8 against real repos as soon as it lands; that's the first point the human can try the tool.

---

## 16. Manual steps for the human [MANUAL]

1. **Pinecone:** create a free account and an API key, and set `PINECONE_API_KEY`. Index creation is handled by `haive init`.
2. **OpenRouter:** set `OPENROUTER_API_KEY`. Confirm the 1,000-requests/day free tier is active. Review the privacy and data-policy settings for free models (some providers may log or train on prompts, and your code goes into prompts).
3. **Model IDs:** check that the default free models in config still exist on openrouter.ai/models, or pick replacements. The eval model sweep can help choose.
4. **Optional Tavily:** set `TAVILY_API_KEY` for better web results.
5. **Install ripgrep:** `brew install ripgrep` or `apt install ripgrep`.
6. **Register repos:** provide local paths (and any GitHub URLs) and define groups in `config.toml`.
7. **First full index:** run it and report anything wrong (missed files, bad chunks, slowness).
8. **Optionally,** supply 2–3 real error traces from your own projects for the debug eval, in addition to the injected bugs.
9. **Glance over** the drafted eval questions (optional; Claude Code writes them).

---

## 17. Definition of done

- All tickets merged; CI green; `uv tool install .` gives a working `haive` command on macOS and Linux/WSL.
- On the user's real repos: every intent works end to end. Answers show [repo]/[web]/[model] labels with verified refs. The debug flow handles traces from at least Python, TypeScript/Node, and one compiled language. The three web modes behave as specified. Threads list with titles and resume correctly.
- The read-only guarantee is tested: a test asserts that no indexed repo's `git status` or file mtimes change after a full run of the eval suite.
- The eval runs and produces the ablation and results tables, and the README shows only measured numbers.
- The docs cover setup, privacy (free-model data policies, secrets denylist), limitations (TS source maps, no patch application, free-tier rate limits), and architecture.
