# haive-mind (`haive`)

A personal, local-first CLI that answers questions about, and helps debug, **your own git repositories**.

haive indexes your repos (code, config, docs, and the last 300 commit messages) into a vector store, then runs a
LangGraph agent that mixes vector search with deterministic tools (ripgrep, file reads, git), optional
web search, and several LLMs. Answers come with verified `file:line` (or commit) citations, and every claim is
labeled by where it came from:

- **[repo]**: backed by code that was retrieved this run and re-checked against your working tree
- **[web]**: backed by a web result fetched this run
- **[model]**: the model's general knowledge
- **[unverified]**: a claim whose citation failed verification (shown, never silently dropped)

It is **read-only toward your repos**: it never writes, stages, commits, or runs anything inside them.
The only state-changing git commands are `git clone` (into haive's own data dir, for `haive add <url>`)
and `git pull --ff-only` (in `haive sync`).

```
$ haive ask "Where does the DCF choose a valuation method?"
╭─ lookup ───────────────────────────────────────────────────────────────╮
│ choose_method in app/valuation/dcf.py picks ddm, fcf, or multiples.    │
╰────────────────────────────────────────────────────────────────────────╯
Location
  • [repo] choose_method decides the valuation method
      valuation/app/valuation/dcf.py:1-7
Walkthrough
  • [repo] Companies paying dividends use the dividend discount model ...
```

## What it can do

| Intent | Example |
|---|---|
| lookup | "Where does the DCF choose a valuation method?" / "How does X work?" |
| recall | "How did I implement OAuth before?" (across all your repos) |
| history | "When and why did I add retries to the API client?" (commits, `git log -S/-L`) |
| debug | `haive debug trace.txt`: ranked hypotheses, evidence, a suggested diff (never applied), how to verify |
| compare | "How do weatherpruf and the valuation app differ on deploy?" |
| architecture | "Break down the valuation project" (a repo or a group: components, connections, data flow, deploy) |
| general | "How do I paginate in FastAPI?" (not about your repos; web if allowed) |

Debug understands Python (including chained exceptions), Node/V8, Java/Kotlin, Go panics, Rust panics and
backtraces, .NET, C/C++ sanitizer and gdb output, and compiler/linter lines (gcc/clang, tsc, rustc,
MSBuild, mypy/ruff/eslint style). Anything else falls back to a generic `path:line` scan.

## Install

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), git, and (recommended) ripgrep. macOS and
Linux/WSL are supported; native Windows is not.

```bash
git clone https://github.com/aman-todi/haive-mind && cd haive-mind
uv tool install .              # exposes the `haive` command
# optional extras: uv tool install '.[chroma]'  (local vector store)  /  '.[tavily]'  (better web search)
```

## Setup (manual steps)

These need you, not the tool:

1. **Pinecone** (default vector store): create a free account and an API key, and set `PINECONE_API_KEY`.
   `haive init` creates the index. Prefer everything local? Use `haive init --backend chroma` (install the
   `chroma` extra) and skip Pinecone entirely.
2. **OpenRouter**: set `OPENROUTER_API_KEY`. Confirm the free tier (1,000 requests/day) is active, and
   **review OpenRouter's privacy and data-policy settings for free models**: some providers may log or
   train on prompts, and your code goes into prompts. See [docs/PRIVACY.md](docs/PRIVACY.md).
3. **Model IDs**: check that the default free models in `config.toml` still exist on
   [openrouter.ai/models](https://openrouter.ai/models), or pick replacements. The eval model sweep helps.
4. **Optional Tavily**: set `TAVILY_API_KEY` for better web results (DuckDuckGo is used otherwise).
5. **Install ripgrep**: `brew install ripgrep` or `apt install ripgrep` (a slow Python fallback exists).
6. **Register repos** and define groups (below).
7. **First full index**: run it and report anything wrong (missed files, bad chunks, slowness).

Keys can be exported in your shell or put in `~/.config/haive/.env` (created by `haive init`, mode 600).
They are never written to `config.toml`.

## Quickstart

```bash
haive init                                   # config + data dirs, validates keys, creates the index
haive add ~/code/intrinsic-valuation --name valuation-backend -g valuation
haive add https://github.com/you/weatherpruf  # URLs are cloned into haive's data dir
haive add --scan ~/code                      # or register every repo in a folder at once
haive index                                  # indexes all registered repos; first run downloads the model
haive ask "Where does the DCF choose a valuation method?"
haive debug crash.txt -q "happens on empty portfolios"
pytest 2>&1 | haive debug -                  # or pipe it in
haive chat                                   # threads are saved, titled, and resumable
```

## Commands

```
haive init                                  create config + data dirs, validate keys, create the vector index
haive add <path|git-url>... [--name N] [-g G]   register one or more repos
haive add --scan <dir>... [-g G]            register every git repo directly inside <dir>
haive remove <name>                         unregister a repo and delete its vectors and rows
haive repos                                 path, last indexed, chunk/commit counts, stale?
haive groups                                groups and their repos (edit groups in config.toml)
haive index [names...] [--full]             no names = all registered repos; incremental unless --full
haive sync                                  git pull --ff-only on every repo, then incremental index
haive status                                vector store, models per role, today's LLM usage, index health

haive ask "<question>"  [-r REPO]... [-g GROUP]... [--web off|ask|auto] [--json] [--trace] [--paid]
haive debug [FILE|-]    [-q "extra question"] [-r/-g ...] [--web ...] [--json] [--trace] [--paid]
haive chat              [--resume ID] [-r/-g ...] [--web ...] [--paid]
haive threads           [--limit N]
haive threads delete <ID>
haive eval              [--suite FILE] [--ablation NAME] [--models ROLE=MODEL ...] [--setup-sample]
```

- `--trace` prints the path through the graph: nodes visited, tools called with timing, LLM calls by role
  and model.
- `--json` prints the structured answer plus metadata (intent, scope, LLM calls, latency, notices).
- In `chat`: `/scope <repos or groups|all>`, `/web off|ask|auto`, `/new`, `/threads`, `/exit`. Pasting a
  multi-line error trace routes it to debug.
- Before every question, haive checks the repos in scope for changes. Up to 50 changed files are
  re-indexed inline (with a one-line notice); beyond that it warns and suggests `haive index`.

### Web search modes

`--web off` never searches. `ask` (the default) asks before each search; when there's no terminal to ask
on (piped input, `--json`), it declines and says so. `auto` searches whenever the agent decides the
answer depends on external library behavior, or when a debug trace fails inside a library.

### Budgets

Each question may use up to `max_llm_calls_per_question` (25) LLM calls; most use 3–5. The grader,
query rewriter, and expand planner always leave room for the answer. `haive status` shows today's calls
against `daily_call_budget` (1,000, matching OpenRouter's free tier); haive warns at 90% and refuses new
questions at 100% unless `--force`.

## Configuration

`~/.config/haive/config.toml` (or `$HAIVE_HOME/config/config.toml`):

```toml
[repos.valuation-backend]
path = "~/code/intrinsic-valuation"

[groups]
valuation = ["valuation-backend", "valuation-frontend", "valuation-infra"]

[index]
max_commits = 300             # commit messages + file lists only (no diffs); git log -S/-L covers code history
max_file_kb = 512
extra_ignore = ["*.ipynb"]

[vector_store]
backend = "pinecone"          # or "chroma"
pinecone_index = "haive"

[embeddings]
model = "jinaai/jina-embeddings-v2-base-code"   # 768-d, runs locally via fastembed
batch_size = 8

[models]                      # OpenRouter model IDs (any OpenAI-compatible endpoint works)
fast = "nvidia/nemotron-3-super-120b-a12b:free"    # router, grader, rewrite, expand planner, titles, summaries
strong = "nvidia/nemotron-3-ultra-550b-a55b:free"  # answer, diagnose, synthesis
strong_paid = "deepseek/deepseek-v4-pro-0813"      # used with --paid or use_paid_strong = true
use_paid_strong = false
fallback = ["openrouter/free"]
requests_per_minute = 20      # client-side throttle
max_retries = 2               # per model (honors Retry-After) before falling back
base_url = "https://openrouter.ai/api/v1"

[limits]
max_llm_calls_per_question = 25
max_grade_loops = 2
max_answer_retries = 2
daily_call_budget = 1000
expand_with_tools = true      # the fast model picks expand tools; false = deterministic heuristics

[web]
mode = "ask"                  # off | ask | auto
provider = "auto"             # tavily if TAVILY_API_KEY is set, else duckduckgo
```

Data lives in `~/.local/share/haive/` on Linux and `~/Library/Application Support/haive/` on macOS:
`haive.db` (the index, not a copy of your code), `checkpoints.db` (chat state), `models/` (the embedding
model, about 600 MB, downloaded once), `clones/`, and `chroma/`.

## Models

haive uses two OpenRouter roles, plus a fallback chain tried when a model keeps failing:

| Role | Default (free) | Alternative (free) | Used for |
|---|---|---|---|
| fast | `nvidia/nemotron-3-super-120b-a12b:free` | `google/gemma-4-26b-a4b-it:free` | router, grader, query rewrite, expand planner, per-repo summaries, chat titles |
| strong | `nvidia/nemotron-3-ultra-550b-a55b:free` | `poolside/laguna-s-2.1:free` | answers and debug diagnoses |
| strong (paid, off by default) | `deepseek/deepseek-v4-pro-0813` ($0.66 / $1.98 per 1M tokens) | `anthropic/claude-sonnet-5.5` ($2 / $10) | the same, when you pass `--paid` or set `use_paid_strong = true` |

Picks were checked against OpenRouter's live model list in October 2026; free models come and go, so
check the list before relying on them and use `haive eval --models fast=… strong=…` to compare. A
typical question makes 3 calls (router, grader, answer); `--paid` only changes the answer call, which
costs about $0.01 with DeepSeek at ~15k tokens of evidence. Paid models also avoid the free tier's
rate limits, and you can choose providers that don't train on prompts.

## How it works

Indexing discovers files with `git ls-files` (so `.gitignore` always applies), skips binaries, lockfiles,
generated files, and a **secrets denylist**, then chunks code with tree-sitter (one chunk per function,
class, or method in Python, JS/TS, Go, Rust, Java, C, C++, and C#) and config/docs by structure (Markdown
headings, YAML/JSON/TOML keys, compose services, Terraform blocks, SQL statements, shell functions).
Chunks are content-addressed, so re-indexing only re-embeds what changed. Commits are indexed by message,
author, date, and changed files; code-level history comes from `git log -S/-L` at question time.

Questions go through a LangGraph state machine: freshness check → router (fast model) → intent-specific
retrieval (code first: symbols, ripgrep, git; vectors after) → grader loop (expand, rewrite, or web) →
answer (strong model, structured output) → verification in code → finalize. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the graph and data flow.

**LangGraph** runs the whole flow: a `StateGraph` with conditional edges, `Send` fan-out for compare and
architecture, `interrupt`/`Command` for clarification and web approval, a `SqliteSaver` checkpointer for
resumable chats, and a tool-calling subgraph (`ToolNode` + `tools_condition`) for the expand step.
**LangChain** provides the pieces around it: `init_chat_model` with an `InMemoryRateLimiter`,
`.with_fallbacks()` across models, LCEL structured-output chains (native tool calling, then
`PydanticOutputParser` with an error-fed repair retry), `ChatPromptTemplate`s with a `MessagesPlaceholder`
for chat history, a callback handler that meters calls and records usage, `@tool`s over the code tools,
`Embeddings` over fastembed, and the `langchain-pinecone` / `langchain-chroma` vector stores.

## Privacy

Your code is embedded locally. Snippets relevant to each question are sent to the LLM provider you
configure (OpenRouter by default), and, with Pinecone, chunk text is stored in your Pinecone index as
metadata. Use Chroma for fully local storage. Secret files (`.env*`, keys, credentials) are never indexed
or sent anywhere, and chunks containing obvious tokens are dropped. Details: [docs/PRIVACY.md](docs/PRIVACY.md).

## Limitations

- **TypeScript source maps aren't read.** Traces pointing at compiled `dist/*.js` are matched to `.ts`
  sources by file stem and function name, which can pick the wrong file in unusual layouts.
- **No patch application.** Debug suggests a unified diff and checks that its context lines exist in the
  file (else it's labeled "approximate"), but never writes it.
- **No code execution**, no call-graph analysis, no MCP server or web UI, no native Windows.
- **Free-tier rate limits**: OpenRouter free models allow about 20 requests/minute and 1,000/day; haive
  throttles to stay under that, so answers can pause briefly under load.
- **Embedding speed on CPU**: the default code embedding model is accurate but heavy. Measured on a 4-core
  cloud container with no GPU: about 1.6 chunks/s, with a 3.7 GB peak (713 chunks in 7.5 minutes). The
  first index of a large repo takes a while; later runs only embed changed chunks. Lower
  `embeddings.batch_size` if memory is tight.
- Freshness re-indexing inline is limited to 50 changed files per repo per question.

## Eval

`haive eval` (from a source checkout) runs a question suite through the full graph and reports intent
accuracy, file hit rate (expected files among verified refs), retrieval recall@10 before any LLM step,
first-try citation validity, debug file+line accuracy (top hypothesis and any), LLM calls per question,
and p50/p95 latency. Ablations: `vector_only`, `ripgrep_only`, `hybrid`, `hybrid+loop` (default),
`pinecone` vs `chroma` (retrieval recall only), or `all`; `--models fast=… strong=…` sweeps models.

Debug cases come from **bug injection into temp copies, never your repos**: `eval/inject_bugs.py` clones a
repo, applies realistic bugs (wrong key, off-by-one, renamed env var, missing None check, bad import, …),
runs the entrypoint to capture a real error, and records the expected file, line, and cause.

```bash
haive eval --setup-sample            # bundled sample repo + 7 injected bugs (Python, Go, Java)
haive index eval-sample eval-sample-bugs
haive eval --suite ~/.local/share/haive/eval/questions.sample.yaml --ablation all
```

Your own suite goes in `eval/questions.yaml` (about 30 questions, 4–5 per intent, drafted from your
registered repos). Results land in `eval/results/<timestamp>/`, and the table below is regenerated from
the latest run, so it only ever shows measured numbers.

<!-- eval-results:start -->
_No eval has been run with real models yet. This table is filled in by `haive eval`._
<!-- eval-results:end -->

## Development

```bash
uv sync                 # dev dependencies included
uv run pytest           # tests never touch the network: LLMs, Pinecone, and web search are faked
uv run ruff check && uv run ruff format --check
```

`HAIVE_HOME=/some/dir` relocates config and data (handy for experiments). Progress and decisions are
tracked in [PROGRESS.md](PROGRESS.md) and [docs/DECISIONS.md](docs/DECISIONS.md); the original spec is
[docs/spec.md](docs/spec.md).

## License

MIT
