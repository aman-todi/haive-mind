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
| recall | "How did I implement OAuth before?" (searches every repo, or the ones you name, and keeps the ones whose code best matches) |
| history | "When and why did I add retries to the API client?" / "What changed in weatherpruf in the last month?" (commits, `git log -S/-L`; time windows like "last 2 weeks", "since August", "last 20 commits", defaulting to 2 weeks) |
| debug | `haive debug trace.txt`: ranked hypotheses, evidence, a suggested diff (never applied), how to verify |
| compare | "How do weatherpruf and the valuation app differ on deploy?" |
| architecture | "Break down the valuation project" (a repo or a group: components, connections, data flow, deploy) |
| general | "How do I paginate in FastAPI?" (not about your repos; web if allowed) |

Debug understands Python (including chained exceptions), Node/V8, Java/Kotlin, Go panics, Rust panics and
backtraces, .NET, C/C++ sanitizer and gdb output, and compiler/linter lines (gcc/clang, tsc, rustc,
MSBuild, mypy/ruff/eslint style). Anything else falls back to a generic `path:line` scan.

Vague questions: when a reasonable reading exists, haive picks it and says so ("Assumed: …",
"History window: the last 2 weeks"). Only when the answers would differ completely ("is it slow?") does
it ask, with a few options; pick one by number or type your own. Non-interactive runs answer as asked
and note what they would have asked.

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
2. **OpenRouter**: set `OPENROUTER_API_KEY` and add a few dollars of credit (paid models are the
   default; see [Models](#models) for the cost, or set `use_paid = false` for free models only), and
   **review OpenRouter's privacy and data-policy settings**: some providers may log or train on prompts,
   and your code goes into prompts. See [docs/PRIVACY.md](docs/PRIVACY.md).
3. **Model IDs**: check that the default models in `config.toml` still exist on
   [openrouter.ai/models](https://openrouter.ai/models), or pick replacements. `haive eval --model …` helps.
4. **Optional Tavily**: set `TAVILY_API_KEY` for better web results (DuckDuckGo is used otherwise).
5. **Install ripgrep**: `brew install ripgrep` or `apt install ripgrep` (a slow Python fallback exists).
6. **Register repos** and define groups (below).
7. **First full index**: run it and report anything wrong (missed files, bad chunks, slowness).

Keys can be exported in your shell or put in the `.env` next to `config.toml` (created by `haive init`,
mode 600; see Configuration for where that is).
They are never written to `config.toml`.

## Quickstart

```bash
haive init                                   # config + data dirs, validates keys, creates the index
haive add ~/code/intrinsic-valuation --name valuation-backend -g valuation
haive add https://github.com/you/weatherpruf  # URLs are cloned into haive's data dir
haive add --scan ~/code                      # or register every repo in a folder at once (skips haive's own clone)
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
haive add --scan <dir>... [-g G]            register every git repo directly inside <dir>, except haive's own source
haive remove <name>                         unregister a repo and delete its vectors and rows
haive repos                                 path, last indexed, chunk/commit counts, stale?
haive groups                                groups and their repos (edit groups in config.toml)
haive index [names...] [--full]             no names = all registered repos; incremental unless --full
haive sync                                  git pull --ff-only on every repo, then incremental index
haive status                                vector store, model tiers, today's LLM usage, index health

haive ask "<question>"  [-r REPO]... [-g GROUP]... [--web off|ask|auto] [--json] [--trace] [--all] [--strong|--ultra] [--paid|--free]
haive debug [FILE|-]    [-q "extra question"] [-r/-g ...] [--web ...] [--json] [--trace] [--all] [--strong|--ultra] [--paid|--free]
haive chat              [--resume ID] [-r/-g ...] [--web ...] [--all] [--strong|--ultra] [--paid|--free]
haive threads           [--limit N]
haive threads delete <ID>
haive eval              [--suite FILE] [--ablation NAME] [--model MODEL] [--limit N] [--setup-sample]
```

- Every answer starts with its response time (time spent at clarify or web prompts is not counted).
- `--trace` prints the path through the graph: nodes visited, tools called with timing, LLM calls by step
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

`config.toml` (with `.env` beside it) lives in `~/.config/haive/` on Linux and
`~/Library/Application Support/haive/` on macOS, or in `$HAIVE_HOME/config/` if set. haive writes
every setting, but those left at their defaults are commented out (`# max_claims = 10`): uncomment a
line to change it. Commented defaults update with haive; the embedding model and vector store are
always written out, since the index was built with them. Every setting, with its default:

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
tier = "strong"               # or "ultra"; per command: --strong / --ultra
use_paid = true               # per command: --paid / --free
# One list answers every step of a question; later entries are fallbacks, in order.
strong = ["nvidia/nemotron-3.5-super-vl:free", "google/gemma-4-26b-a4b-it:free",
          "google/gemma-4-31b-it:free", "openrouter/free"]
ultra = ["nvidia/nemotron-3-ultra-550b-a55b:free", "thinkingmachines/inkling:free",
         "poolside/laguna-s-2.1:free", "openrouter/free"]
strong_paid = ["deepseek/deepseek-v4.1-flash-1017", "openai/gpt-oss-120b", "openrouter/auto"]
ultra_paid = ["deepseek/deepseek-v4-pro-0813", "z-ai/glm-5.2", "openrouter/auto"]
routing = { sort = "price", preferred_max_latency = { p90 = 10 }, preferred_min_throughput = { p50 = 50 } }
requests_per_minute = 20      # client-side throttle
max_retries = 1               # per model (honors Retry-After) before falling back
timeout_seconds = 125         # wall-clock cap on an answer call, then the next model
step_timeout_seconds = 30     # the same for router, grader, rewrite, expand, titles
free_timeout_factor = 2       # free models get twice these limits
base_url = "https://openrouter.ai/api/v1"

[limits]
max_llm_calls_per_question = 25
max_grade_loops = 2
max_answer_retries = 2
max_claims = 10               # per answer: asked of the model and shown (--all shows the rest)
daily_call_budget = 1000
expand_with_tools = true      # the model picks expand tools; false = deterministic heuristics

[web]
mode = "ask"                  # off | ask | auto
provider = "auto"             # tavily if TAVILY_API_KEY is set, else duckduckgo
```

Data lives in `~/.local/share/haive/` on Linux and `~/Library/Application Support/haive/` on macOS
(`$HAIVE_HOME/data/` if set):
`haive.db` (the index, not a copy of your code), `checkpoints.db` (chat state), `models/` (the embedding
model, about 600 MB, downloaded once), `clones/`, and `chroma/`.

## Models

One model answers every step of a question (router, grader, expand planner, answer, titles). You pick
a **tier**: strong or ultra (bigger, slower), paid or free. Each tier is an ordered list; the first model that works is
used, and a model that fails a call (timeout, rate limit, server error) is skipped for the rest of that
question, so a question never mixes models by accident. A model that answers 404 (removed, or its only
provider down) is skipped for the rest of the process.

| Tier | 1st | 2nd | 3rd | Last resort |
|---|---|---|---|---|
| strong-paid (default) | `deepseek/deepseek-v4.1-flash-1017` ($0.15 / $0.60 per 1M tokens) | `openai/gpt-oss-120b` ($0.15 / $0.60) | | `openrouter/auto` |
| ultra-paid (`--ultra`) | `deepseek/deepseek-v4-pro-0813` ($0.66 / $1.98) | `z-ai/glm-5.2` ($0.62 / $2.42) | | `openrouter/auto` |
| strong, free (`--free`) | `nvidia/nemotron-3.5-super-vl:free` | `google/gemma-4-26b-a4b-it:free` | `google/gemma-4-31b-it:free` | `openrouter/free` |
| ultra, free (`--free --ultra`) | `nvidia/nemotron-3-ultra-550b-a55b:free` | `thinkingmachines/inkling:free` | `poolside/laguna-s-2.1:free` | `openrouter/free` |

**Paid or free?** On the author's 19-question suite (October 2026), strong-paid and strong-free were
about equally accurate (100% vs 94% file hits, 99% vs 98% first-try citations), but paid answered in
**13.5 s at the median (29 s p95)** against free's **63 s (176 s p95)**, using 3.6 LLM calls a question
instead of 5.3 (free models fail over more often). The whole suite cost about 1-2 cents on paid. Free
costs nothing, but expect a minute a question, OpenRouter's free rate limits, and providers that may
train on prompts.

Each paid model is served by many providers on OpenRouter, and its default routing favors the cheapest,
which is occasionally one that stalls for minutes. haive asks OpenRouter (`models.routing`) to prefer
providers that start answering within 10 s (p90) and stream at least 50 tokens/s (median) over the
last 5 minutes, and to pick the cheapest of those; slower providers are tried only after them. Set
`routing = {}` for OpenRouter's default. If a call still stalls, haive abandons it after
`models.timeout_seconds` (125 s) for an answer, or `step_timeout_seconds` (30 s) for the short steps
that normally take 1-5 s (twice as long for free models, `free_timeout_factor`), and the next model in
the tier answers the rest of the question.

Answers are capped at `limits.max_claims` (10) claims: the model merges related points, and a history
answer groups related commits into one claim per piece of work. The terminal shows at most that many (spread across sections); `--all`
shows any extra, and `--json` always includes every claim.

Fallbacks are mostly served by other providers than the model before them, so one provider's outage
rarely takes a tier down. Free picks favor speed over size: Nemotron 3.5 Super VL (12B active of 120B)
and Gemma 4 26B A4B (4B active, faster than the dense 31B at similar quality). Picks were checked against OpenRouter's live model list and provider uptime in
October 2026; free models come and go, so check the list before relying on them, and use
`haive eval --model …` to compare models on your own questions. A typical question makes 3-4 calls
(router, grader, answer); on strong-paid that is well under a cent, on ultra-paid roughly $0.01-0.03. Paid models also avoid the free tier's rate limits, and you can choose providers that don't
train on prompts. `openrouter/auto` picks a paid model for you, so its cost per call varies.

## How it works

Indexing discovers files with `git ls-files` (so `.gitignore` always applies), skips binaries, lockfiles,
generated files, and a **secrets denylist**, then chunks code with tree-sitter (one chunk per function,
class, or method in Python, JS/TS, Go, Rust, Java, C, C++, and C#) and config/docs by structure (Markdown
headings, YAML/JSON/TOML keys, compose services, Terraform blocks, SQL statements, shell functions).
Chunks are content-addressed, so re-indexing only re-embeds what changed. Commits are indexed by message,
author, date, and changed files; code-level history comes from `git log -S/-L` at question time.

Questions go through a LangGraph state machine: router → freshness check of the repos in scope →
intent-specific retrieval (code first: symbols, ripgrep, git; vectors after) → grader loop (expand,
rewrite, or web) → answer (structured output) → verification in code → finalize. One model list (the
tier) serves every step. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the graph and data flow.

**LangGraph** runs the whole flow: a `StateGraph` with conditional edges, `Send` fan-out for compare and
architecture, `interrupt`/`Command` for clarification and web approval, a `SqliteSaver` checkpointer for
resumable chats, and a tool-calling subgraph (`ToolNode` + `tools_condition`) for the expand step.
**LangChain** provides the pieces around it: `init_chat_model` with an `InMemoryRateLimiter`,
`.with_fallbacks()` down each tier's model list, LCEL structured-output chains (native tool calling, then
`PydanticOutputParser` with an error-fed repair retry), `ChatPromptTemplate`s with a `MessagesPlaceholder`
for chat history, a callback handler that meters calls and records usage, `@tool`s over the code tools,
`Embeddings` over fastembed, and the `langchain-pinecone` / `langchain-chroma` vector stores.

## Privacy

Your code is embedded locally. Snippets relevant to each question are sent to the LLM provider you
configure (OpenRouter by default), and, with Pinecone, chunk text is stored in your Pinecone index as
metadata. Use Chroma for fully local storage. Secret files (`.env*`, keys, credentials) are never indexed
or sent anywhere, and file chunks containing obvious tokens are dropped. Commit messages are indexed
without a token scan. Details: [docs/PRIVACY.md](docs/PRIVACY.md).

## Limitations

- **TypeScript source maps aren't read.** Traces pointing at compiled `dist/*.js` are matched to `.ts`
  sources by file stem and function name, which can pick the wrong file in unusual layouts.
- **No patch application.** Debug suggests a unified diff and checks that its context lines exist in the
  file (else it's labeled "approximate"), but never writes it.
- **No code execution**, no call-graph analysis, no MCP server or web UI, no native Windows.
- **Free-tier rate limits** (with `--free`): OpenRouter free models allow about 20 requests/minute and
  1,000/day; haive throttles to stay under that, so answers can pause briefly under load.
- **Embedding speed on CPU**: the default code embedding model is accurate but heavy. Measured on a 4-core
  cloud container with no GPU: about 1.6 chunks/s, with a 3.7 GB peak (713 chunks in 7.5 minutes). The
  first index of a large repo takes a while; later runs only embed changed chunks. Lower
  `embeddings.batch_size` if memory is tight.
- Freshness re-indexing inline is limited to 50 changed files per repo per question.

## Eval

`haive eval` (from a source checkout) runs a question suite through the full graph and reports intent
accuracy, file hit rate (expected files among verified refs), retrieval recall@10 before any LLM step,
first-try citation validity, debug file+line accuracy (top hypothesis and any), LLM calls per question,
and p50/p95 latency, plus first-try verification failures grouped by kind and the questions over 60 s with
their slowest LLM call (each question's answer and full trace are saved to `<ablation>_traces.jsonl`). Ablations: `vector_only`,
`ripgrep_only`, `hybrid`, `hybrid+loop` (default), `pinecone` vs `chroma` (retrieval recall only), or
`all`. `--paid/--free` and `--strong/--ultra` pick the tier; `--model …` runs the suite on one model
with no fallbacks, to compare models.

Debug cases come from **bug injection into temp copies, never your repos**: `eval/inject_bugs.py` clones a
repo, applies realistic bugs (wrong key, off-by-one, renamed env var, missing None check, bad import, …),
runs the entrypoint to capture a real error, and records the expected file, line, and cause.

```bash
haive eval --setup-sample            # bundled sample repo + 7 injected bugs (Python, Go, Java)
haive index eval-sample eval-sample-bugs
haive eval --suite "<data dir>/eval/questions.sample.yaml" --ablation all   # path printed by --setup-sample
```

Your own suite goes in `eval/questions.yaml` (about 30 questions, 4–5 per intent, drafted from your
registered repos). Results land in `eval/results/<timestamp>/`, and the table below is regenerated from
the latest run, so it only ever shows measured numbers.

<!-- eval-results:start -->
Measured 2026-10-26 on `questions.yaml` (19 questions; model=deepseek/deepseek-v4.1-flash-1017; store=pinecone; embeddings=jinaai/jina-embeddings-v2-base-code).

| ablation | intent_accuracy | file_hit_rate | recall@10 | citation_validity | debug_top1 | debug_any | llm_calls_mean | latency_p50_s | latency_p95_s | questions | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid+loop | 100% | 100% | 100% | 100% | – | – | 3.32 | 15.42 | 29.47 | 19 | 0 |
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
