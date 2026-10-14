# Progress

## Tickets

| Ticket | Status | Notes |
|---|---|---|
| T0 Foundation | ✅ done | pyproject, paths, config, db + migrations, types, fixtures (temp git repo, fake LLM), CI |
| T1 Discovery + chunking + profile | ✅ done | 9 tree-sitter languages; structural config/docs; secrets denylist + content scan |
| T2 Commit indexing | ✅ done | last N, small-diff rule, incremental `rev-list`, rebase rebuild, pruning |
| T3 Embeddings + vector stores | ✅ done | fastembed + hash embedder; Pinecone (mocked in tests), Chroma, memory |
| T4 Deterministic tools | ✅ done | read-only git wrapper, ripgrep + fallback, file windows, web (ddgs/Tavily) |
| T5 Trace parsing + path mapping | ✅ done | Python, Node, Java/Kotlin, Go, Rust, .NET, sanitizer/gdb, compilers, generic |
| T6 LLM layer | ✅ done | roles + fallback, structured output + repair, 429 backoff, throttle, usage |
| T7 Provenance | ✅ done | claim/ref/diff verification; rich + JSON rendering |
| T8 Index pipeline + repo commands | ✅ done | incremental upsert/delete, renames, progress, freshness check |
| T9 Graph core + `ask` | ✅ done | every conditional edge covered by graph tests with the fake LLM |
| T10 Debug | ✅ done | parse → map → parallel gather → diagnose → diff check; stdin |
| T11 Compare + architecture + groups | ✅ done | `Send` fan-out, per-repo summaries, connection mapping |
| T12 Chat, threads, interrupts | ✅ done | slash commands, titles, resume, clarify, non-TTY defaults |
| T13 Web node + general intent | ✅ done | off/ask/auto, approval interrupt, `need_external` |
| T14 Eval + docs | ✅ done (harness) | metrics, ablations, bug injection, README/ARCHITECTURE/PRIVACY. No real-model results yet |
| T15 Hardening | ⏳ blocked on you | needs keys + your repos (see below) |

## Verified so far

- `uv run pytest`: 130 tests, all passing on Python 3.13 and 3.11. Tests never touch the network.
- `uv run ruff check` and `ruff format --check`: clean.
- Read-only guarantee: a CLI test runs index, ask, debug, and chat and asserts no file in the indexed repo
  (including `.git/`) changed mtime and `git status` is unchanged.
- Bug injection on the sample repo produces real traces for 7 bugs (Python ×5, Go, Java); the source repo
  stays clean.
- `uv build` produces a wheel that includes the prompt templates; `haive --help` lists every command.

## Measurements (this cloud container: 4 CPU cores, no GPU)

| What | Result |
|---|---|
| Freshness check, this repo (114 files) | p50 16 ms, max 26 ms (target ≤ 200 ms) |
| Index with fastembed `jina-embeddings-v2-base-code` + Chroma, this repo | 713 chunks in 450 s (1.6 chunks/s), peak RSS 3.7 GB |
| Index with the hash embedder (pipeline overhead only) | 114 files / 710 chunks in 0.6 s |
| `ask` p50 latency with real models | not measured yet (needs keys) |

## Blocked: needs you (spec §16)

1. Set `PINECONE_API_KEY` (or choose `haive init --backend chroma`) and `OPENROUTER_API_KEY`.
2. Confirm the default free model IDs still exist on openrouter.ai/models, and review the free-model data
   policies.
3. Optionally set `TAVILY_API_KEY`; install ripgrep on your machine.
4. Register your repos and groups, run the first full index, and report anything off.
5. Optionally share 2–3 real error traces for the debug eval.
6. Then T15: draft `eval/questions.yaml` from your repos, run `haive eval --ablation all`, measure `ask`
   p50, and fix what your feedback turns up.

## Known issues / risks

- The default free model IDs in the spec (`google/gemma-4-26b-a4b-it:free`,
  `nvidia/nemotron-3-ultra-550b-a55b:free`) haven't been checked against OpenRouter from here (no key).
- CPU embedding with the default model is slow on large repos (see measurements). A GPU or a smaller
  model would help; changing the model means a new index (dimension must match).
- `tree-sitter` is pinned below 0.26 because 0.26.0 segfaulted (see DECISIONS #3).
- Native structured output (function calling) isn't supported by every free model; haive falls back to
  JSON prompting automatically, which can cost an extra call the first time per model.
