# Progress

Status of the build against [docs/spec.md](docs/spec.md); design changes since the spec are logged in
[docs/DECISIONS.md](docs/DECISIONS.md).

## Done

- T0–T14 (spec §15): foundation, indexing (code, config, docs, commit messages), vector stores
  (Pinecone, Chroma), read-only tools, trace parsing, the LLM layer, provenance checks, the graph for
  every intent, debug, compare and architecture, chat with threads and interrupts, web search, and
  the eval harness.
- Running on the author's machine against 13 real repos (about 9,000 vectors in Pinecone), on free
  and paid OpenRouter models. Changes driven by those runs: commit citations, history time windows,
  clarify-or-assume for vague questions, response time above answers, freshness after the router,
  one model list per question by tier, and skipping models that fail or disappear (DECISIONS #25–#38).
- CI: `ruff` + `pytest` on Ubuntu and macOS, Python 3.11 and 3.12. Tests never touch the network;
  the fake LLM is a real `BaseChatModel`, and a CLI test asserts indexed repos are left untouched.

## Open

- **T15 hardening:** draft `eval/questions.yaml` from the author's repos (it is still the empty
  template), run `haive eval --ablation all`, put the numbers in the README, and fix what they show.
- Measure `ask` latency per tier over a question set (single runs so far: ~27 s paid fast/strong
  mix, ~60 s free).

## Measurements

| What | Result |
|---|---|
| Freshness check, one repo (~110 files) | p50 16 ms (target ≤ 200 ms) |
| Index on CPU, fastembed `jina-embeddings-v2-base-code` | ~1.6 chunks/s, peak RSS 3.7 GB (4-core container) |
| History question, weatherpruf, `--paid` | 27 s, 3 LLM calls, all claims verified |

## Known issues / risks

- Free models come and go, and a free model with a single provider fails when that provider does
  (DECISIONS #37, #38); each tier's list falls back across providers.
- CPU embedding is slow on large repos; changing the embedding model needs a full re-index.
- `tree-sitter` is pinned below 0.26 (0.26.0 segfaulted; DECISIONS #3).
