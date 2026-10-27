# Privacy

haive is a local CLI, but answering questions means some of your code leaves your machine. This page
says exactly what goes where.

## Stays on your machine

All of this is in haive's data dir: `~/Library/Application Support/haive/` on macOS,
`~/.local/share/haive/` on Linux, or `$HAIVE_HOME/data/` if set.

- **Embeddings** are computed locally with fastembed. The model is downloaded once from Hugging Face
  into `models/`; your code is never sent there.
- **SQLite** (`haive.db`) holds file hashes, chunk text, commit metadata, repo profiles, chat thread
  metadata, and LLM usage counts. `checkpoints.db` holds chat history.
- **Chroma** (if you choose it) stores vectors and chunk text under `chroma/`.

## Leaves your machine

| Destination | What | When |
|---|---|---|
| **LLM provider** (OpenRouter by default) | Your question, recent chat turns, and the code/config/doc snippets, commit messages, and trace text retrieved for it | Every question (router, grader, expand planner, answer, titles). By default (`strong`) these go to paid models' providers; with `--tier free`, to free-model providers, some of which may log or train on prompts |
| **Pinecone** (default store) | Vectors plus chunk text and metadata (repo, path, symbol, lines) for everything indexed | `haive index` and freshness re-indexing |
| **Web search** (DuckDuckGo or Tavily) | A search query, which may include an error message fragment or a library name | Only when web mode allows (`ask` asks first; `off` never) |
| **GitHub / your git host** | Normal git traffic | `haive add <url>` (clone) and `haive sync` (pull) |

### Free models on OpenRouter (`--tier free`)

Free models are served by third-party providers whose data policies vary: **some may log prompts or
use them for training.** Since your code goes into prompts, review your OpenRouter privacy settings
(openrouter.ai/settings/privacy) and the policies of the providers behind the models in your
`config.toml` before indexing anything sensitive. You can point `models.base_url` at any
OpenAI-compatible endpoint, including a local one, to keep prompts on your machine.

### Fully local setup

Use `haive init --backend chroma`, a local OpenAI-compatible model server for `models.base_url`, and
`web.mode = "off"`. Then nothing leaves your machine except the one-time model download.

## Secrets

- **Denylist (never indexed, never read into a prompt, never returned by search):** `.env*` (except
  `.env.example`, `.env.sample`, `.env.template`), `*.pem`, `*.key`, `id_rsa*`/`id_dsa*`/`id_ecdsa*`/
  `id_ed25519*`, `*.p12`, `*.pfx`, `credentials*`, `secrets*`, keystores, `.npmrc`, `.pypirc`, `.netrc`.
- **Content scan:** file chunks that contain obvious tokens are dropped: AWS keys,
  `sk-…` style API keys, private key headers, GitHub/Slack/Google/Pinecone/Tavily tokens. Commit
  messages are not scanned (secrets there are rare and git hosts already flag them), so they are
  indexed, and with Pinecone uploaded, as written.
- `.gitignore` is always respected, so ignored files are never indexed.
- haive's own keys live in your environment or the `.env` next to `config.toml` (mode 600), never in
  `config.toml` itself.

These are heuristics. Don't rely on them as your only protection for repos full of real secrets.

## Read-only guarantee

haive never writes to, stages, commits, or executes anything inside an indexed repo. Git access goes
through a wrapper that only allows read-only subcommands, with `GIT_OPTIONAL_LOCKS=0` so even
`git status` doesn't refresh the index file. The two exceptions are `git clone` into haive's own data
dir and `git pull --ff-only` in `haive sync`. A test asserts that no indexed repo's `git status` or file
mtimes (including `.git/`) change after a full run. The eval harness runs code only in temporary
copies it creates.
