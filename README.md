# RecettesBot

A Telegram bot that saves recipes from any URL, searches them with AI-powered semantic similarity, and exports them as PDF recipe cards — entirely in French.

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Telegram Bot API](https://img.shields.io/badge/Telegram-Bot%20API-26A5E4?logo=telegram&logoColor=white)](https://core.telegram.org/bots/api)
[![Claude AI](https://img.shields.io/badge/Claude-Agent%20SDK-D97757?logo=anthropic&logoColor=white)](https://docs.anthropic.com/en/docs/agents-and-tools/claude-agent-sdk)
[![PostgreSQL 18](https://img.shields.io/badge/PostgreSQL-18%20%2B%20pgvector-4169E1?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)

---

## Features

- **AI recipe extraction** — send any URL and Claude extracts the recipe (title, ingredients, steps, servings), translated to French.
- **Semantic search** — find saved recipes by meaning, not just keywords, powered by pgvector HNSW indexes and multilingual sentence embeddings.
- **PDF export** — generate clean A4 recipe cards with DejaVu Sans for full Unicode support.
- **Servings adjustment** — scale ingredient quantities to any number of servings.
- **Shopping list** — generate an interactive Telegram poll from a recipe's ingredients.
- **Access control** — optionally restrict the bot to specific Telegram users or groups.
- **Observability** — optional Logfire (Pydantic) tracing across all handlers.

---

## Architecture

```
┌──────────────┐     ┌──────────────────────────────────────────┐
│  Telegram    │◄───►│  python-telegram-bot (polling)           │
│  User        │     │                                          │
└──────────────┘     │  bot.py ─── ConversationHandler states   │
                     │    │                                     │
                     │    ├── agent.py ── Claude Agent SDK       │
                     │    │               (WebFetch + JSON)      │
                     │    ├── embedding.py ── sentence-transformers
                     │    ├── pdf.py ── fpdf2                    │
                     │    └── db.py ── SQLModel + asyncpg        │
                     │                    │                      │
                     └────────────────────┼──────────────────────┘
                                          │
                              ┌───────────▼───────────┐
                              │  PostgreSQL 18        │
                              │  + pgvector 0.8       │
                              │  (HNSW cosine index)  │
                              └───────────────────────┘
```

### Tech stack

| Layer | Technology |
|-------|-----------|
| Bot framework | python-telegram-bot 22.7 (polling) |
| AI extraction | Claude Agent SDK 0.1.62 |
| Embeddings | sentence-transformers 5.4.1 (paraphrase-multilingual-mpnet-base-v2, 768-dim) |
| Tensor backend | PyTorch 2.11.0 (CPU-only build via `pytorch-cpu` uv index) |
| ORM | SQLModel 0.0.38 (SQLAlchemy + Pydantic) |
| Database | PostgreSQL 18 + pgvector 0.4.2 |
| Async DB driver | asyncpg 0.31.0 (runtime) |
| Sync DB driver | psycopg 3.3.3 (Alembic migrations) |
| Migrations | Alembic 1.18.4 |
| PDF generation | fpdf2 2.8.7 |
| Observability | Logfire 4.32.1 |
| Settings | pydantic-settings 2.13.1 |
| Package manager | uv |
| Linting | Ruff 0.15.11 |
| Type checking | ty 0.0.31 |
| Testing | pytest 9.0.3 + pytest-asyncio |

---

## Prerequisites

- **Python** >= 3.13.13
- **uv** — [install guide](https://docs.astral.sh/uv/getting-started/installation/)
- **Docker** and **Docker Compose** (v2)
- A **Telegram bot token** from [@BotFather](https://t.me/BotFather)
- An **Anthropic API key** from [console.anthropic.com](https://console.anthropic.com/)

---

## Quick start

```bash
# 1. Clone the repository
git clone https://github.com/j-about/RecettesBot.git
cd RecettesBot

# 2. Create your environment file
cp .env.example .env
# Edit .env — set TELEGRAM_BOT_TOKEN and ANTHROPIC_API_KEY at minimum

# 3. Start the stack
docker compose up -d --build
```

The bot service runs Alembic migrations automatically on startup, then begins polling Telegram.

---

## Configuration

All configuration is loaded from environment variables (or `.env`). See [.env.example](.env.example) for full documentation.

### Required

| Variable | Description |
|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather |
| `ANTHROPIC_API_KEY` | Anthropic API key (read directly by the Claude SDK). When unset, the entrypoint falls back to the Claude Code credentials file pointed to by `CLAUDE_CREDENTIALS_FILE`. |

### Database (set automatically by Docker Compose)

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_DB` | `rb_db` | PostgreSQL database name |
| `POSTGRES_USER` | `rb_user` | PostgreSQL user |
| `POSTGRES_PASSWORD` | `changeme` | PostgreSQL password (**change in production**) |

> `DATABASE_URL` and `DATABASE_URL_SYNC` are constructed by Docker Compose from the above variables. Only set them manually for non-Docker deployments.

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_CREDENTIALS_FILE` | `/dev/null` | Host path to a Claude Code `.credentials.json`, bind-mounted into the container as a fallback when `ANTHROPIC_API_KEY` is empty (e.g. `${HOME}/.claude/.credentials.json`) |
| `HF_TOKEN` | `""` | Hugging Face Hub token — enables higher rate limits and faster model downloads |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | HuggingFace model for title embeddings |
| `SEARCH_DISTANCE_THRESHOLD` | `0.65` | Cosine similarity minimum (0.0 = permissive, 1.0 = exact) |
| `SEARCH_RESULT_LIMIT` | `5` | Maximum recipes returned per search |
| `AGENT_TIMEOUT_SECONDS` | `120` | Timeout for Claude recipe extraction |
| `LOGFIRE_TOKEN` | `""` | Pydantic Logfire token (leave empty to disable) |
| `ALLOWED_USER_IDS` | `""` | Comma-separated Telegram user IDs |
| `ALLOWED_GROUP_IDS` | `""` | Comma-separated Telegram group IDs |

When both access-control lists are empty, the bot is open to everyone.

---

## Bot commands

| Command | Description |
|---------|-------------|
| `/ajouter [url]` | Extract and save a recipe from a URL |
| `/chercher [query]` | Search saved recipes by semantic similarity |
| `/personnes [n]` | Adjust servings for the selected recipe |
| `/courses` | Generate a shopping list poll from the selected recipe |
| `/annuler` | Cancel the current conversation |

Arguments shown in brackets are optional at invocation time — if omitted, the bot prompts the user for them in a follow-up message.

The bot registers these commands with Telegram on startup (for both private and group chats), so they appear in the `/` autocomplete menu automatically.

---

## Development

### Local setup

```bash
# Install all dependencies (including dev)
uv sync

# Set up environment variables
cp .env.example .env
# Fill in TELEGRAM_BOT_TOKEN, ANTHROPIC_API_KEY, DATABASE_URL, DATABASE_URL_SYNC
```

### Running tests

The test suite is organized into unit, integration, and end-to-end tests.

```bash
# Unit tests (no external services required)
uv run pytest tests/unit

# Integration tests (requires a running PostgreSQL instance with pgvector)
docker compose up -d db
uv run pytest tests/integration

# All tests with coverage
uv run pytest
```

### Linting and formatting

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run ty check
```

---

## Deployment

### Docker Compose (recommended)

The default `docker-compose.yml` defines two services:

- **db** — `pgvector/pgvector:0.8.2-pg18-trixie` with a persistent volume and health checks.
- **bot** — multi-stage build (uv builder + slim Python runtime), runs as unprivileged `app` user. Uses CPU-only PyTorch wheels to keep the runtime image small; the sentence-transformers model cache is persisted via a Docker volume.

```bash
docker compose up -d --build
docker compose logs -f bot
```

### Ansible

An Ansible playbook is provided for remote deployment.

```bash
# Set deployment target in .env
# HOST_IP, HOST_USER, HOST_SSH_PRIVATE_KEY_FILE

ansible-playbook playbooks/recettesbot_deployment.yaml
```

The playbook syncs the project to `/opt/recettesbot` on the target host and runs `docker compose up --build`.

---

## Project structure

```
RecettesBot/
├── main.py                  # Entry point
├── bot.py                   # Telegram handlers and conversation states
├── agent.py                 # Claude Agent SDK — recipe extraction
├── embedding.py             # Sentence-transformers vectorization
├── pdf.py                   # A4 PDF recipe card generation
├── models.py                # SQLModel ORM (Recipe, Ingredient)
├── db.py                    # Async engine and session factory
├── settings.py              # Pydantic Settings (env var validation)
├── migrations/              # Alembic schema migrations
│   └── versions/            # Migration scripts
├── tests/
│   ├── unit/                # Fast, no external dependencies
│   ├── integration/         # Requires PostgreSQL + pgvector
│   ├── e2e/                 # Full conversation flows
│   └── mocks/               # Fakes, stubs, and factories
├── Dockerfile               # Multi-stage build
├── docker-compose.yml       # PostgreSQL + bot orchestration
├── entrypoint.sh            # Credential setup + unprivileged exec
├── pyproject.toml           # Project metadata, tool config
├── uv.lock                  # Locked dependency tree
├── alembic.ini              # Alembic configuration
├── playbooks/               # Ansible deployment playbook
├── roles/                   # Ansible roles
├── inventory.yaml           # Ansible inventory
└── ansible.cfg              # Ansible configuration
```

---

## License

This project is licensed under the [MIT License](LICENSE).
