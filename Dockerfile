# ---- builder stage --------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim AS builder

ENV UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Install dependencies first (cached unless pyproject.toml / uv.lock change).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Copy source and install the project itself.
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# ---- runtime stage --------------------------------------------------------
FROM python:3.13-slim-trixie

# gosu for dropping root; fonts-dejavu-core for Unicode PDF rendering.
RUN apt-get update && apt-get install -y --no-install-recommends gosu fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# sentence-transformers model cache lives here at runtime.
ENV HF_HOME=/opt/hf-cache

WORKDIR /app

# Copy the entire app directory (venv + source + migrations + alembic.ini).
COPY --from=builder /app /app

# Activate the virtual environment via PATH (no uv in runtime image).
ENV PATH="/app/.venv/bin:$PATH"

# Create unprivileged user with a real home directory for ~/.claude.
RUN groupadd --system app && useradd --system --gid app --create-home app \
    && mkdir -p "$HF_HOME" && chown app:app "$HF_HOME"

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
# Run Alembic migrations then start the bot.
CMD ["sh", "-c", "alembic upgrade head && python -m main"]
