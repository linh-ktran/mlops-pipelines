FROM python:3.12-slim

WORKDIR /app

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dependencies (cached layer — only re-runs if lockfile changes)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code
COPY src/ src/
COPY configs/ configs/

# Install the project itself
RUN uv sync --frozen --no-dev

# Default: run the full pipeline
ENTRYPOINT ["uv", "run", "python", "-m", "src.orchestrator.main"]

