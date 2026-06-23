FROM python:3.12-slim

WORKDIR /app

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy shared library (dependency)
COPY shared/ /shared/

# Install dependencies (cached layer — only re-runs if lockfile changes)
COPY ibm-code-engine-pipeline/pyproject.toml ibm-code-engine-pipeline/uv.lock ibm-code-engine-pipeline/README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code
COPY ibm-code-engine-pipeline/src/ src/
COPY ibm-code-engine-pipeline/configs/ configs/

# Install the project itself
RUN uv sync --frozen --no-dev

# Default: run the full pipeline
ENTRYPOINT ["uv", "run", "python", "-m", "src.orchestrator.main"]

