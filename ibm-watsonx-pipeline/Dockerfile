FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for graphviz
RUN apt-get update && apt-get install -y --no-install-recommends graphviz && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Force CPU-only torch to avoid pulling ~1 GB of CUDA libraries
ENV UV_TORCH_BACKEND=cpu

# Copy shared library (dependency)
COPY shared/ /shared/

# Install dependencies (cached layer — only re-runs if lockfile changes)
COPY ibm-watsonx-pipeline/pyproject.toml ibm-watsonx-pipeline/uv.lock ibm-watsonx-pipeline/README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code
COPY ibm-watsonx-pipeline/src/ src/
COPY ibm-watsonx-pipeline/configs/ configs/

# Install the project itself
RUN uv sync --frozen --no-dev

# Default: run the full pipeline
ENTRYPOINT ["uv", "run", "python", "-m", "src.pipeline.orchestrator"]
