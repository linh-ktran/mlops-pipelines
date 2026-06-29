FROM python:3.12-slim

LABEL maintainer="mlops-team" \
      description="MLOps serving API for electricity price forecasting" \
      org.opencontainers.image.source="https://github.com/mlops-pipelines"

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy shared library (dependency)
COPY shared/ /shared/

# Install dependencies first (better layer caching)
COPY aws-sagemaker-pipeline/pyproject.toml aws-sagemaker-pipeline/uv.lock aws-sagemaker-pipeline/README.md ./
RUN uv sync --frozen --no-dev --no-install-project || true

COPY aws-sagemaker-pipeline/src ./src
COPY aws-sagemaker-pipeline/configs ./configs

RUN uv sync --frozen --no-dev

ENV PYTHONPATH=/app/src

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uv", "run", "python", "-m", "mlops_serving_starter.api.app"]
