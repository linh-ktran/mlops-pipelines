FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy shared library (dependency)
COPY shared/ /shared/

COPY aws-sagemaker-pipeline/pyproject.toml aws-sagemaker-pipeline/uv.lock aws-sagemaker-pipeline/README.md ./
COPY aws-sagemaker-pipeline/src ./src
COPY aws-sagemaker-pipeline/configs ./configs

RUN uv sync --frozen --no-dev

ENV PYTHONPATH=/app/src

EXPOSE 8000
CMD ["uv", "run", "python", "-m", "mlops_serving_starter.api.app"]
