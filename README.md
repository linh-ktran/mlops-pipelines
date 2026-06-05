# MLOps Pipelines

A collection of ML pipeline experiments to learn and test different MLOps platforms and techniques.

## Projects

| Directory | Platform | Description |
|-----------|----------|-------------|
| `aws-sagemaker-pipeline/` | AWS SageMaker + MLflow | SageMaker endpoints, Terraform infra |
| `ibm-code-engine-pipeline/` | IBM Code Engine + COS | Lightweight Python orchestrator |
| `ibm-watsonx-pipeline/` | IBM watsonx.ai + COS | DAG orchestration, model registry, monitoring |
| `shared/` | — | `mlops-core` library (features, training, storage, inference) |

## Shared Library

All projects depend on `shared/` (`mlops-core`) for common logic:

- **`mlops_core.features`** — cyclical time, holidays, lags, rolling stats
- **`mlops_core.training`** — XGBoost train/evaluate with bias correction
- **`mlops_core.storage`** — IBM COS client
- **`mlops_core.inference`** — prediction with bias correction
- **`mlops_core.testing`** — shared pytest fixtures

Usage in each project's `pyproject.toml`:

```toml
dependencies = ["mlops-core[cos]"]

[tool.uv.sources]
mlops-core = { path = "../shared", editable = true }
```
