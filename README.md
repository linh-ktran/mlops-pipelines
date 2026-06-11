# MLOps Pipelines

A collection of ML pipeline experiments to learn and test different MLOps platforms and techniques.

## Use Case: aFRR Capacity Price Forecasting

All projects implement the same use case — forecasting **French aFRR capacity prices** using XGBoost time-series regression, inspired by a real production pipeline.

**Problem:** Predict hourly capacity prices (EUR/MW) for up to 4 days ahead (horizons 1–4).

**Approach:**
- XGBoost regression with early stopping and bias correction
- Lag features (1, 2, 3, 7, 14, 28 days), rolling statistics (7-day mean/min/max)
- Exogenous inputs: FCR prices, consumption/gas/spot/solar/wind forecasts
- Calendar features: French holidays, weekends
- Per-horizon models (each horizon has its own trained model)
- Chronological train/val split (no data leakage)

Each project deploys the same model on a different platform to compare MLOps tooling and infrastructure patterns.

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
