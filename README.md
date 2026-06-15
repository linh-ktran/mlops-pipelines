# MLOps Pipelines

Me experimenting with different MLOps platforms by building the same forecasting pipeline on each one. The goal is to get hands-on experience with real deployment patterns, not just notebook prototypes.

## The use case

All projects tackle the same problem: **forecasting French aFRR capacity prices** (EUR/MW), up to 4 days ahead. It's based on a real production system I worked on.

The model is XGBoost regression with:
- Lag features (1, 2, 3, 7, 14, 28 days) and 7-day rolling stats
- Exogenous inputs — FCR prices, consumption/gas/spot/solar/wind forecasts
- Calendar stuff — French holidays, weekends
- One model per horizon (so 4 models total)
- Chronological train/val split to avoid leakage

Nothing fancy ML-wise — the interesting part is how each platform handles the deployment and orchestration differently.

## What's in here

| Directory | Platform | What it does |
|-----------|----------|--------------|
| `aws-sagemaker-pipeline/` | AWS SageMaker + MLflow | Endpoints, Terraform, the whole AWS dance |
| `ibm-code-engine-pipeline/` | IBM Code Engine + COS | Simple Python orchestrator, no managed ML service |
| `ibm-watsonx-pipeline/` | IBM watsonx.ai + COS | DAG-based, model registry, monitoring |
| `shared/` | — | Common library so I don't repeat myself |

## Shared library (`mlops-core`)

All projects import from `shared/` to keep feature engineering, training, and inference logic in one place:

- `mlops_core.features` — cyclical time encoding, holidays, lags, rolling stats
- `mlops_core.training` — XGBoost train/eval with bias correction
- `mlops_core.storage` — IBM COS client
- `mlops_core.inference` — prediction with bias correction
- `mlops_core.testing` — shared pytest fixtures

Each project references it as an editable dependency:

```toml
dependencies = ["mlops-core[cos]"]

[tool.uv.sources]
mlops-core = { path = "../shared", editable = true }
```
