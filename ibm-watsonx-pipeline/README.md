# IBM watsonx.ai — ML Pipeline with DAG Orchestration

Machine learning pipeline on IBM watsonx.ai with model registry integration, and online deployment.

> **Use case:** aFRR Capacity Price Forecasting — see the [top-level README](../README.md#use-case-afrr-capacity-price-forecasting) for details.

## Key Features

| Feature | Description |
|---------|-------------|
| **Model Registry** | Store and version models in watsonx.ai |
| **Online Deployment** | Deploy models as REST endpoints via Watson Machine Learning |
| **Step Tracking** | Per-node status, timing, and metadata |
| **Monitoring** | Forecast vs actuals comparison with alerting |
| **Resumability** | Skip already-completed steps on re-run |

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                    Pipeline DAG Executor                          │
│                                                                   │
│  load_data → feature_engineering → train_model → evaluate_model   │
│                                                        ↓          │
│              save_to_cos → register_watsonx → deploy_model        │
│                                                    ↓              │
│                              predict → save_forecasts             │
└───────────────────────────────────────────────────────────────────┘
         ↓                              ↓
  IBM Cloud Object Storage       IBM watsonx.ai
  (data, models, forecasts)      (model registry + deployment)
```

## IBM Cloud Services Used

| Service | Purpose | Resource |
|---------|---------|----------|
| watsonx.ai (WML) | Model registry & online deployment | Project: `watsonx-pipeline` |
| Cloud Object Storage | Store data, models, forecasts, DAG state | Bucket: `watsonx-pipeline-bucket` |
| Code Engine (Jobs) | Compute (scheduled execution) | Project: `watsonx-pipeline` |
| Container Registry | Docker image storage | Namespace: `watsonx-ml` |

## Comparison: Code Engine Pipeline vs watsonx Pipeline

| Aspect | Code Engine Pipeline | watsonx Pipeline |
|--------|---------------------|------------------|
| Orchestration | Sequential Python script | Explicit DAG with topological sort |
| Model Storage | Pickle in COS only | COS + watsonx.ai model registry |
| Deployment | N/A (batch only) | Online REST endpoint via WML |
| Visualization | None | Graphviz DAG rendering |
| State Tracking | Logs only | Per-node status + JSON state file |
| Resumability | Full re-run | Skip completed steps |

## Quick Start (local development)

```bash
# Install dependencies with uv
uv sync

# Set environment variables
cp .env.example .env
# Fill in your IBM watsonx + COS credentials

# View the pipeline DAG
uv run python -m src.pipeline.dag

# Run full pipeline
uv run python -m src.pipeline.orchestrator

# Run training only
uv run python -m src.pipeline.orchestrator --mode training

# Run inference only
uv run python -m src.pipeline.orchestrator --mode inference

# Run tests
uv run pytest tests/ -v

# Generate sample data
uv run python scripts/generate_sample_data.py
```

## Pipeline DAG Modes

### Training DAG
```
load_data → feature_engineering → train_model → evaluate_model → save_to_cos → register_watsonx
```

### Inference DAG
```
load_data → feature_engineering → load_model → predict → save_forecasts
```

### Full DAG
```
load_data → feature_engineering → train_model → evaluate_model → save_to_cos
                                                                      ↓
                                  register_watsonx → deploy_model → predict → save_forecasts
```

## Model Saving & Registry

Models are saved in two locations:

1. **IBM COS** — Pickle file for fast local loading
   - `models/model_latest.pkl` (always the most recent)
   - `models/2026_06_03.pkl` (dated version)

2. **watsonx.ai Model Registry** — For deployment and governance
   - Versioned model assets
   - Metadata (metrics, feature names, training date)
   - One-click deployment to online endpoint

## Deploy to IBM Cloud

```bash
# Using the deployment script
chmod +x scripts/deploy.sh
./scripts/deploy.sh

# Or manually deploy a model to watsonx.ai
uv run python scripts/watsonx_deploy_example.py
```

## Next Steps

- [ ] Add watsonx.ai Prompt Lab integration for LLM-based anomaly explanation
- [ ] Add model A/B testing with traffic splitting
- [ ] Add Watson OpenScale for model fairness & drift monitoring
- [ ] Add Apache Airflow DAG export for enterprise scheduling
- [ ] Add multi-model pipeline (ensemble forecasting)

