# IBM watsonx.ai — ML Pipeline with DAG Orchestration

Third iteration of the pipeline. This time on watsonx.ai, with a proper DAG executor, model registry, and online deployment. More "enterprise" than the Code Engine version — wanted to see what watsonx brings to the table.

> **Use case:** aFRR capacity price forecasting — details in the [top-level README](../README.md#the-use-case).

## What's different here

Compared to the Code Engine pipeline (which is just a sequential script), this one has:

- **A real DAG** — steps have explicit dependencies, topologically sorted
- **Model registry** — models go into watsonx.ai, not just a pickle in COS
- **Online deployment** — REST endpoint via Watson Machine Learning
- **Step tracking** — per-node status, timing, metadata
- **Resumability** — if it crashes mid-way, re-run skips completed steps
- **Monitoring** — forecast vs actuals comparison with alerting

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

## How it compares to the Code Engine version

| | Code Engine Pipeline | This one |
|--|---------------------|----------|
| Orchestration | Sequential script | DAG with topological sort |
| Model storage | Pickle in COS | COS + watsonx.ai registry |
| Deployment | Batch only | Online REST endpoint via WML |
| Visualization | None | Graphviz DAG rendering |
| State tracking | Logs | Per-node status + JSON state file |
| Resumability | Re-runs everything | Skips completed steps |

## Local development

```bash
uv sync

# credentials
cp .env.example .env
# fill in watsonx + COS creds

# see the DAG
uv run python -m src.pipeline.dag

# run everything
uv run python -m src.pipeline.orchestrator

# just training or inference
uv run python -m src.pipeline.orchestrator --mode training
uv run python -m src.pipeline.orchestrator --mode inference

# tests
uv run pytest tests/ -v

# generate sample data if you need it
uv run python scripts/generate_sample_data.py
```

## The DAG modes

**Training:**
```
load_data → feature_engineering → train_model → evaluate_model → save_to_cos → register_watsonx
```

**Inference:**
```
load_data → feature_engineering → load_model → predict → save_forecasts
```

**Full (train + deploy + predict):**
```
load_data → feature_engineering → train_model → evaluate_model → save_to_cos
                                                                      ↓
                                  register_watsonx → deploy_model → predict → save_forecasts
```

## Where models live

Two places:

1. **COS** — pickle files for quick local loading
   - `models/model_latest.pkl`
   - `models/2026_06_03.pkl`

2. **watsonx.ai registry** — for deployment and governance
   - Versioned assets with metadata (metrics, features, training date)
   - Deploy to online endpoint with one command

## Deploying

```bash
# automated
chmod +x scripts/deploy.sh
./scripts/deploy.sh

# or manually
uv run python scripts/watsonx_deploy_example.py
```

## IBM Cloud services

| Service | Purpose |
|---------|---------|
| watsonx.ai (WML) | Model registry + online endpoints |
| Cloud Object Storage | Data, models, forecasts, DAG state |
| Code Engine (Jobs) | Scheduled compute |
| Container Registry | Docker images |

## Things I want to try next

- watsonx.ai Prompt Lab for LLM-based anomaly explanations
- A/B testing with traffic splitting between model versions
- Watson OpenScale for fairness and drift monitoring
- Export the DAG as an Airflow DAG for enterprise scheduling
- Ensemble forecasting (multiple models, combined predictions)
