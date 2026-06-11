# IBM Cloud Code Engine — Energy Forecast Pipeline

Automated energy price forecasting pipeline running on IBM Cloud. Replaces an AWS Step Functions + SageMaker architecture with a simpler Python orchestrator on Code Engine.

> **Use case:** aFRR Capacity Price Forecasting — see the [top-level README](../README.md#use-case-afrr-capacity-price-forecasting) for details.

## Architecture

```
Code Engine Cron Subscriptions
         ↓
 ┌────────────────────────────┐
 │  Python Orchestrator       │
 │  (Code Engine Job)         │
 │                            │
 │  1. Load raw data from COS │
 │  2. Feature engineering    │
 │  3. Train XGBoost (weekly) │
 │  4. Generate forecasts     │
 │  5. Save results to COS    │
 └────────────────────────────┘
         ↓
 IBM Cloud Object Storage
 (data, models, forecasts)
```

## IBM Cloud Services Used

| Service                     | Purpose                        | Resource             |
| --------------------------- | ------------------------------ | -------------------- |
| Code Engine (Jobs)          | Compute (scheduled execution)  | Project: `energy-forecast` |
| Cloud Object Storage (COS)  | Store data, models, forecasts  | Bucket: `energy-forecast-bucket` |
| Container Registry          | Docker image storage           | Namespace: `energy-ml` |

## AWS → IBM Cloud Mapping

| AWS                  | IBM Cloud                                   |
| -------------------- | ------------------------------------------- |
| Step Functions       | Python orchestrator inside Code Engine job   |
| SageMaker Training   | XGBoost in Code Engine job                  |
| Lambda (inference)   | Code Engine job                             |
| EventBridge Schedule | Code Engine cron subscriptions              |
| S3                   | IBM Cloud Object Storage                    |

## Schedules

| Schedule             | Job                 | Cron          |
| -------------------- | ------------------- | ------------- |
| Daily inference      | `forecast-inference` | `0 8 * * *`  (08:00 UTC) |
| Weekly training      | `forecast-training`  | `0 6 * * 0`  (Sunday 06:00 UTC) |

## Quick Start (local development)

```bash
# Install dependencies with uv
uv sync

# Set environment variables
cp .env.example .env
# Fill in your IBM COS credentials

# Run full pipeline
uv run python -m src.orchestrator.main

# Run inference only
uv run python -m src.orchestrator.main --mode inference

# Run training only
uv run python -m src.orchestrator.main --mode training

# Run tests
uv run pytest tests/
```

## Deploy to IBM Cloud Code Engine

```bash
# 1. Login and target
ibmcloud login
ibmcloud target -g Test -r eu-de

# 2. Select the Code Engine project
ibmcloud ce project select --name energy-forecast

# 3. Build and push container image
ibmcloud ce buildrun submit \
  --name forecast-build \
  --source . \
  --strategy dockerfile \
  --image private.de.icr.io/energy-ml/energy-forecast:latest \
  --registry-secret icr-secret \
  --wait

# 4. Create secrets
ibmcloud ce secret create --name cos-credentials --from-env-file .env

# 5. Create the training job
ibmcloud ce job create --name forecast-training \
  --image private.de.icr.io/energy-ml/energy-forecast:latest \
  --registry-secret icr-secret \
  --env-from-secret cos-credentials \
  --argument "--mode" --argument "training" \
  --cpu 4 --memory 8G

# 6. Create the inference job
ibmcloud ce job create --name forecast-inference \
  --image private.de.icr.io/energy-ml/energy-forecast:latest \
  --registry-secret icr-secret \
  --env-from-secret cos-credentials \
  --argument "--mode" --argument "inference" \
  --cpu 2 --memory 4G

# 7. Schedule daily inference (08:00 UTC)
ibmcloud ce subscription cron create --name daily-inference \
  --destination-type job \
  --destination forecast-inference \
  --schedule "0 8 * * *"

# 8. Schedule weekly training (Sunday 06:00 UTC)
ibmcloud ce subscription cron create --name weekly-training \
  --destination-type job \
  --destination forecast-training \
  --schedule "0 6 * * 0"
```

## Operations

```bash
# List recent job runs
ibmcloud ce jobrun list

# Trigger a manual run
ibmcloud ce jobrun submit --job forecast-training
ibmcloud ce jobrun submit --job forecast-inference

# View logs
ibmcloud ce jobrun logs -f -n <run-name>

# Check schedules
ibmcloud ce subscription cron list
```

## COS Bucket Structure

```
energy-forecast-bucket/
├── raw/
│   └── electricity_prices_2026_06_02.parquet
├── features/
│   └── 2026_06_02.parquet
├── models/
│   ├── model_latest.pkl
│   └── 2026_06_02.pkl
└── forecasts/
    └── 2026_06_02.parquet
```

## Project Structure

```
ibm-energy-forecast-pipeline/
├── Dockerfile
├── pyproject.toml
├── uv.lock
├── .env.example
├── .ceignore
├── configs/
│   └── pipeline_config.json
├── scripts/
│   ├── deploy.sh
│   ├── generate_and_upload.py
│   └── fetch_real_data.py
├── src/
│   ├── orchestrator/
│   │   ├── main.py            ← entry point (replaces Step Functions)
│   │   └── config.py
│   ├── features/
│   │   └── feature_engineering.py
│   ├── training/
│   │   └── trainer.py
│   ├── inference/
│   │   └── predictor.py
│   └── storage/
│       └── cos_client.py      ← IBM COS wrapper (replaces boto3/S3)
└── tests/
    ├── test_features.py
    └── test_training.py
```

## Environment Variables

| Variable           | Description                               |
| ------------------ | ----------------------------------------- |
| `COS_ENDPOINT`     | IBM COS endpoint (`https://s3.eu-de.cloud-object-storage.appdomain.cloud`) |
| `COS_API_KEY`      | IBM Cloud API key                         |
| `COS_INSTANCE_CRN` | COS service instance CRN                 |
| `COS_BUCKET`       | Bucket name (`energy-forecast-bucket`)    |
| `PIPELINE_MODE`    | `full`, `training`, or `inference`        |
| `RETRAINING_DAY`   | Day of week for retraining (0=Mon, 6=Sun) |

## Next Steps

- [ ] Replace synthetic data with real electricity prices (ENTSO-E or internal source)
- [ ] Add more features (weather, load forecasts, etc.)
- [ ] Add model monitoring (forecast vs actuals comparison)
- [ ] Add Apache Airflow when scaling to multiple models
