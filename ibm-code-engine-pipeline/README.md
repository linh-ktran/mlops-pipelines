# IBM Code Engine — Energy Forecast Pipeline

Same forecasting model as the AWS version, but deployed on IBM Cloud with a much simpler setup. No managed ML service — just a Python script running as a Code Engine job, reading and writing to Cloud Object Storage.

The idea was to see how far you can get without something like SageMaker or Step Functions.

> **Use case:** aFRR capacity price forecasting — details in the [top-level README](../README.md#the-use-case).

## How it works

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

Pretty straightforward. A cron triggers the job, the job does everything sequentially, results go to COS. No DAG framework, no orchestrator service — just Python.

## Services used

| Service | What for |
|---------|----------|
| Code Engine (Jobs) | Running the pipeline on a schedule |
| Cloud Object Storage | Storing everything (data, models, forecasts) |
| Container Registry | Hosting the Docker image |

## If you're coming from AWS

| AWS thing | IBM equivalent here |
|-----------|-------------------|
| Step Functions | Just a Python script in a Code Engine job |
| SageMaker Training | XGBoost running inside the job |
| Lambda (inference) | Same job, different mode |
| EventBridge Schedule | Code Engine cron subscriptions |
| S3 | IBM Cloud Object Storage |

## Schedules

- **Daily inference** at 08:00 UTC — generates tomorrow's forecasts
- **Weekly training** on Sundays at 06:00 UTC — retrains the model on latest data

## Local development

```bash
uv sync

# set up your COS credentials
cp .env.example .env
# edit .env with your actual values

# run the full pipeline
uv run python -m src.orchestrator.main

# or just one part
uv run python -m src.orchestrator.main --mode inference
uv run python -m src.orchestrator.main --mode training

# tests
uv run pytest tests/
```

## Deploying to IBM Cloud

```bash
# login
ibmcloud login
ibmcloud target -g Test -r eu-de

# pick your project
ibmcloud ce project select --name energy-forecast

# build and push the image
ibmcloud ce buildrun submit \
  --name forecast-build \
  --source . \
  --strategy dockerfile \
  --image private.de.icr.io/energy-ml/energy-forecast:latest \
  --registry-secret icr-secret \
  --wait

# store credentials as a secret
ibmcloud ce secret create --name cos-credentials --from-env-file .env

# create the training job
ibmcloud ce job create --name forecast-training \
  --image private.de.icr.io/energy-ml/energy-forecast:latest \
  --registry-secret icr-secret \
  --env-from-secret cos-credentials \
  --argument "--mode" --argument "training" \
  --cpu 4 --memory 8G

# create the inference job
ibmcloud ce job create --name forecast-inference \
  --image private.de.icr.io/energy-ml/energy-forecast:latest \
  --registry-secret icr-secret \
  --env-from-secret cos-credentials \
  --argument "--mode" --argument "inference" \
  --cpu 2 --memory 4G

# set up the schedules
ibmcloud ce subscription cron create --name daily-inference \
  --destination-type job \
  --destination forecast-inference \
  --schedule "0 8 * * *"

ibmcloud ce subscription cron create --name weekly-training \
  --destination-type job \
  --destination forecast-training \
  --schedule "0 6 * * 0"
```

## Day-to-day operations

```bash
# what ran recently?
ibmcloud ce jobrun list

# trigger something manually
ibmcloud ce jobrun submit --job forecast-training
ibmcloud ce jobrun submit --job forecast-inference

# check logs
ibmcloud ce jobrun logs -f -n <run-name>

# verify schedules
ibmcloud ce subscription cron list
```

## COS bucket layout

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

## Project structure

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
│   │   ├── main.py            ← entry point (this is the "Step Functions")
│   │   └── config.py
│   ├── features/
│   │   └── feature_engineering.py
│   ├── training/
│   │   └── trainer.py
│   ├── inference/
│   │   └── predictor.py
│   └── storage/
│       └── cos_client.py      ← IBM COS wrapper
└── tests/
    ├── test_features.py
    └── test_training.py
```

## Environment variables

| Variable | What it is |
|----------|-----------|
| `COS_ENDPOINT` | IBM COS endpoint (e.g. `https://s3.eu-de.cloud-object-storage.appdomain.cloud`) |
| `COS_API_KEY` | Your IBM Cloud API key |
| `COS_INSTANCE_CRN` | COS service instance CRN |
| `COS_BUCKET` | Bucket name (`energy-forecast-bucket`) |
| `PIPELINE_MODE` | `full`, `training`, or `inference` |
| `RETRAINING_DAY` | Day of week for retraining (0=Mon, 6=Sun) |

## What I'd do next

- Hook up real electricity price data (ENTSO-E or similar)
- More features (weather, load forecasts)
- Model monitoring — compare forecasts against actuals once they arrive
- Maybe Airflow if this grows to multiple models
