# aws-sagemaker-pipeline

XGBoost regression model for aFRR capacity price forecasting, deployed on AWS SageMaker.

See [top-level README](../README.md#the-use-case) for context on the use case.

## Stack

| What | Why |
|------|-----|
| uv | dependency management |
| XGBoost | model |
| MLflow | experiment tracking + model registry |
| FastAPI | local serving |
| SageMaker | AWS deployment (endpoints + pipelines) |
| Terraform | infra-as-code |
| GitHub Actions | CI |
| Ruff | lint/format |

## Layout

```
src/mlops_serving_starter/
├── training/           # train XGBoost, feature engineering
├── serving/            # load MLflow model, run predictions
├── api/                # FastAPI /predict endpoint
├── monitoring/         # drift detection (Evidently)
└── sagemaker/          # inference handler, deploy, pipeline builder
scripts/                # data generation
configs/                # hyperparameters
infra/terraform/        # endpoint, EventBridge schedule, CloudWatch alarms
```

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
make install
make generate-data
make train                # horizon 1 = day-ahead
make mlflow-ui            # http://127.0.0.1:5001
```

Train a specific hour:

```bash
uv run python -m mlops_serving_starter.training.train --data data/sample.csv --horizon 1 --hour 10
```

Serve locally:

```bash
make serve MODEL_URI="runs:/<RUN_ID>/model"

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"records":[{"afrr_capacity_price_up_lag_1_h1":25.3, ...}]}'

make test
```

## MLflow model lifecycle

```bash
make train-all            # all 4 horizons
make compare              # compare registered versions
make promote VERSION=10 ALIAS=production
make serve MODEL_URI="models:/afrr-capacity-price-xgboost@production"
```

Aliases: `staging`, `production`, `champion`, `challenger`.

## SageMaker endpoint

```bash
# package model artifact
make package-model MODEL_URI="runs:/<RUN_ID>/model"

# dry-run
make sagemaker-plan \
  IMAGE_URI=<ECR_IMAGE> \
  MODEL_DATA_URL=s3://<BUCKET>/artifacts/model.tar.gz \
  EXECUTION_ROLE_ARN=arn:aws:iam::<ACCOUNT>:role/<ROLE>

# deploy
make sagemaker-apply \
  IMAGE_URI=<ECR_IMAGE> \
  MODEL_DATA_URL=s3://<BUCKET>/artifacts/model.tar.gz \
  EXECUTION_ROLE_ARN=arn:aws:iam::<ACCOUNT>:role/<ROLE> \
  AWS_REGION=<REGION>
```

## SageMaker pipeline

4-step pipeline: processing → training → create model → batch transform.

```bash
make sagemaker-pipeline-plan \
  PIPELINE_NAME=my-pipeline \
  EXECUTION_ROLE_ARN=arn:aws:iam::<ACCOUNT>:role/<ROLE> \
  PROCESSING_IMAGE_URI=<ECR> \
  TRAINING_IMAGE_URI=<ECR> \
  INFERENCE_IMAGE_URI=<ECR> \
  INPUT_DATA_S3_URI=s3://<BUCKET>/input \
  TRANSFORM_INPUT_S3_URI=s3://<BUCKET>/transform-input \
  PROCESSING_OUTPUT_S3_URI=s3://<BUCKET>/processing-output \
  TRAINING_OUTPUT_S3_URI=s3://<BUCKET>/training-output \
  TRANSFORM_OUTPUT_S3_URI=s3://<BUCKET>/transform-output
```

## Infrastructure

```bash
make terraform-init
make terraform-validate

cp infra/terraform/terraform.tfvars.example infra/terraform/terraform.tfvars
terraform -chdir=infra/terraform plan -var-file=terraform.tfvars
```

## Drift monitoring

Uses [Evidently](https://www.evidentlyai.com/) to compare training vs current feature distributions.

```bash
make install-monitoring
make drift-check REFERENCE_DATA=data/sample.csv CURRENT_DATA=data/current.csv

# CI mode — exits 1 on drift
make drift-check-ci REFERENCE_DATA=data/sample.csv CURRENT_DATA=data/current.csv
```

Monitored features: FCR price, consumption/gas/spot/solar/wind forecasts, rolling stats, calendar features.

Available tests: `ks` (default), `wasserstein`, `psi`, `kl_div`, `jensen_shannon`. Reports go to `reports/`.

## TODO

- Bayesian hyperparameter tuning (BayesSearchCV + TimeSeriesSplit)
- Train all 8 models per hour (2 targets × 4 horizons) in one pipeline run
- Staging → production promotion flow in MLflow
- Backtesting with rolling-window evaluation
- Wire EventBridge schedule to trigger retraining
