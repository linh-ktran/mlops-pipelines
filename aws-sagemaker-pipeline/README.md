# aws-sagemaker-pipeline

Using Sagemaker to deploy a simple XGBoost regression model for time-series forecasting.

> **Use case:** aFRR capacity price forecasting — details in the [top-level README](../README.md#the-use-case).

## The stack

- **uv** for dependency management (fast, just works)
- **XGBoost** for the actual model
- **MLflow** for experiment tracking and model registry
- **FastAPI** for serving predictions locally
- **SageMaker** for AWS deployment (endpoints + pipelines)
- **Terraform** for infra (endpoint, scheduled retraining, alarms)
- **GitHub Actions** for CI
- **Ruff** for linting/formatting

## Project layout

```
src/mlops_serving_starter/
├── training/
│   ├── train.py                # train XGBoost + log to MLflow
│   └── feature_engineering.py  # lags, rolling stats, calendar, exogenous features
├── serving/        # load MLflow model, run predictions
├── api/            # FastAPI /predict endpoint
├── monitoring/     # data drift detection with Evidently
└── sagemaker/      # inference handler, deploy script, pipeline builder
scripts/
├── generate_sample_data.py     # generate synthetic aFRR-like data
configs/
├── train_config.json           # model hyperparameters & experiment config
infra/terraform/    # SageMaker endpoint + EventBridge schedule + CloudWatch alarms
```

## Running locally

You need [uv](https://docs.astral.sh/uv/) installed.

```bash
make install          # sets up .venv

# generate some fake time-series data to play with
make generate-data

# train a model (horizon 1 = day-ahead) and log it to MLflow
make train

# or be specific about the hour
uv run python -m mlops_serving_starter.training.train --data data/sample.csv --horizon 1 --hour 10

# check your experiments
make mlflow-ui   # → http://127.0.0.1:5001

# serve it locally (grab the run_id from MLflow)
make serve MODEL_URI="runs:/<RUN_ID>/model"

# hit the API
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"records":[{"afrr_capacity_price_up_lag_1_h1":25.3,"afrr_capacity_price_up_lag_2_h1":24.1,"afrr_capacity_price_up_lag_3_h1":23.5,"afrr_capacity_price_up_lag_7_h1":22.0,"afrr_capacity_price_up_lag_14_h1":21.5,"afrr_capacity_price_up_lag_28_h1":20.0,"rolling_mean_7":23.4,"rolling_min_7":18.2,"rolling_max_7":30.1,"fcr_price_symmetric":15.0,"consumption_forecast":52000,"gas_price_forecast":32.5,"spot_price_forecast":55.0,"solar_forecast":2500,"wind_onshore_forecast":2100,"wind_offshore_forecast":900,"holiday_status":0,"weekend_status":0}]}'

make test
```

## Model lifecycle (MLflow)

```bash
# train all 4 horizons
make train-all

# compare registered model versions
make compare

# promote to production
make promote VERSION=10 ALIAS=production

# serve the production model directly
make serve MODEL_URI="models:/afrr-capacity-price-xgboost@production"
```

Aliases I'm using: `staging`, `production`, `champion`, `challenger`

## Deploying to SageMaker

```bash
# package the model
make package-model MODEL_URI="runs:/<RUN_ID>/model"

# dry-run (just prints what it would do)
make sagemaker-plan \
  IMAGE_URI=<ECR_IMAGE> \
  MODEL_DATA_URL=s3://<BUCKET>/artifacts/model.tar.gz \
  EXECUTION_ROLE_ARN=arn:aws:iam::<ACCOUNT>:role/<ROLE>

# actually create/update the endpoint
make sagemaker-apply \
  IMAGE_URI=<ECR_IMAGE> \
  MODEL_DATA_URL=s3://<BUCKET>/artifacts/model.tar.gz \
  EXECUTION_ROLE_ARN=arn:aws:iam::<ACCOUNT>:role/<ROLE> \
  AWS_REGION=<REGION>
```

## SageMaker Pipeline

This generates a 4-step pipeline (processing → training → create model → batch transform) as JSON that you can register in SageMaker.

```bash
make sagemaker-pipeline-plan \
  PIPELINE_NAME=my-pipeline \
  EXECUTION_ROLE_ARN=arn:aws:iam::<ACCOUNT>:role/<ROLE> \
  PROCESSING_IMAGE_URI=<ECR> TRAINING_IMAGE_URI=<ECR> INFERENCE_IMAGE_URI=<ECR> \
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

# for a real plan
cp infra/terraform/terraform.tfvars.example infra/terraform/terraform.tfvars
terraform -chdir=infra/terraform plan -var-file=terraform.tfvars
```

## Drift monitoring

Using [Evidently](https://www.evidentlyai.com/) to catch distribution shifts in the input features. Basically comparing training data distributions against what's coming in now.

```bash
make install-monitoring

# generate a report (HTML + JSON)
make drift-check REFERENCE_DATA=data/sample.csv CURRENT_DATA=data/current.csv

# CI-friendly version (exits 1 if drift detected)
make drift-check-ci REFERENCE_DATA=data/sample.csv CURRENT_DATA=data/current.csv
```

Monitors: FCR price, consumption/gas/spot/solar/wind forecasts, rolling stats, calendar features.

Supports different statistical tests: `ks` (default), `wasserstein`, `psi`, `kl_div`, `jensen_shannon`.

Reports go to `reports/`.

## Ideas for later

- Bayesian hyperparameter tuning (BayesSearchCV + TimeSeriesSplit)
- Train all 8 models per hour (2 targets × 4 horizons) in one pipeline run
- Proper staging → production promotion flow in MLflow
- Backtesting framework for rolling-window performance evaluation
- Wire up the EventBridge schedule to actually trigger retraining
