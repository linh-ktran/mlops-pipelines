"""Pipeline configuration from env vars."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


@dataclass
class PipelineConfig:
    # watsonx.ai
    watsonx_api_key: str = ""
    watsonx_project_id: str = ""
    watsonx_url: str = "https://eu-de.ml.cloud.ibm.com"
    watsonx_space_id: str = ""

    # COS
    cos_endpoint: str = ""
    cos_api_key: str = ""
    cos_instance_crn: str = ""
    cos_bucket: str = "watsonx-pipeline-bucket"

    # Pipeline
    pipeline_name: str = "watsonx-energy-forecast"
    mode: str = "full"  # full | training | inference | dag
    retraining_day: int = 6  # 0=Mon, 6=Sun

    # Model
    target_variable: str = "electricity_spot_price"
    forecast_horizon_days: int = 14
    model_prefix: str = "models/"
    features_prefix: str = "features/"
    forecasts_prefix: str = "forecasts/"
    raw_data_prefix: str = "raw/"
    dag_prefix: str = "pipeline-metadata/dag/"

    # XGBoost
    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 20,
    })

    # watsonx deployment
    deployment_name: str = "energy-forecast-deployment"
    model_type: str = "xgboost_2.1"
    software_spec: str = "runtime-25.1-py3.12"

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        load_dotenv()
        return cls(
            watsonx_api_key=os.getenv("WATSONX_API_KEY", ""),
            watsonx_project_id=os.getenv("WATSONX_PROJECT_ID", ""),
            watsonx_url=os.getenv("WATSONX_URL", "https://eu-de.ml.cloud.ibm.com"),
            watsonx_space_id=os.getenv("WATSONX_SPACE_ID", ""),
            cos_endpoint=os.getenv("COS_ENDPOINT", ""),
            cos_api_key=os.getenv("COS_API_KEY", ""),
            cos_instance_crn=os.getenv("COS_INSTANCE_CRN", ""),
            cos_bucket=os.getenv("COS_BUCKET", "watsonx-pipeline-bucket"),
            mode=os.getenv("PIPELINE_MODE", "full"),
            retraining_day=int(os.getenv("RETRAINING_DAY", "6")),
        )
