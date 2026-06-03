"""Pipeline configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


@dataclass
class PipelineConfig:
    """Configuration for the energy forecast pipeline."""

    # IBM Cloud Object Storage
    cos_endpoint: str = ""
    cos_api_key: str = ""
    cos_instance_crn: str = ""
    cos_bucket: str = "energy-forecast-bucket"

    # Pipeline settings
    mode: str = "full"  # full | training | inference
    retraining_day: int = 6  # 0=Monday, 6=Sunday

    # Model settings
    forecast_horizon_days: int = 14
    target_variable: str = "electricity_spot_price"
    model_prefix: str = "models/"
    features_prefix: str = "features/"
    forecasts_prefix: str = "forecasts/"
    raw_data_prefix: str = "raw/"

    # XGBoost hyperparameters
    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 20,
    })

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Load configuration from environment variables."""
        load_dotenv()

        return cls(
            cos_endpoint=os.getenv("COS_ENDPOINT", ""),
            cos_api_key=os.getenv("COS_API_KEY", ""),
            cos_instance_crn=os.getenv("COS_INSTANCE_CRN", ""),
            cos_bucket=os.getenv("COS_BUCKET", "energy-forecast-bucket"),
            mode=os.getenv("PIPELINE_MODE", "full"),
            retraining_day=int(os.getenv("RETRAINING_DAY", "6")),
        )

