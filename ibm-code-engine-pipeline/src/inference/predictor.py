"""
Inference module for energy price forecasting.

Loads the latest trained model from COS, generates forecasts, and saves results.
Replaces the AWS Lambda inference function.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import structlog

from src.orchestrator.config import PipelineConfig
from src.storage.cos_client import COSClient
from src.training.trainer import TrainedModel

logger = structlog.get_logger(__name__)


def predict(cos: COSClient, features_df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    Load the latest model from COS and generate forecasts.

    Returns a DataFrame with timestamps and predicted values.
    """
    # Load model
    trained_model = _load_latest_model(cos, config)
    logger.info(
        "inference.model_loaded",
        n_features=len(trained_model.feature_names),
        bias_correction=trained_model.bias_correction,
    )

    # Prepare features (use same columns as training)
    feature_cols = trained_model.feature_names
    missing_cols = [col for col in feature_cols if col not in features_df.columns]
    if missing_cols:
        raise ValueError(f"Missing features in input data: {missing_cols}")

    X = features_df[feature_cols]

    # Generate predictions with bias correction
    raw_predictions = trained_model.model.predict(X)
    predictions = np.clip(raw_predictions + trained_model.bias_correction, a_min=0, a_max=None)

    # Build forecast DataFrame
    forecast_df = pd.DataFrame({
        "timestamp_utc": features_df["timestamp_utc"] if "timestamp_utc" in features_df.columns else range(len(predictions)),
        "forecast_value": predictions,
        "forecast_issued_at": datetime.now(timezone.utc).isoformat(),
    })

    # Add actuals if available (for monitoring)
    if config.target_variable in features_df.columns:
        forecast_df["actual_value"] = features_df[config.target_variable].values

    logger.info("inference.predictions_generated", rows=len(forecast_df))
    return forecast_df


def save_forecasts(cos: COSClient, forecasts_df: pd.DataFrame, config: PipelineConfig) -> str:
    """Save forecast results to COS as parquet."""
    key = cos.generate_dated_key(config.forecasts_prefix, ".parquet")
    cos.write_parquet(forecasts_df, key)
    logger.info("inference.forecasts_saved", key=key, rows=len(forecasts_df))
    return key


def _load_latest_model(cos: COSClient, config: PipelineConfig) -> TrainedModel:
    """Load the latest model from COS."""
    # Try 'model_latest.pkl' first (always updated on training)
    latest_key = f"{config.model_prefix}model_latest.pkl"
    try:
        return cos.read_pickle(latest_key)
    except Exception:
        logger.warning("inference.latest_model_not_found", key=latest_key)

    # Fallback: find most recent dated model
    fallback_key = cos.get_latest_key(config.model_prefix, suffix=".pkl")
    if fallback_key is None:
        raise FileNotFoundError(
            f"No trained model found in COS under prefix '{config.model_prefix}'. "
            "Run training first with --mode training"
        )
    logger.info("inference.using_fallback_model", key=fallback_key)
    return cos.read_pickle(fallback_key)

