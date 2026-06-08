"""
Inference module for energy price forecasting.

Thin wrapper around mlops_core.inference — handles COS model loading and saving.
"""

from __future__ import annotations

import pandas as pd
import structlog

from mlops_core.inference import predict_with_model
from mlops_core.training import TrainedModel

from src.orchestrator.config import PipelineConfig
from src.storage.cos_client import COSClient

logger = structlog.get_logger(__name__)


def predict(cos: COSClient, features_df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Load the latest model from COS and generate forecasts."""
    trained_model = _load_latest_model(cos, config)
    logger.info(
        "inference.model_loaded",
        n_features=len(trained_model.feature_names),
        bias_correction=trained_model.bias_correction,
    )

    return predict_with_model(
        trained_model, features_df, target_variable=config.target_variable
    )


def save_forecasts(cos: COSClient, forecasts_df: pd.DataFrame, config: PipelineConfig) -> str:
    """Save forecast results to COS as parquet."""
    key = cos.generate_dated_key(config.forecasts_prefix, ".parquet")
    cos.write_parquet(forecasts_df, key)
    logger.info("inference.forecasts_saved", key=key, rows=len(forecasts_df))
    return key


def _load_latest_model(cos: COSClient, config: PipelineConfig) -> TrainedModel:
    """Load the latest model from COS."""
    latest_key = f"{config.model_prefix}model_latest.pkl"
    try:
        return cos.read_pickle(latest_key)
    except Exception:
        logger.warning("inference.latest_model_not_found", key=latest_key)

    fallback_key = cos.get_latest_key(config.model_prefix, suffix=".pkl")
    if fallback_key is None:
        raise FileNotFoundError(
            f"No trained model found in COS under prefix '{config.model_prefix}'. "
            "Run training first with --mode training"
        )
    logger.info("inference.using_fallback_model", key=fallback_key)
    return cos.read_pickle(fallback_key)

