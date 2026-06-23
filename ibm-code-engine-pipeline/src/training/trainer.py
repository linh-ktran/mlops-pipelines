"""
Model training module for energy price forecasting.

Thin wrapper around mlops_core.training — handles COS model persistence.
"""

from __future__ import annotations

import structlog

from mlops_core.training import TrainedModel, train_model  # noqa: F401

from src.orchestrator.config import PipelineConfig
from src.storage.cos_client import COSClient

logger = structlog.get_logger(__name__)


def train(features_df, config: PipelineConfig) -> TrainedModel:
    """Train XGBoost model using shared training logic."""
    return train_model(
        features_df,
        target_variable=config.target_variable,
        xgb_params=config.xgb_params,
    )


def save_model(cos: COSClient, trained_model: TrainedModel, config: PipelineConfig) -> str:
    """Save trained model to COS. Also saves as 'model_latest.pkl' for easy loading."""
    # Save dated version
    dated_key = cos.generate_dated_key(config.model_prefix, ".pkl")
    cos.write_pickle(trained_model, dated_key)

    # Also save as "latest" for inference to always find
    latest_key = f"{config.model_prefix}model_latest.pkl"
    cos.write_pickle(trained_model, latest_key)

    logger.info("training.model_saved", dated_key=dated_key, latest_key=latest_key)
    return dated_key
