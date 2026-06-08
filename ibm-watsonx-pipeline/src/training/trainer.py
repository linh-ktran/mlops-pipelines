"""XGBoost training and evaluation — delegates to mlops_core.training."""

from __future__ import annotations

import structlog

from mlops_core.training import TrainedModel, evaluate_model, train_model  # noqa: F401

from src.pipeline.config import PipelineConfig

log = structlog.get_logger(__name__)


def train(features_df, config: PipelineConfig) -> TrainedModel:
    """Train XGBoost model using shared training logic."""
    return train_model(
        features_df,
        target_variable=config.target_variable,
        xgb_params=config.xgb_params,
    )


def evaluate(trained_model: TrainedModel, features_df, config: PipelineConfig) -> dict:
    """Evaluate on full dataset (separate DAG step for tracking)."""
    return evaluate_model(trained_model, features_df, config.target_variable)


def save_model_to_cos(cos, trained_model: TrainedModel, config: PipelineConfig) -> str:
    """Save model to COS (dated + latest)."""
    dated_key = cos.generate_dated_key(config.model_prefix, ".pkl")
    cos.write_pickle(trained_model, dated_key)

    latest_key = f"{config.model_prefix}model_latest.pkl"
    cos.write_pickle(trained_model, latest_key)

    log.info("model saved", dated_key=dated_key)
    return dated_key
