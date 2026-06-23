"""Inference — delegates prediction to mlops_core, handles COS I/O."""

from __future__ import annotations

import structlog

from mlops_core.inference import predict_with_model
from mlops_core.training import TrainedModel

from src.pipeline.config import PipelineConfig

log = structlog.get_logger(__name__)


def load_model(cos, config: PipelineConfig) -> TrainedModel:
    """Load latest model from storage."""
    latest_key = f"{config.model_prefix}model_latest.pkl"
    try:
        return cos.read_pickle(latest_key)
    except Exception:
        pass

    # Fallback: find most recent dated model
    fallback_key = cos.get_latest_key(config.model_prefix, suffix=".pkl")
    if fallback_key is None:
        raise FileNotFoundError(f"No model found in '{config.model_prefix}'. Run training first.")
    log.info("using fallback model", key=fallback_key)
    return cos.read_pickle(fallback_key)


def predict(trained_model: TrainedModel, features_df, config: PipelineConfig):
    """Generate forecasts with bias correction using shared predict logic."""
    return predict_with_model(trained_model, features_df, target_variable=config.target_variable)


def save_forecasts(cos, forecasts_df, config: PipelineConfig) -> str:
    """Save forecasts as parquet."""
    key = cos.generate_dated_key(config.forecasts_prefix, ".parquet")
    cos.write_parquet(forecasts_df, key)
    log.info("forecasts saved", key=key, rows=len(forecasts_df))
    return key
