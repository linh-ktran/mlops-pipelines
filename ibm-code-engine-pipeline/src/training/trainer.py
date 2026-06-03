"""
Model training module for energy price forecasting.

Trains an XGBoost model on the feature-engineered data and saves it to IBM COS.
Replaces SageMaker training jobs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.orchestrator.config import PipelineConfig
from src.storage.cos_client import COSClient

logger = structlog.get_logger(__name__)


@dataclass
class TrainedModel:
    """Container for a trained model and its metadata."""

    model: xgb.XGBRegressor
    metrics: dict
    feature_names: list[str]
    bias_correction: float


def train_model(features_df: pd.DataFrame, config: PipelineConfig) -> TrainedModel:
    """
    Train XGBoost regressor on the features DataFrame.

    Uses chronological train/validation split (last 20% for validation).
    Applies bias correction like the production AWS system.
    """
    target = config.target_variable

    # Separate features from target
    feature_cols = [
        col for col in features_df.columns
        if col not in [target, "timestamp_utc", "timestamp", "datetime"]
    ]

    X = features_df[feature_cols]
    y = features_df[target]

    # Chronological split (no shuffle — time series)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(
        "training.split",
        train_rows=len(X_train),
        val_rows=len(X_val),
        n_features=len(feature_cols),
    )

    # Train XGBoost
    params = config.xgb_params.copy()
    early_stopping = params.pop("early_stopping_rounds", 20)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=early_stopping,
        **params,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Evaluate with bias correction (same as your AWS production pipeline)
    val_preds = model.predict(X_val)
    bias_correction = float((y_val.values - val_preds).mean())
    val_preds_corrected = np.clip(val_preds + bias_correction, a_min=0, a_max=None)

    metrics = {
        "mae": float(mean_absolute_error(y_val, val_preds_corrected)),
        "rmse": float(np.sqrt(mean_squared_error(y_val, val_preds_corrected))),
        "bias_correction": bias_correction,
        "best_iteration": int(model.best_iteration) if hasattr(model, "best_iteration") else -1,
        "n_features": len(feature_cols),
        "train_rows": len(X_train),
        "val_rows": len(X_val),
    }

    logger.info("training.complete", **metrics)

    return TrainedModel(
        model=model,
        metrics=metrics,
        feature_names=feature_cols,
        bias_correction=bias_correction,
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

