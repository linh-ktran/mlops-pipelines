"""XGBoost model training with chronological split and bias correction.

Shared training logic used by all three pipeline implementations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = structlog.get_logger(__name__)

# Columns that should never be used as features
_EXCLUDE_COLS = frozenset(["timestamp_utc", "timestamp", "datetime", "date"])


@dataclass
class TrainedModel:
    """Container for a trained model and its metadata."""

    model: xgb.XGBRegressor
    metrics: dict
    feature_names: list[str]
    bias_correction: float


def train_model(
    features_df: pd.DataFrame,
    target_variable: str,
    xgb_params: dict | None = None,
    test_size: float = 0.2,
    exclude_cols: set[str] | None = None,
) -> TrainedModel:
    """Train an XGBoost regressor on a feature DataFrame.

    Uses a chronological train/validation split (no shuffle — time series).
    Applies bias correction to predictions (mean residual on validation set).

    Args:
        features_df: Feature-engineered DataFrame including the target column.
        target_variable: Name of the target column.
        xgb_params: XGBoost hyperparameters. Uses sensible defaults if not provided.
        test_size: Fraction of data to use for validation (last N%). Default 0.2.
        exclude_cols: Additional column names to exclude from features.

    Returns:
        TrainedModel with model, metrics, feature names, and bias correction.
    """
    if xgb_params is None:
        xgb_params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "early_stopping_rounds": 20,
        }

    # Determine feature columns
    all_exclude = _EXCLUDE_COLS | (exclude_cols or set())
    feature_cols = [
        col for col in features_df.columns
        if col not in all_exclude and col != target_variable
    ]

    X = features_df[feature_cols]
    y = features_df[target_variable]

    # Chronological split (time series — no shuffle)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(
        "training.split",
        train_rows=len(X_train),
        val_rows=len(X_val),
        n_features=len(feature_cols),
    )

    # Train XGBoost
    params = xgb_params.copy()
    early_stopping = params.pop("early_stopping_rounds", 20)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=early_stopping,
        **params,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Bias correction: mean residual on validation set
    val_preds = model.predict(X_val)
    bias_correction = float((y_val.values - val_preds).mean())
    val_preds_corrected = np.clip(val_preds + bias_correction, a_min=0, a_max=None)

    metrics = {
        "mae": float(mean_absolute_error(y_val, val_preds_corrected)),
        "rmse": float(np.sqrt(mean_squared_error(y_val, val_preds_corrected))),
        "r2": float(r2_score(y_val, val_preds_corrected)),
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


def evaluate_model(
    trained_model: TrainedModel,
    features_df: pd.DataFrame,
    target_variable: str,
) -> dict:
    """Evaluate a trained model on a full dataset.

    Useful as a separate DAG step for model validation/gating.

    Args:
        trained_model: A previously trained model.
        features_df: DataFrame containing features and target.
        target_variable: Name of the target column.

    Returns:
        Dictionary with evaluation metrics and a pass/fail flag.
    """
    X = features_df[trained_model.feature_names]
    y = features_df[target_variable]

    preds = trained_model.model.predict(X)
    preds = np.clip(preds + trained_model.bias_correction, a_min=0, a_max=None)

    return {
        "full_mae": float(mean_absolute_error(y, preds)),
        "full_rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "full_r2": float(r2_score(y, preds)),
        "val_mae": trained_model.metrics["mae"],
        "val_rmse": trained_model.metrics["rmse"],
        "model_passed": bool(trained_model.metrics["mae"] < y.std() * 1.5),
    }

