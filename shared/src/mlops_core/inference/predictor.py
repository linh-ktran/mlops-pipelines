"""Base prediction logic with bias correction.

Provides the core predict function used by all pipeline implementations.
Platform-specific model loading and forecast saving are left to each pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import structlog

from mlops_core.training.trainer import TrainedModel

logger = structlog.get_logger(__name__)


def predict_with_model(
    trained_model: TrainedModel,
    features_df: pd.DataFrame,
    target_variable: str | None = None,
    timestamp_col: str = "timestamp_utc",
) -> pd.DataFrame:
    """Generate predictions using a trained model with bias correction.

    Args:
        trained_model: TrainedModel instance with model, feature_names, and bias_correction.
        features_df: DataFrame containing the feature columns.
        target_variable: If provided and present in features_df, include actuals for monitoring.
        timestamp_col: Name of timestamp column to include in output.

    Returns:
        DataFrame with columns: timestamp_utc, forecast_value, forecast_issued_at,
        and optionally actual_value.
    """
    # Validate features
    feature_cols = trained_model.feature_names
    missing_cols = [col for col in feature_cols if col not in features_df.columns]
    if missing_cols:
        raise ValueError(f"Missing features in input data: {missing_cols}")

    X = features_df[feature_cols]

    # Generate predictions with bias correction and floor at zero
    raw_predictions = trained_model.model.predict(X)
    predictions = np.clip(raw_predictions + trained_model.bias_correction, a_min=0, a_max=None)

    # Build forecast DataFrame
    forecast_df = pd.DataFrame({
        "timestamp_utc": (
            features_df[timestamp_col] if timestamp_col in features_df.columns
            else range(len(predictions))
        ),
        "forecast_value": predictions,
        "forecast_issued_at": datetime.now(timezone.utc).isoformat(),
    })

    # Attach actuals if available (useful for drift monitoring)
    if target_variable and target_variable in features_df.columns:
        forecast_df["actual_value"] = features_df[target_variable].values

    logger.info("inference.predictions_generated", rows=len(forecast_df))
    return forecast_df

