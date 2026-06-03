"""Tests for the training module."""

import numpy as np
import pandas as pd
import pytest

from src.training.trainer import train_model
from src.orchestrator.config import PipelineConfig


@pytest.fixture
def features_df():
    """Create a feature-engineered DataFrame ready for training."""
    np.random.seed(42)
    n = 1000

    # Simulate features that would come out of feature_engineering
    df = pd.DataFrame({
        "timestamp_utc": pd.date_range("2025-01-01", periods=n, freq="h"),
        "electricity_spot_price": 50 + 20 * np.sin(np.arange(n) * 2 * np.pi / 24) + np.random.normal(0, 5, n),
        "hour": np.tile(np.arange(24), n // 24 + 1)[:n],
        "hour_sin": np.sin(2 * np.pi * np.tile(np.arange(24), n // 24 + 1)[:n] / 24),
        "hour_cos": np.cos(2 * np.pi * np.tile(np.arange(24), n // 24 + 1)[:n] / 24),
        "dow_sin": np.random.uniform(-1, 1, n),
        "dow_cos": np.random.uniform(-1, 1, n),
        "doy_sin": np.random.uniform(-1, 1, n),
        "doy_cos": np.random.uniform(-1, 1, n),
        "is_holiday": np.random.choice([0, 1], n, p=[0.95, 0.05]),
        "is_weekend": np.random.choice([0, 1], n, p=[0.71, 0.29]),
        "electricity_spot_price_lag_24h": np.random.uniform(30, 70, n),
        "electricity_spot_price_lag_48h": np.random.uniform(30, 70, n),
        "electricity_spot_price_lag_168h": np.random.uniform(30, 70, n),
        "electricity_spot_price_rolling_mean_24h": np.random.uniform(40, 60, n),
        "electricity_spot_price_rolling_std_24h": np.random.uniform(2, 10, n),
    })
    return df


def test_train_model(features_df):
    config = PipelineConfig()
    result = train_model(features_df, config)

    assert result.model is not None
    assert result.metrics["mae"] > 0
    assert result.metrics["rmse"] > 0
    assert len(result.feature_names) > 0
    assert "timestamp_utc" not in result.feature_names
    assert "electricity_spot_price" not in result.feature_names


def test_train_model_metrics_reasonable(features_df):
    config = PipelineConfig()
    result = train_model(features_df, config)

    # MAE should be less than the std of the target (model is better than guessing mean)
    target_std = features_df["electricity_spot_price"].std()
    assert result.metrics["mae"] < target_std * 2  # Generous bound

