"""Shared test fixtures for the aws-sagemaker-pipeline test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_feature_record():
    """A single feature record for prediction tests."""
    return {
        "afrr_capacity_price_up_lag_1_h1": 25.0,
        "rolling_mean_7": 23.0,
        "holiday_status": 0,
        "weekend_status": 0,
    }


@pytest.fixture
def sample_feature_df(sample_feature_record):
    """A DataFrame with multiple feature records for batch tests."""
    records = [sample_feature_record.copy() for _ in range(5)]
    # Add slight variation
    for i, r in enumerate(records):
        r["afrr_capacity_price_up_lag_1_h1"] += i * 2.0
        r["rolling_mean_7"] += i * 0.5
    return pd.DataFrame(records)


@pytest.fixture
def sample_training_df():
    """Synthetic training data for model training tests."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "afrr_capacity_price_up": np.random.exponential(20, n),
            "afrr_capacity_price_down": np.random.exponential(15, n),
            "mfrr_capacity_price_up": np.random.exponential(18, n),
            "mfrr_capacity_price_down": np.random.exponential(12, n),
        }
    )


class FakeModel:
    """A mock model for serving tests."""

    def __init__(self, return_value=1.0):
        self._return_value = return_value
        self.metadata = None

    def predict(self, df):
        return [self._return_value] * len(df)


@pytest.fixture
def fake_model():
    """A fake model instance for testing."""
    return FakeModel()
