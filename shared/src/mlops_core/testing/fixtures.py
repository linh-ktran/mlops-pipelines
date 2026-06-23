"""Reusable pytest fixtures for energy price ML pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Synthetic raw hourly electricity price data (no features yet)."""
    np.random.seed(42)
    n = 1200

    timestamps = pd.date_range("2025-01-01", periods=n, freq="h")
    prices = 50 + 20 * np.sin(np.arange(n) * 2 * np.pi / 24) + np.random.normal(0, 5, n)

    return pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "electricity_spot_price": prices,
        }
    )


@pytest.fixture
def features_df() -> pd.DataFrame:
    """Synthetic feature-engineered DataFrame ready for training.

    Includes all the standard features that the pipelines produce:
    cyclical time, holiday/weekend, lags, and rolling statistics.
    """
    np.random.seed(42)
    n = 1000

    hours = np.tile(np.arange(24), n // 24 + 1)[:n]

    df = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2025-01-01", periods=n, freq="h"),
            "electricity_spot_price": (
                50 + 20 * np.sin(np.arange(n) * 2 * np.pi / 24) + np.random.normal(0, 5, n)
            ),
            "hour": hours,
            "day_of_week": np.tile(np.arange(7), n // 7 + 1)[:n] % 7,
            "day_of_year": np.arange(1, n + 1) % 366,
            "month": np.tile(np.arange(1, 13), n // 12 + 1)[:n],
            "quarter": np.tile([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], n // 12 + 1)[:n],
            "hour_sin": np.sin(2 * np.pi * hours / 24),
            "hour_cos": np.cos(2 * np.pi * hours / 24),
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
            "electricity_spot_price_rolling_mean_72h": np.random.uniform(40, 60, n),
            "electricity_spot_price_rolling_std_72h": np.random.uniform(2, 10, n),
            "electricity_spot_price_rolling_mean_168h": np.random.uniform(40, 60, n),
            "electricity_spot_price_rolling_std_168h": np.random.uniform(2, 10, n),
        }
    )
    return df
