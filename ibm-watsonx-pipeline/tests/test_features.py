"""Tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from mlops_core.features import (
    add_cyclical_datetime_features,
    add_holiday_weekend_features,
    add_lag_features,
    add_rolling_statistics,
    generate_features_pipeline,
)


@pytest.fixture
def sample_df():
    """Create a sample hourly electricity price DataFrame."""
    dates = pd.date_range("2026-01-01", periods=720, freq="h")  # 30 days
    np.random.seed(42)
    prices = 50 + 20 * np.sin(np.arange(720) * 2 * np.pi / 24) + np.random.normal(0, 5, 720)
    return pd.DataFrame(
        {
            "timestamp_utc": dates,
            "electricity_spot_price": prices,
        }
    )


def test_cyclical_features(sample_df):
    result = add_cyclical_datetime_features(sample_df, "timestamp_utc")
    assert "hour_sin" in result.columns
    assert "hour_cos" in result.columns
    assert "dow_sin" in result.columns
    assert "doy_sin" in result.columns
    assert result["hour_sin"].between(-1, 1).all()
    assert result["hour_cos"].between(-1, 1).all()


def test_holiday_weekend_features(sample_df):
    result = add_holiday_weekend_features(sample_df, "timestamp_utc")
    assert "is_holiday" in result.columns
    assert "is_weekend" in result.columns
    assert result["is_weekend"].isin([0, 1]).all()
    jan1_rows = result[result["timestamp_utc"].dt.date == pd.Timestamp("2026-01-01").date()]
    assert jan1_rows["is_holiday"].iloc[0] == 1


def test_lag_features(sample_df):
    result = add_lag_features(sample_df, "electricity_spot_price")
    assert "electricity_spot_price_lag_24h" in result.columns
    assert "electricity_spot_price_lag_168h" in result.columns
    assert result["electricity_spot_price_lag_168h"].iloc[:168].isna().all()
    assert result["electricity_spot_price_lag_24h"].iloc[24] == sample_df["electricity_spot_price"].iloc[0]


def test_rolling_statistics(sample_df):
    result = add_rolling_statistics(sample_df, "electricity_spot_price")
    assert "electricity_spot_price_rolling_mean_24h" in result.columns
    assert "electricity_spot_price_rolling_std_72h" in result.columns
    valid_mean = result["electricity_spot_price_rolling_mean_168h"].dropna()
    assert abs(valid_mean.mean() - sample_df["electricity_spot_price"].mean()) < 5


def test_generate_features_drops_na(sample_df):
    result = generate_features_pipeline(sample_df, "electricity_spot_price")
    assert not result.isna().any().any()
    # Should have fewer rows than input (lag/rolling window creates NaN at start)
    assert len(result) < len(sample_df)
