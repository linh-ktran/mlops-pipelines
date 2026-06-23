"""Tests for the monitoring module."""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.tracker import compute_monitoring_report


@pytest.fixture
def forecast_df():
    """Create a sample forecast DataFrame with actuals."""
    np.random.seed(42)
    n = 100
    actuals = 50 + np.random.normal(0, 10, n)
    forecasts = actuals + np.random.normal(0, 3, n)  # Small error
    return pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-06-01", periods=n, freq="h"),
            "forecast_value": forecasts,
            "actual_value": actuals,
        }
    )


def test_monitoring_report_no_alert(forecast_df):
    report = compute_monitoring_report(forecast_df, mae_threshold=20.0)
    assert report.alert is False
    assert report.mae > 0
    assert report.rmse > 0
    assert report.coverage_rows == 100


def test_monitoring_report_mae_alert(forecast_df):
    # Set very tight threshold to trigger alert
    report = compute_monitoring_report(forecast_df, mae_threshold=0.1)
    assert report.alert is True
    assert "MAE" in report.alert_reason


def test_monitoring_report_bias_alert():
    """Test bias detection."""
    n = 100
    df = pd.DataFrame(
        {
            "forecast_value": np.ones(n) * 70,  # Systematically high
            "actual_value": np.ones(n) * 50,
        }
    )
    report = compute_monitoring_report(df, mae_threshold=100, bias_threshold=5.0)
    assert report.alert is True
    assert "Bias" in report.alert_reason
    assert report.bias == pytest.approx(20.0)


def test_monitoring_report_missing_actuals():
    df = pd.DataFrame({"forecast_value": [1, 2, 3]})
    with pytest.raises(ValueError, match="actual_value"):
        compute_monitoring_report(df)
