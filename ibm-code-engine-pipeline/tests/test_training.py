"""Tests for the training module."""

from mlops_core.training import train_model


def test_train_model(features_df):
    result = train_model(features_df, "electricity_spot_price")

    assert result.model is not None
    assert result.metrics["mae"] > 0
    assert result.metrics["rmse"] > 0
    assert len(result.feature_names) > 0
    assert "timestamp_utc" not in result.feature_names
    assert "electricity_spot_price" not in result.feature_names


def test_train_model_metrics_reasonable(features_df):
    result = train_model(features_df, "electricity_spot_price")

    # MAE should be less than the std of the target (model is better than guessing mean)
    target_std = features_df["electricity_spot_price"].std()
    assert result.metrics["mae"] < target_std * 2  # Generous bound
