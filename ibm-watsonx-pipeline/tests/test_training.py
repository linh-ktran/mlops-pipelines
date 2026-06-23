"""Tests for the training module."""

from mlops_core.training import train_model, evaluate_model


def test_train_model(features_df):
    result = train_model(features_df, "electricity_spot_price")

    assert result.model is not None
    assert result.metrics["mae"] > 0
    assert result.metrics["rmse"] > 0
    assert result.metrics["r2"] is not None
    assert len(result.feature_names) > 0
    assert "timestamp_utc" not in result.feature_names
    assert "electricity_spot_price" not in result.feature_names


def test_train_model_metrics_reasonable(features_df):
    result = train_model(features_df, "electricity_spot_price")

    target_std = features_df["electricity_spot_price"].std()
    assert result.metrics["mae"] < target_std * 2


def test_evaluate_model(features_df):
    trained = train_model(features_df, "electricity_spot_price")
    evaluation = evaluate_model(trained, features_df, "electricity_spot_price")

    assert "full_mae" in evaluation
    assert "full_rmse" in evaluation
    assert "full_r2" in evaluation
    assert "model_passed" in evaluation
    assert evaluation["full_mae"] > 0
