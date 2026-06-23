"""Tests for mlops_core shared library."""

from mlops_core.features import (
    add_cyclical_datetime_features,
    add_holiday_weekend_features,
    add_lag_features,
    add_rolling_statistics,
    generate_features_pipeline,
)
from mlops_core.training import evaluate_model, train_model
from mlops_core.inference import predict_with_model


class TestFeatureEngineering:
    def test_cyclical_features(self, raw_df):
        result = add_cyclical_datetime_features(raw_df.copy(), "timestamp_utc")
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        assert "dow_sin" in result.columns
        assert "dow_cos" in result.columns
        assert "doy_sin" in result.columns
        assert "doy_cos" in result.columns
        # Verify range is [-1, 1]
        assert result["hour_sin"].between(-1, 1).all()

    def test_holiday_features(self, raw_df):
        result = add_holiday_weekend_features(raw_df.copy(), "timestamp_utc")
        assert "is_holiday" in result.columns
        assert "is_weekend" in result.columns
        assert set(result["is_holiday"].unique()).issubset({0, 1})

    def test_lag_features(self, raw_df):
        result = add_lag_features(raw_df.copy(), "electricity_spot_price")
        assert "electricity_spot_price_lag_24h" in result.columns
        assert "electricity_spot_price_lag_48h" in result.columns
        assert "electricity_spot_price_lag_168h" in result.columns

    def test_rolling_statistics(self, raw_df):
        result = add_rolling_statistics(raw_df.copy(), "electricity_spot_price")
        assert "electricity_spot_price_rolling_mean_24h" in result.columns
        assert "electricity_spot_price_rolling_std_24h" in result.columns

    def test_full_pipeline(self, raw_df):
        result = generate_features_pipeline(raw_df.copy(), "electricity_spot_price")
        assert len(result) > 0
        assert not result.isna().any().any()
        assert "hour_sin" in result.columns
        assert "electricity_spot_price_lag_24h" in result.columns


class TestTraining:
    def test_train_model(self, features_df):
        result = train_model(features_df, "electricity_spot_price")
        assert result.model is not None
        assert result.metrics["mae"] > 0
        assert result.metrics["rmse"] > 0
        assert result.metrics["r2"] <= 1.0
        assert len(result.feature_names) > 0
        assert "timestamp_utc" not in result.feature_names
        assert "electricity_spot_price" not in result.feature_names

    def test_train_model_metrics_reasonable(self, features_df):
        result = train_model(features_df, "electricity_spot_price")
        target_std = features_df["electricity_spot_price"].std()
        assert result.metrics["mae"] < target_std * 2  # Better than random

    def test_evaluate_model(self, features_df):
        trained = train_model(features_df, "electricity_spot_price")
        eval_result = evaluate_model(trained, features_df, "electricity_spot_price")
        assert "full_mae" in eval_result
        assert "model_passed" in eval_result
        assert isinstance(eval_result["model_passed"], bool)


class TestInference:
    def test_predict_with_model(self, features_df):
        trained = train_model(features_df, "electricity_spot_price")
        forecasts = predict_with_model(trained, features_df, target_variable="electricity_spot_price")
        assert "forecast_value" in forecasts.columns
        assert "timestamp_utc" in forecasts.columns
        assert "actual_value" in forecasts.columns
        assert len(forecasts) == len(features_df)
        assert (forecasts["forecast_value"] >= 0).all()  # Clipped at 0
