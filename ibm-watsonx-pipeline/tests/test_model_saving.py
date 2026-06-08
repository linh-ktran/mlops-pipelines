"""Tests for model saving and loading (watsonx integration)."""

import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.training.trainer import TrainedModel, train_model, save_model_to_cos
from src.pipeline.config import PipelineConfig


@pytest.fixture
def trained_model():
    """Create a simple trained model for testing save/load."""
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 5)
    y = X[:, 0] * 2 + X[:, 1] + np.random.normal(0, 0.1, n)

    model = xgb.XGBRegressor(n_estimators=10, max_depth=3)
    model.fit(X, y)

    return TrainedModel(
        model=model,
        metrics={"mae": 0.5, "rmse": 0.7, "r2": 0.95, "bias_correction": 0.01},
        feature_names=["f1", "f2", "f3", "f4", "f5"],
        bias_correction=0.01,
    )


def test_model_pickle_roundtrip(trained_model):
    """Test that TrainedModel can be pickled and unpickled."""
    data = pickle.dumps(trained_model)
    loaded = pickle.loads(data)

    assert loaded.feature_names == trained_model.feature_names
    assert loaded.bias_correction == trained_model.bias_correction
    assert loaded.metrics == trained_model.metrics

    # Verify predictions match
    X_test = np.random.randn(10, 5)
    original_preds = trained_model.model.predict(X_test)
    loaded_preds = loaded.model.predict(X_test)
    np.testing.assert_array_almost_equal(original_preds, loaded_preds)


def test_model_save_to_cos(trained_model):
    """Test save_model_to_cos writes both dated and latest keys."""
    config = PipelineConfig()
    mock_cos = MagicMock()
    mock_cos.generate_dated_key.return_value = "models/2026_06_03.pkl"

    key = save_model_to_cos(mock_cos, trained_model, config)

    assert key == "models/2026_06_03.pkl"
    assert mock_cos.write_pickle.call_count == 2  # dated + latest
    # Verify latest key
    latest_call = mock_cos.write_pickle.call_args_list[1]
    assert latest_call[0][1] == "models/model_latest.pkl"


def test_model_prediction_consistency(trained_model):
    """Test that model predictions are deterministic."""
    X_test = np.random.randn(50, 5)
    preds1 = trained_model.model.predict(X_test)
    preds2 = trained_model.model.predict(X_test)
    np.testing.assert_array_equal(preds1, preds2)

