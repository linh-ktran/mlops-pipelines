"""XGBoost training utilities for energy price forecasting."""

from mlops_core.training.trainer import (
    TrainedModel,
    evaluate_model,
    train_model,
)

__all__ = [
    "TrainedModel",
    "evaluate_model",
    "train_model",
]

