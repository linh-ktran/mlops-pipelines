"""
IBM Cloud Code Engine — Energy Forecast Pipeline Orchestrator.

This is the single entry point that runs the entire forecasting workflow:
    1. Generate features from raw data
    2. Train model (if retraining day)
    3. Load latest model
    4. Generate forecasts
    5. Save forecasts to Cloud Object Storage

Replaces AWS Step Functions with a simple Python script scheduled via Code Engine cron.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

import structlog

from src.features.feature_engineering import generate_features
from src.training.trainer import train_model, save_model
from src.inference.predictor import predict, save_forecasts
from src.storage.cos_client import COSClient
from src.orchestrator.config import PipelineConfig

logger = structlog.get_logger(__name__)


def is_retraining_day(config: PipelineConfig) -> bool:
    """Check if today is a scheduled retraining day."""
    today = datetime.now(timezone.utc).weekday()
    return today == config.retraining_day


def run_full_pipeline(config: PipelineConfig) -> dict:
    """Run the complete pipeline: features → (optional training) → inference → save."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logger.info("pipeline.start", run_id=run_id, mode="full")

    cos = COSClient(config)

    # Step 1: Generate features
    logger.info("step.features.start")
    features_df = generate_features(cos, config)
    logger.info("step.features.done", rows=len(features_df))

    # Step 2: Conditionally retrain
    if is_retraining_day(config):
        logger.info("step.training.start", reason="retraining_day")
        model = train_model(features_df, config)
        save_model(cos, model, config)
        logger.info("step.training.done")
    else:
        logger.info("step.training.skipped", reason="not_retraining_day")

    # Step 3: Load latest model and predict
    logger.info("step.inference.start")
    forecasts_df = predict(cos, features_df, config)
    logger.info("step.inference.done", forecast_rows=len(forecasts_df))

    # Step 4: Save forecasts
    logger.info("step.save.start")
    save_forecasts(cos, forecasts_df, config)
    logger.info("step.save.done")

    logger.info("pipeline.complete", run_id=run_id)
    return {"run_id": run_id, "forecast_rows": len(forecasts_df), "status": "success"}


def run_training_only(config: PipelineConfig | None = None) -> dict:
    """Run only the training step."""
    if config is None:
        config = PipelineConfig.from_env()

    logger.info("pipeline.start", mode="training")
    cos = COSClient(config)

    features_df = generate_features(cos, config)
    model = train_model(features_df, config)
    save_model(cos, model, config)

    logger.info("pipeline.complete", mode="training")
    return {"status": "success", "mode": "training"}


def run_inference_only(config: PipelineConfig | None = None) -> dict:
    """Run only the inference step (assumes model already exists in COS)."""
    if config is None:
        config = PipelineConfig.from_env()

    logger.info("pipeline.start", mode="inference")
    cos = COSClient(config)

    features_df = generate_features(cos, config)
    forecasts_df = predict(cos, features_df, config)
    save_forecasts(cos, forecasts_df, config)

    logger.info("pipeline.complete", mode="inference", forecast_rows=len(forecasts_df))
    return {"status": "success", "mode": "inference", "forecast_rows": len(forecasts_df)}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Energy Forecast Pipeline — IBM Cloud Code Engine")
    parser.add_argument(
        "--mode",
        choices=["full", "training", "inference"],
        default=None,
        help="Pipeline mode (overrides PIPELINE_MODE env var)",
    )
    args = parser.parse_args()

    # Configure structured logging
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer() if sys.stdout.isatty() else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )

    config = PipelineConfig.from_env()
    if args.mode:
        config.mode = args.mode

    try:
        if config.mode == "training":
            result = run_training_only(config)
        elif config.mode == "inference":
            result = run_inference_only(config)
        else:
            result = run_full_pipeline(config)

        logger.info("pipeline.result", **result)
    except Exception as e:
        logger.error("pipeline.failed", error=str(e), error_type=type(e).__name__)
        sys.exit(1)


if __name__ == "__main__":
    main()

