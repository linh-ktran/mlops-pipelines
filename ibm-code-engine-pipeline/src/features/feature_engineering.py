"""
Feature engineering for energy price forecasting.

Thin wrapper around mlops_core.features — handles COS I/O and delegates
all transform logic to the shared library.
"""

from __future__ import annotations

import structlog

from mlops_core.features import generate_features_pipeline  # noqa: F401

from src.orchestrator.config import PipelineConfig
from src.storage.cos_client import COSClient

logger = structlog.get_logger(__name__)


def generate_features(cos: COSClient, config: PipelineConfig):
    """Load raw data from COS, run feature pipeline, save results back to COS."""
    # Load raw electricity price data
    raw_key = _get_latest_raw_data_key(cos, config)
    logger.info("features.load_raw", key=raw_key)
    df = cos.read_parquet(raw_key)

    # Run shared feature pipeline
    df = generate_features_pipeline(df, config.target_variable)

    # Save features to COS
    features_key = cos.generate_dated_key(config.features_prefix, ".parquet")
    cos.write_parquet(df, features_key)
    logger.info("features.saved", key=features_key)

    return df


def _get_latest_raw_data_key(cos: COSClient, config: PipelineConfig) -> str:
    """Find the most recent raw data file in COS."""
    keys = cos.list_keys(config.raw_data_prefix)
    parquet_keys = [k for k in keys if k.endswith(".parquet")]
    if not parquet_keys:
        raise FileNotFoundError(f"No raw data files found in COS under prefix '{config.raw_data_prefix}'")
    return sorted(parquet_keys)[-1]
