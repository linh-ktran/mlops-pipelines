"""Feature engineering — thin wrapper around mlops_core.features."""

from __future__ import annotations

import pandas as pd
import structlog

from mlops_core.features import generate_features_pipeline  # noqa: F401

from src.pipeline.config import PipelineConfig

log = structlog.get_logger(__name__)


def load_raw_data(cos, config: PipelineConfig) -> pd.DataFrame:
    """Load the most recent raw parquet from storage."""
    raw_key = _find_latest_raw_file(cos, config)
    log.info("loading raw data", key=raw_key)
    df = cos.read_parquet(raw_key)

    # Normalize timestamp column
    ts_col = "timestamp_utc"
    if ts_col not in df.columns:
        for alt in ["timestamp", "datetime", "date"]:
            if alt in df.columns:
                df = df.rename(columns={alt: ts_col})
                break

    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values(ts_col).reset_index(drop=True)
    log.info("raw data loaded", rows=len(df), columns=list(df.columns))
    return df


def generate_features(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Run full feature pipeline using shared mlops_core transforms."""
    return generate_features_pipeline(df, config.target_variable)


def _find_latest_raw_file(cos, config: PipelineConfig) -> str:
    keys = cos.list_keys(config.raw_data_prefix)
    parquet_keys = [k for k in keys if k.endswith(".parquet")]
    if not parquet_keys:
        raise FileNotFoundError(f"No raw data in '{config.raw_data_prefix}'")
    return sorted(parquet_keys)[-1]
