"""
Feature engineering for energy price forecasting.

Mirrors the logic from the AWS Lambda (espf_feature_engineering) but adapted
for IBM Cloud Object Storage and the Python orchestrator pattern.
"""

from __future__ import annotations

from datetime import datetime, timezone

import holidays
import numpy as np
import pandas as pd
import structlog

from src.orchestrator.config import PipelineConfig
from src.storage.cos_client import COSClient

logger = structlog.get_logger(__name__)


def generate_features(cos: COSClient, config: PipelineConfig) -> pd.DataFrame:
    """
    Full feature engineering pipeline:
        1. Load raw data from COS
        2. Add temporal features (cyclical, holidays)
        3. Add lag features
        4. Add rolling statistics
        5. Save features to COS
        6. Return feature DataFrame
    """
    # Load raw electricity price data
    raw_key = _get_latest_raw_data_key(cos, config)
    logger.info("features.load_raw", key=raw_key)
    df = cos.read_parquet(raw_key)

    # Ensure timestamp column
    timestamp_col = "timestamp_utc"
    if timestamp_col not in df.columns:
        # Try common alternatives
        for alt in ["timestamp", "datetime", "date"]:
            if alt in df.columns:
                df = df.rename(columns={alt: timestamp_col})
                break

    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    logger.info("features.raw_loaded", rows=len(df), columns=list(df.columns))

    # Feature engineering steps
    df = add_cyclical_datetime_features(df, timestamp_col)
    df = add_holiday_weekend_features(df, timestamp_col)
    df = add_lag_features(df, config.target_variable, timestamp_col)
    df = add_rolling_statistics(df, config.target_variable)

    # Drop rows with NaN from lag/rolling features (beginning of series)
    initial_len = len(df)
    df = df.dropna().reset_index(drop=True)
    logger.info("features.dropna", dropped=initial_len - len(df), remaining=len(df))

    # Save features to COS
    features_key = cos.generate_dated_key(config.features_prefix, ".parquet")
    cos.write_parquet(df, features_key)
    logger.info("features.saved", key=features_key)

    return df


def add_cyclical_datetime_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """Add sin/cos cyclical encodings for hour, day-of-week, day-of-year."""
    ts = df[timestamp_col]

    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["day_of_year"] = ts.dt.dayofyear
    df["month"] = ts.dt.month
    df["quarter"] = ts.dt.quarter

    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 366)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 366)

    return df


def add_holiday_weekend_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """Add holiday and weekend binary features (France by default)."""
    years = sorted(df[timestamp_col].dt.year.unique())
    fr_holidays = holidays.France(years=years)

    df["is_holiday"] = df[timestamp_col].dt.date.isin(fr_holidays).astype(int)
    df["is_weekend"] = df[timestamp_col].dt.dayofweek.isin([5, 6]).astype(int)

    return df


def add_lag_features(
    df: pd.DataFrame,
    target: str,
    timestamp_col: str,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Add lagged values of the target variable."""
    if lags is None:
        lags = [24, 48, 168]  # 1 day, 2 days, 1 week (in hours)

    for lag in lags:
        df[f"{target}_lag_{lag}h"] = df[target].shift(lag)

    return df


def add_rolling_statistics(
    df: pd.DataFrame,
    target: str,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add rolling mean and std of the target variable."""
    if windows is None:
        windows = [24, 72, 168]  # 1 day, 3 days, 1 week (in hours)

    for window in windows:
        df[f"{target}_rolling_mean_{window}h"] = df[target].rolling(window).mean()
        df[f"{target}_rolling_std_{window}h"] = df[target].rolling(window).std()

    return df


def _get_latest_raw_data_key(cos: COSClient, config: PipelineConfig) -> str:
    """Find the most recent raw data file in COS."""
    keys = cos.list_keys(config.raw_data_prefix)
    parquet_keys = [k for k in keys if k.endswith(".parquet")]
    if not parquet_keys:
        raise FileNotFoundError(
            f"No raw data files found in COS under prefix '{config.raw_data_prefix}'"
        )
    # Lexicographic sort → latest date
    return sorted(parquet_keys)[-1]

