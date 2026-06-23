"""Feature engineering transforms for energy price forecasting.

All functions are pure DataFrame transforms with no I/O dependencies,
making them easy to test and reuse across pipeline implementations.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


def generate_features_pipeline(
    df: pd.DataFrame,
    target_variable: str,
    timestamp_col: str = "timestamp_utc",
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    country: str = "FR",
) -> pd.DataFrame:
    """Run the full feature engineering pipeline.

    Steps:
        1. Normalize and sort by timestamp
        2. Add cyclical datetime features
        3. Add holiday/weekend features
        4. Add lag features
        5. Add rolling statistics
        6. Drop NaN rows (from lag/rolling warmup)

    Args:
        df: Raw DataFrame with timestamp and target columns.
        target_variable: Name of the target column (e.g. 'electricity_spot_price').
        timestamp_col: Name of the timestamp column.
        lags: Lag periods in hours. Defaults to [24, 48, 168].
        rolling_windows: Rolling window sizes in hours. Defaults to [24, 72, 168].
        country: Country code for holidays. Defaults to 'FR' (France).

    Returns:
        Feature-engineered DataFrame with NaN rows removed.
    """
    # Normalize timestamp column name
    df = _normalize_timestamp_col(df, timestamp_col)

    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    logger.info("features.raw_loaded", rows=len(df), columns=list(df.columns))

    # Feature engineering steps
    df = add_cyclical_datetime_features(df, timestamp_col)
    df = add_holiday_weekend_features(df, timestamp_col, country=country)
    df = add_lag_features(df, target_variable, lags=lags)
    df = add_rolling_statistics(df, target_variable, windows=rolling_windows)

    # Drop NaN rows from lag/rolling warmup period
    initial_len = len(df)
    df = df.dropna().reset_index(drop=True)
    logger.info("features.ready", rows=len(df), dropped=initial_len - len(df))

    return df


def add_cyclical_datetime_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """Add sin/cos cyclical encodings for hour, day-of-week, and day-of-year.

    Also adds raw calendar features: hour, day_of_week, day_of_year, month, quarter.
    """
    ts = df[timestamp_col]

    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["day_of_year"] = ts.dt.dayofyear
    df["month"] = ts.dt.month
    df["quarter"] = ts.dt.quarter

    # Cyclical encodings (avoid discontinuities at period boundaries)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 366)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 366)

    return df


def add_holiday_weekend_features(
    df: pd.DataFrame,
    timestamp_col: str,
    country: str = "FR",
) -> pd.DataFrame:
    """Add binary holiday and weekend indicator features.

    Args:
        df: DataFrame with a datetime timestamp column.
        timestamp_col: Name of the timestamp column.
        country: ISO country code for the holiday calendar. Defaults to France.
    """
    years = sorted(df[timestamp_col].dt.year.unique())
    country_holidays = holidays.country_holidays(country, years=years)

    df["is_holiday"] = df[timestamp_col].dt.date.isin(country_holidays).astype(int)
    df["is_weekend"] = df[timestamp_col].dt.dayofweek.isin([5, 6]).astype(int)

    return df


def add_lag_features(
    df: pd.DataFrame,
    target: str,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Add lagged values of the target variable.

    Args:
        df: DataFrame containing the target column.
        target: Name of the target column.
        lags: List of lag periods in hours. Defaults to [24, 48, 168] (1d, 2d, 1w).
    """
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
    """Add rolling mean and standard deviation of the target variable.

    Args:
        df: DataFrame containing the target column.
        target: Name of the target column.
        windows: List of window sizes in hours. Defaults to [24, 72, 168] (1d, 3d, 1w).
    """
    if windows is None:
        windows = [24, 72, 168]  # 1 day, 3 days, 1 week (in hours)

    for window in windows:
        df[f"{target}_rolling_mean_{window}h"] = df[target].rolling(window).mean()
        df[f"{target}_rolling_std_{window}h"] = df[target].rolling(window).std()

    return df


def _normalize_timestamp_col(df: pd.DataFrame, expected_col: str) -> pd.DataFrame:
    """Rename common timestamp column names to the expected name."""
    if expected_col in df.columns:
        return df
    for alt in ["timestamp", "datetime", "date", "timestamp_utc"]:
        if alt in df.columns:
            return df.rename(columns={alt: expected_col})
    return df
