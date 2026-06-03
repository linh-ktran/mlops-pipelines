"""Generate synthetic electricity spot price data and upload to COS for testing."""

import numpy as np
import pandas as pd

from src.storage.cos_client import COSClient
from src.orchestrator.config import PipelineConfig


def generate_sample_data(days: int = 90) -> pd.DataFrame:
    """Generate realistic hourly electricity spot prices (EUR/MWh)."""
    np.random.seed(42)
    hours = days * 24
    timestamps = pd.date_range("2026-03-01", periods=hours, freq="h", tz="UTC")

    # Base price ~50 EUR/MWh with daily pattern + weekly pattern + noise
    hour_of_day = timestamps.hour
    day_of_week = timestamps.dayofweek

    # Daily pattern: peak at 8-9am and 6-7pm, low at night
    daily_pattern = 15 * np.sin((hour_of_day - 6) * np.pi / 12) + 5 * np.sin((hour_of_day - 3) * np.pi / 6)

    # Weekly pattern: lower on weekends
    weekly_pattern = -8 * (day_of_week >= 5).astype(float)

    # Seasonal trend (slight increase over time)
    trend = np.linspace(0, 5, hours)

    # Random noise
    noise = np.random.normal(0, 8, hours)

    # Combine
    prices = 55 + daily_pattern + weekly_pattern + trend + noise
    prices = np.clip(prices, 0, None)  # No negative prices

    df = pd.DataFrame({
        "timestamp_utc": timestamps.tz_localize(None),  # Store as naive UTC
        "electricity_spot_price": np.round(prices, 2),
    })

    return df


if __name__ == "__main__":
    print("Generating 90 days of synthetic electricity price data...")
    df = generate_sample_data(days=90)
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['timestamp_utc'].min()} → {df['timestamp_utc'].max()}")
    print(f"  Price range: {df['electricity_spot_price'].min():.2f} → {df['electricity_spot_price'].max():.2f} EUR/MWh")
    print()

    # Connect to COS and upload
    config = PipelineConfig.from_env()
    cos = COSClient(config)

    key = "raw/electricity_prices_2026_06_01.parquet"
    cos.write_parquet(df, key)
    print(f"✓ Uploaded to cos://{config.cos_bucket}/{key}")
    print()

    # Verify
    objects = cos.list_keys("raw/")
    print(f"  Objects in raw/: {objects}")

