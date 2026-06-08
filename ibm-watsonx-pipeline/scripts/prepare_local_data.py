"""
Prepare local sample data for testing the pipeline in --local mode.

Creates the expected directory structure in local_storage/ so the
pipeline can load raw data without needing IBM COS.

Usage:
    uv run python scripts/prepare_local_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_sample_data(days: int = 90) -> pd.DataFrame:
    """Generate realistic hourly electricity spot prices (EUR/MWh)."""
    np.random.seed(42)
    hours = days * 24
    timestamps = pd.date_range("2026-03-01", periods=hours, freq="h", tz="UTC")

    hour_of_day = timestamps.hour
    day_of_week = timestamps.dayofweek

    daily_pattern = 15 * np.sin((hour_of_day - 6) * np.pi / 12) + 5 * np.sin((hour_of_day - 3) * np.pi / 6)
    weekly_pattern = -8 * (day_of_week >= 5).astype(float)
    trend = np.linspace(0, 5, hours)
    noise = np.random.normal(0, 8, hours)

    prices = 55 + daily_pattern + weekly_pattern + trend + noise
    prices = np.clip(prices, 0, None)

    df = pd.DataFrame({
        "timestamp_utc": timestamps.tz_localize(None),
        "electricity_spot_price": np.round(prices, 2),
    })
    return df


def main():
    print("=" * 60)
    print("  Preparing local data for watsonx pipeline testing")
    print("=" * 60)
    print()

    # Create local storage directory structure
    bucket_dir = Path("local_storage/watsonx-pipeline-bucket")
    raw_dir = bucket_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Generate and save sample data
    print("→ Generating 90 days of synthetic electricity price data...")
    df = generate_sample_data(days=90)
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['timestamp_utc'].min()} → {df['timestamp_utc'].max()}")
    print(f"  Price range: €{df['electricity_spot_price'].min():.2f} → €{df['electricity_spot_price'].max():.2f}/MWh")
    print()

    # Save as parquet (what the pipeline expects)
    parquet_path = raw_dir / "electricity_prices_2026_06_01.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"✓ Saved to {parquet_path}")
    print()

    # Show directory structure
    print("Local storage structure:")
    print("  local_storage/")
    print("  └── watsonx-pipeline-bucket/")
    print("      └── raw/")
    print(f"          └── {parquet_path.name}")
    print()

    print("You can now run the pipeline locally:")
    print()
    print("  # Training pipeline (generates features, trains model, registers in mock watsonx)")
    print("  uv run python -m src.pipeline.orchestrator --mode training --local")
    print()
    print("  # Full pipeline (training + deployment + inference)")
    print("  uv run python -m src.pipeline.orchestrator --mode full --local")
    print()


if __name__ == "__main__":
    main()

