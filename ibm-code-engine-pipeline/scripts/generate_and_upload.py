"""
Generate realistic sample electricity price data as CSV and Parquet.

This creates files you can:
  1. Inspect locally (CSV)
  2. Upload to COS (Parquet) — already done automatically
  3. Use as a template for when you plug in real data

No API keys needed. No external data sources.
"""

import numpy as np
import pandas as pd

from src.storage.cos_client import COSClient
from src.orchestrator.config import PipelineConfig


def generate_realistic_prices(days: int = 180) -> pd.DataFrame:
    """
    Generate 180 days of realistic French electricity spot prices (EUR/MWh).

    Mimics real patterns:
    - Daily cycle: high during morning/evening peaks, low at night
    - Weekly cycle: lower on weekends
    - Seasonal: higher in winter
    - Price spikes: occasional high prices (cold snaps, low wind)
    - No negative prices (floor at 0)
    """
    np.random.seed(2026)
    hours = days * 24
    timestamps = pd.date_range("2025-12-01", periods=hours, freq="h")

    hour = timestamps.hour
    dow = timestamps.dayofweek
    month = timestamps.month

    # --- Base price (seasonal) ---
    # Winter months (Dec-Feb) ~70 EUR/MWh, Summer ~45 EUR/MWh
    seasonal = np.where(month.isin([12, 1, 2]), 70,
               np.where(month.isin([3, 4, 5]), 55,
               np.where(month.isin([6, 7, 8]), 45, 60)))

    # --- Daily pattern ---
    # Peak hours: 7-9am and 6-8pm
    morning_peak = 20 * np.exp(-0.5 * ((hour - 8) / 2) ** 2)
    evening_peak = 15 * np.exp(-0.5 * ((hour - 19) / 2) ** 2)
    night_dip = -15 * np.exp(-0.5 * ((hour - 3) / 3) ** 2)
    daily = morning_peak + evening_peak + night_dip

    # --- Weekend effect ---
    weekend = -12 * (dow >= 5).astype(float)

    # --- Random noise ---
    noise = np.random.normal(0, 6, hours)

    # --- Price spikes (2% of hours) ---
    spikes = np.zeros(hours)
    spike_mask = np.random.random(hours) < 0.02
    spikes[spike_mask] = np.random.uniform(30, 80, spike_mask.sum())

    # --- Combine ---
    prices = seasonal + daily + weekend + noise + spikes
    prices = np.clip(prices, 0, None)  # Floor at 0
    prices = np.round(prices, 2)

    df = pd.DataFrame({
        "timestamp_utc": timestamps,
        "electricity_spot_price": prices,
    })

    return df


if __name__ == "__main__":
    print("=" * 60)
    print("  Generating sample electricity price data")
    print("  (No API keys needed — this is synthetic but realistic)")
    print("=" * 60)
    print()

    df = generate_realistic_prices(days=180)

    # Save locally as CSV (so you can inspect it)
    csv_path = "data/sample_electricity_prices.csv"
    import os
    os.makedirs("data", exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV locally: {csv_path}")
    print(f"  → {len(df)} rows, {df['timestamp_utc'].min()} to {df['timestamp_utc'].max()}")
    print(f"  → Price range: {df['electricity_spot_price'].min():.2f} - {df['electricity_spot_price'].max():.2f} EUR/MWh")
    print(f"  → Mean price: {df['electricity_spot_price'].mean():.2f} EUR/MWh")
    print()

    # Upload to COS as parquet
    print("Uploading to IBM Cloud Object Storage...")
    config = PipelineConfig.from_env()
    cos = COSClient(config)

    key = "raw/electricity_prices_2026_06_02.parquet"
    cos.write_parquet(df, key)
    print(f"✓ Uploaded to COS: {key}")
    print()

    # Show what's in the bucket
    all_objects = cos.list_keys("")
    print("Current bucket contents:")
    for obj in all_objects:
        print(f"  📄 {obj}")
    print()
    print("You can now run the full pipeline:")
    print("  python -m src.orchestrator.main --mode training")
    print("  python -m src.orchestrator.main --mode inference")

