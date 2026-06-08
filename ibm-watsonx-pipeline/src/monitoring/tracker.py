"""Monitoring — compare forecasts vs actuals, detect drift."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)


@dataclass
class MonitoringReport:
    timestamp: str
    mae: float
    rmse: float
    mean_forecast: float
    mean_actual: float
    bias: float
    coverage_rows: int
    alert: bool
    alert_reason: str = ""


def compute_monitoring_report(
    forecasts_df: pd.DataFrame,
    mae_threshold: float = 20.0,
    bias_threshold: float = 10.0,
) -> MonitoringReport:
    """Compare forecast vs actual values. Raises alert if thresholds exceeded."""
    if "actual_value" not in forecasts_df.columns:
        raise ValueError("Need 'actual_value' column to compute monitoring")

    actuals = forecasts_df["actual_value"].values
    forecasts = forecasts_df["forecast_value"].values

    mae = float(np.mean(np.abs(actuals - forecasts)))
    rmse = float(np.sqrt(np.mean((actuals - forecasts) ** 2)))
    bias = float(np.mean(forecasts - actuals))

    alert = False
    reason = ""
    if mae > mae_threshold:
        alert = True
        reason = f"MAE ({mae:.2f}) > threshold ({mae_threshold})"
    elif abs(bias) > bias_threshold:
        alert = True
        reason = f"Bias ({bias:.2f}) > threshold ({bias_threshold})"

    report = MonitoringReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        mae=mae,
        rmse=rmse,
        mean_forecast=float(np.mean(forecasts)),
        mean_actual=float(np.mean(actuals)),
        bias=bias,
        coverage_rows=len(forecasts_df),
        alert=alert,
        alert_reason=reason,
    )

    if alert:
        log.warning("monitoring alert", reason=reason, mae=mae, bias=bias)
    else:
        log.info("monitoring ok", mae=mae, rmse=rmse, bias=bias)

    return report
