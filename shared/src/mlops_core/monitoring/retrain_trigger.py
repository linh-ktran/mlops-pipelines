"""
Automated retraining trigger — monitors model performance and drift,
then decides whether to kick off a retraining pipeline.

This can be run on a cron schedule (e.g., daily) or invoked by monitoring alerts.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


class RetrainReason(str, Enum):
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    SCHEDULED = "scheduled"
    MANUAL = "manual"


@dataclass
class RetrainDecision:
    should_retrain: bool
    reason: RetrainReason | None
    details: dict
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "should_retrain": self.should_retrain,
            "reason": self.reason.value if self.reason else None,
            "details": self.details,
            "timestamp": self.timestamp,
        }


@dataclass
class PerformanceThresholds:
    """Thresholds that trigger retraining when exceeded."""
    max_mae: float = 8.0            # €/MW
    max_bias: float = 5.0           # €/MW (absolute)
    min_drift_features: int = 3     # number of drifted features
    min_coverage: float = 0.6       # ±10€ coverage ratio
    cooldown_hours: int = 24        # minimum hours between retrains


def evaluate_retraining(
    current_mae: float,
    current_bias: float,
    n_drifted_features: int,
    coverage: float,
    last_retrain_time: datetime | None = None,
    thresholds: PerformanceThresholds | None = None,
) -> RetrainDecision:
    """
    Evaluate whether retraining should be triggered based on current metrics.

    Returns a RetrainDecision with the recommendation and reasoning.
    """
    if thresholds is None:
        thresholds = PerformanceThresholds()

    now = datetime.utcnow()

    # Check cooldown
    if last_retrain_time:
        hours_since = (now - last_retrain_time).total_seconds() / 3600
        if hours_since < thresholds.cooldown_hours:
            logger.info(
                "retraining_cooldown_active",
                hours_since=round(hours_since, 1),
                cooldown_hours=thresholds.cooldown_hours,
            )
            return RetrainDecision(
                should_retrain=False,
                reason=None,
                details={"blocked_by": "cooldown", "hours_remaining": round(thresholds.cooldown_hours - hours_since, 1)},
                timestamp=now.isoformat(),
            )

    details = {
        "mae": current_mae,
        "bias": current_bias,
        "drifted_features": n_drifted_features,
        "coverage": coverage,
        "thresholds": {
            "max_mae": thresholds.max_mae,
            "max_bias": thresholds.max_bias,
            "min_drift_features": thresholds.min_drift_features,
            "min_coverage": thresholds.min_coverage,
        },
    }

    # Priority 1: Performance degradation
    if current_mae > thresholds.max_mae:
        logger.warning("retraining_triggered", reason="mae_exceeded", mae=current_mae)
        return RetrainDecision(
            should_retrain=True,
            reason=RetrainReason.PERFORMANCE_DEGRADATION,
            details={**details, "trigger": "mae_exceeded"},
            timestamp=now.isoformat(),
        )

    if abs(current_bias) > thresholds.max_bias:
        logger.warning("retraining_triggered", reason="bias_exceeded", bias=current_bias)
        return RetrainDecision(
            should_retrain=True,
            reason=RetrainReason.PERFORMANCE_DEGRADATION,
            details={**details, "trigger": "bias_exceeded"},
            timestamp=now.isoformat(),
        )

    if coverage < thresholds.min_coverage:
        logger.warning("retraining_triggered", reason="low_coverage", coverage=coverage)
        return RetrainDecision(
            should_retrain=True,
            reason=RetrainReason.PERFORMANCE_DEGRADATION,
            details={**details, "trigger": "low_coverage"},
            timestamp=now.isoformat(),
        )

    # Priority 2: Data drift
    if n_drifted_features >= thresholds.min_drift_features:
        logger.warning("retraining_triggered", reason="data_drift", n_drifted=n_drifted_features)
        return RetrainDecision(
            should_retrain=True,
            reason=RetrainReason.DATA_DRIFT,
            details={**details, "trigger": "feature_drift"},
            timestamp=now.isoformat(),
        )

    # All good
    logger.info("no_retraining_needed", mae=current_mae, bias=current_bias)
    return RetrainDecision(
        should_retrain=False,
        reason=None,
        details=details,
        timestamp=now.isoformat(),
    )


def save_decision(decision: RetrainDecision, output_path: Path) -> None:
    """Persist the retraining decision for audit trail."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(decision.to_dict(), f, indent=2)
    logger.info("decision_saved", path=str(output_path))


if __name__ == "__main__":
    # Example: could be called by a cron job or monitoring webhook
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate retraining trigger")
    parser.add_argument("--mae", type=float, required=True)
    parser.add_argument("--bias", type=float, required=True)
    parser.add_argument("--drifted-features", type=int, required=True)
    parser.add_argument("--coverage", type=float, required=True)
    parser.add_argument("--output", type=str, default="retrain_decision.json")
    args = parser.parse_args()

    decision = evaluate_retraining(
        current_mae=args.mae,
        current_bias=args.bias,
        n_drifted_features=args.drifted_features,
        coverage=args.coverage,
    )

    save_decision(decision, Path(args.output))
    print(json.dumps(decision.to_dict(), indent=2))

    if decision.should_retrain:
        exit(1)  # Non-zero exit = retraining needed (useful in CI/cron)

