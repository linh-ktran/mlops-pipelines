"""Mock watsonx.ai client for local testing."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import structlog

from src.pipeline.config import PipelineConfig

log = structlog.get_logger(__name__)

REGISTRY_DIR = Path("local_storage/watsonx_registry")


class MockWatsonxClient:
    """Simulates watsonx.ai locally — same interface as WatsonxClient."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        log.info("watsonx mock ready (no cloud connection)")

    def store_model(self, trained_model, config: PipelineConfig) -> str:
        model_id = f"mock-model-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

        with open(REGISTRY_DIR / f"{model_id}.pkl", "wb") as f:
            pickle.dump(trained_model.model, f)

        meta = {
            "model_id": model_id,
            "name": f"{config.pipeline_name}-model",
            "type": config.model_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metrics": trained_model.metrics,
            "feature_names": trained_model.feature_names,
        }
        (REGISTRY_DIR / f"{model_id}_meta.json").write_text(json.dumps(meta, indent=2))

        log.info("model stored (mock)", model_id=model_id)
        return model_id

    def deploy_model(self, model_id: str, config: PipelineConfig) -> str:
        dep_id = f"mock-deploy-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

        meta = {
            "deployment_id": dep_id,
            "model_id": model_id,
            "name": config.deployment_name,
            "status": "ready",
            "endpoint_url": f"https://eu-de.ml.cloud.ibm.com/ml/v4/deployments/{dep_id}/predictions",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (REGISTRY_DIR / f"{dep_id}.json").write_text(json.dumps(meta, indent=2))

        log.info("model deployed (mock)", deployment_id=dep_id, endpoint=meta["endpoint_url"])
        return dep_id

    def score(self, deployment_id: str, payload: dict) -> dict:
        n_rows = len(payload.get("input_data", [{}])[0].get("values", []))
        return {"predictions": [{"values": [[0.0]] * n_rows}]}

    def list_models(self) -> list[dict]:
        return [json.loads(f.read_text()) for f in REGISTRY_DIR.glob("*_meta.json")]

    def list_deployments(self) -> list[dict]:
        return [json.loads(f.read_text()) for f in REGISTRY_DIR.glob("mock-deploy-*.json")]

    def delete_deployment(self, deployment_id: str):
        path = REGISTRY_DIR / f"{deployment_id}.json"
        if path.exists():
            path.unlink()

    def delete_model(self, model_id: str):
        for path in REGISTRY_DIR.glob(f"{model_id}*"):
            path.unlink()
