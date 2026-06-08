"""watsonx.ai client — model registry and deployment."""

from __future__ import annotations

import structlog

from src.pipeline.config import PipelineConfig

log = structlog.get_logger(__name__)


class WatsonxClient:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._client = None
        self._connect()

    def _connect(self):
        from ibm_watsonx_ai import Credentials, APIClient

        creds = Credentials(url=self.config.watsonx_url, api_key=self.config.watsonx_api_key)
        self._client = APIClient(creds)

        if self.config.watsonx_space_id:
            self._client.set.default_space(self.config.watsonx_space_id)
        else:
            self._client.set.default_project(self.config.watsonx_project_id)
        log.info("watsonx connected", url=self.config.watsonx_url)

    def store_model(self, trained_model, config: PipelineConfig) -> str:
        """Register model in watsonx.ai. Returns model ID."""
        sw_spec_id = self._client.software_specifications.get_id_by_name(config.software_spec)

        meta = {
            self._client.repository.ModelMetaNames.NAME: f"{config.pipeline_name}-model",
            self._client.repository.ModelMetaNames.TYPE: config.model_type,
            self._client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_id,
        }

        stored = self._client.repository.store_model(model=trained_model.model, meta_props=meta)
        model_id = self._client.repository.get_model_id(stored)
        log.info("model registered", model_id=model_id)
        return model_id

    def deploy_model(self, model_id: str, config: PipelineConfig) -> str:
        """Deploy as online REST endpoint. Returns deployment ID."""
        meta = {
            self._client.deployments.ConfigurationMetaNames.NAME: config.deployment_name,
            self._client.deployments.ConfigurationMetaNames.ONLINE: {},
        }

        try:
            deployment = self._client.deployments.create(artifact_uid=model_id, meta_props=meta)
            dep_id = self._client.deployments.get_id(deployment)
            log.info("model deployed", deployment_id=dep_id)
            return dep_id
        except Exception as e:
            # Reuse existing deployment if name conflict
            if "already exists" in str(e).lower() or "name" in str(e).lower():
                return self._find_existing_deployment(config.deployment_name)
            raise

    def _find_existing_deployment(self, name: str) -> str:
        details = self._client.deployments.get_details()
        for item in details.get("resources", []):
            if item.get("entity", {}).get("name") == name:
                dep_id = item["metadata"]["id"]
                log.info("reusing deployment", deployment_id=dep_id)
                return dep_id
        raise RuntimeError(f"Deployment '{name}' not found")

    def score(self, deployment_id: str, payload: dict) -> dict:
        """Score data via deployed endpoint."""
        return self._client.deployments.score(deployment_id, payload)

    def list_models(self):
        return self._client.repository.list_models()

    def list_deployments(self):
        return self._client.deployments.list()

    def delete_deployment(self, deployment_id: str):
        self._client.deployments.delete(deployment_id)

    def delete_model(self, model_id: str):
        self._client.repository.delete(model_id)
