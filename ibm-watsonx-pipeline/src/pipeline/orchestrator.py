"""Pipeline orchestrator — executes DAG steps in order."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

import structlog

from src.pipeline.config import PipelineConfig
from src.pipeline.dag import PipelineDAG, build_full_dag, build_inference_dag, build_training_dag
from src.training.trainer import train_model, evaluate_model
from src.training.feature_engineering import generate_features, load_raw_data
from src.inference.predictor import predict, save_forecasts

log = structlog.get_logger(__name__)


def create_storage_client(config: PipelineConfig, local: bool = False):
    if local:
        from src.storage.local_storage import LocalStorageClient
        return LocalStorageClient(config)
    from src.storage.cos_client import COSClient
    return COSClient(config)


def create_watsonx_client(config: PipelineConfig, local: bool = False):
    if local:
        from src.storage.mock_watsonx import MockWatsonxClient
        return MockWatsonxClient(config)
    from src.storage.watsonx_client import WatsonxClient
    return WatsonxClient(config)


class PipelineExecutor:
    def __init__(self, config: PipelineConfig, local: bool = False):
        self.config = config
        self.local = local
        self.cos = create_storage_client(config, local=local)
        self.watsonx = create_watsonx_client(config, local=local)
        self._ctx: dict = {}

    def execute(self, dag: PipelineDAG) -> dict:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dag.run_id = run_id
        log.info("pipeline.start", run_id=run_id, dag=dag.name, local=self.local)

        for node_name in dag.get_execution_order():
            node = dag.nodes[node_name]

            # Skip if a dependency failed
            failed_deps = [d for d in node.dependencies if dag.nodes[d].status.value == "failed"]
            if failed_deps:
                node.mark_skipped(reason=f"dependency failed: {failed_deps}")
                log.warning("step.skipped", step=node_name, failed_deps=failed_deps)
                continue

            node.mark_running()
            log.info("step.start", step=node_name)

            try:
                metadata = self._run_step(node_name)
                node.mark_success(metadata=metadata)
                log.info("step.done", step=node_name, duration=node.duration_seconds, **metadata)
            except Exception as e:
                node.mark_failed(error=str(e))
                log.error("step.failed", step=node_name, error=str(e))

        self._save_dag_state(dag)

        statuses = {name: n.status.value for name, n in dag.nodes.items()}
        success = all(s in ("success", "skipped") for s in statuses.values())
        log.info("pipeline.done", run_id=run_id, success=success, statuses=statuses)
        return {"run_id": run_id, "statuses": statuses, "success": success}

    def _run_step(self, step: str) -> dict:
        match step:
            case "load_data":
                df = load_raw_data(self.cos, self.config)
                self._ctx["raw_df"] = df
                return {"rows": len(df), "columns": len(df.columns)}

            case "feature_engineering":
                df = generate_features(self._ctx["raw_df"], self.config)
                self._ctx["features_df"] = df
                return {"rows": len(df), "n_features": len(df.columns)}

            case "train_model":
                result = train_model(self._ctx["features_df"], self.config)
                self._ctx["trained_model"] = result
                return {"metrics": result.metrics}

            case "evaluate_model":
                result = evaluate_model(self._ctx["trained_model"], self._ctx["features_df"], self.config)
                self._ctx["evaluation"] = result
                return result

            case "save_to_cos":
                from src.training.trainer import save_model_to_cos
                key = save_model_to_cos(self.cos, self._ctx["trained_model"], self.config)
                self._ctx["model_cos_key"] = key
                return {"cos_key": key}

            case "register_watsonx":
                model_id = self.watsonx.store_model(self._ctx["trained_model"], self.config)
                self._ctx["watsonx_model_id"] = model_id
                return {"model_id": model_id}

            case "deploy_model":
                dep_id = self.watsonx.deploy_model(self._ctx["watsonx_model_id"], self.config)
                self._ctx["deployment_id"] = dep_id
                return {"deployment_id": dep_id}

            case "load_model":
                from src.inference.predictor import load_model
                self._ctx["trained_model"] = load_model(self.cos, self.config)
                return {"model_loaded": True}

            case "predict":
                forecasts = predict(self._ctx["trained_model"], self._ctx["features_df"], self.config)
                self._ctx["forecasts_df"] = forecasts
                return {"forecast_rows": len(forecasts)}

            case "save_forecasts":
                key = save_forecasts(self.cos, self._ctx["forecasts_df"], self.config)
                return {"cos_key": key}

            case _:
                raise ValueError(f"Unknown step: {step}")

    def _save_dag_state(self, dag: PipelineDAG):
        state_key = f"{self.config.dag_prefix}{dag.run_id}_state.json"
        self.cos.write_bytes(dag.to_json().encode(), state_key)

        try:
            dot_key = f"{self.config.dag_prefix}{dag.run_id}_dag.dot"
            self.cos.write_bytes(dag.render_graphviz().encode(), dot_key)
        except Exception:
            pass


# ─── Public API ───────────────────────────────────────────────────────────────

def run_full_pipeline(config: PipelineConfig | None = None, local: bool = False) -> dict:
    config = config or PipelineConfig.from_env()
    return PipelineExecutor(config, local=local).execute(build_full_dag(config.pipeline_name))


def run_training_only(config: PipelineConfig | None = None, local: bool = False) -> dict:
    config = config or PipelineConfig.from_env()
    return PipelineExecutor(config, local=local).execute(build_training_dag(config.pipeline_name))


def run_inference_only(config: PipelineConfig | None = None, local: bool = False) -> dict:
    config = config or PipelineConfig.from_env()
    return PipelineExecutor(config, local=local).execute(build_inference_dag(config.pipeline_name))


def main():
    parser = argparse.ArgumentParser(description="watsonx ML Pipeline")
    parser.add_argument("--mode", choices=["full", "training", "inference", "dag"], default=None)
    parser.add_argument("--local", action="store_true", help="Use local storage (no credentials needed)")
    args = parser.parse_args()

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer() if sys.stdout.isatty() else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )

    config = PipelineConfig.from_env()
    if args.mode:
        config.mode = args.mode

    if config.mode == "dag":
        from src.pipeline.dag import main as dag_main
        dag_main()
        return

    runners = {"training": run_training_only, "inference": run_inference_only}
    runner = runners.get(config.mode, run_full_pipeline)

    try:
        result = runner(config, local=args.local)
        if not result["success"]:
            sys.exit(1)
    except Exception as e:
        log.error("pipeline.crashed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
