"""
Pipeline DAG — defines steps, dependencies, and renders visual graph.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineNode:
    name: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    metadata: dict = field(default_factory=dict)

    def mark_running(self):
        self.status = NodeStatus.RUNNING
        self.started_at = datetime.now(timezone.utc).isoformat()

    def mark_success(self, metadata: dict | None = None):
        self.status = NodeStatus.SUCCESS
        self.completed_at = datetime.now(timezone.utc).isoformat()
        if self.started_at:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.completed_at)
            self.duration_seconds = (end - start).total_seconds()
        if metadata:
            self.metadata.update(metadata)

    def mark_failed(self, error: str):
        self.status = NodeStatus.FAILED
        self.completed_at = datetime.now(timezone.utc).isoformat()
        self.metadata["error"] = error

    def mark_skipped(self, reason: str = ""):
        self.status = NodeStatus.SKIPPED
        self.metadata["skip_reason"] = reason


@dataclass
class PipelineDAG:
    name: str
    nodes: dict[str, PipelineNode] = field(default_factory=dict)
    run_id: str = ""

    def add_node(self, name: str, description: str, dependencies: list[str] | None = None) -> PipelineNode:
        node = PipelineNode(name=name, description=description, dependencies=dependencies or [])
        self.nodes[name] = node
        return node

    def get_execution_order(self) -> list[str]:
        """Topological sort. Raises ValueError on cycle."""
        visited: set[str] = set()
        order: list[str] = []
        visiting: set[str] = set()

        def dfs(node_name: str):
            if node_name in visiting:
                raise ValueError(f"Cycle detected at '{node_name}'")
            if node_name in visited:
                return
            visiting.add(node_name)
            for dep in self.nodes[node_name].dependencies:
                dfs(dep)
            visiting.remove(node_name)
            visited.add(node_name)
            order.append(node_name)

        for name in self.nodes:
            dfs(name)
        return order

    def validate(self) -> list[str]:
        """Check DAG structure. Returns errors (empty = valid)."""
        errors = []
        for name, node in self.nodes.items():
            for dep in node.dependencies:
                if dep not in self.nodes:
                    errors.append(f"'{name}' depends on unknown node '{dep}'")
        if not errors:
            try:
                self.get_execution_order()
            except ValueError as e:
                errors.append(str(e))
        return errors

    def get_ready_nodes(self) -> list[str]:
        """Nodes whose dependencies are all done."""
        ready = []
        for name, node in self.nodes.items():
            if node.status != NodeStatus.PENDING:
                continue
            all_done = all(
                self.nodes[dep].status in (NodeStatus.SUCCESS, NodeStatus.SKIPPED)
                for dep in node.dependencies
            )
            if all_done:
                ready.append(name)
        return ready

    def render_graphviz(self, output_path: str | None = None) -> str:
        """Render DAG as graphviz. Saves PNG if output_path given."""
        try:
            import graphviz
        except ImportError:
            return self._to_dot()

        colors = {
            NodeStatus.PENDING: "#E0E0E0",
            NodeStatus.RUNNING: "#FFF9C4",
            NodeStatus.SUCCESS: "#C8E6C9",
            NodeStatus.FAILED: "#FFCDD2",
            NodeStatus.SKIPPED: "#B0BEC5",
        }

        dot = graphviz.Digraph(
            name=self.name,
            format="png",
            graph_attr={"rankdir": "TB", "label": f"Pipeline: {self.name}\\nRun: {self.run_id}", "labelloc": "t", "fontsize": "14"},
        )

        for name, node in self.nodes.items():
            label = f"{name}\\n({node.status.value})"
            if node.duration_seconds is not None:
                label += f"\\n{node.duration_seconds:.1f}s"
            dot.node(name, label=label, style="filled", fillcolor=colors.get(node.status, "#E0E0E0"), shape="box")

        for name, node in self.nodes.items():
            for dep in node.dependencies:
                dot.edge(dep, name)

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                dot.render(str(path.with_suffix("")), cleanup=True)
            except Exception:
                path.with_suffix(".dot").write_text(dot.source)

        return dot.source

    def _to_dot(self) -> str:
        lines = [f'digraph "{self.name}" {{', '  rankdir=TB;']
        for name, node in self.nodes.items():
            lines.append(f'  "{name}" [label="{name}\\n({node.status.value})"];')
        for name, node in self.nodes.items():
            for dep in node.dependencies:
                lines.append(f'  "{dep}" -> "{name}";')
        lines.append("}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "run_id": self.run_id,
            "nodes": {
                name: {
                    "description": node.description,
                    "dependencies": node.dependencies,
                    "status": node.status.value,
                    "started_at": node.started_at,
                    "completed_at": node.completed_at,
                    "duration_seconds": node.duration_seconds,
                    "metadata": node.metadata,
                }
                for name, node in self.nodes.items()
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ─── DAG definitions ─────────────────────────────────────────────────────────

def build_training_dag(pipeline_name: str = "watsonx-energy-forecast") -> PipelineDAG:
    dag = PipelineDAG(name=f"{pipeline_name}-training")
    dag.add_node("load_data", "Load raw data from COS")
    dag.add_node("feature_engineering", "Generate temporal + lag features", ["load_data"])
    dag.add_node("train_model", "Train XGBoost", ["feature_engineering"])
    dag.add_node("evaluate_model", "Evaluate on validation set", ["train_model"])
    dag.add_node("save_to_cos", "Save model to COS", ["evaluate_model"])
    dag.add_node("register_watsonx", "Register in watsonx.ai", ["save_to_cos"])
    return dag


def build_inference_dag(pipeline_name: str = "watsonx-energy-forecast") -> PipelineDAG:
    dag = PipelineDAG(name=f"{pipeline_name}-inference")
    dag.add_node("load_data", "Load raw data from COS")
    dag.add_node("feature_engineering", "Generate temporal + lag features", ["load_data"])
    dag.add_node("load_model", "Load model from COS", ["feature_engineering"])
    dag.add_node("predict", "Generate forecasts", ["load_model"])
    dag.add_node("save_forecasts", "Save results to COS", ["predict"])
    return dag


def build_full_dag(pipeline_name: str = "watsonx-energy-forecast") -> PipelineDAG:
    dag = PipelineDAG(name=f"{pipeline_name}-full")
    dag.add_node("load_data", "Load raw data from COS")
    dag.add_node("feature_engineering", "Generate temporal + lag features", ["load_data"])
    dag.add_node("train_model", "Train XGBoost", ["feature_engineering"])
    dag.add_node("evaluate_model", "Evaluate on validation set", ["train_model"])
    dag.add_node("save_to_cos", "Save model to COS", ["evaluate_model"])
    dag.add_node("register_watsonx", "Register in watsonx.ai", ["save_to_cos"])
    dag.add_node("deploy_model", "Deploy to REST endpoint", ["register_watsonx"])
    dag.add_node("predict", "Generate forecasts", ["deploy_model"])
    dag.add_node("save_forecasts", "Save results to COS", ["predict"])
    return dag


def main():
    """CLI: print DAG execution order and render visualization."""
    import sys

    dag = build_full_dag()
    errors = dag.validate()
    if errors:
        print(f"DAG errors: {errors}", file=sys.stderr)
        sys.exit(1)

    print("Pipeline DAG — Execution Order:")
    print("=" * 50)
    for i, name in enumerate(dag.get_execution_order(), 1):
        node = dag.nodes[name]
        deps = f" (after: {', '.join(node.dependencies)})" if node.dependencies else ""
        print(f"  {i}. {name}{deps}")
        print(f"     {node.description}")

    print()
    print(dag.render_graphviz(output_path="dag_output/pipeline_dag.png"))


if __name__ == "__main__":
    main()
