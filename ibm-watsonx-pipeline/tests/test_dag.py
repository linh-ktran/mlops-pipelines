"""Tests for the pipeline DAG module."""

import pytest

from src.pipeline.dag import (
    PipelineDAG,
    PipelineNode,
    NodeStatus,
    build_training_dag,
    build_inference_dag,
    build_full_dag,
)


class TestPipelineNode:
    def test_initial_status(self):
        node = PipelineNode(name="test", description="Test node")
        assert node.status == NodeStatus.PENDING

    def test_mark_running(self):
        node = PipelineNode(name="test", description="Test node")
        node.mark_running()
        assert node.status == NodeStatus.RUNNING
        assert node.started_at is not None

    def test_mark_success(self):
        node = PipelineNode(name="test", description="Test node")
        node.mark_running()
        node.mark_success(metadata={"rows": 100})
        assert node.status == NodeStatus.SUCCESS
        assert node.completed_at is not None
        assert node.duration_seconds is not None
        assert node.metadata["rows"] == 100

    def test_mark_failed(self):
        node = PipelineNode(name="test", description="Test node")
        node.mark_running()
        node.mark_failed(error="something broke")
        assert node.status == NodeStatus.FAILED
        assert node.metadata["error"] == "something broke"

    def test_mark_skipped(self):
        node = PipelineNode(name="test", description="Test node")
        node.mark_skipped(reason="not needed")
        assert node.status == NodeStatus.SKIPPED
        assert node.metadata["skip_reason"] == "not needed"


class TestPipelineDAG:
    def test_add_node(self):
        dag = PipelineDAG(name="test-dag")
        dag.add_node("step1", "First step")
        dag.add_node("step2", "Second step", ["step1"])
        assert len(dag.nodes) == 2
        assert dag.nodes["step2"].dependencies == ["step1"]

    def test_execution_order_simple(self):
        dag = PipelineDAG(name="test-dag")
        dag.add_node("a", "Step A")
        dag.add_node("b", "Step B", ["a"])
        dag.add_node("c", "Step C", ["b"])
        order = dag.get_execution_order()
        assert order == ["a", "b", "c"]

    def test_execution_order_diamond(self):
        """Test diamond dependency: a → b, a → c, b → d, c → d"""
        dag = PipelineDAG(name="test-dag")
        dag.add_node("a", "Step A")
        dag.add_node("b", "Step B", ["a"])
        dag.add_node("c", "Step C", ["a"])
        dag.add_node("d", "Step D", ["b", "c"])
        order = dag.get_execution_order()
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_cycle_detection(self):
        dag = PipelineDAG(name="test-dag")
        dag.add_node("a", "Step A", ["c"])
        dag.add_node("b", "Step B", ["a"])
        dag.add_node("c", "Step C", ["b"])
        with pytest.raises(ValueError, match="Cycle detected"):
            dag.get_execution_order()

    def test_validate_missing_dependency(self):
        dag = PipelineDAG(name="test-dag")
        dag.add_node("a", "Step A", ["nonexistent"])
        errors = dag.validate()
        assert len(errors) > 0
        assert "nonexistent" in errors[0]

    def test_validate_valid_dag(self):
        dag = build_training_dag()
        errors = dag.validate()
        assert errors == []

    def test_get_ready_nodes(self):
        dag = PipelineDAG(name="test-dag")
        dag.add_node("a", "Step A")
        dag.add_node("b", "Step B", ["a"])
        dag.add_node("c", "Step C", ["a"])

        # Initially only 'a' is ready (no dependencies)
        assert dag.get_ready_nodes() == ["a"]

        # After 'a' succeeds, 'b' and 'c' are ready
        dag.nodes["a"].mark_success()
        ready = dag.get_ready_nodes()
        assert "b" in ready
        assert "c" in ready

    def test_render_graphviz_dot(self):
        dag = build_training_dag()
        dot_source = dag.render_graphviz()
        assert "load_data" in dot_source
        assert "train_model" in dot_source
        assert "->" in dot_source

    def test_to_json(self):
        dag = build_training_dag()
        dag.run_id = "test_run_001"
        json_str = dag.to_json()
        assert "test_run_001" in json_str
        assert "load_data" in json_str
        assert "pending" in json_str


class TestDAGBuilders:
    def test_training_dag_structure(self):
        dag = build_training_dag()
        assert "load_data" in dag.nodes
        assert "feature_engineering" in dag.nodes
        assert "train_model" in dag.nodes
        assert "evaluate_model" in dag.nodes
        assert "save_to_cos" in dag.nodes
        assert "register_watsonx" in dag.nodes
        # Validate execution is possible
        order = dag.get_execution_order()
        assert len(order) == 6

    def test_inference_dag_structure(self):
        dag = build_inference_dag()
        assert "load_data" in dag.nodes
        assert "predict" in dag.nodes
        assert "save_forecasts" in dag.nodes
        order = dag.get_execution_order()
        assert len(order) == 5

    def test_full_dag_structure(self):
        dag = build_full_dag()
        assert "deploy_model" in dag.nodes
        order = dag.get_execution_order()
        assert len(order) == 9
        # deploy must come after register
        assert order.index("register_watsonx") < order.index("deploy_model")
