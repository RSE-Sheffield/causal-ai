"""Tests for causal_ai.utils.visualise."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pytest

from causal_ai.utils.visualise import (
    _compute_layout,
    _format_node_label,
    _get_adjustment_nodes,
    _sanitise_filename,
    draw_summary_heatmap,
    draw_test_on_dag,
    load_dag_as_networkx,
    visualise_results,
)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from conftest import SAMPLE_DAG_DOT, SAMPLE_RESULTS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dag_dot_file(tmp_path):
    """Write a sample .dot file and return its path."""
    p = tmp_path / "dag.dot"
    p.write_text(SAMPLE_DAG_DOT)
    return p


@pytest.fixture
def results_json_file(tmp_path):
    """Write sample results JSON and return its path."""
    p = tmp_path / "results.json"
    p.write_text(json.dumps(SAMPLE_RESULTS))
    return p


@pytest.fixture
def dag(dag_dot_file):
    """Load the sample DAG as a NetworkX DiGraph."""
    return load_dag_as_networkx(dag_dot_file)


# ---------------------------------------------------------------------------
# load_dag_as_networkx
# ---------------------------------------------------------------------------

class TestLoadDag:
    def test_returns_digraph(self, dag):
        assert isinstance(dag, nx.DiGraph)

    def test_expected_nodes(self, dag):
        assert set(dag.nodes()) == {
            "fp_precision", "gpu_memory_peak_mb",
            "batch_size", "training_time_seconds", "seed",
        }

    def test_expected_edges(self, dag):
        assert ("fp_precision", "gpu_memory_peak_mb") in dag.edges()
        assert ("batch_size", "training_time_seconds") in dag.edges()
        assert len(dag.edges()) == 2

    def test_ignores_meta_nodes(self, dag):
        for meta in ("node", "edge", "graph", ""):
            assert meta not in dag.nodes()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_format_node_label_replaces_underscores(self):
        assert _format_node_label("gpu_memory_peak_mb") == "gpu\nmemory\npeak\nmb"

    def test_format_node_label_no_underscores(self):
        assert _format_node_label("seed") == "seed"

    def test_sanitise_filename_causal(self):
        result = _sanitise_filename("fp_precision --> gpu_memory_peak_mb")
        assert "causes" in result
        assert "/" not in result and " " not in result

    def test_sanitise_filename_independence(self):
        result = _sanitise_filename("seed _||_ gpu_memory_peak_mb")
        assert "indep" in result

    def test_sanitise_filename_conditional(self):
        result = _sanitise_filename("a _||_ b | [c, d]")
        assert "given" in result

    def test_sanitise_filename_truncates(self):
        long_name = "a" * 200
        assert len(_sanitise_filename(long_name)) <= 120

    def test_get_adjustment_nodes_from_result(self):
        test = {"result": {"adjustment_set": ["fp_precision", "batch_size"]}}
        assert _get_adjustment_nodes(test) == ["fp_precision", "batch_size"]

    def test_get_adjustment_nodes_empty(self):
        test = {"result": {"adjustment_set": []}, "name": "a --> b"}
        assert _get_adjustment_nodes(test) == []

    def test_get_adjustment_nodes_from_name(self):
        test = {
            "result": {"adjustment_set": []},
            "name": "a _||_ b | ['c', 'd']",
        }
        nodes = _get_adjustment_nodes(test)
        assert "c" in nodes and "d" in nodes


# ---------------------------------------------------------------------------
# _compute_layout
# ---------------------------------------------------------------------------

class TestComputeLayout:
    def test_returns_positions_for_all_nodes(self, dag):
        pos = _compute_layout(dag.copy())
        assert set(pos.keys()) == set(dag.nodes())

    def test_positions_are_deterministic(self, dag):
        pos1 = _compute_layout(dag.copy())
        pos2 = _compute_layout(dag.copy())
        for node in pos1:
            assert pos1[node] == pytest.approx(pos2[node])

    def test_positions_are_tuples(self, dag):
        pos = _compute_layout(dag.copy())
        for v in pos.values():
            assert len(v) == 2


# ---------------------------------------------------------------------------
# draw_test_on_dag
# ---------------------------------------------------------------------------

class TestDrawTestOnDag:
    def test_draws_without_error(self, dag):
        pos = _compute_layout(dag.copy())
        fig, ax = plt.subplots()
        draw_test_on_dag(dag, pos, SAMPLE_RESULTS[0], test_index=1, ax=ax)
        plt.close(fig)

    def test_draws_failed_test(self, dag):
        pos = _compute_layout(dag.copy())
        fig, ax = plt.subplots()
        draw_test_on_dag(dag, pos, SAMPLE_RESULTS[2], test_index=3, ax=ax)
        plt.close(fig)

    def test_title_contains_status(self, dag):
        pos = _compute_layout(dag.copy())
        fig, ax = plt.subplots()
        draw_test_on_dag(dag, pos, SAMPLE_RESULTS[0], test_index=1, ax=ax)
        assert "Passed" in ax.get_title()
        plt.close(fig)

        fig, ax = plt.subplots()
        draw_test_on_dag(dag, pos, SAMPLE_RESULTS[2], test_index=3, ax=ax)
        assert "Failed" in ax.get_title()
        plt.close(fig)

    def test_skipped_test(self, dag):
        pos = _compute_layout(dag.copy())
        skipped = {**SAMPLE_RESULTS[0], "skip": True}
        fig, ax = plt.subplots()
        draw_test_on_dag(dag, pos, skipped, test_index=1, ax=ax)
        assert "Skipped" in ax.get_title()
        plt.close(fig)


# ---------------------------------------------------------------------------
# draw_summary_heatmap
# ---------------------------------------------------------------------------

class TestDrawSummaryHeatmap:
    def test_creates_file(self, tmp_path):
        out = tmp_path / "summary.png"
        result = draw_summary_heatmap(SAMPLE_RESULTS, out)
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_handles_all_passed(self, tmp_path):
        all_passed = [{**r, "passed": True, "skip": False} for r in SAMPLE_RESULTS]
        out = tmp_path / "all_pass.png"
        draw_summary_heatmap(all_passed, out)
        assert out.exists()

    def test_handles_all_failed(self, tmp_path):
        all_failed = [{**r, "passed": False, "skip": False} for r in SAMPLE_RESULTS]
        out = tmp_path / "all_fail.png"
        draw_summary_heatmap(all_failed, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# visualise_results (end-to-end)
# ---------------------------------------------------------------------------

class TestVisualiseResults:
    def test_creates_output_structure(self, dag_dot_file, results_json_file, tmp_path):
        out_dir = tmp_path / "vis_output"
        visualise_results(dag_dot_file, results_json_file, out_dir)

        assert (out_dir / "summary.png").exists()
        assert (out_dir / "causal_tests").is_dir()

        test_pngs = list((out_dir / "causal_tests").glob("*.png"))
        assert len(test_pngs) == len(SAMPLE_RESULTS)

    def test_filenames_are_numbered(self, dag_dot_file, results_json_file, tmp_path):
        out_dir = tmp_path / "vis_output"
        visualise_results(dag_dot_file, results_json_file, out_dir)

        test_pngs = sorted((out_dir / "causal_tests").glob("*.png"))
        assert test_pngs[0].name.startswith("01_")
        assert test_pngs[-1].name.startswith(f"{len(SAMPLE_RESULTS):02d}_")
