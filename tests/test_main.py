"""Tests for causal_ai.main — loading CTF artifacts and summarising results."""

import json
from pathlib import Path

import pandas as pd
import pytest

from causal_ai.main import (
    compare_clusters,
    load_cluster_data,
    load_dag,
    load_results,
    load_runtime_data,
    load_tests,
    summarise_results,
)
from tests.conftest import (
    SAMPLE_DAG_DOT,
    SAMPLE_RESULTS,
    SAMPLE_RUNTIME_CSV,
    SAMPLE_TESTS,
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

class TestLoadDag:
    def test_loads_dot_string(self, tmp_path):
        f = tmp_path / "dag.dot"
        f.write_text(SAMPLE_DAG_DOT)
        result = load_dag(f)
        assert "digraph dag" in result
        assert "fp_precision -> gpu_memory_peak_mb" in result


class TestLoadTests:
    def test_loads_tests_with_wrapper(self, tmp_path):
        f = tmp_path / "tests.json"
        f.write_text(json.dumps(SAMPLE_TESTS))
        result = load_tests(f)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["name"] == "seed _||_ gpu_memory_peak_mb"

    def test_loads_bare_list(self, tmp_path):
        f = tmp_path / "tests.json"
        f.write_text(json.dumps(SAMPLE_TESTS["tests"]))
        result = load_tests(f)
        assert isinstance(result, list)
        assert len(result) == 2


class TestLoadResults:
    def test_loads_results_list(self, tmp_path):
        f = tmp_path / "results.json"
        f.write_text(json.dumps(SAMPLE_RESULTS))
        result = load_results(f)
        assert isinstance(result, list)
        assert len(result) == 3
        assert "passed" in result[0]


class TestLoadRuntimeData:
    def test_loads_csv_as_dataframe(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text(SAMPLE_RUNTIME_CSV)
        df = load_runtime_data(f)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert "learning_rate" in df.columns
        assert "gpu_memory_peak_mb" in df.columns


# ---------------------------------------------------------------------------
# Summarise results
# ---------------------------------------------------------------------------

class TestSummariseResults:
    def test_counts(self):
        summary = summarise_results(SAMPLE_RESULTS)
        assert summary["total"] == 3
        assert summary["passed"] == 2
        assert summary["failed_count"] == 1
        assert summary["skipped_count"] == 0

    def test_failed_test_names(self):
        summary = summarise_results(SAMPLE_RESULTS)
        assert "batch_size --> training_time_seconds" in summary["failed_tests"]

    def test_empty_results(self):
        summary = summarise_results([])
        assert summary["total"] == 0
        assert summary["passed"] == 0
        assert summary["failed_count"] == 0

    def test_skipped_tests(self):
        results_with_skip = [
            {"name": "skipped_test", "skip": True, "passed": False},
            {"name": "normal_test", "skip": False, "passed": True},
        ]
        summary = summarise_results(results_with_skip)
        assert summary["skipped_count"] == 1
        assert "skipped_test" in summary["skipped_tests"]
        assert summary["failed_count"] == 0  # skipped tests aren't counted as failed


# ---------------------------------------------------------------------------
# Compare clusters
# ---------------------------------------------------------------------------

class TestCompareClusters:
    def test_two_identical_clusters(self):
        cluster_results = {
            "stanage": SAMPLE_RESULTS,
            "bede": SAMPLE_RESULTS,
        }
        comparison = compare_clusters(cluster_results)
        assert len(comparison["per_cluster"]) == 2
        assert comparison["divergent_tests"] == []

    def test_divergent_tests_detected(self):
        bede_results = json.loads(json.dumps(SAMPLE_RESULTS))
        bede_results[2]["passed"] = True  # flip the failed test

        comparison = compare_clusters({
            "stanage": SAMPLE_RESULTS,
            "bede": bede_results,
        })
        assert "batch_size --> training_time_seconds" in comparison["divergent_tests"]

    def test_single_cluster(self):
        comparison = compare_clusters({"stanage": SAMPLE_RESULTS})
        assert len(comparison["per_cluster"]) == 1
        assert comparison["divergent_tests"] == []


# ---------------------------------------------------------------------------
# Load cluster data (auto-discovery)
# ---------------------------------------------------------------------------

class TestLoadClusterData:
    def test_loads_all_artifacts(self, cluster_dir):
        artifacts = load_cluster_data(cluster_dir)
        assert "dag" in artifacts
        assert "tests" in artifacts
        assert "results" in artifacts
        assert "runtime_data" in artifacts
        assert isinstance(artifacts["runtime_data"], pd.DataFrame)

    def test_missing_artifacts_omitted(self, tmp_path):
        d = tmp_path / "empty_cluster"
        d.mkdir()
        artifacts = load_cluster_data(d)
        assert artifacts == {}

    def test_partial_artifacts(self, tmp_path):
        d = tmp_path / "partial"
        d.mkdir()
        (d / "dag.dot").write_text(SAMPLE_DAG_DOT)
        artifacts = load_cluster_data(d)
        assert "dag" in artifacts
        assert "results" not in artifacts
