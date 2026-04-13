"""Main orchestration logic for causal-ai.

Provides functions to load CTF artifacts (DAGs, tests, results, runtime data)
and summarise causal test outcomes across HPC clusters.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def load_dag(path: Path) -> str:
    """Load a causal DAG from a DOT file.

    Args:
        path: Path to the .dot file.

    Returns:
        The DOT source string.
    """
    return path.read_text()


def load_tests(path: Path) -> List[Dict[str, Any]]:
    """Load causal test definitions from a JSON file.

    Args:
        path: Path to the causal tests JSON.

    Returns:
        List of test definition dicts.
    """
    with open(path) as f:
        data = json.load(f)
    return data.get("tests", data) if isinstance(data, dict) else data


def load_results(path: Path) -> List[Dict[str, Any]]:
    """Load causal test results from a JSON file.

    Args:
        path: Path to the causal test results JSON.

    Returns:
        List of result dicts.
    """
    with open(path) as f:
        return json.load(f)


def load_runtime_data(path: Path) -> pd.DataFrame:
    """Load runtime data from a CSV file.

    Args:
        path: Path to the runtime data CSV.

    Returns:
        DataFrame of runtime observations.
    """
    return pd.read_csv(path)


def summarise_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Produce a summary of causal test results.

    Args:
        results: List of result dicts as returned by :func:`load_results`.

    Returns:
        Dictionary with total, passed, failed counts and lists of
        failed/skipped test names.
    """
    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    skipped = [r["name"] for r in results if r.get("skip")]
    failed = [r["name"] for r in results if not r.get("passed") and not r.get("skip")]

    return {
        "total": total,
        "passed": passed,
        "failed_count": len(failed),
        "skipped_count": len(skipped),
        "failed_tests": failed,
        "skipped_tests": skipped,
    }


def compare_clusters(
    cluster_results: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Compare causal test results across HPC clusters.

    Args:
        cluster_results: Mapping of cluster name to its list of result dicts.

    Returns:
        Dictionary with per-cluster summaries and cross-cluster comparison.
    """
    summaries = {
        name: summarise_results(results) for name, results in cluster_results.items()
    }

    # Find tests that passed on one cluster but failed on another
    cluster_names = list(summaries.keys())
    divergent: List[str] = []
    if len(cluster_names) == 2:
        a, b = cluster_names
        failed_a = set(summaries[a]["failed_tests"])
        failed_b = set(summaries[b]["failed_tests"])
        divergent = sorted((failed_a - failed_b) | (failed_b - failed_a))

    return {
        "per_cluster": summaries,
        "divergent_tests": divergent,
    }


def load_cluster_data(data_dir: Path) -> Dict[str, Any]:
    """Load all CTF artifacts for a single cluster directory.

    Expects the directory to contain a DAG (.dot), tests (.json with 'test'
    in the name), results (.json with 'result' in the name), and runtime
    data (.csv).

    Args:
        data_dir: Path to a cluster data directory.

    Returns:
        Dictionary with keys: dag, tests, results, runtime_data.
    """
    dag_file = next(data_dir.glob("*.dot"), None)
    test_files = list(data_dir.glob("*test*[!result]*.json"))
    result_files = list(data_dir.glob("*result*.json"))
    csv_files = list(data_dir.glob("*.csv"))

    artifacts: Dict[str, Any] = {}

    if dag_file:
        artifacts["dag"] = load_dag(dag_file)
    if test_files:
        artifacts["tests"] = load_tests(test_files[0])
    if result_files:
        artifacts["results"] = load_results(result_files[0])
    if csv_files:
        artifacts["runtime_data"] = load_runtime_data(csv_files[0])

    return artifacts
