"""CLI entry point for causal-ai.

Usage:
    python -m causal_ai summary <data_dir>
    python -m causal_ai compare <data_dir>
    python -m causal_ai visualise --dag <dot> --results <json> [--output_dir <dir>]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from causal_ai.utils import (
    compare_clusters,
    load_cluster_data,
    summarise_results,
    visualise_results,
)

logger = logging.getLogger("causal_ai")


def cmd_summary(args: argparse.Namespace) -> None:
    """Print a summary of causal test results for a single cluster directory."""
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit(f"Error: {data_dir} is not a directory")

    artifacts = load_cluster_data(data_dir)
    if "results" not in artifacts:
        sys.exit(f"Error: no results JSON found in {data_dir}")

    summary = summarise_results(artifacts["results"])
    print(f"Cluster directory: {data_dir}")
    print(f"  Total tests:   {summary['total']}")
    print(f"  Passed:        {summary['passed']}")
    print(f"  Failed:        {summary['failed_count']}")
    print(f"  Skipped:       {summary['skipped_count']}")
    if summary["failed_tests"]:
        print("\n  Failed tests:")
        for name in summary["failed_tests"]:
            print(f"    - {name}")

    if args.json:
        print("\n" + json.dumps(summary, indent=2))


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare causal test results across cluster subdirectories."""
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit(f"Error: {data_dir} is not a directory")

    # Discover cluster subdirectories (each should contain CTF artifacts)
    cluster_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not cluster_dirs:
        sys.exit(f"Error: no cluster subdirectories found in {data_dir}")

    cluster_results = {}
    for cluster_dir in cluster_dirs:
        artifacts = load_cluster_data(cluster_dir)
        if "results" in artifacts:
            cluster_results[cluster_dir.name] = artifacts["results"]
        else:
            logger.warning("No results found in %s, skipping", cluster_dir)

    if not cluster_results:
        sys.exit("Error: no cluster results found")

    comparison = compare_clusters(cluster_results)

    for name, summary in comparison["per_cluster"].items():
        print(f"\n[{name}]")
        print(f"  Total: {summary['total']}  Passed: {summary['passed']}  "
              f"Failed: {summary['failed_count']}  Skipped: {summary['skipped_count']}")
        if summary["failed_tests"]:
            print("  Failed:")
            for t in summary["failed_tests"]:
                print(f"    - {t}")

    if comparison["divergent_tests"]:
        print("\nDivergent tests (passed on one cluster, failed on another):")
        for t in comparison["divergent_tests"]:
            print(f"  - {t}")
    else:
        print("\nNo divergent tests — both clusters agree on all outcomes.")

    if args.json:
        print("\n" + json.dumps(comparison, indent=2))


def cmd_visualise(args: argparse.Namespace) -> None:
    """Generate visualisations of causal test results on the DAG."""
    dag_path = Path(args.dag)
    results_path = Path(args.results)
    output_dir = Path(args.output_dir)

    if not dag_path.is_file():
        sys.exit(f"Error: DAG file not found: {dag_path}")
    if not results_path.is_file():
        sys.exit(f"Error: results file not found: {results_path}")

    visualise_results(dag_path, results_path, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="causal_ai",
        description="Causal inference for AI/ML workflows",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # summary
    p_summary = subparsers.add_parser("summary", help="Summarise results for one cluster")
    p_summary.add_argument("data_dir", help="Path to a cluster data directory")
    p_summary.add_argument("--json", action="store_true", help="Also print raw JSON output")

    # compare
    p_compare = subparsers.add_parser("compare", help="Compare results across clusters")
    p_compare.add_argument("data_dir", help="Path to parent directory containing cluster subdirs")
    p_compare.add_argument("--json", action="store_true", help="Also print raw JSON output")

    # visualise
    p_vis = subparsers.add_parser("visualise", help="Visualise causal test results on the DAG")
    p_vis.add_argument("--dag", required=True, help="Path to the DAG .dot file")
    p_vis.add_argument("--results", required=True, help="Path to the causal test results JSON")
    p_vis.add_argument("--output_dir", default="visualisations",
                       help="Output directory for PNGs (default: visualisations/)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.command == "summary":
        cmd_summary(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "visualise":
        cmd_visualise(args)


if __name__ == "__main__":
    main()
