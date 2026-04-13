"""Visualise causal test results on the causal DAG.

Follows the plotting conventions from the Causal Testing Framework tutorial:
https://causal-testing-framework.readthedocs.io/en/latest/tutorials/visualising_causal_test_results/

Produces:
  - Individual PNG files for each causal test (DAG with highlighted edges)
  - A summary heatmap showing pass/fail status across all tests

Usage:
    python -m causal_ai visualise \\
        --dag examples/digits_dann/data/stanage/dag.dot \\
        --results examples/digits_dann/data/stanage/causal_test_results_stanage.json \\
        --output_dir output/visualisations
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)


def load_dag_as_networkx(dot_path: Path) -> nx.DiGraph:
    """Parse a DOT file into a NetworkX DiGraph.

    Args:
        dot_path: Path to the .dot file.

    Returns:
        A NetworkX directed graph.
    """
    import pydot

    graphs = pydot.graph_from_dot_file(str(dot_path))
    graph = graphs[0]
    G = nx.DiGraph()

    for node in graph.get_nodes():
        name = node.get_name().strip('"')
        if name not in ("node", "edge", "graph", ""):
            G.add_node(name)

    for edge in graph.get_edge_list():
        src = edge.get_source().strip('"')
        dst = edge.get_destination().strip('"')
        G.add_edge(src, dst)

    return G


def _get_adjustment_nodes(test: Dict[str, Any]) -> List[str]:
    """Extract adjustment set node names from a test result."""
    adj = test.get("result", {}).get("adjustment_set", [])
    if adj:
        return adj

    name = test.get("name", "")
    if " | " in name and "[" in name:
        bracket = name.split("[")[-1].split("]")[0]
        return [v.strip().strip("'\"") for v in bracket.split(",")]

    return []


def _compute_layout(dag: nx.DiGraph) -> Dict[str, tuple]:
    """Compute a stable layered layout for the DAG.

    The layout is deterministic and based only on the DAG structure,
    so it remains identical across all test plots.
    """
    for layer, nodes in enumerate(sorted(nx.topological_generations(dag))):
        for node in nodes:
            dag.nodes[node]["layer"] = layer

    pos = nx.multipartite_layout(dag, subset_key="layer", align="horizontal")

    # Scale for spacing
    for node in pos:
        x, y = pos[node]
        pos[node] = (x * 2.5, y * 2.0)

    return pos


def _format_node_label(name: str) -> str:
    """Wrap long node names so they fit inside nodes."""
    name = name.replace("_", "\n")
    return name


def _sanitise_filename(name: str) -> str:
    """Convert a test name into a safe filename."""
    name = name.replace(" --> ", "_causes_")
    name = name.replace(" _||_ ", "_indep_")
    name = name.replace(" | ", "_given_")
    name = re.sub(r"[^\w\-.]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:120]


def draw_test_on_dag(
    dag: nx.DiGraph,
    pos: Dict[str, tuple],
    test: Dict[str, Any],
    test_index: int,
    ax: plt.Axes,
) -> None:
    """Draw a single causal test result on the DAG.

    Colour scheme (following CTF tutorial):
      - Treatment node: orange (C1)
      - Outcome node: blue (C0)
      - Adjustment set nodes: purple (C4)
      - Solid line: edge exists in the DAG
      - Dashed line: tested relationship not in DAG (independence test)
      - Green: test passed
      - Red: test failed

    Args:
        dag: The causal DAG as a NetworkX DiGraph.
        pos: Pre-computed node positions (must be the same for every plot).
        test: A single test result dict from the CTF results JSON.
        test_index: 1-based index of the test (used in the title).
        ax: Matplotlib axes to draw on.
    """
    treatment = test["result"]["treatment"]
    outcome = test["result"]["outcome"]
    adjustment_nodes = _get_adjustment_nodes(test)
    passed = test.get("passed", False)
    skipped = test.get("skip", False)

    test_edge = (treatment, outcome)
    edge_in_dag = test_edge in dag.edges()

    # Node colours
    colour_map = []
    for node in dag.nodes():
        if node == treatment:
            colour_map.append("C1")
        elif node == outcome:
            colour_map.append("C0")
        elif node in adjustment_nodes:
            colour_map.append("C4")
        else:
            colour_map.append("white")

    # Node labels with line-wrapped names
    node_labels = {n: _format_node_label(n) for n in dag.nodes()}

    # Draw all DAG edges in black (solid)
    nx.draw_networkx_edges(
        dag,
        pos=pos,
        ax=ax,
        edgelist=list(dag.edges()),
        edge_color="black",
        width=1.0,
        style="-",
        arrows=True,
        arrowsize=12,
        node_size=5000,
        min_source_margin=25,
        min_target_margin=25,
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        dag,
        pos=pos,
        ax=ax,
        node_size=5000,
        node_color=colour_map,
        edgecolors="black",
        linewidths=1.0,
    )

    # Draw labels
    nx.draw_networkx_labels(
        dag,
        pos=pos,
        ax=ax,
        labels=node_labels,
        font_size=7,
    )

    # Test edge colour
    if skipped:
        test_colour = "C7"
    elif passed:
        test_colour = "C2"
    else:
        test_colour = "C3"

    # Solid if in DAG, dashed if not
    test_style = "-" if edge_in_dag else "--"

    # Draw the test edge on top
    nx.draw_networkx_edges(
        dag if edge_in_dag else nx.DiGraph([(treatment, outcome)]),
        pos=pos,
        ax=ax,
        edgelist=[test_edge],
        edge_color=[test_colour],
        width=2.5,
        style=test_style,
        arrows=True,
        arrowsize=15,
        node_size=5000,
        min_source_margin=25,
        min_target_margin=25,
    )

    # Legend
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="C1",
            markersize=9,
            markeredgecolor="black",
            label="Treatment",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="C0",
            markersize=9,
            markeredgecolor="black",
            label="Outcome",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="C4",
            markersize=9,
            markeredgecolor="black",
            label="Adjustment set",
        ),
        Line2D([], [], color="C2", ls="-", lw=1.5, label="Passed (in DAG)"),
        Line2D([], [], color="C2", ls="--", lw=1.5, label="Passed (not in DAG)"),
        Line2D([], [], color="C3", ls="-", lw=1.5, label="Failed (in DAG)"),
        Line2D([], [], color="C3", ls="--", lw=1.5, label="Failed (not in DAG)"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=7,
        framealpha=0.9,
        edgecolor="0.8",
    )

    status = "Skipped" if skipped else ("Passed" if passed else "Failed")
    ax.set_title(f"Test {test_index}: {test['name']}  [{status}]", fontsize=9)
    ax.margins(0.25)
    ax.axis("off")


def draw_summary_heatmap(
    results: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Draw a summary of all causal test results.

    Args:
        results: List of causal test result dicts.
        output_path: Path to save the summary PNG.

    Returns:
        Path to the saved PNG.
    """
    labels = []
    statuses = []

    for test in results:
        skipped = test.get("skip", False)
        passed = test.get("passed", False)
        labels.append(test["name"])
        if skipped:
            statuses.append(0)
        elif passed:
            statuses.append(1)
        else:
            statuses.append(-1)

    n_tests = len(results)
    n_passed = statuses.count(1)
    n_failed = statuses.count(-1)
    n_skipped = statuses.count(0)

    colours = {-1: "#d9534f", 0: "#999999", 1: "#5cb85c"}
    status_text = {-1: "Failed", 0: "Skipped", 1: "Passed"}

    fig_height = max(3, n_tests * 0.38 + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    bar_height = 0.7
    for i, (label, status) in enumerate(zip(labels, statuses)):
        ax.barh(
            i,
            1,
            color=colours[status],
            height=bar_height,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.text(
            0.5,
            i,
            status_text[status],
            ha="center",
            va="center",
            fontsize=8,
            color="white",
        )

    ax.set_yticks(range(n_tests))
    ax.set_yticklabels(labels, fontsize=7, fontfamily="monospace")
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.invert_yaxis()

    ax.set_title(
        f"Causal Test Results Summary\n"
        f"Passed: {n_passed}   Failed: {n_failed}   Skipped: {n_skipped}   Total: {n_tests}",
        fontsize=10,
        pad=12,
    )

    # Legend outside the plot area
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#5cb85c", label="Passed"),
        plt.Rectangle((0, 0), 1, 1, fc="#d9534f", label="Failed"),
        plt.Rectangle((0, 0), 1, 1, fc="#999999", label="Skipped"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, edgecolor="0.8")

    ax.spines[:].set_visible(False)
    ax.tick_params(left=False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def visualise_results(
    dag_path: Path,
    results_path: Path,
    output_dir: Path,
) -> Path:
    """Generate visualisations of causal test results.

    Creates:
      - One PNG per test in output_dir/causal_tests/
      - A summary heatmap at output_dir/summary.png

    Args:
        dag_path: Path to the causal DAG .dot file.
        results_path: Path to the causal test results JSON.
        output_dir: Directory to save all output PNGs.

    Returns:
        Path to the output directory.
    """
    dag = load_dag_as_networkx(dag_path)

    with open(results_path) as f:
        results = json.load(f)

    # Compute layout once so all plots share the same node positions
    pos = _compute_layout(dag.copy())

    # Create output directories
    tests_dir = output_dir / "causal_tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    # Generate individual test plots
    for i, test in enumerate(results):
        fig, ax = plt.subplots(figsize=(14, 8))
        draw_test_on_dag(dag, pos, test, test_index=i + 1, ax=ax)
        fig.tight_layout()

        filename = f"{i + 1:02d}_{_sanitise_filename(test['name'])}.png"
        filepath = tests_dir / filename
        fig.savefig(str(filepath), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved %s", filepath.name)

    # Generate summary heatmap
    summary_path = output_dir / "summary.png"
    draw_summary_heatmap(results, summary_path)

    print(f"Saved {len(results)} individual test plots to {tests_dir}/")
    print(f"Saved summary heatmap to {summary_path}")

    return output_dir
