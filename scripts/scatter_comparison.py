#!/usr/bin/env python3
"""
Script to generate scatter plots comparing modularity scores of Girvan-Newman vs Louvain algorithms
for repositories where both algorithms were run.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_benchmark_results(filepath):
    """Load the benchmark results from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return data["benchmark_results"]


def filter_common_repos(results):
    """
    Filter results to include only repos where both algorithms were run
    on the same graph type.
    Returns two lists: modularity scores for louvain (x-axis) and girvan_newman (y-axis).
    """
    # Group by (project_name, graph_type)
    grouped = {}
    for result in results:
        key = (result["project_name"], result["graph_type"])
        if key not in grouped:
            grouped[key] = {}
        grouped[key][result["algorithm"]] = result

    # Find repos where both algorithms were run
    louvain_modularity = []
    girvan_newman_modularity = []

    for key, algorithms in grouped.items():
        if "girvan_newman" in algorithms and "louvain" in algorithms:
            louvain_modularity.append(algorithms["louvain"]["modularity"])
            girvan_newman_modularity.append(algorithms["girvan_newman"]["modularity"])

    return louvain_modularity, girvan_newman_modularity


def create_scatterplot(louvain_modularity, girvan_newman_modularity, output_path):
    """Create and save a scatter plot comparing the two algorithms."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create scatter plot
    ax.scatter(
        louvain_modularity,
        girvan_newman_modularity,
        alpha=0.6,
        s=100,
        c="#4ECDC4",
        edgecolors="black",
        linewidth=1,
    )

    # Add 45-degree reference line
    min_val = min(min(louvain_modularity), min(girvan_newman_modularity))
    max_val = max(max(louvain_modularity), max(girvan_newman_modularity))
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        linewidth=2,
        label="Equal Performance (45° line)",
        alpha=0.7,
    )

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # Labels and title
    ax.set_xlabel("Louvain Modularity Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("Girvan-Newman Modularity Score", fontsize=12, fontweight="bold")
    ax.set_title(
        "Modularity Score Comparison: Girvan-Newman vs Louvain",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Add legend
    ax.legend(loc="upper left", fontsize=10)

    # Add statistics text
    correlation = np.corrcoef(louvain_modularity, girvan_newman_modularity)[0, 1]
    stats_text = (
        f"Statistics:\n"
        f"  Correlation: {correlation:.4f}\n"
        f"  Sample size: {len(louvain_modularity)}\n\n"
        f"Louvain (x-axis):\n"
        f"  Mean: {np.mean(louvain_modularity):.4f}\n"
        f"  Std: {np.std(louvain_modularity):.4f}\n\n"
        f"Girvan-Newman (y-axis):\n"
        f"  Mean: {np.mean(girvan_newman_modularity):.4f}\n"
        f"  Std: {np.std(girvan_newman_modularity):.4f}"
    )

    ax.text(
        0.98,
        0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Make axes equal for better comparison
    ax.set_aspect("equal", adjustable="box")

    # Tight layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Scatter plot saved to: {output_path}")

    # Show plot
    plt.show()

    return fig


def main():
    # Define paths
    project_root = Path(__file__).parent.parent
    benchmark_file = project_root / "codevoyant_output" / "benchmark_results.json"
    output_file = (
        project_root / "codevoyant_output" / "algorithm_comparison_scatter.png"
    )

    # Load results
    results = load_benchmark_results(benchmark_file)

    # Filter for common repos
    louvain_modularity, girvan_newman_modularity = filter_common_repos(results)

    # Create scatter plot
    create_scatterplot(louvain_modularity, girvan_newman_modularity, output_file)


if __name__ == "__main__":
    main()
