#!/usr/bin/env python3
"""
Script to generate box plots comparing modularity scores of Girvan-Newman and Louvain algorithms
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
    Returns two lists: modularity scores for girvan_newman and louvain.
    """
    # Group by (project_name, graph_type)
    grouped = {}
    for result in results:
        key = (result["project_name"], result["graph_type"])
        if key not in grouped:
            grouped[key] = {}
        grouped[key][result["algorithm"]] = result

    # Find repos where both algorithms were run
    girvan_newman_modularity = []
    louvain_modularity = []

    for key, algorithms in grouped.items():
        if "girvan_newman" in algorithms and "louvain" in algorithms:
            girvan_newman_modularity.append(algorithms["girvan_newman"]["modularity"])
            louvain_modularity.append(algorithms["louvain"]["modularity"])

    return girvan_newman_modularity, louvain_modularity


def create_boxplot(girvan_newman_modularity, louvain_modularity, output_path):
    """Create and save a box plot comparing the two algorithms."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for box plot
    data = [girvan_newman_modularity, louvain_modularity]
    labels = ["Girvan-Newman", "Louvain"]

    # Create box plot
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)

    # Customize colors
    colors = ["#FF6B6B", "#4ECDC4"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Customize whiskers, caps, and medians
    for whisker in bp["whiskers"]:
        whisker.set(linewidth=1.5)
    for cap in bp["caps"]:
        cap.set(linewidth=1.5)
    for median in bp["medians"]:
        median.set(linewidth=2, color="black")
    for mean in bp["means"]:
        mean.set(
            marker="D", markerfacecolor="red", markeredgecolor="black", markersize=8
        )

    # Add grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Labels and title
    ax.set_ylabel("Modularity Score", fontsize=12, fontweight="bold")
    ax.set_title(
        "Algorithm Modularity Comparison",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Add statistics text
    stats_text = (
        f"Girvan-Newman:\n"
        f"  Mean: {np.mean(girvan_newman_modularity):.4f}\n"
        f"  Median: {np.median(girvan_newman_modularity):.4f}\n"
        f"  Std: {np.std(girvan_newman_modularity):.4f}\n\n"
        f"Louvain:\n"
        f"  Mean: {np.mean(louvain_modularity):.4f}\n"
        f"  Median: {np.median(louvain_modularity):.4f}\n"
        f"  Std: {np.std(louvain_modularity):.4f}\n\n"
        f"Sample size: {len(girvan_newman_modularity)}"
    )

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Tight layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Box plot saved to: {output_path}")

    # Show plot
    plt.show()

    return fig


def main():
    # Define paths
    project_root = Path(__file__).parent.parent
    benchmark_file = project_root / "codevoyant_output" / "benchmark_results.json"
    output_file = (
        project_root / "codevoyant_output" / "algorithm_comparison_boxplot.png"
    )

    # Load results
    results = load_benchmark_results(benchmark_file)

    # Filter for common repos
    girvan_newman_modularity, louvain_modularity = filter_common_repos(results)

    # Create box plot
    create_boxplot(girvan_newman_modularity, louvain_modularity, output_file)


if __name__ == "__main__":
    main()
