"""
Module for plotting benchmark results.
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .runner import BenchmarkResult


class BenchmarkPlotter:
    """Creates visualizations of benchmark results."""

    @staticmethod
    def _extract_common_repos_data(results: List[BenchmarkResult], metric: str):
        """
        Extract metric data for repos where both algorithms were run.

        Args:
            results: List of BenchmarkResult objects
            metric: Name of the metric attribute (e.g., 'modularity', 'density', 'conductance')

        Returns:
            Tuple of (louvain_values, girvan_newman_values) as lists
        """
        from collections import defaultdict
        from typing import Dict, Tuple

        # Group by (project_name, graph_type)
        grouped: Dict[Tuple[str, str], Dict[str, BenchmarkResult]] = defaultdict(dict)
        for result in results:
            key = (result.project_name, result.graph_type)
            grouped[key][result.algorithm] = result

        # Find repos where both algorithms were run
        louvain_values = []
        girvan_newman_values = []

        for key, algorithms in grouped.items():
            if "girvan_newman" in algorithms and "louvain" in algorithms:
                louvain_values.append(getattr(algorithms["louvain"], metric))
                girvan_newman_values.append(
                    getattr(algorithms["girvan_newman"], metric)
                )

        return louvain_values, girvan_newman_values

    @staticmethod
    def _create_scatterplot(
        louvain_values,
        girvan_newman_values,
        output_path: Path,
        metric_name: str,
        higher_is_better: bool,
    ) -> None:
        """
        Create and save a scatter plot comparing the two algorithms.

        Args:
            louvain_values: List of metric values for Louvain algorithm
            girvan_newman_values: List of metric values for Girvan-Newman algorithm
            output_path: Path to save the plot
            metric_name: Display name of the metric
            higher_is_better: Whether higher values are better for this metric
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Create scatter plot
        ax.scatter(
            louvain_values,
            girvan_newman_values,
            alpha=0.6,
            s=100,
            c="#4ECDC4",
            edgecolors="black",
            linewidth=1,
        )

        # Add 45-degree reference line
        min_val = min(min(louvain_values), min(girvan_newman_values))
        max_val = max(max(louvain_values), max(girvan_newman_values))
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
        ax.set_xlabel(f"Louvain {metric_name}", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"Girvan-Newman {metric_name}", fontsize=12, fontweight="bold")

        comparison_text = "Higher is Better" if higher_is_better else "Lower is Better"
        ax.set_title(
            f"{metric_name} Comparison: Girvan-Newman vs Louvain\n({comparison_text})",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # Add legend
        ax.legend(loc="upper left", fontsize=10)

        # Add statistics text
        correlation = np.corrcoef(louvain_values, girvan_newman_values)[0, 1]
        stats_text = (
            f"Statistics:\n"
            f"  Correlation: {correlation:.4f}\n"
            f"  Sample size: {len(louvain_values)}\n\n"
            f"Louvain (x-axis):\n"
            f"  Mean: {np.mean(louvain_values):.4f}\n"
            f"  Std: {np.std(louvain_values):.4f}\n\n"
            f"Girvan-Newman (y-axis):\n"
            f"  Mean: {np.mean(girvan_newman_values):.4f}\n"
            f"  Std: {np.std(girvan_newman_values):.4f}"
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
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {output_path}")

    @staticmethod
    def plot_modularity_scatter(
        results: List[BenchmarkResult], output_path: Path
    ) -> None:
        """
        Create scatter plot comparing modularity scores between algorithms.

        Args:
            results: List of BenchmarkResult objects
            output_path: Path to save the plot
        """
        louvain_values, girvan_values = BenchmarkPlotter._extract_common_repos_data(
            results, "modularity"
        )

        if not louvain_values:
            print("  No graphs with both algorithms to compare modularity")
            return

        BenchmarkPlotter._create_scatterplot(
            louvain_values,
            girvan_values,
            output_path,
            "Modularity",
            higher_is_better=True,
        )

    @staticmethod
    def plot_density_scatter(results: List[BenchmarkResult], output_path: Path) -> None:
        """
        Create scatter plot comparing density scores between algorithms.

        Args:
            results: List of BenchmarkResult objects
            output_path: Path to save the plot
        """
        louvain_values, girvan_values = BenchmarkPlotter._extract_common_repos_data(
            results, "density"
        )

        if not louvain_values:
            print("  No graphs with both algorithms to compare density")
            return

        BenchmarkPlotter._create_scatterplot(
            louvain_values, girvan_values, output_path, "Density", higher_is_better=True
        )

    @staticmethod
    def plot_conductance_scatter(
        results: List[BenchmarkResult], output_path: Path
    ) -> None:
        """
        Create scatter plot comparing conductance scores between algorithms.

        Args:
            results: List of BenchmarkResult objects
            output_path: Path to save the plot
        """
        louvain_values, girvan_values = BenchmarkPlotter._extract_common_repos_data(
            results, "conductance"
        )

        if not louvain_values:
            print("  No graphs with both algorithms to compare conductance")
            return

        BenchmarkPlotter._create_scatterplot(
            louvain_values,
            girvan_values,
            output_path,
            "Conductance",
            higher_is_better=False,
        )

    @staticmethod
    def generate_all_plots(results: List[BenchmarkResult], output_dir: Path) -> None:
        """
        Generate all benchmark plots.

        Args:
            results: List of BenchmarkResult objects
            output_dir: Directory to save plots
        """
        print("\nGenerating scatter plots...")

        # Plot modularity scatter
        modularity_plot_path = output_dir / "benchmark_modularity_scatter.png"
        BenchmarkPlotter.plot_modularity_scatter(results, modularity_plot_path)

        # Plot density scatter
        density_plot_path = output_dir / "benchmark_density_scatter.png"
        BenchmarkPlotter.plot_density_scatter(results, density_plot_path)

        # Plot conductance scatter
        conductance_plot_path = output_dir / "benchmark_conductance_scatter.png"
        BenchmarkPlotter.plot_conductance_scatter(results, conductance_plot_path)

        print("\nAll plots generated successfully!")
