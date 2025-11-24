"""
Module for plotting benchmark results.
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

from .runner import BenchmarkResult


class BenchmarkPlotter:
    """Creates visualizations of benchmark results."""

    @staticmethod
    def plot_execution_time_vs_nodes(
        results: List[BenchmarkResult], output_path: Path
    ) -> None:
        """
        Plot execution time vs number of nodes.

        Args:
            results: List of BenchmarkResult objects
            output_path: Path to save the plot
        """
        # Separate results by algorithm and graph type
        louvain_call = []
        louvain_dep = []
        girvan_call = []
        girvan_dep = []

        for result in results:
            if result.algorithm == "louvain":
                if result.graph_type == "call_graph":
                    louvain_call.append(result)
                else:
                    louvain_dep.append(result)
            else:  # girvan_newman
                if result.graph_type == "call_graph":
                    girvan_call.append(result)
                else:
                    girvan_dep.append(result)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot for call graphs
        BenchmarkPlotter._plot_single_graph_type(
            ax1, louvain_call, girvan_call, "Call Graphs", "num_nodes"
        )

        # Plot for dependency graphs
        BenchmarkPlotter._plot_single_graph_type(
            ax2, louvain_dep, girvan_dep, "Dependency Graphs", "num_nodes"
        )

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {output_path}")

    @staticmethod
    def plot_execution_time_vs_edges(
        results: List[BenchmarkResult], output_path: Path
    ) -> None:
        """
        Plot execution time vs number of edges.

        Args:
            results: List of BenchmarkResult objects
            output_path: Path to save the plot
        """
        # Separate results by algorithm and graph type
        louvain_call = []
        louvain_dep = []
        girvan_call = []
        girvan_dep = []

        for result in results:
            if result.algorithm == "louvain":
                if result.graph_type == "call_graph":
                    louvain_call.append(result)
                else:
                    louvain_dep.append(result)
            else:  # girvan_newman
                if result.graph_type == "call_graph":
                    girvan_call.append(result)
                else:
                    girvan_dep.append(result)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot for call graphs
        BenchmarkPlotter._plot_single_graph_type(
            ax1, louvain_call, girvan_call, "Call Graphs", "num_edges"
        )

        # Plot for dependency graphs
        BenchmarkPlotter._plot_single_graph_type(
            ax2, louvain_dep, girvan_dep, "Dependency Graphs", "num_edges"
        )

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {output_path}")

    @staticmethod
    def _plot_single_graph_type(
        ax,
        louvain_results: List[BenchmarkResult],
        girvan_results: List[BenchmarkResult],
        title: str,
        x_metric: str,
    ) -> None:
        """
        Plot a single graph type (call or dependency).

        Args:
            ax: Matplotlib axis
            louvain_results: Results for Louvain algorithm
            girvan_results: Results for Girvan-Newman algorithm
            title: Plot title
            x_metric: 'num_nodes' or 'num_edges'
        """
        # Sort results by x_metric for proper line plotting
        louvain_results = sorted(louvain_results, key=lambda r: getattr(r, x_metric))
        girvan_results = sorted(girvan_results, key=lambda r: getattr(r, x_metric))

        # Extract data for Louvain
        louvain_x = [getattr(r, x_metric) for r in louvain_results]
        louvain_y = [r.execution_time for r in louvain_results]
        louvain_labels = [r.project_name for r in louvain_results]

        # Extract data for Girvan-Newman
        girvan_x = [getattr(r, x_metric) for r in girvan_results]
        girvan_y = [r.execution_time for r in girvan_results]
        girvan_labels = [r.project_name for r in girvan_results]

        # Plot lines
        ax.plot(
            louvain_x,
            louvain_y,
            marker="o",
            linewidth=2,
            markersize=8,
            label="Louvain",
            color="#2E86AB",
        )
        ax.plot(
            girvan_x,
            girvan_y,
            marker="s",
            linewidth=2,
            markersize=8,
            label="Girvan-Newman",
            color="#A23B72",
        )

        # Add labels for each point
        for x, y, label in zip(louvain_x, louvain_y, louvain_labels):
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                color="#2E86AB",
            )

        for x, y, label in zip(girvan_x, girvan_y, girvan_labels):
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(0, -15),
                ha="center",
                fontsize=9,
                color="#A23B72",
            )

        # Set labels and title
        xlabel = "Number of Nodes" if x_metric == "num_nodes" else "Number of Edges"
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Execution Time (seconds)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    @staticmethod
    def plot_modularity_comparison(
        results: List[BenchmarkResult], output_path: Path
    ) -> None:
        """
        Plot modularity comparison for graphs where both algorithms ran.

        Args:
            results: List of BenchmarkResult objects
            output_path: Path to save the plot
        """
        # Group results by project_name and graph_type
        from collections import defaultdict
        from typing import Dict, Tuple

        grouped: Dict[Tuple[str, str], Dict[str, BenchmarkResult]] = defaultdict(dict)
        for result in results:
            key = (result.project_name, result.graph_type)
            grouped[key][result.algorithm] = result

        # Filter to only include graphs where both algorithms ran
        comparison_data = []
        for (project_name, graph_type), algorithms in grouped.items():
            if "louvain" in algorithms and "girvan_newman" in algorithms:
                comparison_data.append(
                    {
                        "project_name": project_name,
                        "graph_type": graph_type,
                        "louvain_mod": algorithms["louvain"].modularity,
                        "girvan_mod": algorithms["girvan_newman"].modularity,
                        "num_nodes": algorithms["louvain"].num_nodes,
                    }
                )

        if not comparison_data:
            print("  No graphs with both algorithms to compare modularity")
            return

        # Separate by graph type
        call_graphs = [d for d in comparison_data if d["graph_type"] == "call_graph"]
        dep_graphs = [
            d for d in comparison_data if d["graph_type"] == "dependency_graph"
        ]

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot call graphs
        if call_graphs:
            BenchmarkPlotter._plot_modularity_comparison_single(
                ax1, call_graphs, "Call Graphs"
            )
        else:
            ax1.text(
                0.5,
                0.5,
                "No call graphs with both algorithms",
                ha="center",
                va="center",
            )
            ax1.set_title("Call Graphs", fontsize=14, fontweight="bold")

        # Plot dependency graphs
        if dep_graphs:
            BenchmarkPlotter._plot_modularity_comparison_single(
                ax2, dep_graphs, "Dependency Graphs"
            )
        else:
            ax2.text(
                0.5,
                0.5,
                "No dependency graphs with both algorithms",
                ha="center",
                va="center",
            )
            ax2.set_title("Dependency Graphs", fontsize=14, fontweight="bold")

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {output_path}")

    @staticmethod
    def _plot_modularity_comparison_single(ax, data: list, title: str) -> None:
        """
        Plot modularity comparison for a single graph type.

        Args:
            ax: Matplotlib axis
            data: List of dictionaries with comparison data
            title: Plot title
        """
        # Sort by number of nodes
        data = sorted(data, key=lambda d: d["num_nodes"])

        x_positions = range(len(data))
        louvain_mods = [d["louvain_mod"] for d in data]
        girvan_mods = [d["girvan_mod"] for d in data]
        labels = [d["project_name"] for d in data]

        width = 0.35

        # Create bars
        ax.bar(
            [x - width / 2 for x in x_positions],
            louvain_mods,
            width,
            label="Louvain",
            color="#2E86AB",
            alpha=0.8,
        )
        ax.bar(
            [x + width / 2 for x in x_positions],
            girvan_mods,
            width,
            label="Girvan-Newman",
            color="#A23B72",
            alpha=0.8,
        )

        # Customize plot
        ax.set_xlabel("Project", fontsize=12)
        ax.set_ylabel("Modularity", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, max(max(louvain_mods), max(girvan_mods)) * 1.1)

    @staticmethod
    def generate_all_plots(results: List[BenchmarkResult], output_dir: Path) -> None:
        """
        Generate all benchmark plots.

        Args:
            results: List of BenchmarkResult objects
            output_dir: Directory to save plots
        """
        print("\nGenerating plots...")

        # Plot execution time vs nodes
        nodes_plot_path = output_dir / "benchmark_execution_time_vs_nodes.png"
        BenchmarkPlotter.plot_execution_time_vs_nodes(results, nodes_plot_path)

        # Plot execution time vs edges
        edges_plot_path = output_dir / "benchmark_execution_time_vs_edges.png"
        BenchmarkPlotter.plot_execution_time_vs_edges(results, edges_plot_path)

        # Plot modularity comparison
        modularity_plot_path = output_dir / "benchmark_modularity_comparison.png"
        BenchmarkPlotter.plot_modularity_comparison(results, modularity_plot_path)

        print("\nAll plots generated successfully!")
