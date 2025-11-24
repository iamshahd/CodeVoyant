#!/usr/bin/env python
"""
Main script for running benchmark comparisons of community detection algorithms.

This script compares Louvain and Girvan-Newman algorithms on multiple graph files
and generates line plots showing execution time vs graph size.
"""

import argparse
import json
import sys
from pathlib import Path

from src.benchmark.loader import GraphLoader
from src.benchmark.plotter import BenchmarkPlotter
from src.benchmark.runner import BenchmarkRunner


def get_root_dir() -> Path:
    """
    Get the root directory of the project.

    Returns:
        Path to the root directory
    """
    # Assuming this script is in src/benchmark/run.py
    return Path(__file__).parent.parent.parent


def discover_graph_files(root_dir: Path) -> list[Path]:
    """
    Automatically discover call_graph.json and dependency_graph.json files
    in subdirectories of codevoyant_output.

    Args:
        root_dir: Root directory of the project

    Returns:
        List of paths to discovered graph files
    """
    output_dir = root_dir / "codevoyant_output"
    graph_files: list[Path] = []

    if not output_dir.exists():
        print(f"Warning: Directory {output_dir} does not exist")
        return graph_files

    # Iterate through subdirectories
    for subdir in output_dir.iterdir():
        if subdir.is_dir():
            # Check for call_graph.json
            call_graph = subdir / "call_graph.json"
            if call_graph.exists():
                graph_files.append(call_graph)

            # Check for dependency_graph.json
            dep_graph = subdir / "dependency_graph.json"
            if dep_graph.exists():
                graph_files.append(dep_graph)

    return sorted(graph_files)


def save_results_to_json(results, output_path: Path):
    """
    Save benchmark results to a JSON file.

    Args:
        results: List of BenchmarkResult objects
        output_path: Path to save the JSON file
    """
    json_data = {
        "benchmark_results": [
            {
                "project_name": result.project_name,
                "graph_type": result.graph_type,
                "algorithm": result.algorithm,
                "num_nodes": result.num_nodes,
                "num_edges": result.num_edges,
                "execution_time": result.execution_time,
                "num_communities": result.num_communities,
                "modularity": result.modularity,
            }
            for result in results
        ],
        "summary": {
            "total_benchmarks": len(results),
            "algorithms": list(set(r.algorithm for r in results)),
            "projects": list(set(r.project_name for r in results)),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"Saved benchmark results to {output_path}")


def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark community detection algorithms on graph files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with auto-discovered graph files from codevoyant_output
  python -m src.benchmark.run
  
  # Run with custom graph files
  python -m src.benchmark.run --files graphs/graph1.json graphs/graph2.json
  
  # Specify custom output directory
  python -m src.benchmark.run --output ./benchmark_results
        """,
    )

    parser.add_argument(
        "--files",
        nargs="+",
        help="List of JSON graph files to benchmark (if not provided, auto-discovers files in codevoyant_output)",
    )

    parser.add_argument(
        "--output",
        help="Output directory for plots (default: root_dir/codevoyant_output)",
    )

    args = parser.parse_args()

    # Get root directory
    root_dir = get_root_dir()

    # Determine which files to process
    if args.files:
        graph_files = [Path(f) for f in args.files]
    else:
        # Auto-discover graph files
        graph_files = discover_graph_files(root_dir)
        if not graph_files:
            print("Error: No graph files found in codevoyant_output subdirectories")
            print(
                "Please ensure subdirectories contain call_graph.json or dependency_graph.json files"
            )
            sys.exit(1)

    # Validate that all files exist
    missing_files = [f for f in graph_files if not f.exists()]
    if missing_files:
        print("Error: The following files do not exist:")
        for f in missing_files:
            print(f"  - {f}")
        sys.exit(1)

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = root_dir / "codevoyant_output"

    print("=" * 70)
    print("Community Detection Algorithm Benchmark")
    print("=" * 70)
    print(f"\nRoot directory: {root_dir}")
    print(f"Output directory: {output_dir}")
    print(f"\nProcessing {len(graph_files)} graph files:")
    for f in graph_files:
        print(f"  - {f.relative_to(root_dir)}")

    # Load graphs
    print("\n" + "=" * 70)
    print("Loading graphs...")
    print("=" * 70)
    graphs = GraphLoader.load_graphs_from_paths(graph_files)
    print(f"Loaded {len(graphs)} graphs successfully")

    # Run benchmarks
    print("\n" + "=" * 70)
    print("Running benchmarks...")
    print("=" * 70)
    results = BenchmarkRunner.run_benchmarks(graphs)

    # Save results to JSON
    print("\n" + "=" * 70)
    print("Saving results to JSON...")
    print("=" * 70)
    json_output_path = output_dir / "benchmark_results.json"
    save_results_to_json(results, json_output_path)

    # Generate plots
    print("\n" + "=" * 70)
    print("Generating plots...")
    print("=" * 70)
    BenchmarkPlotter.generate_all_plots(results, output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)
    print(f"\nTotal graphs processed: {len(graphs)}")
    print(f"Total benchmark runs: {len(results)}")
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - {output_dir / 'benchmark_results.json'}")
    print(f"  - {output_dir / 'benchmark_execution_time_vs_nodes.png'}")
    print(f"  - {output_dir / 'benchmark_execution_time_vs_edges.png'}")
    print(f"  - {output_dir / 'benchmark_modularity_comparison.png'}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
