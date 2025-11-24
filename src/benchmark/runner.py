"""
Module for running benchmark comparisons of community detection algorithms.
"""

import time
from dataclasses import dataclass
from typing import List, Tuple

import networkx as nx  # type: ignore[import-untyped]

from ..algo.factory import CommunityDetectionFactory


@dataclass
class BenchmarkResult:
    """Result of running a single algorithm on a graph."""

    project_name: str
    graph_type: str
    algorithm: str
    num_nodes: int
    num_edges: int
    execution_time: float
    num_communities: int
    modularity: float
    density: float
    conductance: float


class BenchmarkRunner:
    """Runs community detection algorithms and collects performance metrics."""

    ALGORITHMS = ["louvain", "girvan_newman"]

    @staticmethod
    def run_algorithm(
        algorithm: str, graph: nx.Graph
    ) -> Tuple[List, float, float, float, float]:
        """
        Run a single community detection algorithm.

        Args:
            algorithm: Algorithm name ('louvain' or 'girvan_newman')
            graph: NetworkX graph

        Returns:
            Tuple of (communities, execution_time, modularity, density, conductance)
        """
        # Measure execution time
        start_time = time.time()

        # Run algorithm
        detector = CommunityDetectionFactory.create(algorithm, graph)
        communities = detector.detect_communities()

        execution_time = time.time() - start_time

        # Calculate quality metrics
        modularity = detector.get_modularity()
        density = detector.get_density()
        conductance = detector.get_conductance()

        return communities, execution_time, modularity, density, conductance

    @staticmethod
    def run_benchmark_on_graph(
        project_name: str, graph_type: str, graph: nx.Graph
    ) -> List[BenchmarkResult]:
        """
        Run all algorithms on a single graph.

        Args:
            project_name: Name of the project (e.g., 'httpx', 'django')
            graph_type: Type of graph ('call_graph' or 'dependency_graph')
            graph: NetworkX graph

        Returns:
            List of BenchmarkResult objects
        """
        results = []
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()

        for algorithm in BenchmarkRunner.ALGORITHMS:
            # Skip Girvan-Newman for graphs with 500+ nodes (too slow)
            if algorithm == "girvan_newman" and num_nodes >= 500:
                print(f"  {algorithm}: Skipped (graph too large: {num_nodes} nodes)")
                continue

            try:
                communities, exec_time, modularity, density, conductance = (
                    BenchmarkRunner.run_algorithm(algorithm, graph)
                )

                result = BenchmarkResult(
                    project_name=project_name,
                    graph_type=graph_type,
                    algorithm=algorithm,
                    num_nodes=num_nodes,
                    num_edges=num_edges,
                    execution_time=exec_time,
                    num_communities=len(communities),
                    modularity=modularity,
                    density=density,
                    conductance=conductance,
                )
                results.append(result)

                print(
                    f"  {algorithm}: {exec_time:.4f}s, "
                    f"{len(communities)} communities, "
                    f"modularity: {modularity:.4f}, "
                    f"density: {density:.4f}, "
                    f"conductance: {conductance:.4f}"
                )

            except Exception as e:
                print(f"  Error running {algorithm}: {e}")

        return results

    @staticmethod
    def run_benchmarks(
        graphs: List[Tuple[str, str, nx.Graph]],
    ) -> List[BenchmarkResult]:
        """
        Run benchmarks on all provided graphs.

        Args:
            graphs: List of tuples (project_name, graph_type, graph)

        Returns:
            List of all BenchmarkResult objects
        """
        all_results = []

        for project_name, graph_type, graph in graphs:
            print(f"\nBenchmarking {project_name}/{graph_type}...")
            print(
                f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}"
            )

            results = BenchmarkRunner.run_benchmark_on_graph(
                project_name, graph_type, graph
            )
            all_results.extend(results)

        return all_results
