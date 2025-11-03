"""
Factory and convenience functions for community detection algorithms.
"""

from typing import List, Set, Optional, Dict, Any

import networkx as nx

from .base import CommunityDetectionAlgorithm
from .community_detection import (
    LouvainCommunityDetection,
    LabelPropagationCommunityDetection,
    GreedyModularityCommunityDetection,
    GirvanNewmanCommunityDetection,
)
from .community_utils import CommunityAnalyzer, CommunityVisualizer


class CommunityDetectionFactory:
    """
    Factory for creating community detection algorithm instances.
    """

    _algorithms = {
        "louvain": LouvainCommunityDetection,
        "label_propagation": LabelPropagationCommunityDetection,
        "greedy_modularity": GreedyModularityCommunityDetection,
        "girvan_newman": GirvanNewmanCommunityDetection,
    }

    @classmethod
    def create(
        cls,
        algorithm: str,
        graph: nx.Graph,
        **kwargs: Any
    ) -> CommunityDetectionAlgorithm:
        """
        Create a community detection algorithm instance.

        Args:
            algorithm: Algorithm name ('louvain', 'label_propagation', 'greedy_modularity', 'girvan_newman')
            graph: NetworkX graph
            **kwargs: Algorithm-specific parameters

        Returns:
            Community detection algorithm instance

        Raises:
            ValueError: If algorithm name is not recognized
        """
        if algorithm not in cls._algorithms:
            available = ", ".join(cls._algorithms.keys())
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. Available algorithms: {available}"
            )

        return cls._algorithms[algorithm](graph, **kwargs)

    @classmethod
    def available_algorithms(cls) -> List[str]:
        """
        Get list of available algorithm names.

        Returns:
            List of algorithm names
        """
        return list(cls._algorithms.keys())

    @classmethod
    def register_algorithm(
        cls,
        name: str,
        algorithm_class: type[CommunityDetectionAlgorithm]
    ) -> None:
        """
        Register a custom community detection algorithm.

        Args:
            name: Name for the algorithm
            algorithm_class: Algorithm class (must inherit from CommunityDetectionAlgorithm)
        """
        if not issubclass(algorithm_class, CommunityDetectionAlgorithm):
            raise ValueError(
                "Algorithm class must inherit from CommunityDetectionAlgorithm"
            )
        
        cls._algorithms[name] = algorithm_class


def detect_communities(
    graph: nx.Graph,
    algorithm: str = "louvain",
    **kwargs: Any
) -> List[Set[str]]:
    """
    Convenience function to detect communities in a graph.

    Args:
        graph: NetworkX graph
        algorithm: Algorithm name (default: 'louvain')
        **kwargs: Algorithm-specific parameters

    Returns:
        List of communities (sets of node IDs)

    Example:
        >>> communities = detect_communities(graph, algorithm='louvain', resolution=1.5)
        >>> print(f"Found {len(communities)} communities")
    """
    detector = CommunityDetectionFactory.create(algorithm, graph, **kwargs)
    return detector.detect_communities()


def analyze_communities(
    graph: nx.Graph,
    communities: List[Set[str]]
) -> CommunityAnalyzer:
    """
    Convenience function to create a community analyzer.

    Args:
        graph: NetworkX graph
        communities: List of detected communities

    Returns:
        CommunityAnalyzer instance

    Example:
        >>> analyzer = analyze_communities(graph, communities)
        >>> metrics = analyzer.get_all_metrics()
        >>> print(f"Modularity: {analyzer.get_modularity():.4f}")
    """
    return CommunityAnalyzer(graph, communities)


def visualize_communities(
    graph: nx.Graph,
    communities: List[Set[str]]
) -> CommunityVisualizer:
    """
    Convenience function to create a community visualizer.

    Args:
        graph: NetworkX graph
        communities: List of detected communities

    Returns:
        CommunityVisualizer instance

    Example:
        >>> visualizer = visualize_communities(graph, communities)
        >>> node_colors = visualizer.get_node_colors()
        >>> graph_with_attrs = visualizer.add_community_attributes()
    """
    return CommunityVisualizer(graph, communities)


def compare_algorithms(
    graph: nx.Graph,
    algorithms: Optional[List[str]] = None,
    **shared_kwargs: Any
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple community detection algorithms on the same graph.

    Args:
        graph: NetworkX graph
        algorithms: List of algorithm names (None = all available)
        **shared_kwargs: Parameters to pass to all algorithms

    Returns:
        Dictionary mapping algorithm names to their results and statistics

    Example:
        >>> results = compare_algorithms(graph, algorithms=['louvain', 'label_propagation'])
        >>> for algo, stats in results.items():
        ...     print(f"{algo}: {stats['num_communities']} communities, modularity={stats['modularity']:.4f}")
    """
    import time
    
    if algorithms is None:
        algorithms = CommunityDetectionFactory.available_algorithms()

    results = {}

    for algo_name in algorithms:
        try:
            start_time = time.perf_counter()
            
            detector = CommunityDetectionFactory.create(
                algo_name,
                graph,
                **shared_kwargs
            )
            communities = detector.detect_communities()
            
            end_time = time.perf_counter()
            runtime = end_time - start_time
            
            stats = detector.get_community_stats()
            
            results[algo_name] = {
                "communities": communities,
                "num_communities": stats["num_communities"],
                "modularity": stats["modularity"],
                "avg_community_size": stats["avg_community_size"],
                "min_community_size": stats["min_community_size"],
                "max_community_size": stats["max_community_size"],
                "runtime_seconds": runtime,
                "algorithm_params": detector.params,
            }
        except Exception as e:
            results[algo_name] = {
                "error": str(e)
            }

    return results
