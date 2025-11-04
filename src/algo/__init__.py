"""
Community detection algorithms for graph analysis.

This module provides various community detection algorithms including:
- Louvain method (fast, hierarchical)
- Label Propagation (very fast, good for large networks)
- Greedy Modularity Maximization
- Girvan-Newman (slower, hierarchical)

Basic usage:
    >>> from src.algo import detect_communities, analyze_communities
    >>> import networkx as nx
    >>>
    >>> # Create or load a graph
    >>> graph = nx.karate_club_graph()
    >>>
    >>> # Detect communities
    >>> communities = detect_communities(graph, algorithm='louvain')
    >>> print(f"Found {len(communities)} communities")
    >>>
    >>> # Analyze the communities
    >>> analyzer = analyze_communities(graph, communities)
    >>> print(f"Modularity: {analyzer.get_modularity():.4f}")

Advanced usage:
    >>> from src.algo import CommunityDetectionFactory, CommunityAnalyzer
    >>>
    >>> # Create a specific algorithm with custom parameters
    >>> detector = CommunityDetectionFactory.create(
    ...     'louvain',
    ...     graph,
    ...     resolution=1.5,
    ...     seed=42
    ... )
    >>> communities = detector.detect_communities()
    >>> stats = detector.get_community_stats()
    >>>
    >>> # Compare multiple algorithms
    >>> from src.algo import compare_algorithms
    >>> results = compare_algorithms(graph, algorithms=['louvain', 'label_propagation'])
"""

from .base import CommunityDetectionAlgorithm
from .community_detection import (
    LouvainCommunityDetection,
    LabelPropagationCommunityDetection,
    GreedyModularityCommunityDetection,
    GirvanNewmanCommunityDetection,
)
from .community_utils import CommunityAnalyzer, CommunityVisualizer
from .factory import (
    CommunityDetectionFactory,
    detect_communities,
    analyze_communities,
    visualize_communities,
    compare_algorithms,
)

__all__ = [
    # Base class
    "CommunityDetectionAlgorithm",
    # Algorithm implementations
    "LouvainCommunityDetection",
    "LabelPropagationCommunityDetection",
    "GreedyModularityCommunityDetection",
    "GirvanNewmanCommunityDetection",
    # Utilities
    "CommunityAnalyzer",
    "CommunityVisualizer",
    # Factory and convenience functions
    "CommunityDetectionFactory",
    "detect_communities",
    "analyze_communities",
    "visualize_communities",
    "compare_algorithms",
]
