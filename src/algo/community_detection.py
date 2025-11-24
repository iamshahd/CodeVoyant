from typing import Any, Callable, Dict, List, Optional, Set

import networkx as nx
from networkx.algorithms import community

from .base import CommunityDetectionAlgorithm


class LouvainLibrary(CommunityDetectionAlgorithm):
    """
    Louvain community detection algorithm.

    The Louvain method is a greedy optimization method that attempts to optimize
    the "modularity" of a partition of the network. It's one of the fastest and
    most widely used community detection algorithms.

    Reference:
        Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008).
        Fast unfolding of communities in large networks.
        Journal of statistical mechanics: theory and experiment, 2008(10), P10008.
    """

    def __init__(
        self,
        graph: nx.Graph,
        weight: str = "weight",
        resolution: float = 1.0,
        seed: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Initialize Louvain community detection.

        Args:
            graph: NetworkX graph
            weight: Edge attribute key containing edge weights (None for unweighted)
            resolution: Resolution parameter (default 1.0). Higher values lead to more communities
            seed: Random seed for reproducibility
            **kwargs: Additional parameters passed to base class
        """
        super().__init__(graph, **kwargs)
        self.weight = weight
        self.resolution = resolution
        self.seed = seed
        self.hierarchy: Optional[List[Dict[str, int]]] = None

    def detect_communities(self) -> List[Set[str]]:
        """
        Detect communities using the Louvain algorithm.

        Returns:
            List of sets, where each set contains node IDs belonging to a community
        """
        # Use NetworkX's Louvain implementation
        communities_generator = community.louvain_communities(
            self.graph, weight=self.weight, resolution=self.resolution, seed=self.seed
        )

        self.communities = [set(comm) for comm in communities_generator]
        return self.communities

    def detect_communities_hierarchical(self) -> List[List[Set[str]]]:
        """
        Detect communities and return the full hierarchy of partitions.

        The Louvain algorithm works in iterations, producing increasingly
        coarse-grained community structures. This method returns all levels.

        Returns:
            List of partitions, where each partition is a list of communities
        """
        partitions = community.louvain_partitions(
            self.graph, weight=self.weight, resolution=self.resolution, seed=self.seed
        )

        hierarchy = []
        for partition in partitions:
            level_communities = [set(comm) for comm in partition]
            hierarchy.append(level_communities)

        # Store the finest partition as the main result
        if hierarchy:
            self.communities = hierarchy[-1]

        return hierarchy

    def get_partition_at_level(self, level: int = -1) -> List[Set[str]]:
        """
        Get the partition at a specific hierarchical level.

        Args:
            level: Hierarchy level (0 is coarsest, -1 is finest/default)

        Returns:
            List of communities at the specified level
        """
        hierarchy = self.detect_communities_hierarchical()
        return hierarchy[level] if hierarchy else []


class LabelPropagationCommunityDetection(CommunityDetectionAlgorithm):
    """
    Label Propagation community detection algorithm.

    A semi-synchronous label propagation method where each node adopts the label
    that most of its neighbors have. Fast and works well on large networks.

    Reference:
        Raghavan, U. N., Albert, R., & Kumara, S. (2007).
        Near linear time algorithm to detect community structures in large-scale networks.
        Physical review E, 76(3), 036106.
    """

    def __init__(
        self,
        graph: nx.Graph,
        weight: Optional[str] = "weight",
        seed: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Initialize Label Propagation community detection.

        Args:
            graph: NetworkX graph
            weight: Edge attribute key containing edge weights
            seed: Random seed for reproducibility
            **kwargs: Additional parameters passed to base class
        """
        super().__init__(graph, **kwargs)
        self.weight = weight
        self.seed = seed

    def detect_communities(self) -> List[Set[str]]:
        """
        Detect communities using Label Propagation.

        Returns:
            List of sets, where each set contains node IDs belonging to a community
        """
        communities_generator = community.label_propagation_communities(
            self.graph,
        )

        self.communities = [set(comm) for comm in communities_generator]
        return self.communities


class GreedyModularityCommunityDetection(CommunityDetectionAlgorithm):
    """
    Greedy Modularity Maximization community detection.

    Uses Clauset-Newman-Moore greedy modularity maximization to find communities.
    Works by progressively merging communities to maximize modularity.

    Reference:
        Clauset, A., Newman, M. E., & Moore, C. (2004).
        Finding community structure in very large networks.
        Physical review E, 70(6), 066111.
    """

    def __init__(
        self,
        graph: nx.Graph,
        weight: Optional[str] = "weight",
        resolution: float = 1.0,
        cutoff: int = 1,
        **kwargs: Any,
    ):
        """
        Initialize Greedy Modularity community detection.

        Args:
            graph: NetworkX graph
            weight: Edge attribute key containing edge weights
            resolution: Resolution parameter for modularity (default 1.0)
            cutoff: Minimum number of communities to find (stops when reached)
            **kwargs: Additional parameters passed to base class
        """
        super().__init__(graph, **kwargs)
        self.weight = weight
        self.resolution = resolution
        self.cutoff = cutoff

    def detect_communities(self) -> List[Set[str]]:
        """
        Detect communities using Greedy Modularity Maximization.

        Returns:
            List of sets, where each set contains node IDs belonging to a community
        """
        communities_generator = community.greedy_modularity_communities(
            self.graph,
            weight=self.weight,
            resolution=self.resolution,
            cutoff=self.cutoff,
            best_n=None,
        )

        self.communities = [set(comm) for comm in communities_generator]
        return self.communities


class GirvanNewmanCommunityDetection(CommunityDetectionAlgorithm):
    """
    Girvan-Newman community detection algorithm.

    Detects communities by progressively removing edges with high betweenness
    centrality. Produces a hierarchical decomposition of the network.
    Note: This algorithm is slow for large networks (O(m^2 n) complexity).

    Reference:
        Girvan, M., & Newman, M. E. (2002).
        Community structure in social and biological networks.
        Proceedings of the national academy of sciences, 99(12), 7821-7826.
    """

    def __init__(
        self,
        graph: nx.Graph,
        most_valuable_edge: Optional[Callable] = None,
        **kwargs: Any,
    ):
        """
        Initialize Girvan-Newman community detection.

        Args:
            graph: NetworkX graph
            most_valuable_edge: Function to determine edge importance (default: edge betweenness)
            **kwargs: Additional parameters passed to base class
        """
        super().__init__(graph, **kwargs)
        self.most_valuable_edge = most_valuable_edge

    def detect_communities(
        self, num_communities: Optional[int] = None
    ) -> List[Set[str]]:
        """
        Detect communities using Girvan-Newman algorithm.

        Args:
            num_communities: Target number of communities (if None, uses optimal modularity)

        Returns:
            List of sets, where each set contains node IDs belonging to a community
        """
        communities_generator = community.girvan_newman(
            self.graph, most_valuable_edge=self.most_valuable_edge
        )

        if num_communities is not None:
            # Get partition with specified number of communities
            for _ in range(num_communities - 1):
                partition = next(communities_generator)
            self.communities = [set(comm) for comm in partition]
        else:
            # Find partition with best modularity
            best_modularity = -1
            best_partition = None

            for partition in communities_generator:
                partition_list = [set(comm) for comm in partition]
                mod = nx.community.modularity(self.graph, partition_list)

                if mod > best_modularity:
                    best_modularity = mod
                    best_partition = partition_list
                else:
                    # Modularity started decreasing, use previous best
                    break

            self.communities = (
                best_partition if best_partition else [set(self.graph.nodes())]
            )

        return self.communities

    def get_dendrogram(self, max_levels: int = 10) -> List[List[Set[str]]]:
        """
        Get the hierarchical dendrogram of community splits.

        Args:
            max_levels: Maximum number of hierarchy levels to compute

        Returns:
            List of partitions at each level of the hierarchy
        """
        communities_generator = community.girvan_newman(
            self.graph, most_valuable_edge=self.most_valuable_edge
        )

        dendrogram = []
        for i, partition in enumerate(communities_generator):
            if i >= max_levels:
                break
            dendrogram.append([set(comm) for comm in partition])

        return dendrogram
