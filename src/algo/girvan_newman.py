"""
Hand-implemented Girvan-Newman community detection algorithm.

References:
- Girvan, M., & Newman, M. E. J. (2002). Community structure in social and biological networks. Proceedings of the National Academy of Sciences, 99(12), 7821-7826.
"""

from collections import deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import networkx as nx  # type: ignore[import-untyped]

from src.algo.base import CommunityDetectionAlgorithm


class GirvanNewmanCommunityDetection(CommunityDetectionAlgorithm):
    """
    Girvan-Newman community detection algorithm implementation.

    The Girvan-Newman algorithm is a hierarchical method that detects communities by
    progressively removing edges with the highest betweenness. Edge betweenness
    is the number of shortest paths between all pairs of nodes that pass through that edge.

    The algorithm:
    1. Calculate edge betweenness for all edges
    2. Remove edge(s) with highest betweenness
    3. Recalculate betweenness for all edges
    4. Repeat until stopping criterion is met (target communities or modularity peak)

    Note: This algorithm is slow for large networks (O(m^2 n) complexity).
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
        self.modularity_history: List[float] = []
        self.dendrogram: List[List[Set[str]]] = []
        # Store original graph for modularity calculations
        self.original_graph = self.graph.copy()

    def _calculate_edge_betweenness(
        self, graph: nx.Graph
    ) -> Dict[Tuple[str, str], float]:
        """
        Calculate edge betweenness for all edges using a BFS-based approach.

        Edge betweenness is the raw count of shortest paths passing through each edge.
        This is NOT normalized betweenness centrality, but the unnormalized score
        used for ranking edges in the Girvan-Newman algorithm.

        This implementation uses BFS to find shortest paths from each node and
        accumulates the contribution of each edge to shortest paths (similar to
        the accumulation phase in Brandes' algorithm).

        Args:
            graph: The graph to calculate edge betweenness on

        Returns:
            Dictionary mapping edge tuples (u, v) where u < v to betweenness scores
        """
        # Initialize betweenness scores for each edge (canonical form: smaller node first)
        betweenness: Dict[Tuple[str, str], float] = {}
        for u, v in graph.edges():
            edge = tuple(sorted([u, v]))
            betweenness[edge] = 0.0

        nodes = list(graph.nodes())

        # For each node as source, compute shortest paths and accumulate edge scores
        for source in nodes:
            # BFS to find shortest paths
            stack: List[
                str
            ] = []  # Nodes in order of non-increasing distance from source
            predecessors: Dict[str, List[str]] = {node: [] for node in nodes}
            sigma: Dict[str, float] = {
                node: 0.0 for node in nodes
            }  # Number of shortest paths
            sigma[source] = 1.0
            distance = {node: -1 for node in nodes}
            distance[source] = 0

            queue = deque([source])

            # Forward pass: BFS to compute shortest paths
            while queue:
                current = queue.popleft()
                stack.append(current)

                for neighbor in graph.neighbors(current):
                    # First time visiting this neighbor
                    if distance[neighbor] < 0:
                        queue.append(neighbor)
                        distance[neighbor] = distance[current] + 1

                    # Shortest path to neighbor via current
                    if distance[neighbor] == distance[current] + 1:
                        sigma[neighbor] += sigma[current]
                        predecessors[neighbor].append(current)

            # Backward pass: accumulate edge betweenness
            delta = {node: 0.0 for node in nodes}

            while stack:
                node = stack.pop()

                for predecessor in predecessors[node]:
                    # Calculate contribution of this edge
                    contribution = (sigma[predecessor] / sigma[node]) * (
                        1.0 + delta[node]
                    )

                    # Add to edge betweenness (canonical form)
                    edge = tuple(sorted([predecessor, node]))
                    if edge in betweenness:
                        betweenness[edge] += contribution

                    delta[predecessor] += contribution

        # For undirected graphs, we count each edge twice (once from each direction)
        # so divide by 2 to get the correct betweenness
        for edge in betweenness:
            betweenness[edge] /= 2.0

        return betweenness

    def _get_connected_components(self, graph: nx.Graph) -> List[Set[str]]:
        """
        Get all connected components in the current graph.

        Args:
            graph: The graph to get connected components from

        Returns:
            List of sets, each containing nodes in a connected component
        """
        return [set(component) for component in nx.connected_components(graph)]

    def _calculate_modularity(
        self, communities: List[Set[str]], graph: nx.Graph
    ) -> float:
        """
        Calculate modularity Q of a partition.

        Q = (1/2m) * sum_ij [A_ij - (k_i * k_j)/(2m)] * delta(c_i, c_j)

        Where:
        - m is the number of edges
        - A_ij is the adjacency matrix
        - k_i is the degree of node i
        - delta(c_i, c_j) is 1 if nodes i and j are in the same community, 0 otherwise

        Args:
            communities: List of communities (sets of nodes)
            graph: The graph to calculate modularity on (typically the original graph)

        Returns:
            Modularity score (ranges from -0.5 to 1.0)
        """
        if not communities:
            return 0.0

        # Build community membership map
        node_to_comm = {}
        for comm_id, community in enumerate(communities):
            for node in community:
                node_to_comm[node] = comm_id

        # Calculate modularity on the provided graph
        m = graph.number_of_edges()
        if m == 0:
            return 0.0

        q = 0.0
        for node_i in graph.nodes():
            # Skip nodes not in any community (safety check)
            if node_i not in node_to_comm:
                continue

            for node_j in graph.nodes():
                # Skip nodes not in any community (safety check)
                if node_j not in node_to_comm:
                    continue

                if node_to_comm[node_i] == node_to_comm[node_j]:
                    # A_ij
                    a_ij = 1.0 if graph.has_edge(node_i, node_j) else 0.0

                    # (k_i * k_j) / (2m)
                    k_i = graph.degree(node_i)
                    k_j = graph.degree(node_j)
                    expected = (k_i * k_j) / (2.0 * m)

                    q += a_ij - expected

        return q / (2.0 * m)

    def detect_communities(
        self, num_communities: Optional[int] = None
    ) -> List[Set[str]]:
        """
        Detect communities using the Girvan-Newman algorithm.

        Args:
            num_communities: Target number of communities (if None, uses optimal modularity)

        Returns:
            List of sets, where each set contains node IDs belonging to a community
        """
        # Reset state for new run
        self.modularity_history = []
        self.dendrogram = []

        # Reset node_to_community mapping (will be rebuilt by base class if needed)
        self.node_to_community = None

        # Start with the full graph as a working copy
        working_graph = self.original_graph.copy()

        # Handle graphs with no edges - each node is its own community
        if working_graph.number_of_edges() == 0:
            self.communities = [set([node]) for node in working_graph.nodes()]
            self.dendrogram.append(self.communities)
            return self.communities

        # Get initial connected components
        initial_communities = self._get_connected_components(working_graph)

        # Store initial state in dendrogram
        self.dendrogram.append(initial_communities)

        if num_communities is not None:
            # Get partition with specified number of communities
            current_num_communities = len(initial_communities)

            while (
                working_graph.number_of_edges() > 0
                and current_num_communities < num_communities
            ):
                # Calculate edge betweenness on the current working graph
                if self.most_valuable_edge is not None:
                    # Use custom edge selection function
                    edge_to_remove = self.most_valuable_edge(working_graph)
                    if edge_to_remove and working_graph.has_edge(*edge_to_remove):
                        working_graph.remove_edge(*edge_to_remove)
                else:
                    # Use default betweenness calculation
                    betweenness = self._calculate_edge_betweenness(working_graph)

                    if not betweenness:
                        break

                    # Find edge(s) with maximum betweenness
                    max_betweenness = max(betweenness.values())
                    edges_to_remove = [
                        edge
                        for edge, score in betweenness.items()
                        if abs(score - max_betweenness) < 1e-9
                    ]

                    # Remove edge(s) with highest betweenness
                    for edge in edges_to_remove:
                        if working_graph.has_edge(*edge):
                            working_graph.remove_edge(*edge)

                # Get current communities (connected components)
                current_communities = self._get_connected_components(working_graph)
                self.dendrogram.append(current_communities)
                current_num_communities = len(current_communities)

            self.communities = self._get_connected_components(working_graph)
        else:
            # Find partition with best modularity
            best_modularity = float("-inf")
            best_partition = initial_communities

            # Calculate initial modularity
            initial_mod = self._calculate_modularity(
                initial_communities, self.original_graph
            )
            self.modularity_history.append(initial_mod)
            if initial_mod > best_modularity:
                best_modularity = initial_mod
                best_partition = initial_communities

            # Keep removing edges and track modularity
            iterations_since_improvement = 0

            while working_graph.number_of_edges() > 0:
                # Calculate edge betweenness on the current working graph
                if self.most_valuable_edge is not None:
                    # Use custom edge selection function
                    edge_to_remove = self.most_valuable_edge(working_graph)
                    if edge_to_remove and working_graph.has_edge(*edge_to_remove):
                        working_graph.remove_edge(*edge_to_remove)
                else:
                    # Use default betweenness calculation
                    betweenness = self._calculate_edge_betweenness(working_graph)

                    if not betweenness:
                        break

                    # Find edge(s) with maximum betweenness
                    max_betweenness = max(betweenness.values())
                    edges_to_remove = [
                        edge
                        for edge, score in betweenness.items()
                        if abs(score - max_betweenness) < 1e-9
                    ]

                    # Remove edge(s) with highest betweenness
                    for edge in edges_to_remove:
                        if working_graph.has_edge(*edge):
                            working_graph.remove_edge(*edge)

                # Get current communities (connected components)
                current_communities = self._get_connected_components(working_graph)
                self.dendrogram.append(current_communities)

                # Calculate modularity on the ORIGINAL graph with current partition
                mod = self._calculate_modularity(
                    current_communities, self.original_graph
                )
                self.modularity_history.append(mod)

                if mod > best_modularity:
                    best_modularity = mod
                    best_partition = current_communities
                    iterations_since_improvement = 0
                else:
                    iterations_since_improvement += 1
                    # Stop if modularity has been decreasing consistently
                    # Use a dynamic threshold based on graph size
                    max_patience = max(10, working_graph.number_of_nodes() // 3)
                    if iterations_since_improvement >= max_patience:
                        break

            self.communities = best_partition

        return self.communities

    def get_dendrogram(self, max_levels: int = 10) -> List[List[Set[str]]]:
        """
        Get the hierarchical dendrogram of community splits.

        Args:
            max_levels: Maximum number of hierarchy levels to compute

        Returns:
            List of partitions at each level of the hierarchy
        """
        # If dendrogram already computed from detect_communities, return it
        if self.dendrogram:
            return self.dendrogram[:max_levels]

        # Otherwise, compute it fresh
        dendrogram = []
        working_graph = self.original_graph.copy()

        # Handle graphs with no edges
        if working_graph.number_of_edges() == 0:
            return [[set([node]) for node in working_graph.nodes()]]

        # Store initial state
        initial_communities = self._get_connected_components(working_graph)
        dendrogram.append(initial_communities)

        # Keep removing edges up to max_levels
        level = 1
        while working_graph.number_of_edges() > 0 and level < max_levels:
            # Calculate edge betweenness on the current working graph
            if self.most_valuable_edge is not None:
                # Use custom edge selection function
                edge_to_remove = self.most_valuable_edge(working_graph)
                if edge_to_remove and working_graph.has_edge(*edge_to_remove):
                    working_graph.remove_edge(*edge_to_remove)
            else:
                # Use default betweenness calculation
                betweenness = self._calculate_edge_betweenness(working_graph)

                if not betweenness:
                    break

                # Find edge(s) with maximum betweenness
                max_betweenness = max(betweenness.values())
                edges_to_remove = [
                    edge
                    for edge, score in betweenness.items()
                    if abs(score - max_betweenness) < 1e-9
                ]

                # Remove edge(s) with highest betweenness
                for edge in edges_to_remove:
                    if working_graph.has_edge(*edge):
                        working_graph.remove_edge(*edge)

            # Get current communities (connected components)
            current_communities = self._get_connected_components(working_graph)
            dendrogram.append(current_communities)
            level += 1

        return dendrogram

    def get_modularity_history(self) -> List[float]:
        """
        Get the history of modularity values at each step.

        Returns:
            List of modularity values
        """
        return self.modularity_history
