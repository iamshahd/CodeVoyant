"""
Hand-implemented Girvan-Newman community detection algorithm.

References:
- Girvan, M., & Newman, M. E. J. (2002). Community structure in social and biological networks. Proceedings of the National Academy of Sciences, 99(12), 7821-7826.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

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
    """

    def __init__(
        self,
        graph: nx.Graph,
        num_communities: Optional[int] = None,
        use_modularity: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize Girvan-Newman community detection.

        Args:
            graph: NetworkX graph
            num_communities: Target number of communities. If specified, this takes
                           precedence over use_modularity and the algorithm stops
                           when this number is reached.
            use_modularity: If True and num_communities is None, automatically determine
                          optimal number of communities by maximizing modularity.
                          Ignored if num_communities is specified.
            **kwargs: Additional parameters passed to base class
        """
        super().__init__(graph, **kwargs)
        self.num_communities = num_communities
        self.use_modularity = use_modularity
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

    def detect_communities(self) -> List[Set[str]]:
        """
        Detect communities using the Girvan-Newman algorithm.

        Returns:
            List of sets, where each set contains node IDs belonging to a community
        """
        # Reset state for new run
        self.modularity_history = []
        self.dendrogram = []

        # Start with the full graph as a working copy
        working_graph = self.original_graph.copy()

        # Handle graphs with no edges - each node is its own community
        if working_graph.number_of_edges() == 0:
            self.communities = [set([node]) for node in working_graph.nodes()]
            self.dendrogram.append(self.communities)
            self.node_to_community = self._build_node_to_community_mapping()
            return self.communities

        # Get initial connected components
        initial_communities = self._get_connected_components(working_graph)

        # Track the best partition if using modularity
        best_modularity = -1.0
        best_communities = initial_communities

        # Store initial state in dendrogram
        self.dendrogram.append(initial_communities)

        # Keep removing edges until we reach desired communities or run out of edges
        while working_graph.number_of_edges() > 0:
            # Calculate edge betweenness on the current working graph
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

            # Get current communities (connected components) from the working graph
            current_communities = self._get_connected_components(working_graph)
            self.dendrogram.append(current_communities)

            # Calculate modularity on the ORIGINAL graph with the current partition
            modularity = self._calculate_modularity(
                current_communities, self.original_graph
            )
            self.modularity_history.append(modularity)

            # Track best partition
            if modularity > best_modularity:
                best_modularity = modularity
                best_communities = current_communities

            # Check stopping criteria
            if self.num_communities is not None:
                # Stop when we reach or exceed the target number of communities
                if len(current_communities) >= self.num_communities:
                    # If we have exactly the right number, use current
                    # If we overshot, we already stored it in the dendrogram
                    if len(current_communities) == self.num_communities:
                        self.communities = current_communities
                    else:
                        # We overshot - use the previous partition from dendrogram
                        # which had fewer communities
                        for prev_partition in reversed(self.dendrogram[:-1]):
                            if len(prev_partition) <= self.num_communities:
                                self.communities = prev_partition
                                break
                        else:
                            # Fallback to current if we can't find a better one
                            self.communities = current_communities
                    self.node_to_community = self._build_node_to_community_mapping()
                    return self.communities

        # Use best partition by modularity or final partition
        if self.use_modularity:
            self.communities = best_communities
        else:
            self.communities = self._get_connected_components(working_graph)

        self.node_to_community = self._build_node_to_community_mapping()
        return self.communities

    def _build_node_to_community_mapping(self) -> Dict[str, int]:
        """
        Build mapping from node to community ID.

        Returns:
            Dictionary mapping node IDs to community indices
        """
        if self.communities is None:
            return {}

        mapping = {}
        for comm_id, community in enumerate(self.communities):
            for node in community:
                mapping[node] = comm_id
        return mapping

    def get_node_to_community_mapping(self) -> Dict[str, int]:
        """
        Get a mapping from node ID to community ID.

        Returns:
            Dictionary mapping node IDs to their community IDs (empty if detection hasn't run)
        """
        if self.node_to_community is None:
            return {}
        return self.node_to_community

    def get_dendrogram(self) -> List[List[Set[str]]]:
        """
        Get the dendrogram (hierarchy) of community splits.

        Returns:
            List where each element is a list of communities at that step
        """
        return self.dendrogram

    def get_modularity_history(self) -> List[float]:
        """
        Get the history of modularity values at each step.

        Returns:
            List of modularity values
        """
        return self.modularity_history
