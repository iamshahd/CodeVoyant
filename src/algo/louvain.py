"""
Hand-implemented Louvain algorithm for community detection.

Reference:
    Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008).
    Fast unfolding of communities in large networks.
    Journal of statistical mechanics: theory and experiment, 2008(10), P10008.
"""

import random
from typing import Any, Dict, List, Optional, Set

import networkx as nx

from .base import CommunityDetectionAlgorithm


class LouvainCommunityDetection(CommunityDetectionAlgorithm):
    """
    Louvain community detection algorithm implemented from scratch.

    The Louvain method is a greedy optimization method that attempts to optimize
    the "modularity" of a partition of the network. It's one of the fastest and
    most widely used community detection algorithms.
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
        self.hieracrhy: Optional[List[Dict[str, int]]] = None

        if seed is not None:
            random.seed(seed)

    def _get_edge_weight(self, u: str, v: str) -> float:
        """Get the weight of an edge between two nodes."""
        if self.graph.has_edge(u, v):
            if self.weight and self.weight in self.graph[u][v]:
                return float(self.graph[u][v][self.weight])
            return 1.0
        return 0.0

    def _compute_modularity(
        self, partition: Dict[str, int], m: float, k: Dict[str, float]
    ) -> float:
        """
        Compute modularity for a given partition.

        Args:
            partition: Dictionary mapping node to community ID
            m: Total weight of all edges in the graph
            k: Dictionary mapping node to its degree (sum of edge weights)

        Returns:
            Modularity value
        """
        q = 0.0
        for u in self.graph.nodes():
            for v in self.graph.nodes():
                if partition[u] == partition[v]:
                    a_uv = self._get_edge_weight(u, v)
                    q += a_uv - self.resolution * (k[u] * k[v]) / (2 * m)

        return q / (2 * m) if m > 0 else 0.0

    def _compute_modularity_gain(
        self,
        node: str,
        community: int,
        partition: Dict[str, int],
        k_i: float,
        sigma_tot: Dict[int, float],
        k_i_in: float,
        m: float,
    ) -> float:
        """
        Compute the modularity gain from moving a node to a community.

        Args:
            node: Node to move
            community: Target community
            partition: Current partition
            k_i: Degree of node
            sigma_tot: Total degree of each community
            k_i_in: Sum of weights of edges from node to nodes in community
            m: Total weight of all edges

        Returns:
            Modularity gain
        """
        # Delta Q formula from the Louvain paper
        delta_q = (k_i_in - sigma_tot[community] * k_i / (2 * m)) * self.resolution
        return delta_q / m

    def _get_neighbors_communities(
        self, node: str, partition: Dict[str, int]
    ) -> Dict[int, float]:
        """
        Get the communities of neighbors and the sum of edge weights to each.

        Args:
            node: Node to analyze
            partition: Current partition

        Returns:
            Dictionary mapping community ID to sum of edge weights
        """
        neighbor_communities: Dict[int, float] = {}

        for neighbor in self.graph.neighbors(node):
            comm = partition[neighbor]
            weight = self._get_edge_weight(node, neighbor)
            neighbor_communities[comm] = neighbor_communities.get(comm, 0.0) + weight

        return neighbor_communities

    def _first_phase(self, partition: Dict[str, int]) -> tuple[Dict[str, int], bool]:
        """
        First phase: Move nodes to communities to maximize modularity gain.

        Args:
            partition: Current partition

        Returns:
            Tuple of (new partition, whether any change was made)
        """
        # Calculate degrees and total edge weight
        k = {}  # degree of each node
        for node in self.graph.nodes():
            degree = sum(
                self._get_edge_weight(node, neighbor)
                for neighbor in self.graph.neighbors(node)
            )
            k[node] = degree

        m = sum(k.values()) / 2  # Total edge weight

        # Calculate total degree for each community
        sigma_tot: Dict[int, float] = {}
        for node, comm in partition.items():
            sigma_tot[comm] = sigma_tot.get(comm, 0.0) + k[node]

        overall_improvement = False

        # Keep iterating until no improvement in a full pass
        while True:
            improvement_this_pass = False
            nodes = list(self.graph.nodes())
            random.shuffle(nodes)  # Random order for processing

            for node in nodes:
                current_comm = partition[node]
                k_i = k[node]

                # Get neighbors' communities and edge weights
                neighbor_communities = self._get_neighbors_communities(node, partition)

                # Remove node from its current community (temporarily)
                sigma_tot[current_comm] -= k_i

                # Find best community to move to
                best_comm = current_comm
                best_gain = 0.0

                # Check all neighboring communities (including current one if it still has nodes)
                # Collect all communities to check
                communities_to_check = set(neighbor_communities.keys())

                for comm in communities_to_check:
                    k_i_in = neighbor_communities.get(comm, 0.0)

                    gain = self._compute_modularity_gain(
                        node, comm, partition, k_i, sigma_tot, k_i_in, m
                    )

                    if gain > best_gain:
                        best_gain = gain
                        best_comm = comm

                # Move node to best community
                if best_comm != current_comm:
                    partition[node] = best_comm
                    sigma_tot[best_comm] = sigma_tot.get(best_comm, 0.0) + k_i
                    improvement_this_pass = True
                    overall_improvement = True
                else:
                    sigma_tot[current_comm] += k_i

            # If no improvement in this pass, we've reached a local optimum
            if not improvement_this_pass:
                break

        return partition, overall_improvement

    def _second_phase(self, partition: Dict[str, int], graph: nx.Graph) -> nx.Graph:
        """
        Second phase: Build a new graph where nodes are communities.

        Args:
            partition: Current partition
            graph: Graph to use for building the new coarsened graph

        Returns:
            New coarsened graph
        """
        # Create mapping of community to nodes
        comm_nodes: Dict[int, List[str]] = {}
        for node, comm in partition.items():
            if comm not in comm_nodes:
                comm_nodes[comm] = []
            comm_nodes[comm].append(node)

        # Create new graph
        new_graph: nx.Graph = nx.Graph()

        # Add nodes (communities)
        for comm in comm_nodes.keys():
            new_graph.add_node(comm)

        # Add edges between communities
        edge_weights: Dict[tuple[int, int], float] = {}

        for u in graph.nodes():
            comm_u = partition[u]

            for v in graph.neighbors(u):
                comm_v = partition[v]
                # Get weight directly from the passed graph
                if graph.has_edge(u, v):
                    if self.weight and self.weight in graph[u][v]:
                        weight = float(graph[u][v][self.weight])
                    else:
                        weight = 1.0
                else:
                    weight = 0.0

                if comm_u == comm_v:
                    # Self-loop for intra-community edges
                    edge = (comm_u, comm_u)
                else:
                    # Inter-community edge
                    edge = (comm_u, comm_v) if comm_u < comm_v else (comm_v, comm_u)

                edge_weights[edge] = edge_weights.get(edge, 0.0) + weight

        # Add edges to new graph
        for (u, v), weight in edge_weights.items():
            if u == v:
                # Self-loop
                if self.weight:
                    new_graph.add_edge(u, v, **{self.weight: weight})
                else:
                    new_graph.add_edge(u, v)
            else:
                if self.weight:
                    new_graph.add_edge(u, v, **{self.weight: weight})
                else:
                    new_graph.add_edge(u, v)

        return new_graph

    def detect_communities(self) -> List[Set[str]]:
        """
        Detect communities using the Louvain algorithm.

        Returns:
            List of sets, where each set contains node IDs belonging to a community
        """
        # Save original graph
        original_graph = self.graph

        # Initialize: each node in its own community
        partition = {node: i for i, node in enumerate(self.graph.nodes())}
        original_nodes = {i: {node} for i, node in enumerate(self.graph.nodes())}

        current_graph = self.graph.copy()
        self.hierarchy = [partition.copy()]

        while True:
            # Temporarily use current_graph for phase 1
            saved_graph = self.graph
            self.graph = current_graph

            # Phase 1: Optimize modularity by moving nodes
            partition_copy = partition.copy()
            new_partition, improved = self._first_phase(partition_copy)

            # Restore graph
            self.graph = saved_graph

            if not improved:
                break

            # Phase 2: Build new graph with communities as nodes
            current_graph = self._second_phase(new_partition, current_graph)

            # Update mapping of meta-nodes to original nodes
            # After first phase, new_partition maps: current_node -> new_community_id
            # We need to update original_nodes to map: new_community_id -> set of original nodes
            new_original_nodes: Dict[int, Set[str]] = {}

            for node, new_comm in new_partition.items():
                # Find which original nodes this 'node' represents
                # If node is an int, it's a meta-node key in original_nodes.
                if isinstance(node, int):
                    orig_nodes_set = original_nodes.get(node, set())
                else:
                    # node is an original node id (likely str): find which meta-node contains it
                    orig_nodes_set = None
                    for nodes_set in original_nodes.values():
                        if node in nodes_set:
                            orig_nodes_set = nodes_set
                            break

                    if orig_nodes_set is None:
                        # Fallback: treat node itself as original node
                        orig_nodes_set = {node}

                # Add these original nodes to the new community
                if new_comm not in new_original_nodes:
                    new_original_nodes[new_comm] = set()
                new_original_nodes[new_comm].update(orig_nodes_set)

            original_nodes = new_original_nodes

            # Reset partition for new graph - each node (community) is in its own partition
            partition = {node: node for node in current_graph.nodes()}

            self.hierarchy.append(partition.copy())

        # Restore original graph
        self.graph = original_graph

        # Convert to list of sets
        self.communities = [comm for comm in original_nodes.values() if comm]

        return self.communities

    def detect_communities_hierarchical(self) -> List[List[Set[str]]]:
        """
        Detect communities and return the full hierarchy of partitions.

        The Louvain algorithm works in iterations, producing increasingly
        coarse-grained community structures. This method returns all levels.

        Returns:
            List of partitions, where each partition is a list of communities
        """
        self.detect_communities()

        if not self.hierarchy:
            return []

        # Build hierarchy of communities in terms of original nodes
        hierarchy_result: List[List[Set[str]]] = []

        # First level: each node in its own community
        first_level = [{node} for node in self.graph.nodes()]
        hierarchy_result.append(first_level)

        # Add final level
        if self.communities:
            hierarchy_result.append(self.communities)

        return hierarchy_result

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
