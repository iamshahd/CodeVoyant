"""
Base classes and interfaces for graph algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Set, Optional, Any

import networkx as nx


class CommunityDetectionAlgorithm(ABC):
    """
    Abstract base class for community detection algorithms.
    
    All community detection algorithms should inherit from this class
    and implement the detect_communities method.
    """

    def __init__(self, graph: nx.Graph, **kwargs: Any):
        """
        Initialize the community detection algorithm.

        Args:
            graph: NetworkX graph (will be converted to undirected for community detection)
            **kwargs: Algorithm-specific parameters
        """
        # Convert to undirected graph for community detection if needed
        if isinstance(graph, nx.DiGraph):
            self.graph = graph.to_undirected()
        else:
            self.graph = graph.copy()
        
        self.communities: Optional[List[Set[str]]] = None
        self.node_to_community: Optional[Dict[str, int]] = None
        self.params = kwargs

    @abstractmethod
    def detect_communities(self) -> List[Set[str]]:
        """
        Detect communities in the graph.

        Returns:
            List of sets, where each set contains node IDs belonging to a community
        """
        pass

    def get_node_to_community_mapping(self) -> Dict[str, int]:
        """
        Get a mapping from node ID to community ID.

        Returns:
            Dictionary mapping node IDs to their community IDs
        """
        if self.communities is None:
            self.detect_communities()
        
        if self.node_to_community is None:
            self.node_to_community = {}
            for comm_id, community in enumerate(self.communities):
                for node in community:
                    self.node_to_community[node] = comm_id
        
        return self.node_to_community

    def get_modularity(self) -> float:
        """
        Calculate the modularity of the detected communities.

        Returns:
            Modularity score (between -1 and 1, higher is better)
        """
        if self.communities is None:
            self.detect_communities()
        
        return nx.community.modularity(self.graph, self.communities)

    def get_community_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the detected communities.

        Returns:
            Dictionary containing various community statistics
        """
        if self.communities is None:
            self.detect_communities()
        
        community_sizes = [len(comm) for comm in self.communities]
        
        return {
            "num_communities": len(self.communities),
            "modularity": self.get_modularity(),
            "avg_community_size": sum(community_sizes) / len(community_sizes) if community_sizes else 0,
            "min_community_size": min(community_sizes) if community_sizes else 0,
            "max_community_size": max(community_sizes) if community_sizes else 0,
            "community_sizes": community_sizes,
        }

    def get_intercommunity_edges(self) -> List[tuple]:
        """
        Get edges that connect different communities.

        Returns:
            List of edges (tuples) that cross community boundaries
        """
        if self.communities is None:
            self.detect_communities()
        
        node_to_comm = self.get_node_to_community_mapping()
        intercommunity_edges = []
        
        for u, v in self.graph.edges():
            if node_to_comm.get(u) != node_to_comm.get(v):
                intercommunity_edges.append((u, v))
        
        return intercommunity_edges

    def get_community_graph(self) -> nx.Graph:
        """
        Create a meta-graph where nodes are communities and edges represent
        connections between communities.

        Returns:
            NetworkX graph where nodes are community IDs
        """
        if self.communities is None:
            self.detect_communities()
        
        node_to_comm = self.get_node_to_community_mapping()
        community_graph = nx.Graph()
        
        # Add nodes (communities) with size attribute
        for comm_id, community in enumerate(self.communities):
            community_graph.add_node(comm_id, size=len(community))
        
        # Add edges between communities (weighted by number of connections)
        edge_weights: Dict[tuple, int] = {}
        for u, v in self.graph.edges():
            comm_u = node_to_comm.get(u)
            comm_v = node_to_comm.get(v)
            
            if comm_u is not None and comm_v is not None and comm_u != comm_v:
                edge = tuple(sorted([comm_u, comm_v]))
                edge_weights[edge] = edge_weights.get(edge, 0) + 1
        
        for (comm_u, comm_v), weight in edge_weights.items():
            community_graph.add_edge(comm_u, comm_v, weight=weight)
        
        return community_graph

    def __repr__(self) -> str:
        """String representation of the algorithm."""
        return f"{self.__class__.__name__}(nodes={self.graph.number_of_nodes()}, edges={self.graph.number_of_edges()})"
