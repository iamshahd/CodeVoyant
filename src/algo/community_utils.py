"""
Utilities for analyzing and visualizing communities.
"""

from typing import Dict, List, Set, Optional, Any, Tuple
from pathlib import Path

import networkx as nx


class CommunityAnalyzer:
    """
    Analyze properties and quality of detected communities.
    """

    def __init__(self, graph: nx.Graph, communities: List[Set[str]]):
        """
        Initialize community analyzer.

        Args:
            graph: The original graph
            communities: List of detected communities
        """
        self.graph = graph
        self.communities = communities
        self.node_to_community = self._build_node_mapping()

    def _build_node_mapping(self) -> Dict[str, int]:
        """Build mapping from node to community ID."""
        mapping = {}
        for comm_id, community in enumerate(self.communities):
            for node in community:
                mapping[node] = comm_id
        return mapping

    def get_internal_edges(self, community: Set[str]) -> int:
        """
        Count edges within a community.

        Args:
            community: Set of nodes in the community

        Returns:
            Number of internal edges
        """
        subgraph = self.graph.subgraph(community)
        return subgraph.number_of_edges()

    def get_external_edges(self, community: Set[str]) -> int:
        """
        Count edges connecting a community to other nodes.

        Args:
            community: Set of nodes in the community

        Returns:
            Number of external edges
        """
        internal = self.get_internal_edges(community)
        total = sum(1 for node in community for _ in self.graph.neighbors(node))
        return total - (2 * internal)  # Each internal edge is counted twice

    def get_community_density(self, community: Set[str]) -> float:
        """
        Calculate the density of a community.

        Args:
            community: Set of nodes in the community

        Returns:
            Density value (0 to 1)
        """
        if len(community) <= 1:
            return 0.0
        
        subgraph = self.graph.subgraph(community)
        return nx.density(subgraph)

    def get_conductance(self, community: Set[str]) -> float:
        """
        Calculate the conductance of a community.
        
        Conductance measures the ratio of edges leaving the community
        to the total edges connected to the community.
        Lower values indicate better communities.

        Args:
            community: Set of nodes in the community

        Returns:
            Conductance value (0 to 1)
        """
        internal = self.get_internal_edges(community)
        external = self.get_external_edges(community)
        total = 2 * internal + external
        
        if total == 0:
            return 0.0
        
        return external / total

    def get_community_metrics(self, comm_id: int) -> Dict[str, Any]:
        """
        Get comprehensive metrics for a specific community.

        Args:
            comm_id: Community ID

        Returns:
            Dictionary of metrics
        """
        community = self.communities[comm_id]
        
        return {
            "community_id": comm_id,
            "size": len(community),
            "internal_edges": self.get_internal_edges(community),
            "external_edges": self.get_external_edges(community),
            "density": self.get_community_density(community),
            "conductance": self.get_conductance(community),
        }

    def get_all_metrics(self) -> List[Dict[str, Any]]:
        """
        Get metrics for all communities.

        Returns:
            List of metric dictionaries, one per community
        """
        return [
            self.get_community_metrics(i)
            for i in range(len(self.communities))
        ]

    def get_modularity(self) -> float:
        """
        Calculate modularity of the partition.

        Returns:
            Modularity score (-1 to 1, higher is better)
        """
        return nx.community.modularity(self.graph, self.communities)

    def get_coverage(self) -> float:
        """
        Calculate coverage of the partition.
        
        Coverage is the fraction of edges that fall within communities.

        Returns:
            Coverage value (0 to 1)
        """
        return nx.community.partition_quality(self.graph, self.communities)[0]

    def get_performance(self) -> float:
        """
        Calculate performance of the partition.
        
        Performance is the ratio of correctly classified pairs
        (both in same community or both in different communities).

        Returns:
            Performance value (0 to 1)
        """
        return nx.community.partition_quality(self.graph, self.communities)[1]

    def compare_communities(
        self,
        other_communities: List[Set[str]]
    ) -> Dict[str, float]:
        """
        Compare two community partitions using various similarity metrics.

        Args:
            other_communities: Another partition to compare against

        Returns:
            Dictionary of similarity metrics
        """
        # Convert to label format for comparison
        nodes = sorted(self.graph.nodes())
        
        labels1 = [self.node_to_community.get(node, -1) for node in nodes]
        
        other_mapping = {}
        for comm_id, community in enumerate(other_communities):
            for node in community:
                other_mapping[node] = comm_id
        labels2 = [other_mapping.get(node, -1) for node in nodes]

        # Use scikit-learn metrics if available, otherwise basic comparison
        try:
            from sklearn.metrics import (
                adjusted_rand_score,
                normalized_mutual_info_score,
            )
            
            return {
                "adjusted_rand_index": adjusted_rand_score(labels1, labels2),
                "normalized_mutual_info": normalized_mutual_info_score(labels1, labels2),
            }
        except ImportError:
            # Fallback to simple overlap metric
            overlap = sum(1 for l1, l2 in zip(labels1, labels2) if l1 == l2)
            return {
                "agreement_ratio": overlap / len(nodes) if nodes else 0.0,
            }

    def export_communities(
        self,
        output_path: str,
        include_metrics: bool = True
    ) -> None:
        """
        Export communities to a file.

        Args:
            output_path: Path to output file
            include_metrics: Whether to include community metrics
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output, "w") as f:
            f.write("# Community Detection Results\n\n")
            f.write(f"Total communities: {len(self.communities)}\n")
            f.write(f"Total nodes: {self.graph.number_of_nodes()}\n")
            f.write(f"Total edges: {self.graph.number_of_edges()}\n")
            f.write(f"Modularity: {self.get_modularity():.4f}\n\n")
            
            for comm_id, community in enumerate(self.communities):
                f.write(f"\n## Community {comm_id}\n")
                f.write(f"Size: {len(community)}\n")
                
                if include_metrics:
                    metrics = self.get_community_metrics(comm_id)
                    f.write(f"Density: {metrics['density']:.4f}\n")
                    f.write(f"Conductance: {metrics['conductance']:.4f}\n")
                
                f.write("Members:\n")
                for node in sorted(community):
                    f.write(f"  - {node}\n")


class CommunityVisualizer:
    """
    Visualize communities in graphs.
    """

    def __init__(self, graph: nx.Graph, communities: List[Set[str]]):
        """
        Initialize community visualizer.

        Args:
            graph: The original graph
            communities: List of detected communities
        """
        self.graph = graph
        self.communities = communities
        self.node_to_community = self._build_node_mapping()

    def _build_node_mapping(self) -> Dict[str, int]:
        """Build mapping from node to community ID."""
        mapping = {}
        for comm_id, community in enumerate(self.communities):
            for node in community:
                mapping[node] = comm_id
        return mapping

    def get_node_colors(self, colormap: str = "tab20") -> Dict[str, str]:
        """
        Get color mapping for nodes based on their community.

        Args:
            colormap: Matplotlib colormap name

        Returns:
            Dictionary mapping node IDs to color strings
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            
            num_communities = len(self.communities)
            cmap = plt.get_cmap(colormap)
            
            # Generate colors for communities
            colors = [
                mcolors.rgb2hex(cmap(i / max(num_communities - 1, 1)))
                for i in range(num_communities)
            ]
            
            # Map nodes to colors
            node_colors = {}
            for node, comm_id in self.node_to_community.items():
                node_colors[node] = colors[comm_id]
            
            return node_colors
            
        except ImportError:
            # Fallback to simple color scheme
            simple_colors = [
                "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
                "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5"
            ]
            node_colors = {}
            for node, comm_id in self.node_to_community.items():
                node_colors[node] = simple_colors[comm_id % len(simple_colors)]
            
            return node_colors

    def add_community_attributes(self) -> nx.Graph:
        """
        Create a copy of the graph with community attributes added to nodes.

        Returns:
            Graph with 'community' attribute on each node
        """
        g = self.graph.copy()
        
        for node, comm_id in self.node_to_community.items():
            g.nodes[node]["community"] = comm_id
        
        return g

    def create_community_subgraphs(self) -> List[nx.Graph]:
        """
        Create separate subgraphs for each community.

        Returns:
            List of subgraphs, one per community
        """
        subgraphs = []
        
        for community in self.communities:
            subgraph = self.graph.subgraph(community).copy()
            subgraphs.append(subgraph)
        
        return subgraphs

    def get_layout_with_communities(
        self,
        layout: str = "spring",
        **layout_kwargs: Any
    ) -> Dict[str, Tuple[float, float]]:
        """
        Generate a graph layout that considers community structure.

        Args:
            layout: Layout algorithm ('spring', 'kamada_kawai', etc.)
            **layout_kwargs: Additional arguments for the layout algorithm

        Returns:
            Dictionary mapping node IDs to (x, y) positions
        """
        if layout == "spring":
            # Use communities to bias the layout
            pos = nx.spring_layout(
                self.graph,
                **layout_kwargs
            )
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self.graph, **layout_kwargs)
        else:
            # Default to spring layout
            pos = nx.spring_layout(self.graph, **layout_kwargs)
        
        return pos
