"""
Module for loading graphs from JSON files for benchmarking.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
from networkx.readwrite import json_graph


class GraphLoader:
    """Loads graphs from JSON files for benchmarking."""

    @staticmethod
    def load_graph_from_json(file_path: Path) -> nx.Graph:
        """
        Load a NetworkX graph from JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            NetworkX graph
        """
        with open(file_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

        # Convert from node-link format to NetworkX graph
        graph = json_graph.node_link_graph(graph_data, edges="links")

        # Convert to undirected for community detection
        if graph.is_directed():
            graph = graph.to_undirected()

        return graph

    @staticmethod
    def load_graphs_from_paths(
        file_paths: List[Path],
    ) -> List[Tuple[str, str, nx.Graph]]:
        """
        Load multiple graphs from file paths.

        Args:
            file_paths: List of paths to JSON files

        Returns:
            List of tuples containing (project_name, graph_type, graph)
            where project_name is like 'httpx', 'django', 'rich'
            and graph_type is 'call_graph' or 'dependency_graph'
        """
        graphs = []

        for file_path in file_paths:
            graph = GraphLoader.load_graph_from_json(file_path)

            # Extract project name and graph type from path
            # e.g., codevoyant_output/httpx/call_graph.json
            parts = file_path.parts
            project_name = parts[-2]  # httpx, django, rich
            graph_type = file_path.stem  # call_graph or dependency_graph

            graphs.append((project_name, graph_type, graph))

        return graphs

    @staticmethod
    def get_graph_metadata(graph: nx.Graph) -> Dict[str, int]:
        """
        Get metadata about the graph.

        Args:
            graph: NetworkX graph

        Returns:
            Dictionary with 'num_nodes' and 'num_edges'
        """
        return {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
        }
