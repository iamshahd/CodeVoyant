"""Graph loading and network visualization utilities."""

import json
from typing import Any, Dict, Optional

import networkx as nx
from networkx.readwrite import json_graph
from pyvis.network import Network  # type: ignore[import-untyped]


def load_graph_from_json(file_path: str) -> nx.Graph:
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

    return graph


def create_pyvis_network_with_communities(
    graph_data: Dict[str, Any],
    communities: Optional[list] = None,
    node_colors: Optional[Dict[str, str]] = None,
    show_community_labels: bool = True,
) -> Network:
    """
    Create a Pyvis network with community-based coloring.

    Args:
        graph_data: Graph data in node-link format
        communities: List of communities (sets of node IDs)
        node_colors: Dictionary mapping node IDs to color strings
        show_community_labels: Whether to show community ID in node title

    Returns:
        Pyvis Network object
    """
    net = Network(height="750px", width="100%", notebook=False)

    # Configure physics for better layout
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {"iterations": 150}
        }
    }
    """)

    # Build node to community mapping if communities provided
    node_to_community = {}
    if communities:
        for comm_id, community in enumerate(communities):
            for node in community:
                node_to_community[str(node)] = comm_id

    # Add nodes with community colors
    for node in graph_data.get("nodes", []):
        node_id = node["id"]
        label = node.get("label", str(node_id))

        # Determine color
        color = None
        title = label
        if node_colors and str(node_id) in node_colors:
            color = node_colors[str(node_id)]

        # Add community info to title
        if show_community_labels and str(node_id) in node_to_community:
            comm_id = node_to_community[str(node_id)]
            title = f"{label}\nCommunity: {comm_id}"

        net.add_node(node_id, label=label, title=title, color=color if color else None)

    # Add edges
    for link in graph_data.get("links", []):
        net.add_edge(link["source"], link["target"], title=link.get("title", ""))

    return net
