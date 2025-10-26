"""
Module for serializing graphs to various formats for UI consumption.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import networkx as nx
from networkx.readwrite import json_graph


class GraphSerializer:
    """Serializes NetworkX graphs to various formats."""
    
    @staticmethod
    def to_json(graph: nx.DiGraph, output_path: str, format_type: str = 'node_link') -> None:
        """
        Serialize a NetworkX graph to JSON format.
        
        Args:
            graph: NetworkX graph to serialize
            output_path: Path where the JSON file will be saved
            format_type: Type of JSON format ('node_link', 'cytoscape', 'd3')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == 'node_link':
            data = GraphSerializer._to_node_link(graph)
        elif format_type == 'cytoscape':
            data = GraphSerializer._to_cytoscape(graph)
        elif format_type == 'd3':
            data = GraphSerializer._to_d3_force(graph)
        else:
            raise ValueError(f"Unknown format type: {format_type}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Graph saved to {output_path}")
    
    @staticmethod
    def _to_node_link(graph: nx.DiGraph) -> Dict:
        """
        Convert graph to node-link format (standard NetworkX format).
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dictionary in node-link format
        """
        return json_graph.node_link_data(graph)
    
    @staticmethod
    def _to_cytoscape(graph: nx.DiGraph) -> Dict:
        """
        Convert graph to Cytoscape.js format.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dictionary in Cytoscape format
        """
        elements = []
        
        # Add nodes
        for node, data in graph.nodes(data=True):
            element = {
                'data': {
                    'id': str(node),
                    'label': data.get('function_name', str(node)),
                    **data
                }
            }
            elements.append(element)
        
        # Add edges
        for source, target, data in graph.edges(data=True):
            element = {
                'data': {
                    'id': f"{source}->{target}",
                    'source': str(source),
                    'target': str(target),
                    **data
                }
            }
            elements.append(element)
        
        return {'elements': elements}
    
    @staticmethod
    def _to_d3_force(graph: nx.DiGraph) -> Dict:
        """
        Convert graph to D3.js force-directed layout format.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dictionary in D3 format
        """
        nodes = []
        links = []
        
        # Create node index mapping
        node_indices = {node: idx for idx, node in enumerate(graph.nodes())}
        
        # Add nodes
        for node, data in graph.nodes(data=True):
            node_data = {
                'id': str(node),
                'name': data.get('function_name', str(node)),
                **data
            }
            nodes.append(node_data)
        
        # Add links
        for source, target, data in graph.edges(data=True):
            link_data = {
                'source': node_indices[source],
                'target': node_indices[target],
                **data
            }
            links.append(link_data)
        
        return {
            'nodes': nodes,
            'links': links
        }
    
    @staticmethod
    def to_graphml(graph: nx.DiGraph, output_path: str) -> None:
        """
        Serialize graph to GraphML format.
        
        Args:
            graph: NetworkX graph to serialize
            output_path: Path where the GraphML file will be saved
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        nx.write_graphml(graph, output_path)
        print(f"Graph saved to {output_path}")
    
    @staticmethod
    def to_gexf(graph: nx.DiGraph, output_path: str) -> None:
        """
        Serialize graph to GEXF format (for Gephi).
        
        Args:
            graph: NetworkX graph to serialize
            output_path: Path where the GEXF file will be saved
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        nx.write_gexf(graph, output_path)
        print(f"Graph saved to {output_path}")
    
    @staticmethod
    def to_pickle(graph: nx.DiGraph, output_path: str) -> None:
        """
        Serialize graph to pickle format (for Python).
        
        Args:
            graph: NetworkX graph to serialize
            output_path: Path where the pickle file will be saved
        """
        import pickle
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(graph, f)
        
        print(f"Graph saved to {output_path}")
    
    @staticmethod
    def from_pickle(input_path: str) -> nx.DiGraph:
        """
        Load graph from pickle format.
        
        Args:
            input_path: Path to the pickle file
            
        Returns:
            NetworkX graph
        """
        import pickle
        
        with open(input_path, 'rb') as f:
            graph = pickle.load(f)
        
        return graph
    
    @staticmethod
    def to_summary(graph: nx.DiGraph, output_path: str) -> None:
        """
        Create a human-readable summary of the graph.
        
        Args:
            graph: NetworkX graph
            output_path: Path where the summary file will be saved
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Graph Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Nodes: {graph.number_of_nodes()}\n")
            f.write(f"Total Edges: {graph.number_of_edges()}\n")
            f.write(f"Graph Density: {nx.density(graph):.4f}\n\n")
            
            # Degree statistics
            if graph.number_of_nodes() > 0:
                in_degrees = dict(graph.in_degree())
                out_degrees = dict(graph.out_degree())
                
                f.write("Degree Statistics:\n")
                f.write("-" * 50 + "\n")
                
                # Top nodes by in-degree
                f.write("\nTop 10 nodes by in-degree (most called):\n")
                sorted_in = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
                for node, degree in sorted_in:
                    f.write(f"  {node}: {degree}\n")
                
                # Top nodes by out-degree
                f.write("\nTop 10 nodes by out-degree (calls most others):\n")
                sorted_out = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
                for node, degree in sorted_out:
                    f.write(f"  {node}: {degree}\n")
            
            # Connected components
            if not nx.is_empty(graph):
                weakly_connected = list(nx.weakly_connected_components(graph))
                f.write(f"\nWeakly Connected Components: {len(weakly_connected)}\n")
                
                if len(weakly_connected) <= 10:
                    for i, component in enumerate(weakly_connected, 1):
                        f.write(f"  Component {i}: {len(component)} nodes\n")
        
        print(f"Summary saved to {output_path}")
