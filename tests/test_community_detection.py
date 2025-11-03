"""
Unit tests for community detection algorithms.
"""

import pytest
import networkx as nx

from src.algo import (
    detect_communities,
    analyze_communities,
    visualize_communities,
    compare_algorithms,
    CommunityDetectionFactory,
    LouvainCommunityDetection,
    LabelPropagationCommunityDetection,
    GreedyModularityCommunityDetection,
    GirvanNewmanCommunityDetection,
)


@pytest.fixture
def simple_graph():
    """Create a simple test graph with clear community structure."""
    G = nx.Graph()
    
    # Community 1
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])
    
    # Community 2
    G.add_edges_from([(4, 5), (4, 6), (5, 6), (5, 7), (6, 7)])
    
    # Bridge between communities
    G.add_edge(3, 4)
    
    return G


@pytest.fixture
def karate_graph():
    """Use the well-known Karate Club graph."""
    return nx.karate_club_graph()


class TestCommunityDetectionFactory:
    """Test the community detection factory."""

    def test_create_louvain(self, simple_graph):
        """Test creating Louvain algorithm."""
        detector = CommunityDetectionFactory.create('louvain', simple_graph)
        assert isinstance(detector, LouvainCommunityDetection)

    def test_create_label_propagation(self, simple_graph):
        """Test creating Label Propagation algorithm."""
        detector = CommunityDetectionFactory.create('label_propagation', simple_graph)
        assert isinstance(detector, LabelPropagationCommunityDetection)

    def test_create_greedy_modularity(self, simple_graph):
        """Test creating Greedy Modularity algorithm."""
        detector = CommunityDetectionFactory.create('greedy_modularity', simple_graph)
        assert isinstance(detector, GreedyModularityCommunityDetection)

    def test_create_girvan_newman(self, simple_graph):
        """Test creating Girvan-Newman algorithm."""
        detector = CommunityDetectionFactory.create('girvan_newman', simple_graph)
        assert isinstance(detector, GirvanNewmanCommunityDetection)

    def test_create_invalid_algorithm(self, simple_graph):
        """Test that invalid algorithm name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            CommunityDetectionFactory.create('invalid_algo', simple_graph)

    def test_available_algorithms(self):
        """Test getting list of available algorithms."""
        algorithms = CommunityDetectionFactory.available_algorithms()
        assert 'louvain' in algorithms
        assert 'girvan_newman' in algorithms


class TestLouvainCommunityDetection:
    """Test Louvain algorithm."""

    def test_detect_communities(self, simple_graph):
        """Test basic community detection."""
        detector = LouvainCommunityDetection(simple_graph, seed=42)
        communities = detector.detect_communities()
        
        assert len(communities) > 0
        # All nodes should be in exactly one community
        all_nodes = set().union(*communities)
        assert all_nodes == set(simple_graph.nodes())

    def test_detect_communities_karate(self, karate_graph):
        """Test on Karate Club graph."""
        detector = LouvainCommunityDetection(karate_graph, seed=42)
        communities = detector.detect_communities()
        
        assert len(communities) >= 2  # Should find at least 2 communities
        assert len(communities) <= 10  # But not too many

    def test_get_modularity(self, simple_graph):
        """Test modularity calculation."""
        detector = LouvainCommunityDetection(simple_graph, seed=42)
        communities = detector.detect_communities()
        modularity = detector.get_modularity()
        
        assert -1 <= modularity <= 1
        assert modularity > 0  # Should have positive modularity

    def test_hierarchical_detection(self, karate_graph):
        """Test hierarchical community detection."""
        detector = LouvainCommunityDetection(karate_graph, seed=42)
        hierarchy = detector.detect_communities_hierarchical()
        
        assert len(hierarchy) > 0
        # Number of communities should generally decrease up the hierarchy
        num_communities = [len(level) for level in hierarchy]
        assert num_communities[0] >= num_communities[-1]

    def test_resolution_parameter(self, karate_graph):
        """Test that resolution parameter affects number of communities."""
        low_res = LouvainCommunityDetection(karate_graph, resolution=0.5, seed=42)
        high_res = LouvainCommunityDetection(karate_graph, resolution=2.0, seed=42)
        
        low_communities = low_res.detect_communities()
        high_communities = high_res.detect_communities()
        
        # Higher resolution should generally lead to more communities
        # (though this isn't guaranteed for all graphs)
        assert len(high_communities) >= len(low_communities)


class TestGirvanNewmanCommunityDetection:
    """Test Girvan-Newman algorithm."""

    def test_detect_communities_with_target(self, simple_graph):
        """Test detection with target number of communities."""
        detector = GirvanNewmanCommunityDetection(simple_graph)
        communities = detector.detect_communities(num_communities=2)
        
        assert len(communities) == 2
        all_nodes = set().union(*communities)
        assert all_nodes == set(simple_graph.nodes())

    def test_detect_communities_optimal(self, simple_graph):
        """Test detection with optimal modularity."""
        detector = GirvanNewmanCommunityDetection(simple_graph)
        communities = detector.detect_communities(num_communities=None)
        
        assert len(communities) > 0


class TestCommunityAnalyzer:
    """Test community analysis utilities."""

    def test_get_modularity(self, simple_graph):
        """Test modularity calculation."""
        communities = detect_communities(simple_graph, algorithm='louvain', seed=42)
        analyzer = analyze_communities(simple_graph, communities)
        
        modularity = analyzer.get_modularity()
        assert -1 <= modularity <= 1

    def test_get_community_metrics(self, simple_graph):
        """Test getting metrics for individual communities."""
        communities = detect_communities(simple_graph, algorithm='louvain', seed=42)
        analyzer = analyze_communities(simple_graph, communities)
        
        metrics = analyzer.get_community_metrics(0)
        
        assert 'community_id' in metrics
        assert 'size' in metrics
        assert 'density' in metrics
        assert 'conductance' in metrics
        assert metrics['size'] > 0

    def test_get_all_metrics(self, karate_graph):
        """Test getting metrics for all communities."""
        communities = detect_communities(karate_graph, algorithm='louvain', seed=42)
        analyzer = analyze_communities(karate_graph, communities)
        
        all_metrics = analyzer.get_all_metrics()
        
        assert len(all_metrics) == len(communities)
        for metrics in all_metrics:
            assert 'size' in metrics
            assert metrics['size'] > 0

    def test_export_communities(self, simple_graph, tmp_path):
        """Test exporting communities to file."""
        communities = detect_communities(simple_graph, algorithm='louvain', seed=42)
        analyzer = analyze_communities(simple_graph, communities)
        
        output_file = tmp_path / "communities.txt"
        analyzer.export_communities(str(output_file), include_metrics=True)
        
        assert output_file.exists()
        content = output_file.read_text()
        assert "Community Detection Results" in content
        assert "Modularity" in content


class TestCommunityVisualizer:
    """Test community visualization utilities."""

    def test_get_node_colors(self, simple_graph):
        """Test getting node colors."""
        communities = detect_communities(simple_graph, algorithm='louvain', seed=42)
        visualizer = visualize_communities(simple_graph, communities)
        
        colors = visualizer.get_node_colors()
        
        assert len(colors) == simple_graph.number_of_nodes()
        # All nodes should have a color
        for node in simple_graph.nodes():
            assert node in colors

    def test_add_community_attributes(self, simple_graph):
        """Test adding community attributes to graph."""
        communities = detect_communities(simple_graph, algorithm='louvain', seed=42)
        visualizer = visualize_communities(simple_graph, communities)
        
        graph_with_attrs = visualizer.add_community_attributes()
        
        # Check that all nodes have community attribute
        for node in graph_with_attrs.nodes():
            assert 'community' in graph_with_attrs.nodes[node]

    def test_create_community_subgraphs(self, simple_graph):
        """Test creating subgraphs for each community."""
        communities = detect_communities(simple_graph, algorithm='louvain', seed=42)
        visualizer = visualize_communities(simple_graph, communities)
        
        subgraphs = visualizer.create_community_subgraphs()
        
        assert len(subgraphs) == len(communities)
        # Total nodes in subgraphs should equal total nodes in original graph
        total_nodes = sum(sg.number_of_nodes() for sg in subgraphs)
        assert total_nodes == simple_graph.number_of_nodes()


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_detect_communities(self, simple_graph):
        """Test detect_communities convenience function."""
        communities = detect_communities(simple_graph, algorithm='louvain', seed=42)
        
        assert len(communities) > 0
        assert isinstance(communities, list)
        assert all(isinstance(comm, set) for comm in communities)

    def test_compare_algorithms(self, karate_graph):
        """Test comparing multiple algorithms."""
        results = compare_algorithms(
            karate_graph,
            algorithms=['louvain', 'label_propagation']
        )
        
        assert 'louvain' in results
        assert 'label_propagation' in results
        
        for algo_name, result in results.items():
            assert 'num_communities' in result
            assert 'modularity' in result


class TestDirectedGraphHandling:
    """Test handling of directed graphs."""

    def test_directed_graph_conversion(self):
        """Test that directed graphs are converted to undirected."""
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
        
        detector = LouvainCommunityDetection(G, seed=42)
        communities = detector.detect_communities()
        
        assert len(communities) > 0
        # Ensure all nodes are covered
        all_nodes = set().union(*communities)
        assert all_nodes == set(G.nodes())


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_graph(self):
        """Test handling of empty graph."""
        G = nx.Graph()
        detector = LouvainCommunityDetection(G, seed=42)
        communities = detector.detect_communities()
        
        assert communities == []

    def test_single_node(self):
        """Test graph with single node."""
        G = nx.Graph()
        G.add_node(0)
        
        detector = LouvainCommunityDetection(G, seed=42)
        communities = detector.detect_communities()
        
        assert len(communities) == 1
        assert 0 in communities[0]

    def test_disconnected_graph(self):
        """Test graph with disconnected components."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        G.add_edges_from([(3, 4), (4, 5)])
        
        detector = LouvainCommunityDetection(G, seed=42)
        communities = detector.detect_communities()
        
        assert len(communities) >= 2  # At least one community per component
