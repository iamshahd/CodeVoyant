"""
Comprehensive unit tests for the Girvan-Newman community detection implementation.

This test suite thoroughly validates the hand-implemented GirvanNewmanCommunityDetection
algorithm, testing edge cases, correctness, and algorithm-specific features.
"""

import networkx as nx
import pytest

from src.algo.girvan_newman import GirvanNewmanCommunityDetection


@pytest.fixture
def simple_graph():
    """Create a simple test graph with clear community structure."""
    G = nx.Graph()
    # Community 1: nodes 0-3 (densely connected)
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])
    # Community 2: nodes 4-7 (densely connected)
    G.add_edges_from([(4, 5), (4, 6), (5, 6), (5, 7), (6, 7)])
    # Bridge between communities
    G.add_edge(3, 4)
    return G


@pytest.fixture
def triangle_graph():
    """Create a simple triangle graph."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    return G


@pytest.fixture
def barbell_graph():
    """Create a barbell graph (two cliques connected by a bridge)."""
    return nx.barbell_graph(5, 1)


@pytest.fixture
def karate_graph():
    """Use the well-known Karate Club graph."""
    return nx.karate_club_graph()


@pytest.fixture
def disconnected_graph():
    """Create a graph with disconnected components."""
    G = nx.Graph()
    # Component 1
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    # Component 2
    G.add_edges_from([(3, 4), (4, 5), (5, 3)])
    # Component 3 (isolated nodes)
    G.add_node(6)
    G.add_node(7)
    return G


class TestGirvanNewmanBasics:
    """Test basic functionality and initialization."""

    def test_initialization_default(self, simple_graph):
        """Test default initialization."""
        detector = GirvanNewmanCommunityDetection(simple_graph)
        assert detector.graph is not None
        assert detector.num_communities is None
        assert detector.use_modularity is True
        assert detector.communities is None
        assert detector.modularity_history == []
        assert detector.dendrogram == []

    def test_initialization_with_num_communities(self, simple_graph):
        """Test initialization with target number of communities."""
        detector = GirvanNewmanCommunityDetection(simple_graph, num_communities=3)
        assert detector.num_communities == 3
        assert detector.use_modularity is True

    def test_initialization_without_modularity(self, simple_graph):
        """Test initialization with modularity disabled."""
        detector = GirvanNewmanCommunityDetection(simple_graph, use_modularity=False)
        assert detector.use_modularity is False

    def test_directed_graph_conversion(self):
        """Test that directed graphs are converted to undirected."""
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])

        detector = GirvanNewmanCommunityDetection(G)
        # Should be undirected after conversion
        assert isinstance(detector.graph, nx.Graph)
        assert not isinstance(detector.graph, nx.DiGraph)


class TestEdgeBetweennessCalculation:
    """Test edge betweenness calculation."""

    def test_edge_betweenness_triangle(self, triangle_graph):
        """Test edge betweenness on a simple triangle."""
        detector = GirvanNewmanCommunityDetection(triangle_graph)
        betweenness = detector._calculate_edge_betweenness(triangle_graph)

        # In a triangle, all edges should have equal betweenness
        values = list(betweenness.values())
        assert len(values) == 3
        assert all(abs(v - values[0]) < 1e-9 for v in values)

    def test_edge_betweenness_simple_path(self):
        """Test edge betweenness on a simple path."""
        G = nx.path_graph(5)
        detector = GirvanNewmanCommunityDetection(G)
        betweenness = detector._calculate_edge_betweenness(G)

        # In a path, middle edges should have higher betweenness
        # Edge (1,2) and (2,3) should have highest betweenness
        edge_12 = tuple(sorted([1, 2]))
        edge_23 = tuple(sorted([2, 3]))

        assert betweenness[edge_12] == betweenness[edge_23]
        assert betweenness[edge_12] > betweenness[tuple(sorted([0, 1]))]
        assert betweenness[edge_23] > betweenness[tuple(sorted([3, 4]))]

    def test_edge_betweenness_star(self):
        """Test edge betweenness on a star graph."""
        G = nx.star_graph(5)  # Center node 0, leaves 1-5
        detector = GirvanNewmanCommunityDetection(G)
        betweenness = detector._calculate_edge_betweenness(G)

        # All edges should have equal betweenness in a star
        values = list(betweenness.values())
        assert all(abs(v - values[0]) < 1e-9 for v in values)

    def test_edge_betweenness_canonical_form(self, simple_graph):
        """Test that edge betweenness uses canonical form (u < v)."""
        detector = GirvanNewmanCommunityDetection(simple_graph)
        betweenness = detector._calculate_edge_betweenness(simple_graph)

        # All edges should be in canonical form
        for edge in betweenness.keys():
            u, v = edge
            assert u < v


class TestConnectedComponents:
    """Test connected components detection."""

    def test_connected_components_single(self, simple_graph):
        """Test on a fully connected graph."""
        detector = GirvanNewmanCommunityDetection(simple_graph)
        components = detector._get_connected_components(simple_graph)

        assert len(components) == 1
        assert len(components[0]) == simple_graph.number_of_nodes()

    def test_connected_components_multiple(self, disconnected_graph):
        """Test on a disconnected graph."""
        detector = GirvanNewmanCommunityDetection(disconnected_graph)
        components = detector._get_connected_components(disconnected_graph)

        # Should have 4 components (2 triangles + 2 isolated nodes)
        assert len(components) == 4

        # Check that all nodes are accounted for
        all_nodes = set().union(*components)
        assert all_nodes == set(disconnected_graph.nodes())

    def test_connected_components_after_edge_removal(self, barbell_graph):
        """Test components after removing bridge edge."""
        detector = GirvanNewmanCommunityDetection(barbell_graph)

        # Remove the bridge edge
        G = barbell_graph.copy()
        betweenness = detector._calculate_edge_betweenness(G)
        max_betweenness = max(betweenness.values())
        bridge_edge = [
            edge
            for edge, score in betweenness.items()
            if abs(score - max_betweenness) < 1e-9
        ][0]
        G.remove_edge(*bridge_edge)

        components = detector._get_connected_components(G)
        assert len(components) == 2


class TestModularityCalculation:
    """Test modularity calculation."""

    def test_modularity_single_community(self, simple_graph):
        """Test modularity with all nodes in one community."""
        detector = GirvanNewmanCommunityDetection(simple_graph)
        communities = [set(simple_graph.nodes())]
        modularity = detector._calculate_modularity(communities, simple_graph)

        # Modularity should be 0 for a single community
        assert abs(modularity) < 1e-9

    def test_modularity_range(self, simple_graph):
        """Test that modularity is in valid range."""
        detector = GirvanNewmanCommunityDetection(simple_graph)
        communities = detector.detect_communities()
        modularity = detector._calculate_modularity(communities, simple_graph)

        # Modularity should be between -0.5 and 1.0
        assert -0.5 <= modularity <= 1.0

    def test_modularity_optimal_partition(self, simple_graph):
        """Test modularity on optimal partition."""
        detector = GirvanNewmanCommunityDetection(simple_graph)
        # Manually create the expected optimal partition
        communities = [{0, 1, 2, 3}, {4, 5, 6, 7}]
        modularity = detector._calculate_modularity(communities, simple_graph)

        # Should have positive modularity for good partition
        assert modularity > 0

    def test_modularity_worst_partition(self, simple_graph):
        """Test modularity on worst partition (each node separate)."""
        detector = GirvanNewmanCommunityDetection(simple_graph)
        communities = [{node} for node in simple_graph.nodes()]
        modularity = detector._calculate_modularity(communities, simple_graph)

        # Should have negative modularity for poor partition
        assert modularity < 0

    def test_modularity_empty_communities(self, simple_graph):
        """Test modularity with empty community list."""
        detector = GirvanNewmanCommunityDetection(simple_graph)
        modularity = detector._calculate_modularity([], simple_graph)
        assert modularity == 0.0

    def test_modularity_empty_graph(self):
        """Test modularity on empty graph."""
        G = nx.Graph()
        detector = GirvanNewmanCommunityDetection(G)
        modularity = detector._calculate_modularity([], G)
        assert modularity == 0.0


class TestCommunityDetection:
    """Test the main community detection algorithm."""

    def test_detect_with_target_communities(self, simple_graph):
        """Test detection with target number of communities."""
        detector = GirvanNewmanCommunityDetection(simple_graph, num_communities=2)
        communities = detector.detect_communities()

        assert len(communities) == 2
        # All nodes should be assigned
        all_nodes = set().union(*communities)
        assert all_nodes == set(simple_graph.nodes())
        # No overlapping communities
        for i, comm1 in enumerate(communities):
            for j, comm2 in enumerate(communities):
                if i != j:
                    assert comm1.isdisjoint(comm2)

    def test_detect_with_modularity(self, simple_graph):
        """Test detection using modularity maximization."""
        detector = GirvanNewmanCommunityDetection(simple_graph, use_modularity=True)
        communities = detector.detect_communities()

        assert len(communities) > 0
        # All nodes should be assigned
        all_nodes = set().union(*communities)
        assert all_nodes == set(simple_graph.nodes())

    def test_detect_karate_club(self, karate_graph):
        """Test on Karate Club graph."""
        detector = GirvanNewmanCommunityDetection(karate_graph, use_modularity=True)
        communities = detector.detect_communities()

        # Should find reasonable number of communities
        assert 2 <= len(communities) <= 6
        # All nodes should be assigned
        all_nodes = set().union(*communities)
        assert all_nodes == set(karate_graph.nodes())

    def test_detect_triangle(self, triangle_graph):
        """Test on simple triangle (should stay as one community)."""
        detector = GirvanNewmanCommunityDetection(triangle_graph, use_modularity=True)
        communities = detector.detect_communities()

        # Triangle should be one community (or possibly split to 3 if all edges removed)
        assert len(communities) >= 1

    def test_detect_disconnected(self, disconnected_graph):
        """Test on disconnected graph."""
        detector = GirvanNewmanCommunityDetection(
            disconnected_graph, use_modularity=True
        )
        communities = detector.detect_communities()

        # Should find at least the disconnected components
        assert len(communities) >= 4

    def test_state_after_detection(self, simple_graph):
        """Test that detector state is properly set after detection."""
        detector = GirvanNewmanCommunityDetection(simple_graph, num_communities=2)
        communities = detector.detect_communities()

        # Check that communities are stored
        assert detector.communities == communities
        # Check node to community mapping exists
        assert detector.node_to_community is not None
        assert len(detector.node_to_community) == simple_graph.number_of_nodes()

    def test_multiple_detections(self, simple_graph):
        """Test running detection multiple times."""
        detector = GirvanNewmanCommunityDetection(simple_graph, num_communities=2)

        communities1 = detector.detect_communities()
        communities2 = detector.detect_communities()

        # Should get same results (deterministic algorithm)
        assert len(communities1) == len(communities2)


class TestNodeToCommunityMapping:
    """Test node to community mapping functionality."""

    def test_mapping_after_detection(self, simple_graph):
        """Test that mapping is created after detection."""
        detector = GirvanNewmanCommunityDetection(simple_graph, num_communities=2)
        detector.detect_communities()

        mapping = detector.get_node_to_community_mapping()
        assert len(mapping) == simple_graph.number_of_nodes()
        assert all(node in mapping for node in simple_graph.nodes())

    def test_mapping_consistency(self, simple_graph):
        """Test that mapping is consistent with communities."""
        detector = GirvanNewmanCommunityDetection(simple_graph, num_communities=2)
        communities = detector.detect_communities()
        mapping = detector.get_node_to_community_mapping()

        # Check that each node's mapping corresponds to correct community
        for comm_id, community in enumerate(communities):
            for node in community:
                assert mapping[node] == comm_id

    def test_mapping_before_detection(self, simple_graph):
        """Test mapping before detection is run."""
        detector = GirvanNewmanCommunityDetection(simple_graph)
        mapping = detector.get_node_to_community_mapping()
        assert mapping == {}


class TestDendrogram:
    """Test dendrogram (hierarchical) functionality."""

    def test_dendrogram_structure(self, simple_graph):
        """Test dendrogram structure."""
        detector = GirvanNewmanCommunityDetection(simple_graph, num_communities=2)
        detector.detect_communities()
        dendrogram = detector.get_dendrogram()

        # Should have at least 2 levels (initial + after first split)
        assert len(dendrogram) >= 2
        # First level should be entire graph
        assert len(dendrogram[0]) == 1
        assert len(dendrogram[0][0]) == simple_graph.number_of_nodes()

    def test_dendrogram_progression(self, simple_graph):
        """Test that dendrogram shows progression of splits."""
        detector = GirvanNewmanCommunityDetection(simple_graph, num_communities=3)
        detector.detect_communities()
        dendrogram = detector.get_dendrogram()

        # Number of communities should generally increase
        num_communities = [len(level) for level in dendrogram]
        # Should be monotonically non-decreasing
        for i in range(len(num_communities) - 1):
            assert num_communities[i] <= num_communities[i + 1]

    def test_dendrogram_node_coverage(self, simple_graph):
        """Test that all nodes are covered at each dendrogram level."""
        detector = GirvanNewmanCommunityDetection(simple_graph, num_communities=2)
        detector.detect_communities()
        dendrogram = detector.get_dendrogram()

        for level in dendrogram:
            all_nodes = set().union(*level)
            assert all_nodes == set(simple_graph.nodes())


class TestModularityHistory:
    """Test modularity history tracking."""

    def test_modularity_history_recorded(self, simple_graph):
        """Test that modularity history is recorded."""
        detector = GirvanNewmanCommunityDetection(simple_graph, use_modularity=True)
        detector.detect_communities()
        history = detector.get_modularity_history()

        # Should have recorded modularity at each step
        assert len(history) > 0

    def test_modularity_history_range(self, simple_graph):
        """Test that modularity values are in valid range."""
        detector = GirvanNewmanCommunityDetection(simple_graph, use_modularity=True)
        detector.detect_communities()
        history = detector.get_modularity_history()

        # All values should be in valid range
        for modularity in history:
            assert -0.5 <= modularity <= 1.0

    def test_modularity_peak_detected(self, barbell_graph):
        """Test that algorithm finds modularity peak."""
        detector = GirvanNewmanCommunityDetection(barbell_graph, use_modularity=True)
        detector.detect_communities()
        history = detector.get_modularity_history()

        # Should have a peak (or plateau)
        assert len(history) > 0
        max_modularity = max(history)
        # Best partition should have positive modularity
        assert max_modularity > 0


class TestAlgorithmProperties:
    """Test algorithmic properties and invariants."""

    def test_deterministic(self, simple_graph):
        """Test that algorithm is deterministic."""
        detector1 = GirvanNewmanCommunityDetection(simple_graph, num_communities=2)
        communities1 = detector1.detect_communities()

        detector2 = GirvanNewmanCommunityDetection(simple_graph, num_communities=2)
        communities2 = detector2.detect_communities()

        # Should get identical results
        assert len(communities1) == len(communities2)
        # Convert to sorted lists of sorted tuples for comparison
        sorted1 = sorted([sorted(list(c)) for c in communities1])
        sorted2 = sorted([sorted(list(c)) for c in communities2])
        assert sorted1 == sorted2

    def test_no_overlapping_communities(self, karate_graph):
        """Test that communities don't overlap."""
        detector = GirvanNewmanCommunityDetection(karate_graph, num_communities=3)
        communities = detector.detect_communities()

        # Check no overlaps
        for i, comm1 in enumerate(communities):
            for j, comm2 in enumerate(communities):
                if i != j:
                    assert comm1.isdisjoint(comm2)

    def test_all_nodes_assigned(self, karate_graph):
        """Test that all nodes are assigned to a community."""
        detector = GirvanNewmanCommunityDetection(karate_graph, use_modularity=True)
        communities = detector.detect_communities()

        all_nodes = set().union(*communities)
        assert all_nodes == set(karate_graph.nodes())

    def test_progressive_splitting(self, barbell_graph):
        """Test that communities split progressively (not merged)."""
        detector = GirvanNewmanCommunityDetection(barbell_graph, num_communities=3)
        detector.detect_communities()
        dendrogram = detector.get_dendrogram()

        # Number of communities should never decrease
        num_communities = [len(level) for level in dendrogram]
        for i in range(len(num_communities) - 1):
            assert num_communities[i] <= num_communities[i + 1]


class TestIntegrationWithNetworkX:
    """Test integration with NetworkX graphs and data structures."""

    def test_with_node_attributes(self):
        """Test with graph containing node attributes."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
        nx.set_node_attributes(G, {i: f"node_{i}" for i in G.nodes()}, "label")

        detector = GirvanNewmanCommunityDetection(G, num_communities=2)
        communities = detector.detect_communities()

        assert len(communities) == 2

    def test_with_edge_weights(self):
        """Test with weighted edges."""
        G = nx.Graph()
        G.add_weighted_edges_from(
            [
                (0, 1, 1.0),
                (1, 2, 1.0),
                (2, 0, 1.0),
                (3, 4, 1.0),
                (4, 5, 1.0),
                (5, 3, 1.0),
                (2, 3, 0.1),  # Weak bridge
            ]
        )

        detector = GirvanNewmanCommunityDetection(G, num_communities=2)
        communities = detector.detect_communities()

        # Should split at the weak bridge
        assert len(communities) == 2

    def test_with_string_node_ids(self):
        """Test with string node IDs."""
        G = nx.Graph()
        G.add_edges_from(
            [
                ("A", "B"),
                ("B", "C"),
                ("C", "A"),
                ("D", "E"),
                ("E", "F"),
                ("F", "D"),
                ("C", "D"),
            ]
        )

        detector = GirvanNewmanCommunityDetection(G, num_communities=2)
        communities = detector.detect_communities()

        assert len(communities) == 2
        all_nodes = set().union(*communities)
        assert all_nodes == set(G.nodes())

    def test_preserves_original_graph(self, simple_graph):
        """Test that original graph is not modified."""
        original_edges = set(simple_graph.edges())
        original_nodes = set(simple_graph.nodes())

        detector = GirvanNewmanCommunityDetection(simple_graph, num_communities=2)
        detector.detect_communities()

        # Original graph should be unchanged
        assert set(simple_graph.edges()) == original_edges
        assert set(simple_graph.nodes()) == original_nodes


class TestComparisonWithExpectedResults:
    """Test against known expected results for specific graphs."""

    def test_two_cliques_bridge(self):
        """Test on two cliques connected by a single edge."""
        G = nx.Graph()
        # Clique 1
        G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
        # Clique 2
        G.add_edges_from([(4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7)])
        # Bridge
        G.add_edge(3, 4)

        detector = GirvanNewmanCommunityDetection(G, num_communities=2)
        communities = detector.detect_communities()

        assert len(communities) == 2
        # Check that each clique is in its own community
        comm_sizes = sorted([len(c) for c in communities])
        assert comm_sizes == [4, 4]

    def test_path_graph_split(self):
        """Test splitting a path graph."""
        G = nx.path_graph(6)  # 0-1-2-3-4-5

        detector = GirvanNewmanCommunityDetection(G, num_communities=2)
        communities = detector.detect_communities()

        assert len(communities) == 2
        # Should split roughly in the middle
        sizes = sorted([len(c) for c in communities])
        assert sizes[0] >= 2 and sizes[1] >= 2

    def test_ring_of_cliques(self):
        """Test on ring of cliques."""
        G = nx.ring_of_cliques(4, 3)  # 4 cliques of size 3

        detector = GirvanNewmanCommunityDetection(G, num_communities=4)
        communities = detector.detect_communities()

        # Should identify close to 4 communities
        assert len(communities) >= 3
        assert len(communities) <= 5
