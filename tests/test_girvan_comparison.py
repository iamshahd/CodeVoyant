"""
Comprehensive tests comparing hand-implemented Girvan-Newman algorithm
with NetworkX library implementation.

This test suite verifies that GirvanNewmanCommunityDetection (hand-implemented)
produces identical or equivalent results to GirvanNewmanLibrary (NetworkX wrapper).
"""

from typing import List, Set

import networkx as nx
import pytest

from src.algo.community_detection import GirvanNewmanLibrary
from src.algo.girvan_newman import GirvanNewmanCommunityDetection


def normalize_communities(communities: List[Set]) -> List[Set]:
    """
    Normalize communities for comparison by sorting them.

    Args:
        communities: List of community sets

    Returns:
        Sorted list of sorted sets (as frozensets for hashability)
    """
    return sorted(
        [set(comm) for comm in communities], key=lambda x: (len(x), sorted(x))
    )


def communities_are_equivalent(comm1: List[Set], comm2: List[Set]) -> bool:
    """
    Check if two community partitions are equivalent.

    Two partitions are equivalent if they contain the same communities,
    regardless of order.

    Args:
        comm1: First community partition
        comm2: Second community partition

    Returns:
        True if partitions are equivalent
    """
    norm1 = normalize_communities(comm1)
    norm2 = normalize_communities(comm2)
    return norm1 == norm2


def verify_valid_partition(graph: nx.Graph, communities: List[Set]) -> bool:
    """
    Verify that a community partition is valid.

    Args:
        graph: The graph
        communities: The community partition

    Returns:
        True if partition is valid
    """
    # All nodes should be in exactly one community
    all_nodes = set().union(*communities) if communities else set()
    if all_nodes != set(graph.nodes()):
        return False

    # No node should appear in multiple communities
    node_count = sum(len(comm) for comm in communities)
    if node_count != len(all_nodes):
        return False

    return True


def calculate_modularity(graph: nx.Graph, communities: List[Set]) -> float:
    """
    Calculate modularity using NetworkX for verification.

    Args:
        graph: The graph
        communities: The community partition

    Returns:
        Modularity score
    """
    if not communities:
        return 0.0
    return nx.community.modularity(graph, communities)


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
def barbell_graph():
    """Create a barbell graph (two cliques connected by a bridge)."""
    return nx.barbell_graph(5, 1)


@pytest.fixture
def karate_graph():
    """Use the well-known Karate Club graph."""
    return nx.karate_club_graph()


@pytest.fixture
def triangle_graph():
    """Create a simple triangle graph."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    return G


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


@pytest.fixture
def path_graph():
    """Create a path graph."""
    return nx.path_graph(10)


class TestGirvanNewmanBasicComparison:
    """Test basic functionality comparison between implementations."""

    def test_simple_graph_with_target_communities(self, simple_graph):
        """Test on simple graph with target number of communities."""
        target = 2

        hand_impl = GirvanNewmanCommunityDetection(simple_graph, num_communities=target)
        library = GirvanNewmanLibrary(simple_graph)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities(num_communities=target)

        # Verify both are valid partitions
        assert verify_valid_partition(simple_graph, hand_communities)
        assert verify_valid_partition(simple_graph, lib_communities)

        # Should have same number of communities
        assert len(hand_communities) == target
        assert len(lib_communities) == target

        # Should produce equivalent partitions
        assert communities_are_equivalent(hand_communities, lib_communities)

    def test_barbell_graph_two_communities(self, barbell_graph):
        """Test on barbell graph with target of 2 communities."""
        target = 2

        hand_impl = GirvanNewmanCommunityDetection(
            barbell_graph, num_communities=target
        )
        library = GirvanNewmanLibrary(barbell_graph)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities(num_communities=target)

        # Verify both are valid partitions
        assert verify_valid_partition(barbell_graph, hand_communities)
        assert verify_valid_partition(barbell_graph, lib_communities)

        # Should have exactly 2 communities
        assert len(hand_communities) == 2
        assert len(lib_communities) == 2

        # Should produce equivalent partitions
        assert communities_are_equivalent(hand_communities, lib_communities)

    def test_triangle_graph(self, triangle_graph):
        """Test on simple triangle graph."""
        hand_impl = GirvanNewmanCommunityDetection(triangle_graph, num_communities=2)
        library = GirvanNewmanLibrary(triangle_graph)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities(num_communities=2)

        assert verify_valid_partition(triangle_graph, hand_communities)
        assert verify_valid_partition(triangle_graph, lib_communities)

        # Should produce equivalent partitions
        assert communities_are_equivalent(hand_communities, lib_communities)

    def test_empty_graph(self):
        """Test behavior on empty graph."""
        G = nx.Graph()

        hand_impl = GirvanNewmanCommunityDetection(G)
        library = GirvanNewmanLibrary(G)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities(num_communities=1)

        assert hand_communities == []
        assert lib_communities == []

    def test_single_node(self):
        """Test behavior on single node graph."""
        G = nx.Graph()
        G.add_node(0)

        hand_impl = GirvanNewmanCommunityDetection(G, num_communities=1)
        library = GirvanNewmanLibrary(G)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities(num_communities=1)

        assert len(hand_communities) == 1
        assert len(lib_communities) == 1
        assert 0 in hand_communities[0]
        assert 0 in lib_communities[0]

    def test_disconnected_components(self, disconnected_graph):
        """Test on graph with disconnected components."""
        # Get initial number of components
        num_components = nx.number_connected_components(disconnected_graph)

        hand_impl = GirvanNewmanCommunityDetection(
            disconnected_graph, num_communities=num_components
        )
        library = GirvanNewmanLibrary(disconnected_graph)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities(num_communities=num_components)

        assert verify_valid_partition(disconnected_graph, hand_communities)
        assert verify_valid_partition(disconnected_graph, lib_communities)

        # Should identify disconnected components
        assert len(hand_communities) >= num_components
        assert len(lib_communities) >= num_components


class TestGirvanNewmanModularity:
    """Test modularity-based stopping criterion."""

    def test_modularity_optimization_simple(self, simple_graph):
        """Test modularity optimization on simple graph."""
        hand_impl = GirvanNewmanCommunityDetection(
            simple_graph, use_modularity=True, num_communities=None
        )
        library = GirvanNewmanLibrary(simple_graph)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities(num_communities=None)

        # Verify both are valid partitions
        assert verify_valid_partition(simple_graph, hand_communities)
        assert verify_valid_partition(simple_graph, lib_communities)

        # Calculate modularity for both
        hand_modularity = calculate_modularity(simple_graph, hand_communities)
        lib_modularity = calculate_modularity(simple_graph, lib_communities)

        # Both should find high modularity partitions
        assert hand_modularity > 0.0
        assert lib_modularity > 0.0

        # Should be very close (within 10% due to potential stopping differences)
        assert abs(hand_modularity - lib_modularity) < 0.1 * max(
            hand_modularity, lib_modularity
        )

    def test_modularity_optimization_barbell(self, barbell_graph):
        """Test modularity optimization on barbell graph."""
        hand_impl = GirvanNewmanCommunityDetection(
            barbell_graph, use_modularity=True, num_communities=None
        )
        library = GirvanNewmanLibrary(barbell_graph)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities(num_communities=None)

        assert verify_valid_partition(barbell_graph, hand_communities)
        assert verify_valid_partition(barbell_graph, lib_communities)

        # Calculate modularity for both
        hand_modularity = calculate_modularity(barbell_graph, hand_communities)
        lib_modularity = calculate_modularity(barbell_graph, lib_communities)

        # Both should find high modularity partitions
        assert hand_modularity > 0.3  # Barbell should have clear structure
        assert lib_modularity > 0.3

    def test_modularity_history_tracked(self, simple_graph):
        """Test that modularity history is tracked in hand implementation."""
        hand_impl = GirvanNewmanCommunityDetection(
            simple_graph, use_modularity=True, num_communities=None
        )

        hand_communities = hand_impl.detect_communities()

        # Should have modularity history
        history = hand_impl.get_modularity_history()
        assert len(history) > 0

        # Modularity should increase then decrease (showing peak detection)
        # or stay relatively stable
        assert all(isinstance(m, float) for m in history)


class TestGirvanNewmanTargetCommunities:
    """Test with specific target number of communities."""

    def test_various_target_counts(self, karate_graph):
        """Test with various target community counts."""
        for target in [2, 3, 4, 5]:
            hand_impl = GirvanNewmanCommunityDetection(
                karate_graph, num_communities=target
            )
            library = GirvanNewmanLibrary(karate_graph)

            hand_communities = hand_impl.detect_communities()
            lib_communities = library.detect_communities(num_communities=target)

            # Verify both are valid partitions
            assert verify_valid_partition(karate_graph, hand_communities)
            assert verify_valid_partition(karate_graph, lib_communities)

            # Should have target number of communities (or close)
            assert (
                len(hand_communities) == target or len(hand_communities) == target - 1
            )
            assert len(lib_communities) == target

    def test_target_exceeds_nodes(self, triangle_graph):
        """Test when target communities exceeds number of nodes."""
        target = 5  # More than 3 nodes

        hand_impl = GirvanNewmanCommunityDetection(
            triangle_graph, num_communities=target
        )
        library = GirvanNewmanLibrary(triangle_graph)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities(num_communities=target)

        # Both should handle this gracefully
        assert verify_valid_partition(triangle_graph, hand_communities)
        assert verify_valid_partition(triangle_graph, lib_communities)

        # Can't have more communities than nodes
        assert len(hand_communities) <= 3
        assert len(lib_communities) <= 3

    def test_target_one_community(self, simple_graph):
        """Test with target of 1 community."""
        target = 1

        hand_impl = GirvanNewmanCommunityDetection(simple_graph, num_communities=target)
        library = GirvanNewmanLibrary(simple_graph)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities(num_communities=target)

        assert len(hand_communities) == 1
        assert len(lib_communities) == 1

        # Should contain all nodes
        assert hand_communities[0] == set(simple_graph.nodes())
        assert lib_communities[0] == set(simple_graph.nodes())


class TestGirvanNewmanHierarchy:
    """Test hierarchical dendrogram functionality."""

    def test_dendrogram_structure(self, simple_graph):
        """Test that dendrogram is properly structured."""
        hand_impl = GirvanNewmanCommunityDetection(simple_graph, num_communities=2)

        hand_communities = hand_impl.detect_communities()
        dendrogram = hand_impl.get_dendrogram()

        # Should have multiple levels
        assert len(dendrogram) > 0

        # First level should be entire graph as one community
        assert len(dendrogram[0]) == 1 or len(
            dendrogram[0]
        ) == nx.number_connected_components(simple_graph)

        # Last level should match final communities
        # (or close to it if we stopped early)
        assert len(dendrogram[-1]) >= len(hand_communities)

        # Each level should be a valid partition
        for partition in dendrogram:
            assert verify_valid_partition(simple_graph, partition)

    def test_dendrogram_progression(self, barbell_graph):
        """Test that dendrogram shows progressive community splitting."""
        hand_impl = GirvanNewmanCommunityDetection(barbell_graph, num_communities=2)

        _ = hand_impl.detect_communities()
        dendrogram = hand_impl.get_dendrogram()

        # Number of communities should generally increase through levels
        community_counts = [len(partition) for partition in dendrogram]

        # Should be monotonically increasing (or mostly increasing)
        for i in range(len(community_counts) - 1):
            assert community_counts[i] <= community_counts[i + 1]

    def test_library_dendrogram_comparison(self, simple_graph):
        """Compare dendrogram structure between implementations."""
        max_levels = 5

        hand_impl = GirvanNewmanCommunityDetection(simple_graph)
        library = GirvanNewmanLibrary(simple_graph)

        # Run detection to populate dendrogram
        hand_impl.detect_communities()
        hand_dendrogram = hand_impl.get_dendrogram()

        lib_dendrogram = library.get_dendrogram(max_levels=max_levels)

        # Should have similar number of levels (at least in the first few)
        min_levels = min(len(hand_dendrogram), len(lib_dendrogram), max_levels)

        # Compare first few levels
        for i in range(min(3, min_levels)):
            hand_partition = hand_dendrogram[i]
            lib_partition = lib_dendrogram[i]

            # Should have same number of communities at each level
            assert len(hand_partition) == len(lib_partition)

            # Should be equivalent partitions
            assert communities_are_equivalent(hand_partition, lib_partition)


class TestGirvanNewmanEdgeCases:
    """Test edge cases and special graph structures."""

    def test_complete_graph(self):
        """Test on complete graph (no clear community structure)."""
        G = nx.complete_graph(6)

        hand_impl = GirvanNewmanCommunityDetection(G, num_communities=2)
        library = GirvanNewmanLibrary(G)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities(num_communities=2)

        assert verify_valid_partition(G, hand_communities)
        assert verify_valid_partition(G, lib_communities)

        # Should split somehow
        assert len(hand_communities) == 2
        assert len(lib_communities) == 2

    def test_star_graph(self):
        """Test on star graph."""
        G = nx.star_graph(5)

        hand_impl = GirvanNewmanCommunityDetection(G, num_communities=2)
        library = GirvanNewmanLibrary(G)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities(num_communities=2)

        assert verify_valid_partition(G, hand_communities)
        assert verify_valid_partition(G, lib_communities)

        assert len(hand_communities) == 2
        assert len(lib_communities) == 2

    def test_path_graph(self, path_graph):
        """Test on path graph."""
        hand_impl = GirvanNewmanCommunityDetection(path_graph, num_communities=3)
        library = GirvanNewmanLibrary(path_graph)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities(num_communities=3)

        assert verify_valid_partition(path_graph, hand_communities)
        assert verify_valid_partition(path_graph, lib_communities)

        assert len(hand_communities) == 3
        assert len(lib_communities) == 3

    def test_cycle_graph(self):
        """Test on cycle graph."""
        G = nx.cycle_graph(8)

        hand_impl = GirvanNewmanCommunityDetection(G, num_communities=2)
        library = GirvanNewmanLibrary(G)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities(num_communities=2)

        assert verify_valid_partition(G, hand_communities)
        assert verify_valid_partition(G, lib_communities)

        assert len(hand_communities) == 2
        assert len(lib_communities) == 2


class TestGirvanNewmanNodeMapping:
    """Test node-to-community mapping functionality."""

    def test_node_to_community_mapping(self, simple_graph):
        """Test that node-to-community mapping is correct."""
        hand_impl = GirvanNewmanCommunityDetection(simple_graph, num_communities=2)

        hand_communities = hand_impl.detect_communities()
        node_mapping = hand_impl.get_node_to_community_mapping()

        # Every node should be in the mapping
        assert len(node_mapping) == simple_graph.number_of_nodes()

        # Verify mapping is consistent with communities
        for comm_id, community in enumerate(hand_communities):
            for node in community:
                assert node_mapping[node] == comm_id

    def test_mapping_before_detection(self, simple_graph):
        """Test mapping before detection returns empty dict."""
        hand_impl = GirvanNewmanCommunityDetection(simple_graph)

        node_mapping = hand_impl.get_node_to_community_mapping()

        assert node_mapping == {}


class TestGirvanNewmanConsistency:
    """Test consistency and determinism of the algorithm."""

    def test_deterministic_behavior(self, simple_graph):
        """Test that algorithm produces consistent results on same input."""
        results = []

        for _ in range(3):
            hand_impl = GirvanNewmanCommunityDetection(
                simple_graph.copy(), num_communities=2
            )
            communities = hand_impl.detect_communities()
            results.append(normalize_communities(communities))

        # All results should be identical
        assert all(r == results[0] for r in results)

    def test_graph_not_modified(self, simple_graph):
        """Test that original graph is not modified."""
        original_edges = set(simple_graph.edges())
        original_nodes = set(simple_graph.nodes())

        hand_impl = GirvanNewmanCommunityDetection(simple_graph, num_communities=2)
        hand_impl.detect_communities()

        # Graph should be unchanged
        assert set(simple_graph.edges()) == original_edges
        assert set(simple_graph.nodes()) == original_nodes


class TestGirvanNewmanPerformance:
    """Test algorithm on various graph sizes."""

    def test_small_dense_graph(self):
        """Test on small dense graph."""
        G = nx.dense_gnm_random_graph(20, 100, seed=42)

        hand_impl = GirvanNewmanCommunityDetection(G, num_communities=3)
        library = GirvanNewmanLibrary(G)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities(num_communities=3)

        assert verify_valid_partition(G, hand_communities)
        assert verify_valid_partition(G, lib_communities)

        # Should find requested number of communities
        assert len(hand_communities) == 3
        assert len(lib_communities) == 3

    def test_karate_club_detailed(self, karate_graph):
        """Detailed test on Karate Club graph."""
        target = 2

        hand_impl = GirvanNewmanCommunityDetection(karate_graph, num_communities=target)
        library = GirvanNewmanLibrary(karate_graph)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities(num_communities=target)

        # Verify both are valid
        assert verify_valid_partition(karate_graph, hand_communities)
        assert verify_valid_partition(karate_graph, lib_communities)

        # Should produce equivalent partitions
        assert communities_are_equivalent(hand_communities, lib_communities)

        # Check modularity
        hand_modularity = calculate_modularity(karate_graph, hand_communities)
        lib_modularity = calculate_modularity(karate_graph, lib_communities)

        # Both should have good modularity
        assert hand_modularity > 0.3
        assert lib_modularity > 0.3

        # Should be very close
        assert abs(hand_modularity - lib_modularity) < 0.05
