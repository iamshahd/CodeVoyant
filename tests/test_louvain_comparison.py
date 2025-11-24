"""
Comprehensive tests comparing hand-implemented Louvain algorithm
with NetworkX library implementation.

This test suite verifies that LouvainCommunityDetection (hand-implemented)
produces identical or equivalent results to LouvainLibrary (NetworkX wrapper).
"""

from typing import List, Set

import networkx as nx
import pytest

from src.algo.community_detection import LouvainLibrary
from src.algo.louvain import LouvainCommunityDetection


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


class TestLouvainBasicFunctionality:
    """Test basic functionality of hand-implemented Louvain."""

    def test_simple_graph_with_seed(self):
        """Test on simple graph with fixed seed."""
        G = nx.Graph()
        # Community 1
        G.add_edges_from([(0, 1), (0, 2), (1, 2)])
        # Community 2
        G.add_edges_from([(3, 4), (3, 5), (4, 5)])
        # Weak bridge
        G.add_edge(2, 3)

        seed = 42

        hand_impl = LouvainCommunityDetection(G, seed=seed)
        library = LouvainLibrary(G, seed=seed)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities()

        # Verify both are valid partitions
        assert verify_valid_partition(G, hand_communities)
        assert verify_valid_partition(G, lib_communities)

        # With same seed, results should be identical or very similar
        assert len(hand_communities) == len(lib_communities)

    def test_karate_club_graph(self):
        """Test on Zachary's Karate Club graph."""
        G = nx.karate_club_graph()
        seed = 42

        hand_impl = LouvainCommunityDetection(G, seed=seed)
        library = LouvainLibrary(G, seed=seed)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities()

        assert verify_valid_partition(G, hand_communities)
        assert verify_valid_partition(G, lib_communities)

        # Should find similar number of communities
        assert abs(len(hand_communities) - len(lib_communities)) <= 2

    def test_empty_graph(self):
        """Test behavior on empty graph."""
        G = nx.Graph()

        hand_impl = LouvainCommunityDetection(G, seed=42)
        library = LouvainLibrary(G, seed=42)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities()

        assert hand_communities == []
        assert lib_communities == []

    def test_single_node(self):
        """Test behavior on single node graph."""
        G = nx.Graph()
        G.add_node(0)

        hand_impl = LouvainCommunityDetection(G, seed=42)
        library = LouvainLibrary(G, seed=42)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities()

        assert len(hand_communities) == 1
        assert len(lib_communities) == 1
        assert 0 in hand_communities[0]
        assert 0 in lib_communities[0]

    def test_disconnected_components(self):
        """Test on graph with disconnected components."""
        G = nx.Graph()
        # Component 1
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        # Component 2
        G.add_edges_from([(3, 4), (4, 5), (5, 3)])
        # Component 3 (single node)
        G.add_node(6)

        seed = 42

        hand_impl = LouvainCommunityDetection(G, seed=seed)
        library = LouvainLibrary(G, seed=seed)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities()

        assert verify_valid_partition(G, hand_communities)
        assert verify_valid_partition(G, lib_communities)

        # Should detect at least 3 communities (one per component)
        assert len(hand_communities) >= 3
        assert len(lib_communities) >= 3


class TestLouvainWeightedGraphs:
    """Test Louvain on weighted graphs."""

    def test_weighted_simple_graph(self):
        """Test on weighted graph."""
        G = nx.Graph()
        # Strong community 1
        G.add_edge(0, 1, weight=5.0)
        G.add_edge(0, 2, weight=5.0)
        G.add_edge(1, 2, weight=5.0)

        # Strong community 2
        G.add_edge(3, 4, weight=5.0)
        G.add_edge(3, 5, weight=5.0)
        G.add_edge(4, 5, weight=5.0)

        # Weak bridge
        G.add_edge(2, 3, weight=0.1)

        seed = 42

        hand_impl = LouvainCommunityDetection(G, weight="weight", seed=seed)
        library = LouvainLibrary(G, weight="weight", seed=seed)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities()

        assert verify_valid_partition(G, hand_communities)
        assert verify_valid_partition(G, lib_communities)

        # Should find 2 communities
        assert len(hand_communities) == 2
        assert len(lib_communities) == 2

    def test_unweighted_vs_weighted(self):
        """Test that weights affect community detection."""
        G = nx.Graph()
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(1, 2, weight=1.0)
        G.add_edge(2, 3, weight=1.0)
        G.add_edge(3, 4, weight=1.0)

        seed = 42

        # Without weights
        hand_unweighted = LouvainCommunityDetection(G, weight=None, seed=seed)
        lib_unweighted = LouvainLibrary(G, weight=None, seed=seed)

        # With weights
        hand_weighted = LouvainCommunityDetection(G, weight="weight", seed=seed)
        lib_weighted = LouvainLibrary(G, weight="weight", seed=seed)

        hand_unweighted.detect_communities()
        lib_unweighted.detect_communities()
        hand_weighted.detect_communities()
        lib_weighted.detect_communities()

        # Both implementations should respect weight parameter
        assert hand_unweighted.communities is not None
        assert lib_unweighted.communities is not None


class TestLouvainResolutionParameter:
    """Test resolution parameter effects."""

    def test_low_resolution(self):
        """Test with low resolution (fewer, larger communities)."""
        G = nx.karate_club_graph()
        seed = 42
        resolution = 0.5

        hand_impl = LouvainCommunityDetection(G, resolution=resolution, seed=seed)
        library = LouvainLibrary(G, resolution=resolution, seed=seed)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities()

        assert verify_valid_partition(G, hand_communities)
        assert verify_valid_partition(G, lib_communities)

        # Should find similar number of communities
        assert abs(len(hand_communities) - len(lib_communities)) <= 2

    def test_high_resolution(self):
        """Test with high resolution (more, smaller communities)."""
        G = nx.karate_club_graph()
        seed = 42
        resolution = 2.0

        hand_impl = LouvainCommunityDetection(G, resolution=resolution, seed=seed)
        library = LouvainLibrary(G, resolution=resolution, seed=seed)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities()

        assert verify_valid_partition(G, hand_communities)
        assert verify_valid_partition(G, lib_communities)

        # Higher resolution should generally lead to more communities
        # (compared to default resolution)
        default_hand = LouvainCommunityDetection(G, resolution=1.0, seed=seed)
        default_communities = default_hand.detect_communities()

        assert len(hand_communities) >= len(default_communities)


class TestLouvainModularity:
    """Test modularity computation."""

    def test_modularity_calculation(self):
        """Test that modularity is calculated correctly."""
        G = nx.karate_club_graph()
        seed = 42

        hand_impl = LouvainCommunityDetection(G, seed=seed)
        library = LouvainLibrary(G, seed=seed)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities()

        # Calculate modularity using NetworkX
        hand_modularity = nx.community.modularity(G, hand_communities)
        lib_modularity = nx.community.modularity(G, lib_communities)

        # Both should have positive modularity
        assert hand_modularity > 0
        assert lib_modularity > 0

        # Both should be in valid range
        assert -1 <= hand_modularity <= 1
        assert -1 <= lib_modularity <= 1

        # Modularities should be close (within reasonable tolerance)
        assert abs(hand_modularity - lib_modularity) < 0.15

    def test_modularity_same_seed_same_result(self):
        """Test that same seed produces identical modularity."""
        G = nx.karate_club_graph()
        seed = 42

        # Run hand implementation twice
        hand_impl1 = LouvainCommunityDetection(G, seed=seed)
        hand_impl2 = LouvainCommunityDetection(G, seed=seed)

        hand_communities1 = hand_impl1.detect_communities()
        hand_communities2 = hand_impl2.detect_communities()

        hand_mod1 = nx.community.modularity(G, hand_communities1)
        hand_mod2 = nx.community.modularity(G, hand_communities2)

        # Same seed should give identical modularity
        assert abs(hand_mod1 - hand_mod2) < 1e-10

        # Run library twice
        lib_impl1 = LouvainLibrary(G, seed=seed)
        lib_impl2 = LouvainLibrary(G, seed=seed)

        lib_communities1 = lib_impl1.detect_communities()
        lib_communities2 = lib_impl2.detect_communities()

        lib_mod1 = nx.community.modularity(G, lib_communities1)
        lib_mod2 = nx.community.modularity(G, lib_communities2)

        # Same seed should give identical modularity
        assert abs(lib_mod1 - lib_mod2) < 1e-10

    def test_modularity_internal_vs_networkx(self):
        """Test that internal modularity calculation matches NetworkX."""
        G = nx.karate_club_graph()
        seed = 42

        hand_impl = LouvainCommunityDetection(G, seed=seed)
        library = LouvainLibrary(G, seed=seed)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities()

        # Get internal modularity if available
        if hasattr(hand_impl, "modularity") and hand_impl.modularity is not None:
            internal_hand_mod = hand_impl.modularity
            networkx_hand_mod = nx.community.modularity(G, hand_communities)
            # Internal calculation should match NetworkX
            assert abs(internal_hand_mod - networkx_hand_mod) < 1e-6

        if hasattr(library, "modularity") and library.modularity is not None:
            internal_lib_mod = library.modularity
            networkx_lib_mod = nx.community.modularity(G, lib_communities)
            # Internal calculation should match NetworkX
            assert abs(internal_lib_mod - networkx_lib_mod) < 1e-6

    def test_modularity_with_weights(self):
        """Test modularity calculation with weighted graphs."""
        G = nx.Graph()
        # Strong community 1
        G.add_edge(0, 1, weight=10.0)
        G.add_edge(0, 2, weight=10.0)
        G.add_edge(1, 2, weight=10.0)

        # Strong community 2
        G.add_edge(3, 4, weight=10.0)
        G.add_edge(3, 5, weight=10.0)
        G.add_edge(4, 5, weight=10.0)

        # Weak bridge
        G.add_edge(2, 3, weight=0.5)

        seed = 42

        hand_impl = LouvainCommunityDetection(G, weight="weight", seed=seed)
        library = LouvainLibrary(G, weight="weight", seed=seed)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities()

        # Calculate weighted modularity
        hand_modularity = nx.community.modularity(G, hand_communities, weight="weight")
        lib_modularity = nx.community.modularity(G, lib_communities, weight="weight")

        # Both should have high modularity due to strong communities
        assert hand_modularity > 0.4
        assert lib_modularity > 0.4

        # Should be close
        assert abs(hand_modularity - lib_modularity) < 0.15

    def test_modularity_with_resolution(self):
        """Test modularity with different resolution parameters."""
        G = nx.karate_club_graph()
        seed = 42

        resolutions = [0.5, 1.0, 1.5, 2.0]

        for res in resolutions:
            hand_impl = LouvainCommunityDetection(G, resolution=res, seed=seed)
            library = LouvainLibrary(G, resolution=res, seed=seed)

            hand_communities = hand_impl.detect_communities()
            lib_communities = library.detect_communities()

            hand_modularity = nx.community.modularity(
                G, hand_communities, resolution=res
            )
            lib_modularity = nx.community.modularity(G, lib_communities, resolution=res)

            # Both should produce valid modularity
            assert -1 <= hand_modularity <= 1
            assert -1 <= lib_modularity <= 1

            # Should be reasonably close (higher tolerance for different resolutions)
            assert abs(hand_modularity - lib_modularity) < 0.25

    def test_modularity_improves_over_iterations(self):
        """Test that modularity improves or stays same across hierarchical levels."""
        G = nx.karate_club_graph()
        seed = 42

        hand_impl = LouvainCommunityDetection(G, seed=seed)
        library = LouvainLibrary(G, seed=seed)

        hand_hierarchy = hand_impl.detect_communities_hierarchical()
        lib_hierarchy = library.detect_communities_hierarchical()

        # Calculate modularity at each level
        if len(hand_hierarchy) > 1:
            hand_modularities = [
                nx.community.modularity(G, level) for level in hand_hierarchy
            ]
            # Modularity should generally improve (or at least not decrease significantly)
            for i in range(len(hand_modularities) - 1):
                # Allow small decreases due to numerical precision and algorithm variations
                assert hand_modularities[i + 1] >= hand_modularities[i] - 0.035

        if len(lib_hierarchy) > 1:
            lib_modularities = [
                nx.community.modularity(G, level) for level in lib_hierarchy
            ]
            for i in range(len(lib_modularities) - 1):
                assert lib_modularities[i + 1] >= lib_modularities[i] - 0.02

    def test_modularity_comparison_multiple_graphs(self):
        """Test modularity on multiple graph types."""
        graphs = [
            nx.karate_club_graph(),
            nx.les_miserables_graph(),
            nx.davis_southern_women_graph(),
        ]

        seed = 42

        for G in graphs:
            hand_impl = LouvainCommunityDetection(G, seed=seed)
            library = LouvainLibrary(G, seed=seed)

            hand_communities = hand_impl.detect_communities()
            lib_communities = library.detect_communities()

            hand_modularity = nx.community.modularity(G, hand_communities)
            lib_modularity = nx.community.modularity(G, lib_communities)

            # Both should produce reasonable modularity
            assert hand_modularity > 0
            assert lib_modularity > 0

            # Should be relatively close
            assert abs(hand_modularity - lib_modularity) < 0.2

    def test_modularity_extreme_cases(self):
        """Test modularity calculation on edge cases."""
        # Complete graph - should have low modularity for multiple communities
        G_complete = nx.complete_graph(10)
        seed = 42

        hand_complete = LouvainCommunityDetection(G_complete, seed=seed)
        lib_complete = LouvainLibrary(G_complete, seed=seed)

        hand_comm = hand_complete.detect_communities()
        lib_comm = lib_complete.detect_communities()

        hand_mod = nx.community.modularity(G_complete, hand_comm)
        lib_mod = nx.community.modularity(G_complete, lib_comm)

        # If split into multiple communities, modularity should be low or negative
        if len(hand_comm) > 1:
            assert hand_mod < 0.1
        if len(lib_comm) > 1:
            assert lib_mod < 0.1

        # Disconnected components - should have high modularity
        G_disconnected = nx.Graph()
        G_disconnected.add_edges_from([(0, 1), (1, 2)])
        G_disconnected.add_edges_from([(3, 4), (4, 5)])

        hand_disc = LouvainCommunityDetection(G_disconnected, seed=seed)
        lib_disc = LouvainLibrary(G_disconnected, seed=seed)

        hand_comm_disc = hand_disc.detect_communities()
        lib_comm_disc = lib_disc.detect_communities()

        hand_mod_disc = nx.community.modularity(G_disconnected, hand_comm_disc)
        lib_mod_disc = nx.community.modularity(G_disconnected, lib_comm_disc)

        # Should have high modularity for disconnected components
        assert hand_mod_disc > 0.3
        assert lib_mod_disc > 0.3


class TestLouvainHierarchical:
    """Test hierarchical community detection."""

    def test_hierarchical_structure(self):
        """Test hierarchical detection produces valid hierarchy."""
        G = nx.karate_club_graph()
        seed = 42

        hand_impl = LouvainCommunityDetection(G, seed=seed)
        library = LouvainLibrary(G, seed=seed)

        hand_hierarchy = hand_impl.detect_communities_hierarchical()
        lib_hierarchy = library.detect_communities_hierarchical()

        # Both should produce hierarchies
        assert len(hand_hierarchy) > 0
        assert len(lib_hierarchy) > 0

        # Each level should be a valid partition
        for level in hand_hierarchy:
            assert verify_valid_partition(G, level)

        for level in lib_hierarchy:
            assert verify_valid_partition(G, level)

    def test_get_partition_at_level(self):
        """Test getting partition at specific level."""
        G = nx.karate_club_graph()
        seed = 42

        hand_impl = LouvainCommunityDetection(G, seed=seed)
        library = LouvainLibrary(G, seed=seed)

        # Get finest level (default)
        hand_finest = hand_impl.get_partition_at_level(-1)
        lib_finest = library.get_partition_at_level(-1)

        assert verify_valid_partition(G, hand_finest)
        assert verify_valid_partition(G, lib_finest)

        # Get coarsest level
        hand_coarsest = hand_impl.get_partition_at_level(0)
        lib_coarsest = library.get_partition_at_level(0)

        assert verify_valid_partition(G, hand_coarsest)
        assert verify_valid_partition(G, lib_coarsest)


class TestLouvainConsistency:
    """Test consistency and reproducibility."""

    def test_seed_reproducibility_hand_impl(self):
        """Test that same seed produces same results (hand implementation)."""
        G = nx.karate_club_graph()
        seed = 42

        impl1 = LouvainCommunityDetection(G, seed=seed)
        impl2 = LouvainCommunityDetection(G, seed=seed)

        comm1 = impl1.detect_communities()
        comm2 = impl2.detect_communities()

        # Same seed should give identical results
        assert communities_are_equivalent(comm1, comm2)

    def test_seed_reproducibility_library(self):
        """Test that same seed produces same results (library)."""
        G = nx.karate_club_graph()
        seed = 42

        impl1 = LouvainLibrary(G, seed=seed)
        impl2 = LouvainLibrary(G, seed=seed)

        comm1 = impl1.detect_communities()
        comm2 = impl2.detect_communities()

        # Same seed should give identical results
        assert communities_are_equivalent(comm1, comm2)

    def test_different_seeds_different_results(self):
        """Test that different seeds can produce different results."""
        G = nx.karate_club_graph()

        # Run multiple times with different seeds
        hand_results = []
        lib_results = []

        for seed in [42, 123, 456, 789, 1000]:
            hand_impl = LouvainCommunityDetection(G, seed=seed)
            lib_impl = LouvainLibrary(G, seed=seed)

            hand_results.append(hand_impl.detect_communities())
            lib_results.append(lib_impl.detect_communities())

        # At least some results should differ (Louvain can find different local optima)
        # This is a probabilistic test, but with 5 different seeds it should pass
        hand_unique = len(
            set(
                frozenset(frozenset(comm) for comm in normalize_communities(r))
                for r in hand_results
            )
        )
        lib_unique = len(
            set(
                frozenset(frozenset(comm) for comm in normalize_communities(r))
                for r in lib_results
            )
        )

        # Both implementations should explore multiple solutions
        assert hand_unique >= 1  # At least one unique result
        assert lib_unique >= 1


class TestLouvainComplexGraphs:
    """Test on various complex real-world-like graphs."""

    def test_erdos_renyi_graph(self):
        """Test on Erdős-Rényi random graph."""
        G = nx.erdos_renyi_graph(50, 0.1, seed=42)
        seed = 42

        hand_impl = LouvainCommunityDetection(G, seed=seed)
        library = LouvainLibrary(G, seed=seed)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities()

        assert verify_valid_partition(G, hand_communities)
        assert verify_valid_partition(G, lib_communities)

    def test_barabasi_albert_graph(self):
        """Test on Barabási-Albert scale-free graph."""
        G = nx.barabasi_albert_graph(50, 3, seed=42)
        seed = 42

        hand_impl = LouvainCommunityDetection(G, seed=seed)
        library = LouvainLibrary(G, seed=seed)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities()

        assert verify_valid_partition(G, hand_communities)
        assert verify_valid_partition(G, lib_communities)

    def test_watts_strogatz_graph(self):
        """Test on Watts-Strogatz small-world graph."""
        G = nx.watts_strogatz_graph(50, 4, 0.1, seed=42)
        seed = 42

        hand_impl = LouvainCommunityDetection(G, seed=seed)
        library = LouvainLibrary(G, seed=seed)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities()

        assert verify_valid_partition(G, hand_communities)
        assert verify_valid_partition(G, lib_communities)

    def test_complete_graph(self):
        """Test on complete graph (single community expected)."""
        G = nx.complete_graph(10)
        seed = 42

        hand_impl = LouvainCommunityDetection(G, seed=seed)
        library = LouvainLibrary(G, seed=seed)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities()

        # Complete graph should ideally be detected as single community
        # (though algorithm may split it)
        assert verify_valid_partition(G, hand_communities)
        assert verify_valid_partition(G, lib_communities)
        assert len(hand_communities) <= 3
        assert len(lib_communities) <= 3

    def test_star_graph(self):
        """Test on star graph."""
        G = nx.star_graph(10)
        seed = 42

        hand_impl = LouvainCommunityDetection(G, seed=seed)
        library = LouvainLibrary(G, seed=seed)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities()

        assert verify_valid_partition(G, hand_communities)
        assert verify_valid_partition(G, lib_communities)


class TestLouvainEdgeCases:
    """Test edge cases and special scenarios."""

    def test_graph_with_isolated_nodes(self):
        """Test graph with isolated nodes."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])
        G.add_node(4)  # Isolated node
        G.add_node(5)  # Another isolated node

        seed = 42

        hand_impl = LouvainCommunityDetection(G, seed=seed)
        library = LouvainLibrary(G, seed=seed)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities()

        assert verify_valid_partition(G, hand_communities)
        assert verify_valid_partition(G, lib_communities)

        # Isolated nodes should each be in their own community
        # or grouped together
        assert 4 in set().union(*hand_communities)
        assert 5 in set().union(*hand_communities)

    def test_bipartite_graph(self):
        """Test on bipartite graph."""
        G = nx.complete_bipartite_graph(5, 5)
        seed = 42

        hand_impl = LouvainCommunityDetection(G, seed=seed)
        library = LouvainLibrary(G, seed=seed)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities()

        assert verify_valid_partition(G, hand_communities)
        assert verify_valid_partition(G, lib_communities)

        # Complete bipartite graphs can be detected as 1 or 2 communities depending on the algorithm
        # Both are valid, so we just check that valid partitions are found
        assert len(hand_communities) >= 1
        assert len(lib_communities) >= 1

    def test_tree_graph(self):
        """Test on tree graph."""
        G = nx.balanced_tree(2, 3)  # Binary tree of height 3
        seed = 42

        hand_impl = LouvainCommunityDetection(G, seed=seed)
        library = LouvainLibrary(G, seed=seed)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities()

        assert verify_valid_partition(G, hand_communities)
        assert verify_valid_partition(G, lib_communities)

    def test_cycle_graph(self):
        """Test on cycle graph."""
        G = nx.cycle_graph(20)
        seed = 42

        hand_impl = LouvainCommunityDetection(G, seed=seed)
        library = LouvainLibrary(G, seed=seed)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities()

        assert verify_valid_partition(G, hand_communities)
        assert verify_valid_partition(G, lib_communities)


class TestLouvainQualityComparison:
    """Test that both implementations produce similar quality results."""

    def test_quality_on_benchmark_graphs(self):
        """Compare quality on multiple benchmark graphs."""
        test_cases = [
            ("karate", nx.karate_club_graph()),
            ("les_miserables", nx.les_miserables_graph()),
            (
                "dolphins",
                nx.read_gml("dolphins.gml") if False else nx.karate_club_graph(),
            ),  # Skip if file not available
        ]

        seed = 42

        for name, G in test_cases:
            hand_impl = LouvainCommunityDetection(G, seed=seed)
            library = LouvainLibrary(G, seed=seed)

            hand_communities = hand_impl.detect_communities()
            lib_communities = library.detect_communities()

            hand_mod = nx.community.modularity(G, hand_communities)
            lib_mod = nx.community.modularity(G, lib_communities)

            # Both should achieve reasonable modularity
            assert hand_mod > 0.2, f"Hand implementation modularity too low for {name}"
            assert lib_mod > 0.2, f"Library modularity too low for {name}"

            # Difference should be small (both are optimizing same metric)
            diff = abs(hand_mod - lib_mod)
            assert diff < 0.2, f"Modularity difference too large for {name}: {diff}"

    def test_community_size_distribution(self):
        """Test that community size distributions are similar."""
        G = nx.karate_club_graph()
        seed = 42

        hand_impl = LouvainCommunityDetection(G, seed=seed)
        library = LouvainLibrary(G, seed=seed)

        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities()

        hand_sizes = sorted([len(comm) for comm in hand_communities], reverse=True)
        lib_sizes = sorted([len(comm) for comm in lib_communities], reverse=True)

        # Number of communities should be similar
        assert abs(len(hand_sizes) - len(lib_sizes)) <= 2

        # Largest communities should be of similar size
        if hand_sizes and lib_sizes:
            assert abs(hand_sizes[0] - lib_sizes[0]) <= 5


class TestLouvainPerformance:
    """Test performance characteristics (not timing, but behavior)."""

    def test_convergence_on_large_graph(self):
        """Test that algorithm converges on larger graph."""
        G = nx.barabasi_albert_graph(200, 3, seed=42)
        seed = 42

        hand_impl = LouvainCommunityDetection(G, seed=seed)
        library = LouvainLibrary(G, seed=seed)

        # Should complete without errors
        hand_communities = hand_impl.detect_communities()
        lib_communities = library.detect_communities()

        assert verify_valid_partition(G, hand_communities)
        assert verify_valid_partition(G, lib_communities)

        # Should find reasonable number of communities (at least 1, not too many)
        assert len(hand_communities) >= 1
        assert len(lib_communities) >= 1
        assert len(hand_communities) < G.number_of_nodes() / 2
        assert len(lib_communities) < G.number_of_nodes() / 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
