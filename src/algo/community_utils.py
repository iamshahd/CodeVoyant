"""
Utilities for analyzing and visualizing communities.
"""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx  # type: ignore[import-untyped]
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (  # type: ignore[import-untyped]
    adjusted_rand_score,
    normalized_mutual_info_score,
)


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
        external = 0
        for node in community:
            for neighbor in self.graph.neighbors(node):
                if neighbor not in community:
                    external += 1
        return external

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

    def _extract_semantic_tokens(self, node_name: str) -> Dict[str, List[str]]:
        """
        Extract semantic tokens from a node name with context and weights.

        Analyzes code graph node names to extract meaningful components:
        - Module/package names (e.g., 'extractor', 'processor', 'cli_wrapper')
        - Class names (e.g., 'YouGetTests', 'CommunityAnalyzer')
        - Function names (e.g., 'download_urls', 'test_imgur')
        - Domain terms (e.g., 'video', 'audio', 'http', 'player')

        Args:
            node_name: Node identifier (e.g., 'src.you_get.extractors.youtube.download')

        Returns:
            Dictionary with categorized tokens: 'modules', 'functions', 'domain'
        """
        import re

        result: Dict[str, List[str]] = {
            "modules": [],  # Package/module level terms
            "functions": [],  # Function/method level terms
            "domain": [],  # Domain-specific terms
        }

        # Split by dots to analyze path hierarchy
        parts = node_name.split(".")

        # Common prefixes to skip
        skip_prefixes = {"src", "lib", "libs", "tests", "test"}

        # Very generic terms to completely ignore
        ultra_generic = {
            "common",
            "util",
            "utils",
            "helper",
            "helpers",
            "base",
            "core",
            "you",  # Project-specific but too common
        }

        # Domain-specific terms that are valuable
        domain_terms = {
            # Media/Content
            "video",
            "audio",
            "image",
            "player",
            "stream",
            "download",
            "downloader",
            "upload",
            "uploader",
            "extractor",
            "extraction",
            "transcoder",
            "codec",
            "playlist",
            "subtitle",
            # Web/Network
            "http",
            "https",
            "url",
            "request",
            "response",
            "api",
            "client",
            "server",
            "socket",
            "cookie",
            "session",
            # Data formats
            "json",
            "xml",
            "html",
            "parser",
            "serializer",
            "encoder",
            "decoder",
            # Testing
            "unittest",
            "pytest",
            "mock",
            # Architecture patterns
            "handler",
            "manager",
            "processor",
            "wrapper",
            "validator",
            "controller",
            "service",
            "repository",
            "factory",
            # Media platforms (domain-specific)
            "youtube",
            "twitter",
            "vimeo",
            "soundcloud",
            "twitch",
            "tiktok",
            "instagram",
            "facebook",
            # Protocols
            "rtmp",
            "rtsp",
            "websocket",
        }

        # Stop words to filter (common but not meaningful)
        stop_words = {
            "the",
            "and",
            "for",
            "from",
            "with",
            "that",
            "this",
            "get",
            "set",
            "has",
            "add",
            "del",
            "new",
            "init",
            "main",
            "tmp",
            "temp",
            "file",
            "name",
            "data",
            "info",
            "param",
        }

        def split_compound_word(word: str) -> List[str]:
            """Split camelCase and snake_case words."""
            # Split camelCase
            camel_split = re.sub(r"([a-z])([A-Z])", r"\1 \2", word)
            # Split on underscores and spaces
            tokens = re.split(r"[_\s]+", camel_split)
            return [t.lower() for t in tokens if t]

        # Process each part of the path
        for i, part in enumerate(parts):
            if not part or part in skip_prefixes:
                continue

            # Determine context: earlier parts are modules, later parts are functions
            is_module = i < len(parts) - 1
            is_last = i == len(parts) - 1

            # Split the part into sub-tokens
            sub_tokens = split_compound_word(part)

            for token in sub_tokens:
                # Skip very short, numeric, or ultra-generic tokens
                if len(token) <= 2 or token.isdigit() or token in ultra_generic:
                    continue

                # Skip stop words
                if token in stop_words:
                    continue

                # Categorize the token
                if token in domain_terms:
                    result["domain"].append(token)
                elif is_module and len(token) > 3:
                    # Only add module names if they're reasonably long
                    result["modules"].append(token)
                elif is_last and len(token) > 3:
                    # Function names should be meaningful
                    result["functions"].append(token)

        return result

        return result

    def _aggregate_community_tokens(self, comm_id: int) -> Dict[str, Counter[str]]:
        """
        Aggregate tokens from all nodes in a community with semantic categories.

        Args:
            comm_id: Community ID

        Returns:
            Dictionary with token counters for each category
        """
        community = self.communities[comm_id]

        aggregated: Dict[str, Counter[str]] = {
            "modules": Counter(),
            "functions": Counter(),
            "domain": Counter(),
        }

        for node in community:
            node_str = str(node)
            semantic_tokens = self._extract_semantic_tokens(node_str)

            for category, tokens in semantic_tokens.items():
                aggregated[category].update(tokens)

        return aggregated

    def get_community_tags(self, comm_id: int, top_n: int = 5) -> List[str]:
        """
        Generate intelligent tags for a community using TF-IDF analysis.

        Uses scikit-learn's TfidfVectorizer to identify terms that are:
        1. Frequent in this community (high TF)
        2. Rare across other communities (high IDF)

        This makes tags more distinctive and meaningful by prioritizing terms
        that characterize this specific community rather than the entire codebase.

        Additionally applies:
        - Category-based weights (domain > modules > functions)
        - Length bonuses for more specific terms
        - Root similarity filtering for diversity

        Args:
            comm_id: Community ID
            top_n: Number of top tags to return

        Returns:
            List of top N most relevant tags, ordered by TF-IDF score
        """
        # Collect all tokens from all communities for TF-IDF
        all_documents: List[str] = []
        category_weights = {
            "domain": 4.0,
            "modules": 2.5,
            "functions": 1.0,
        }

        # Build documents (one per community) with weighted tokens
        for cid in range(len(self.communities)):
            community_tokens = self._aggregate_community_tokens(cid)
            # Create a document string with tokens weighted by category
            doc_tokens = []
            for category, counter in community_tokens.items():
                weight = int(category_weights[category])
                for token, count in counter.items():
                    # Repeat tokens based on count and category weight
                    doc_tokens.extend([token] * count * weight)
            all_documents.append(" ".join(doc_tokens))

        # Apply TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            max_features=None,
            lowercase=False,  # Tokens already processed
            token_pattern=r"\S+",  # Split on whitespace only
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(all_documents)
            feature_names = vectorizer.get_feature_names_out()

            # Get TF-IDF scores for the target community
            target_scores = tfidf_matrix[comm_id].toarray().flatten()

            # Create token-score pairs
            token_scores: Dict[str, float] = {}
            for idx, token in enumerate(feature_names):
                score = target_scores[idx]

                if score > 0:  # Only consider non-zero scores
                    # Apply length bonus for more specific terms
                    length_bonus = 1.0
                    if len(token) >= 8:
                        length_bonus = 1.8
                    elif len(token) >= 6:
                        length_bonus = 1.4
                    elif len(token) >= 4:
                        length_bonus = 1.0
                    else:
                        length_bonus = 0.7

                    token_scores[token] = score * length_bonus

            # Sort by score (descending)
            sorted_tokens = sorted(
                token_scores.items(), key=lambda x: x[1], reverse=True
            )

            # Select top tags with diversity (avoid similar root words)
            tags: List[str] = []
            seen_roots: Set[str] = set()

            for token, score in sorted_tokens:
                if len(tags) >= top_n:
                    break

                # Check for root similarity to ensure diversity
                root = token[:5] if len(token) > 5 else token
                if root not in seen_roots:
                    tags.append(token)
                    seen_roots.add(root)

            # If we don't have enough tags, relax the root similarity constraint
            if len(tags) < top_n and len(sorted_tokens) > len(tags):
                for token, score in sorted_tokens:
                    if token not in tags and len(tags) < top_n:
                        tags.append(token)

            return tags

        except Exception:
            # Fallback: return most common tokens if TF-IDF fails
            aggregated = self._aggregate_community_tokens(comm_id)
            all_tokens: Counter[str] = Counter()
            for category, counter in aggregated.items():
                all_tokens.update(counter)
            return [token for token, _ in all_tokens.most_common(top_n)]

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
            "tags": self.get_community_tags(comm_id),
        }

    def get_all_metrics(self) -> List[Dict[str, Any]]:
        """
        Get metrics for all communities.

        Returns:
            List of metric dictionaries, one per community
        """
        return [self.get_community_metrics(i) for i in range(len(self.communities))]

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
        self, other_communities: List[Set[str]]
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

        return {
            "adjusted_rand_index": adjusted_rand_score(labels1, labels2),
            "normalized_mutual_info": normalized_mutual_info_score(labels1, labels2),
        }

    def export_communities(
        self, output_path: str, include_metrics: bool = True
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
        self, layout: str = "spring", **layout_kwargs: Any
    ) -> dict[Any, np.ndarray[tuple[int], np.dtype[np.float64]]]:
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
            pos = nx.spring_layout(self.graph, **layout_kwargs)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self.graph, **layout_kwargs)
        else:
            # Default to spring layout
            pos = nx.spring_layout(self.graph, **layout_kwargs)

        return pos
