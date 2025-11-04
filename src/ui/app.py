import atexit
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import networkx as nx
import pandas as pd
import streamlit as st
from pyvis.network import Network  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.algo import (
    CommunityDetectionFactory,
    analyze_communities,
    compare_algorithms,
    detect_communities,
    visualize_communities,
)

# Files to clean up on exit
_TEMP_FILES = {"temp.json", "graph.html"}


def _cleanup_temp_files():
    """Clean up temporary files on exit."""
    for path in list(_TEMP_FILES):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


atexit.register(_cleanup_temp_files)


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
    from networkx.readwrite import json_graph

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


def display_community_stats(analyzer, communities):
    """
    Display community statistics in the sidebar.

    Args:
        analyzer: CommunityAnalyzer instance
        communities: List of communities
    """
    st.sidebar.markdown("### 📊 Community Statistics")

    # Overall stats
    modularity = analyzer.get_modularity()
    coverage = analyzer.get_coverage()
    performance = analyzer.get_performance()

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Communities", len(communities))
        st.metric("Modularity", f"{modularity:.3f}")
    with col2:
        st.metric("Coverage", f"{coverage:.3f}")
        st.metric("Performance", f"{performance:.3f}")

    # Per-community details
    with st.sidebar.expander("📋 Community Details"):
        metrics = analyzer.get_all_metrics()
        for metric in sorted(metrics, key=lambda x: x["size"], reverse=True):
            st.markdown(f"""
            **Community {metric["community_id"]}**
            - Size: {metric["size"]} nodes
            - Density: {metric["density"]:.3f}
            - Conductance: {metric["conductance"]:.3f}
            """)


@st.cache_data
def run_algorithm_comparison(_graph):
    """
    Run algorithm comparison (cached to avoid re-running on UI changes).

    Args:
        _graph: NetworkX graph (underscore prefix to prevent hashing)

    Returns:
        Dictionary of algorithm results
    """
    return compare_algorithms(_graph)


def display_algorithm_comparison(graph, temp_file_path):
    """
    Display comparison of different community detection algorithms.

    Args:
        graph: NetworkX graph
        temp_file_path: Path to temporary file
    """
    st.markdown("### 🔬 Algorithm Comparison")

    st.info(
        "ℹ️ This comparison runs all algorithms independently and is cached per graph upload."
    )

    with st.spinner("Running algorithm comparison..."):
        results = run_algorithm_comparison(graph)

    # Create comparison table
    comparison_data = []
    for algo_name, result in results.items():
        if "error" not in result:
            runtime_ms = result["runtime_seconds"] * 1000  # Convert to milliseconds
            comparison_data.append(
                {
                    "Algorithm": algo_name,
                    "Communities": result["num_communities"],
                    "Modularity": f"{result['modularity']:.4f}",
                    "Runtime (ms)": f"{runtime_ms:.2f}",
                    "Avg Size": f"{result['avg_community_size']:.2f}",
                    "Min Size": result["min_community_size"],
                    "Max Size": result["max_community_size"],
                }
            )

    st.dataframe(comparison_data, width="stretch")

    # Find best algorithm by modularity and fastest
    valid_results = {k: v for k, v in results.items() if "error" not in v}

    if valid_results:
        col1, col2 = st.columns(2)

        with col1:
            best_modularity = max(
                valid_results.items(), key=lambda x: x[1].get("modularity", -1)
            )
            st.success(
                f"🏆 Best modularity: **{best_modularity[0]}** ({best_modularity[1]['modularity']:.4f})"
            )

        with col2:
            fastest = min(
                valid_results.items(),
                key=lambda x: x[1].get("runtime_seconds", float("inf")),
            )
            runtime_ms = fastest[1]["runtime_seconds"] * 1000
            st.success(f"⚡ Fastest: **{fastest[0]}** ({runtime_ms:.2f} ms)")

        # Additional insights
        st.markdown("#### 📊 Performance Insights")

        # Calculate speed vs quality tradeoff
        for algo_name, result in sorted(
            valid_results.items(), key=lambda x: x[1]["runtime_seconds"]
        ):
            runtime_ms = result["runtime_seconds"] * 1000
            modularity = result["modularity"]

            # Create a visual indicator
            speed_emoji = (
                "⚡⚡⚡" if runtime_ms < 10 else "⚡⚡" if runtime_ms < 50 else "⚡"
            )
            quality_emoji = (
                "⭐⭐⭐" if modularity > 0.4 else "⭐⭐" if modularity > 0.3 else "⭐"
            )

            st.markdown(f"""
            **{algo_name}**: {speed_emoji} Speed | {quality_emoji} Quality
            - Runtime: {runtime_ms:.2f} ms | Modularity: {modularity:.4f} | Communities: {result["num_communities"]}
            """)
    else:
        st.error("No valid results to compare.")


def main():
    st.set_page_config(layout="wide", page_title="CodeVoyant - Graph Visualization")

    st.title("🔍 CodeVoyant - Graph Visualization with Community Detection")

    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a graph file (JSON format)",
        type=["json"],
        help="Upload a graph in node-link JSON format",
    )

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = os.path.join("temp.json")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Parse the uploaded JSON file
        with open(temp_file_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

        # Load as NetworkX graph for analysis
        try:
            graph = load_graph_from_json(temp_file_path)
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()

            st.sidebar.markdown("### 📈 Graph Info")
            st.sidebar.metric("Nodes", num_nodes)
            st.sidebar.metric("Edges", num_edges)

        except Exception as e:
            st.error(f"Error loading graph: {e}")
            return

        # Community detection options
        st.sidebar.markdown("---")
        enable_communities = st.sidebar.checkbox(
            "🎨 Enable Community Detection",
            value=True,
            help="Detect and visualize communities in the graph",
        )

        communities = None
        node_colors = None
        analyzer = None

        if enable_communities:
            st.sidebar.markdown("### 🔧 Community Detection")

            # Algorithm selection
            available_algos = CommunityDetectionFactory.available_algorithms()
            algorithm = st.sidebar.selectbox(
                "Algorithm",
                available_algos,
                index=0,  # Default to Louvain
                help="Choose the community detection algorithm",
            )

            # Algorithm-specific parameters
            params = {}

            if algorithm == "louvain":
                resolution = st.sidebar.slider(
                    "Resolution",
                    min_value=0.1,
                    max_value=3.0,
                    value=1.0,
                    step=0.1,
                    help="Higher values lead to more communities",
                )
                params["resolution"] = resolution
                params["seed"] = 42

            elif algorithm == "girvan_newman":
                num_communities = st.sidebar.number_input(
                    "Target Communities",
                    min_value=2,
                    max_value=20,
                    value=5,
                    help="Target number of communities (leave empty for optimal)",
                )
                # Store for later use
                params["_target_communities"] = num_communities

            # Visualization options
            show_labels = st.sidebar.checkbox(
                "Show community labels",
                value=True,
                help="Show community ID in node tooltips",
            )

            color_scheme = st.sidebar.selectbox(
                "Color scheme",
                ["tab20", "tab10", "Set3", "Pastel1", "Set1"],
                help="Color scheme for communities",
            )

            # Detect communities
            with st.spinner(f"Detecting communities using {algorithm}..."):
                try:
                    if algorithm == "girvan_newman":
                        # Special handling for Girvan-Newman
                        detector = CommunityDetectionFactory.create(algorithm, graph)
                        communities = detector.detect_communities(
                            num_communities=params.get("_target_communities")
                        )
                    else:
                        # Use convenience function
                        communities = detect_communities(
                            graph, algorithm=algorithm, **params
                        )

                    # Analyze communities
                    analyzer = analyze_communities(graph, communities)

                    # Get colors
                    visualizer = visualize_communities(graph, communities)
                    node_colors = visualizer.get_node_colors(colormap=color_scheme)

                    # Display stats
                    display_community_stats(analyzer, communities)

                except Exception as e:
                    st.error(f"Error detecting communities: {e}")
                    enable_communities = False

        # Main content area with tabs
        tab1, tab2, tab3 = st.tabs(["📊 Visualization", "📈 Analysis", "🔬 Comparison"])

        with tab1:
            st.markdown("### Graph Visualization")

            if enable_communities and communities:
                st.info(f"✨ Detected {len(communities)} communities using {algorithm}")

            # Create and display the graph
            net = create_pyvis_network_with_communities(
                graph_data,
                communities=communities if enable_communities else None,
                node_colors=node_colors if enable_communities else None,
                show_community_labels=show_labels if enable_communities else False,
            )

            # Generate and display the graph
            net.write_html("graph.html")
            with open("graph.html", "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=750, scrolling=True)

            # Download options
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Download Graph HTML"):
                    with open("graph.html", "r", encoding="utf-8") as f:
                        st.download_button(
                            "Download",
                            f.read(),
                            file_name="graph_visualization.html",
                            mime="text/html",
                        )

            with col2:
                if enable_communities and communities and analyzer:
                    if st.button("📄 Download Community Report"):
                        # Create report
                        report = []
                        report.append("# Community Detection Report\n")
                        report.append(f"\nAlgorithm: {algorithm}\n")
                        report.append(f"Communities: {len(communities)}\n")
                        report.append(f"Modularity: {analyzer.get_modularity():.4f}\n")
                        report.append(f"Coverage: {analyzer.get_coverage():.4f}\n\n")

                        metrics = analyzer.get_all_metrics()
                        for metric in metrics:
                            report.append(f"\n## Community {metric['community_id']}\n")
                            report.append(f"- Size: {metric['size']}\n")
                            report.append(f"- Density: {metric['density']:.4f}\n")
                            report.append(
                                f"- Conductance: {metric['conductance']:.4f}\n"
                            )

                        st.download_button(
                            "Download",
                            "".join(report),
                            file_name="community_report.md",
                            mime="text/markdown",
                        )

        with tab2:
            if enable_communities and communities and analyzer:
                st.markdown("### Community Analysis")

                # Metrics overview
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Modularity", f"{analyzer.get_modularity():.4f}")
                with col2:
                    st.metric("Coverage", f"{analyzer.get_coverage():.4f}")
                with col3:
                    st.metric("Performance", f"{analyzer.get_performance():.4f}")
                with col4:
                    st.metric("Communities", len(communities))

                # Community size distribution
                st.markdown("#### Community Size Distribution")
                metrics = analyzer.get_all_metrics()
                sizes = [m["size"] for m in metrics]

                size_df = pd.DataFrame(
                    {"Community ID": range(len(sizes)), "Size": sizes}
                )
                st.bar_chart(size_df.set_index("Community ID"))

                # Detailed metrics table
                st.markdown("#### Detailed Metrics")
                metrics_df = pd.DataFrame(metrics)
                st.dataframe(
                    metrics_df.sort_values("size", ascending=False), width="stretch"
                )

                # Top communities
                st.markdown("#### Largest Communities")
                top_communities = sorted(
                    metrics, key=lambda x: x["size"], reverse=True
                )[:5]
                for i, metric in enumerate(top_communities, 1):
                    with st.expander(
                        f"{i}. Community {metric['community_id']} ({metric['size']} nodes)"
                    ):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Internal Edges", metric["internal_edges"])
                        with col2:
                            st.metric("Density", f"{metric['density']:.3f}")
                        with col3:
                            st.metric("Conductance", f"{metric['conductance']:.3f}")
            else:
                st.info("Enable community detection to see analysis.")

        with tab3:
            if num_nodes < 1000:  # Only for reasonably sized graphs
                display_algorithm_comparison(graph, temp_file_path)
            else:
                st.warning(
                    "Algorithm comparison is disabled for large graphs (>1000 nodes) to avoid performance issues."
                )

        # Clean up temp file
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except Exception:
            pass

    else:
        # Welcome screen
        st.info("👆 Please upload a graph file to visualize.")

        st.markdown("""
        ### 📖 How to use:
        
        1. **Upload a graph** in JSON format (node-link format)
        2. **Enable community detection** to identify modules/clusters
        3. **Choose an algorithm**:
           - **Louvain** (recommended) - Fast and accurate
           - **Label Propagation** - Very fast for large graphs
           - **Greedy Modularity** - Deterministic results
           - **Girvan-Newman** - Hierarchical for small graphs
        4. **Adjust parameters** to fine-tune the detection
        5. **Explore** the visualization and analysis tabs
        
        ### 🎨 Features:
        - Interactive graph visualization
        - Community-based node coloring
        - Detailed community metrics
        - Algorithm comparison
        - Export results
        """)


if __name__ == "__main__":
    main()
