"""Visualization tab rendering utilities."""

import pandas as pd  # ignore [import-untyped]
import streamlit as st

from .comparison import display_algorithm_comparison


def render_visualization_tab(
    graph_data,
    graph,
    communities,
    node_colors,
    enable_communities,
    algorithm,
    show_labels,
    analyzer,
):
    """
    Render the visualization tab.

    Args:
        graph_data: Graph data in node-link format
        graph: NetworkX graph
        communities: List of communities
        node_colors: Dictionary mapping node IDs to color strings
        enable_communities: Whether community detection is enabled
        algorithm: Algorithm name
        show_labels: Whether to show community labels
        analyzer: CommunityAnalyzer instance
    """
    from .graph_loader import create_pyvis_network_with_communities

    st.markdown("### Graph Visualization")

    if enable_communities and communities:
        st.info(f"Detected {len(communities)} communities using {algorithm}")

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
        if st.button("Download Graph HTML"):
            with open("graph.html", "r", encoding="utf-8") as f:
                st.download_button(
                    "Download",
                    f.read(),
                    file_name="graph_visualization.html",
                    mime="text/html",
                )

    with col2:
        if enable_communities and communities and analyzer:
            if st.button("Download Community Report"):
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
                    report.append(f"- Conductance: {metric['conductance']:.4f}\n")

                st.download_button(
                    "Download",
                    "".join(report),
                    file_name="community_report.md",
                    mime="text/markdown",
                )


def render_analysis_tab(enable_communities, communities, analyzer):
    """
    Render the analysis tab.

    Args:
        enable_communities: Whether community detection is enabled
        communities: List of communities
        analyzer: CommunityAnalyzer instance
    """
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

        size_df = pd.DataFrame({"Community ID": range(len(sizes)), "Size": sizes})
        st.bar_chart(size_df.set_index("Community ID"))

        # Detailed metrics table
        st.markdown("#### Detailed Metrics")
        metrics_df = pd.DataFrame(metrics)
        st.dataframe(metrics_df.sort_values("size", ascending=False), width="stretch")

        # Top communities
        st.markdown("#### Largest Communities")
        top_communities = sorted(metrics, key=lambda x: x["size"], reverse=True)[:5]
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


def render_comparison_tab(graph, num_nodes):
    """
    Render the comparison tab.

    Args:
        graph: NetworkX graph
        num_nodes: Number of nodes in the graph
    """
    if num_nodes < 1000:  # Only for reasonably sized graphs
        display_algorithm_comparison(graph)
    else:
        st.warning(
            "Algorithm comparison is disabled for large graphs (>1000 nodes) to avoid performance issues."
        )
