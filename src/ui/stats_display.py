"""Statistics and metrics display utilities."""

import streamlit as st


def display_community_stats(analyzer, communities):
    """
    Display community statistics in the sidebar.

    Args:
        analyzer: CommunityAnalyzer instance
        communities: List of communities
    """
    st.sidebar.markdown("### Community Statistics")

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
    with st.sidebar.expander("Community Details"):
        metrics = analyzer.get_all_metrics()
        for metric in sorted(metrics, key=lambda x: x["size"], reverse=True):
            st.markdown(f"""
            **Community {metric["community_id"]}**
            - Size: {metric["size"]} nodes
            - Density: {metric["density"]:.3f}
            - Conductance: {metric["conductance"]:.3f}
            """)


def display_graph_info(num_nodes, num_edges):
    """
    Display basic graph information in the sidebar.

    Args:
        num_nodes: Number of nodes in the graph
        num_edges: Number of edges in the graph
    """
    st.sidebar.markdown("### Graph Information")
    st.sidebar.metric("Nodes", num_nodes)
    st.sidebar.metric("Edges", num_edges)
