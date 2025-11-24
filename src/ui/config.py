"""Sidebar configuration and parameter handling."""

import streamlit as st

from src.algo import CommunityDetectionFactory


def configure_sidebar(graph):
    """
    Configure sidebar with all community detection options.

    Args:
        graph: NetworkX graph

    Returns:
        Dictionary containing all configuration parameters
    """
    st.sidebar.title("Configuration")

    # Community detection options
    st.sidebar.markdown("---")
    enable_communities = st.sidebar.checkbox(
        "Enable Community Detection",
        value=True,
        help="Detect and visualize communities in the graph",
    )

    config = {
        "enable_communities": enable_communities,
        "algorithm": None,
        "params": {},
        "show_labels": True,
        "color_scheme": "tab20",
    }

    if enable_communities:
        st.sidebar.markdown("### Community Detection")

        # Algorithm selection
        available_algos = CommunityDetectionFactory.available_algorithms()
        algorithm = st.sidebar.selectbox(
            "Algorithm",
            available_algos,
            index=0,  # Default to Louvain
            help="Choose the community detection algorithm",
        )
        config["algorithm"] = algorithm

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
                help="Target number of communities",
            )
            # Store for later use
            params["_target_communities"] = num_communities

        config["params"] = params

        # Visualization options
        show_labels = st.sidebar.checkbox(
            "Show community labels",
            value=True,
            help="Show community ID in node tooltips",
        )
        config["show_labels"] = show_labels

        color_scheme = st.sidebar.selectbox(
            "Color scheme",
            ["tab20", "tab10", "Set3", "Pastel1", "Set1"],
            help="Color scheme for communities",
        )
        config["color_scheme"] = color_scheme

    return config
