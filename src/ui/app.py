"""Main application for CodeVoyant graph visualization and community detection."""

import atexit
import json
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.algo import (
    CommunityDetectionFactory,
    analyze_communities,
    detect_communities,
    visualize_communities,
)
from src.ui.config import configure_sidebar
from src.ui.graph_loader import load_graph_from_json
from src.ui.stats_display import display_community_stats, display_graph_info
from src.ui.visualization_tabs import (
    render_analysis_tab,
    render_comparison_tab,
    render_visualization_tab,
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


def process_community_detection(graph, config):
    """
    Process community detection based on configuration.

    Args:
        graph: NetworkX graph
        config: Configuration dictionary from sidebar

    Returns:
        Tuple of (communities, node_colors, analyzer)
    """
    if not config["enable_communities"]:
        return None, None, None

    algorithm = config["algorithm"]
    params = config["params"]
    color_scheme = config["color_scheme"]

    try:
        if algorithm == "girvan_newman":
            # Special handling for Girvan-Newman
            detector = CommunityDetectionFactory.create(algorithm, graph)
            communities = detector.detect_communities(
                num_communities=params.get("_target_communities")
            )
        else:
            # Use convenience function
            communities = detect_communities(graph, algorithm=algorithm, **params)

        # Analyze communities
        analyzer = analyze_communities(graph, communities)

        # Get colors
        visualizer = visualize_communities(graph, communities)
        node_colors = visualizer.get_node_colors(colormap=color_scheme)

        return communities, node_colors, analyzer

    except Exception as e:
        st.error(f"Error detecting communities: {e}")
        return None, None, None


def display_welcome_screen():
    """Display the welcome screen when no file is uploaded."""
    st.info("Please upload a graph file to visualize.")

    st.markdown("""
    ### How to use:
    
    1. **Upload a graph** in JSON format (node-link format)
    2. **Enable community detection** to identify modules/clusters
    3. **Choose an algorithm**:
       - **Louvain** (recommended) - Fast and accurate
       - **Label Propagation** - Very fast for large graphs
       - **Greedy Modularity** - Deterministic results
       - **Girvan-Newman** - Hierarchical for small graphs
    4. **Adjust parameters** to fine-tune the detection
    5. **Explore** the visualization and analysis tabs
    
    ### Features:
    - Interactive graph visualization
    - Community-based node coloring
    - Detailed community metrics
    - Algorithm comparison
    - Export results
    """)


def main():
    """Main application entry point."""
    st.set_page_config(layout="wide", page_title="CodeVoyant - Graph Visualization")

    st.title("CodeVoyant - Graph Visualization with Community Detection")

    # Sidebar configuration
    st.sidebar.title("Configuration")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a graph file (JSON format)",
        type=["json"],
        help="Upload a graph in node-link JSON format",
    )

    if uploaded_file is None:
        display_welcome_screen()
        return

    # Save the uploaded file temporarily
    temp_file_path = "temp.json"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Parse the uploaded JSON file
    try:
        with open(temp_file_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    # Load as NetworkX graph for analysis
    try:
        graph = load_graph_from_json(temp_file_path)
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()

        display_graph_info(num_nodes, num_edges)

    except Exception as e:
        st.error(f"Error loading graph: {e}")
        return

    # Configure sidebar and get settings
    st.sidebar.markdown("---")
    config = configure_sidebar(graph)

    # Process community detection
    communities, node_colors, analyzer = process_community_detection(graph, config)

    # Display community stats if available
    if config["enable_communities"] and communities and analyzer:
        display_community_stats(analyzer, communities)

    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["Visualization", "Analysis", "Comparison"])

    with tab1:
        render_visualization_tab(
            graph_data,
            graph,
            communities,
            node_colors,
            config["enable_communities"],
            config["algorithm"],
            config["show_labels"],
            analyzer,
        )

    with tab2:
        render_analysis_tab(config["enable_communities"], communities, analyzer)

    with tab3:
        render_comparison_tab(graph, num_nodes)

    # Clean up temp file
    try:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    except Exception:
        pass


if __name__ == "__main__":
    main()
