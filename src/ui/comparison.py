"""Algorithm comparison and benchmarking utilities."""

import streamlit as st

from src.algo import compare_algorithms


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


def display_algorithm_comparison(graph):
    """
    Display comparison of different community detection algorithms.

    Args:
        graph: NetworkX graph
    """
    st.markdown("### Algorithm Comparison")

    st.info(
        "This comparison runs all algorithms independently and is cached per graph upload."
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
                f"Best modularity: **{best_modularity[0]}** ({best_modularity[1]['modularity']:.4f})"
            )

        with col2:
            fastest = min(
                valid_results.items(),
                key=lambda x: x[1].get("runtime_seconds", float("inf")),
            )
            runtime_ms = fastest[1]["runtime_seconds"] * 1000
            st.success(f"Fastest: **{fastest[0]}** ({runtime_ms:.2f} ms)")

        # Additional insights
        st.markdown("#### Performance Insights")

        # Calculate speed vs quality tradeoff
        for algo_name, result in sorted(
            valid_results.items(), key=lambda x: x[1]["runtime_seconds"]
        ):
            runtime_ms = result["runtime_seconds"] * 1000
            modularity = result["modularity"]

            # Create performance indicators
            speed_rating = (
                "Fast" if runtime_ms < 10 else "Medium" if runtime_ms < 50 else "Slow"
            )
            quality_rating = (
                "High" if modularity > 0.4 else "Medium" if modularity > 0.3 else "Low"
            )

            st.markdown(f"""
            **{algo_name}**: Speed: {speed_rating} | Quality: {quality_rating}
            - Runtime: {runtime_ms:.2f} ms | Modularity: {modularity:.4f} | Communities: {result["num_communities"]}
            """)
    else:
        st.error("No valid results to compare.")
