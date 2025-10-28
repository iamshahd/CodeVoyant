import atexit
import json
import os

import streamlit as st
from pyvis.network import Network  # type: ignore[import-untyped]

# files to clean up on exit
_TEMP_FILES = {"temp.json", "graph.html"}


def _cleanup_temp_files():
    for path in list(_TEMP_FILES):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


atexit.register(_cleanup_temp_files)
st.set_page_config(layout="wide", page_title="CodeVoyant")


def main():
    st.title("Graph Visualization with Pyvis")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a graph file (JSON format)", type=["json"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = os.path.join("temp.json")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Parse the uploaded JSON file in Node-Link format
        with open(temp_file_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

        # Add nodes and edges to the graph
        net = Network(height="750px", width="100%", notebook=False)
        for node in graph_data.get("nodes", []):
            net.add_node(node["id"], label=node.get("label", str(node["id"])))
        for link in graph_data.get("links", []):
            net.add_edge(link["source"], link["target"], title=link.get("title", ""))

        # Generate and display the graph using a custom HTML template
        net.write_html("graph.html")
        with open("graph.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=750, scrolling=True)

        # Remove the temporary file (graph.html will also be removed on exit)
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except Exception:
            pass
    else:
        st.info("Please upload a graph file to visualize.")


if __name__ == "__main__":
    main()
