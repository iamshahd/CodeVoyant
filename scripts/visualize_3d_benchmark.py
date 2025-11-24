#!/usr/bin/env python3
"""
3D Visualization of Benchmark Results
Shows the relationship between number of nodes, number of edges, and execution time.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

# Load benchmark results
with open("codevoyant_output/benchmark_results.json", "r") as f:
    data = json.load(f)

# Extract data
results = data["benchmark_results"]
nodes = [r["num_nodes"] for r in results]
edges = [r["num_edges"] for r in results]
exec_time = [r["execution_time"] for r in results]
algorithms = [r["algorithm"] for r in results]
graph_types = [r["graph_type"] for r in results]

# Create figure
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")

# Color by algorithm
colors = []
for algo in algorithms:
    if algo == "louvain":
        colors.append("blue")
    elif algo == "girvan_newman":
        colors.append("red")
    else:
        colors.append("green")

# Create scatter plot
scatter = ax.scatter(nodes, edges, exec_time, c=colors, marker="o", alpha=0.6, s=50)

# Labels and title
ax.set_xlabel("Number of Nodes", fontsize=12, labelpad=10)
ax.set_ylabel("Number of Edges", fontsize=12, labelpad=10)
ax.set_zlabel("Execution Time (seconds)", fontsize=12, labelpad=10)
ax.set_title("3D Visualization: Nodes vs Edges vs Execution Time", fontsize=14, pad=20)

# Add legend
legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="blue",
        markersize=10,
        label="Louvain",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="red",
        markersize=10,
        label="Girvan-Newman",
    ),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

# Adjust viewing angle for better visualization
ax.view_init(elev=20, azim=45)

# Add grid
ax.grid(True, alpha=0.3)


# Create rotation animation
def rotate(angle):
    ax.view_init(elev=20, azim=angle)
    return ax


print("Creating rotating animation for 3D visualization...")
angles = np.linspace(0, 360, 360)  # 360 frames for smooth, slower rotation
anim = FuncAnimation(fig, rotate, frames=angles, interval=50)

# Save as GIF
writer = PillowWriter(fps=25)
anim.save("codevoyant_output/3d_benchmark_visualization.gif", writer=writer, dpi=150)
print(
    "Rotating 3D visualization saved to: codevoyant_output/3d_benchmark_visualization.gif"
)

# Also save a static image
plt.savefig(
    "codevoyant_output/3d_benchmark_visualization.png", dpi=300, bbox_inches="tight"
)

# Create a second view with log scale for execution time
fig2 = plt.figure(figsize=(14, 10))
ax2 = fig2.add_subplot(111, projection="3d")

# Use log scale for execution time to better see the distribution
exec_time_log = [np.log10(t) if t > 0 else -10 for t in exec_time]

scatter2 = ax2.scatter(
    nodes, edges, exec_time_log, c=colors, marker="o", alpha=0.6, s=50
)

ax2.set_xlabel("Number of Nodes", fontsize=12, labelpad=10)
ax2.set_ylabel("Number of Edges", fontsize=12, labelpad=10)
ax2.set_zlabel("Log₁₀(Execution Time) (seconds)", fontsize=12, labelpad=10)
ax2.set_title(
    "3D Visualization (Log Scale): Nodes vs Edges vs Execution Time",
    fontsize=14,
    pad=20,
)

ax2.legend(handles=legend_elements, loc="upper left", fontsize=10)
ax2.view_init(elev=20, azim=45)
ax2.grid(True, alpha=0.3)


# Create rotation animation for log scale version
def rotate2(angle):
    ax2.view_init(elev=20, azim=angle)
    return ax2


print("Creating rotating animation for log-scale 3D visualization...")
anim2 = FuncAnimation(fig2, rotate2, frames=angles, interval=50)

# Save as GIF
anim2.save(
    "codevoyant_output/3d_benchmark_visualization_log.gif", writer=writer, dpi=150
)
print(
    "Rotating 3D visualization (log scale) saved to: codevoyant_output/3d_benchmark_visualization_log.gif"
)

# Also save a static image
plt.savefig(
    "codevoyant_output/3d_benchmark_visualization_log.png", dpi=300, bbox_inches="tight"
)

# Print statistics
print("\n=== Statistics ===")
print(f"Total data points: {len(results)}")
print(f"Nodes range: {min(nodes)} - {max(nodes)}")
print(f"Edges range: {min(edges)} - {max(edges)}")
print(f"Execution time range: {min(exec_time):.6f} - {max(exec_time):.2f} seconds")
print(f"\nLouvain algorithm count: {algorithms.count('louvain')}")
print(f"Girvan-Newman algorithm count: {algorithms.count('girvan_newman')}")
print("\n✓ GIF animations created successfully!")
