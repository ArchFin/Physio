import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

# Parameters
R_new = 0.1          # Probability of adding a new node
epsilon = 1e-3       # Probability of node removal
num_steps = 10     # Total steps for the simulation

# Initialize graph with a small initial seed
G = nx.Graph()
G.add_edge(0, 1)
initial_nodes = list(G.nodes)

# Function to perform preferential attachment
def preferential_attachment_choice(graph):
    degrees = np.array([graph.degree(n) for n in graph.nodes()])
    if degrees.sum() == 0:
        return np.random.choice(list(graph.nodes()))
    probabilities = degrees / degrees.sum()
    return np.random.choice(list(graph.nodes()), p=probabilities)

# Simulation process
for step in range(num_steps):
    # Decide if adding a new node or an edge between existing nodes
    if np.random.rand() < R_new:
        # Add a new node and connect it to an existing node based on preferential attachment
        new_node = max(G.nodes) + 1  # Add node with a new index
        target_node = preferential_attachment_choice(G)
        G.add_edge(new_node, target_node)
    else:
        # Add an edge between two existing nodes with preferential attachment
        node1 = preferential_attachment_choice(G)
        node2 = preferential_attachment_choice(G)
        # Ensure we don't add self-loops or duplicate edges
        while node1 == node2 or G.has_edge(node1, node2):
            node2 = preferential_attachment_choice(G)
        G.add_edge(node1, node2)
    
    # Node removal step
    for node in list(G.nodes):
        if np.random.rand() < epsilon:
            G.remove_node(node)

# Collect degree distribution data
degrees = [G.degree(n) for n in G.nodes()]
degree_count = Counter(degrees)
deg, freq = zip(*sorted(degree_count.items()))

# Plotting the degree distribution
plt.figure(figsize=(10, 6))
plt.loglog(deg, freq, 'bo-', markerfacecolor='blue')
plt.xlabel("Degree (k)")
plt.ylabel("Number of nodes with degree k")
plt.title("Degree Distribution of Simulated Network")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.show()