import networkx as nx
import numpy as np

def create_testing_ring(num_nodes):
    """Creates a basic circular ring network for simple debugging."""
    G = nx.cycle_graph(num_nodes)
    return nx.to_numpy_array(G)

def create_financial_network(num_nodes, attachments=3):
    """
    Creates a Barabasi-Albert scale-free network.
    This mimics financial networks where a few massive banks (hubs) 
    have the vast majority of connections.
    """
    # attachments = number of edges to attach from a new node to existing nodes
    G = nx.barabasi_albert_graph(num_nodes, attachments)
    return nx.to_numpy_array(G)

# This file handles the network topology. It generates the structural backbone of our simulation
# We will write functions to create a simple ring network for testing, and a "Scale-Free" network 
# which mimics highly interconnected financial systems (where a few nodes act as major banking hubs).
