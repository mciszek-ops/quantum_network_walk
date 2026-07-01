import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from network_generator import generate_financial_network
from quantum_engine import compute_quantum_walk

print("?? Running Quantum Financial Network Simulation...")
G = generate_financial_network(5)
A = nx.to_numpy_array(G)
time_points = np.linspace(0, 2.0, 100)

prob_history = compute_quantum_walk(A, time_points)

print("?? Plotting results...")
plt.figure(figsize=(10, 6))
for node in range(A.shape[0]):
    plt.plot(time_points, prob_history[:, node], label=f"Node {node}")
plt.title("Continuous-Time Quantum Walk on Financial Network")
plt.xlabel("Time Steps")
plt.ylabel("Probability Distribution")
plt.grid(True)
plt.legend()
plt.show()
print("? Done!")
