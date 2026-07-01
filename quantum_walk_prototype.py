import numpy as np
import scipy.linalg as la
import networkx as nx
import matplotlib.pyplot as plt
num_nodes = 5
G = nx.cycle_graph(num_nodes)
A = nx.to_numpy_array(G)
H = -A 
psi_0 = np.zeros(num_nodes, dtype=complex)
psi_0[0] = 1.0 + 0.0j
time_points = np.linspace(0, 2.0, 100)
prob_history = []
for t in time_points:
    U = la.expm(-1j * H * t)
    psi_t = np.dot(U, psi_0)
    prob_history.append(np.abs(psi_t)**2)
prob_history = np.array(prob_history)
plt.figure(figsize=(10, 6))
for node in range(num_nodes):
    plt.plot(time_points, prob_history[:, node], label=f"Node {node}")
plt.title("Continuous-Time Quantum Walk on a 5-Node Ring")
plt.xlabel("Time")
plt.ylabel("Probability")
plt.grid(True)
plt.legend()
plt.show()
