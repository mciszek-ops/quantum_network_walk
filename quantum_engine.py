import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

def run_classical_solver(adjacency_matrix, time_points, start_node=0):
    """
    Standard solver using dense matrices. 
    Fine for small networks, but chokes on big data due to la.expm().
    """
    num_nodes = len(adjacency_matrix)
    H = -adjacency_matrix
    
    # Initialize initial state vector (100% probability at start_node)
    psi_0 = np.zeros(num_nodes, dtype=complex)
    psi_0[start_node] = 1.0 + 0.0j
    
    prob_history = []
    for t in time_points:
        # Computes the full matrix exponential: U = e^(-i * H * t)
        U = la.expm(-1j * H * t)
        psi_t = np.dot(U, psi_0)
        prob_history.append(np.abs(psi_t)**2)
        
    return np.array(prob_history)

def run_hpc_sparse_solver(adjacency_matrix, time_points, start_node=0):
    """
    Optimized HPC solver using compressed sparse row (CSR) matrices.
    Uses Krylov subspace projection instead of full matrix exponentiation.
    """
    num_nodes = len(adjacency_matrix)
    # Convert matrix to sparse format to save massive amounts of RAM
    H_sparse = sparse.csr_matrix(-adjacency_matrix)
    
    psi_0 = np.zeros(num_nodes, dtype=complex)
    psi_0[start_node] = 1.0 + 0.0j
    
    prob_history = []
    for t in time_points:
        # Instead of calculating the giant matrix U, expm_multiply calculates 
        # the directly evolved vector: psi_t = e^(-i * H * t) * psi_0
        psi_t = spla.expm_multiply(-1j * H_sparse * t, psi_0)
        prob_history.append(np.abs(psi_t)**2)
        
    return np.array(prob_history)

# This takes the network layout, converts it into a 
# quantum Hamiltonian and computes how the quantum 
# state spreads across the nodes over time.
