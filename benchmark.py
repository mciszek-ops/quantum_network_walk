import time
import numpy as np
import matplotlib.pyplot as plt
from network_generator import create_testing_ring
from quantum_engine import run_classical_solver, run_hpc_sparse_solver

def run_performance_test():
    # Test network sizes scaling up
    sizes = [10, 50, 100, 200, 400]
    time_points = np.linspace(0, 1.0, 5) # Short timeline just to gauge performance
    
    dense_times = []
    sparse_times = []
    
    print("Starting HPC Benchmarks...")
    for N in sizes:
        print(f"Testing network size: {N} nodes...")
        A = create_testing_ring(N)
        
        # Benchmark Dense Method
        start = time.time()
        _ = run_classical_solver(A, time_points)
        dense_times.append(time.time() - start)
        
        # Benchmark Sparse HPC Method
        start = time.time()
        _ = run_hpc_sparse_solver(A, time_points)
        sparse_times.append(time.time() - start)
        
    # Plot performance results
    plt.figure(figsize=(9, 5))
    plt.plot(sizes, dense_times, 'o--', color='red', label='Dense Solver (O(N³))')
    plt.plot(sizes, sparse_times, 's-', color='green', label='HPC Sparse Solver')
    plt.title('HPC Optimization: Dense vs. Sparse Quantum Walk Execution')
    plt.xlabel('Network Size (Number of Nodes)')
    plt.ylabel('Execution Time (Seconds)')
    plt.yscale('log') # Log scale helps see the massive difference clearly
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.savefig('hpc_benchmark_results.png')
    print("Benchmark completed! Plot saved as 'hpc_benchmark_results.png'.")

if __name__ == "__main__":
    run_performance_test()

# This script runs both solvers on progressively larger network sizes, 
# records exactly how many seconds each method takes, 
# and saves a performance graph to prove  
# high-performance optimization worked.