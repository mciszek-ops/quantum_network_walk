# Continuous-Time Quantum Walk on Financial Networks

This repository contains a Python-based simulation designed to model information, risk, and liquidity propagation across network topologies using Continuous-Time Quantum Walks (CTQW). 

## Technical Overview

Traditional financial risk models generally rely on classical random walks or diffusion processes, where shocks propagate incrementally from node to node, similar to fluid moving through a pipeline. This simulation replaces classical diffusion with a quantum walk framework. Instead of following discrete, single-path trajectories, the initial network state propagates as a wave across all available paths simultaneously, creating constructive and destructive interference patterns. 

By analyzing these wave mechanics over a network topology, the model isolates structural vulnerabilities and transmission speeds that standard Markovian diffusion models frequently miss.

## Network Topology Selection

The engine defaults to a 5-node cyclic ring network. This specific configuration serves two purposes:
* **Symmetry and Inference Tracking:** A 5-node ring is the minimal non-trivial cyclic structure required to observe genuine quantum interference effects without the data being skewed by the oversimplified spatial symmetries found in 3-node or 4-node systems.
* **Benchmarking Baseline:** Because matrix exponentiation scales rapidly with network size, a 5-node matrix establishes a fast computational baseline to verify code execution before expanding the architecture to larger, denser datasets.

## Core Architecture

The codebase is divided into four functional scripts:

* `network_generator.py`: Constructs the graph topology. Nodes represent financial institutions or assets, and edges represent transaction pipelines or risk exposures.
* `quantum_engine.py`: Contains the core mathematical logic. It constructs the system Hamiltonian from the network adjacency matrix ($H = -A$) and computes the state evolution operator using the matrix exponential ($e^{-1j \cdot H \cdot t}$).
* `main.py`: Integrates the network configuration and physics engine, executes the simulation loop, and maps the probability distributions over time using matplotlib.
* `benchmark.py`: Measures execution speed and memory scaling limits to evaluate hardware efficiency.

## Local Installation and Execution

### Dependencies
The simulation requires standard scientific computing libraries. Install them via your terminal:

```powershell
pip install numpy scipy networkx matplotlib
