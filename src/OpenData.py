#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multiplex Network Data Handling and Generation

This module provides utilities for loading and generating multiplex network data:
- Reading multiplex network data from various file formats
- Generating synthetic multiplex networks using Stochastic Block Models (SBM)
- Supporting both symmetric and perturbed network generation

Key Features:
- Flexible file format support (CSV and custom formats)
- SBM-based network generation with community structure
- Controlled perturbation of network layers
- Comprehensive input validation

Author: Martin Guillemaud
Created: Dec 30, 2024
"""

## LIBRARIES 

import pandas as pd
import networkx as nx
import numpy as np
import random 


## FUNCTIONS 

def ReadMultiplexData(filenames):
    """
    Reads multiplex network data from a list of files and constructs graph representations for each layer.

    Parameters
    ----------
    filenames : list of str
        A list of file paths where:
        - If the first file ends with '.csv', it contains adjacency matrices for each layer.
        - Otherwise, the files contain node names, layer names, and edge data in the following order:
          1. Node names file
          2. Layer names file
          3. Edges file

    Returns
    -------
    tuple
        - G_tot : list of networkx.Graph
            A list of graph objects, one for each layer.
        - matrices_tot : list of numpy.ndarray
            A list of adjacency matrices corresponding to each graph.

    Raises
    ------
    TypeError
        If `filenames` is not a list or its elements are not strings.
    ValueError
        If the list is empty or does not have the expected format.
        If the file format or content is invalid or inconsistent.
    FileNotFoundError
        If any of the files in `filenames` cannot be found.
    """
    # --- Verify filenames ---
    if not isinstance(filenames, list):
        raise TypeError(f"Expected 'filenames' to be a list, but got {type(filenames).__name__}.")
    if not all(isinstance(fname, str) for fname in filenames):
        raise TypeError("All elements in 'filenames' must be strings representing file paths.")
    if len(filenames) < 1:
        raise ValueError("The 'filenames' list must contain at least one file path.")
    
    # Check for CSV format if applicable
    if filenames[0].endswith('csv'):
        if not all(fname.endswith('csv') for fname in filenames):
            raise ValueError("If the first file is a CSV, all files in 'filenames' must be CSVs.")
    else:
        # Ensure there are exactly 3 files for non-CSV format
        if len(filenames) != 3:
            raise ValueError("For non-CSV format, 'filenames' must contain exactly 3 files: nodes, layers, and edges.")

    # --- Check if input is in CSV format ---
    if filenames[0].endswith('csv'):
        # Initialize containers
        G_tot = []
        matrices_tot = []

        # Process each file in the list
        for filename in filenames:
            # Read adjacency matrix from CSV
            df = pd.read_csv(filename, header=None)
            adj_matrix = df.iloc[:, :-1].to_numpy()

            # Create a graph from the adjacency matrix
            G = nx.from_numpy_array(adj_matrix)

            # Save graph and its adjacency matrix
            G_tot.append(G)
            matrices_tot.append(nx.to_numpy_array(G))
        
        # Assign layer names as sequential integers
        layers_names = np.arange(len(G_tot))

    else:
        ## --- Process Node Names File --- ##
        # Read and process the node names file
        with open(filenames[0], "r") as f:
            lines = f.readlines()

        # Extract and clean node name data
        result = [x.split(' ') for x in lines[1:]]  # Skip header
        nodes_names = {int(data[0]): data[1][:-2] for data in result if data}

        ## --- Process Layer Names File --- ##
        # Read and process the layer names file
        with open(filenames[1], "r") as f:
            lines = f.readlines()

        # Extract and clean layer name data
        result = [x.split(' ') for x in lines[1:]]  # Skip header
        layers_names = {int(data[0]): data[1][:-1] for data in result if data}

        ## --- Process Edges File --- ##
        # Read and process the edges file
        with open(filenames[2], "r") as f:
            lines = f.readlines()

        # Extract and clean edge data
        result = [list(map(int, filter(None, x.split(' ')))) for x in lines]
        edges_pos = np.array(result)

        ## --- Create Graphs for Each Layer --- ##
        G_tot = []
        matrices_tot = []

        # Iterate over layers and build graphs
        for layer_id, layer_name in layers_names.items():
            print(f'Creating layer: {layer_name} (ID: {layer_id})')

            # Create an empty graph with predefined nodes
            G = nx.Graph()
            G.add_nodes_from(nodes_names.keys())

            # Add edges corresponding to the current layer
            for edge in edges_pos[edges_pos[:, 0] == layer_id]:
                G.add_edge(edge[1], edge[2], weight=edge[3])

            # Save graph and its adjacency matrix
            G_tot.append(G)
            matrices_tot.append(nx.to_numpy_array(G))
    
    return G_tot, matrices_tot

def generate_binary_value(p):
    """
    Generate a binary value (0 or 1) with specified probability.
    
    Parameters
    ----------
    p : float
        Probability of generating 1 (must be between 0 and 1)
    
    Returns
    -------
    int
        0 or 1, where 1 occurs with probability p
    """
    return 1 if random.uniform(0, 1) < p else 0


def generate_sbm_multilayer(n_nodes=100, n_layers=3, n_communities=3, 
                            p_intra=0.16, p_inter=0.05, mu=1000, beta=10):
    """
    Generates a Stochastic Block Model (SBM) multilayer network with specified intra- 
    and inter-layer connectivity probabilities.

    Parameters
    ----------
    n_nodes : int, optional
        Number of nodes per layer. Must be a positive integer. Default is 100.
    n_layers : int, optional
        Number of layers in the SBM multilayer network. Must be a positive integer. Default is 3.
    n_communities : int, optional
        Number of communities (blocks) in the network. Must be a positive integer <= n_nodes. Default is 3.
    p_intra : float, optional
        Probability of intra-community edges within a layer. Must be in [0, 1]. Default is 0.16.
    p_inter : float, optional
        Probability of inter-community edges within a layer. Must be in [0, 1]. Default is 0.05.
    mu : float, optional
        Multiplicative factor for inter-layer self-connections. Must be non-negative. Default is 1000.
    beta : float, optional
        Scaling factor for inter-layer connectivity probabilities. Must be positive. Default is 10.

    Returns
    -------
    tuple
        - G_tot : list of networkx.Graph
            List of graphs representing individual layers.
        - mat_tot = list of np.array
            List of the connectivity maytrices of the layers
        - G_global : numpy.ndarray
            Global adjacency matrix of the multilayer network.
        - node_com : dict
            Mapping of node IDs to community (block) IDs.

    Raises
    ------
    ValueError
        If any of the input parameters are invalid.

    Notes
    -----
    - The intra-layer connections follow a standard SBM structure, where nodes are grouped 
      into communities, and edges are added with different probabilities depending on 
      whether nodes belong to the same or different communities.
    - Inter-layer connections are added probabilistically, scaled by the parameter `beta`.

    Example
    -------
    >>> G_tot, G_global, node_com = generate_sbm_multilayer()
    """
    # --- Validate Parameters ---
    if not isinstance(n_nodes, int) or n_nodes <= 0:
        raise ValueError(f"'n_nodes' must be a positive integer. Got {n_nodes}.")
    if not isinstance(n_layers, int) or n_layers <= 0:
        raise ValueError(f"'n_layers' must be a positive integer. Got {n_layers}.")
    if not isinstance(n_communities, int) or n_communities <= 0 or n_communities > n_nodes:
        raise ValueError(f"'n_communities' must be a positive integer <= n_nodes. Got {n_communities}.")
    if not (isinstance(p_intra, (int, float)) and 0 <= p_intra <= 1):
        raise ValueError(f"'p_intra' must be a float in [0, 1]. Got {p_intra}.")
    if not (isinstance(p_inter, (int, float)) and 0 <= p_inter <= 1):
        raise ValueError(f"'p_inter' must be a float in [0, 1]. Got {p_inter}.")
    if not (isinstance(mu, (int, float)) and mu >= 0):
        raise ValueError(f"'mu' must be a non-negative float. Got {mu}.")
    if not (isinstance(beta, (int, float)) and beta > 0):
        raise ValueError(f"'beta' must be a positive float. Got {beta}.")

    # --- Parameters ---
    p_intra_l = p_intra / beta
    p_inter_l = p_inter / beta

    # --- Define Communities ---
    nodes = np.arange(n_nodes)
    np.random.shuffle(nodes)  # Shuffle node order to randomize community assignments

    # Divide nodes into approximately equal-sized communities
    avg_group_size = len(nodes) // n_communities
    remainder = len(nodes) % n_communities
    groups = []
    start_idx = 0
    for i in range(n_communities):
        group_size = avg_group_size + (1 if i < remainder else 0)
        groups.append(nodes[start_idx:start_idx + group_size])
        start_idx += group_size

    # Create a dictionary mapping nodes to their community (block)
    node_com = {node: i_com for i_com, group in enumerate(groups) for node in group}

    # --- Intra-Layer Connections (SBM Structure) ---
    G_tot = []
    mat_tot = []
    for layer in range(n_layers):
        G = nx.Graph()
        G.add_nodes_from(nodes)

        # Add edges probabilistically based on community membership
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if node_com[i] == node_com[j]:  # Same community
                    if generate_binary_value(p_intra) == 1:
                        G.add_edge(i, j)
                else:  # Different communities
                    if generate_binary_value(p_inter) == 1:
                        G.add_edge(i, j)
        
        # Save the graph and its adjacency matrix
        matrix = nx.to_numpy_array(G)
        G_tot.append(G)
        mat_tot.append(matrix)

    # --- Inter-Layer Connections ---
    G_global = np.zeros((n_nodes * n_layers, n_nodes * n_layers))

    # Populate intra-layer connections in the global matrix
    for layer in range(n_layers):
        G_global[layer * n_nodes:(layer + 1) * n_nodes, 
                 layer * n_nodes:(layer + 1) * n_nodes] = mat_tot[layer]

    # Add inter-layer connections based on SBM rules
    for l1 in range(n_layers):
        for l2 in range(l1 + 1, n_layers):
            G_inter = np.identity(n_nodes) * mu  # Initialize inter-layer connections with self-connections
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if node_com[i] == node_com[j]:  # Same community
                        if generate_binary_value(p_intra_l) == 1:
                            G_inter[i, j] = G_inter[j, i] = 1
                    else:  # Different communities
                        if generate_binary_value(p_inter_l) == 1:
                            G_inter[i, j] = G_inter[j, i] = 1

            # Update the global matrix with inter-layer connections
            G_global[l1 * n_nodes:(l1 + 1) * n_nodes, 
                     l2 * n_nodes:(l2 + 1) * n_nodes] = G_inter
            G_global[l2 * n_nodes:(l2 + 1) * n_nodes, 
                     l1 * n_nodes:(l1 + 1) * n_nodes] = G_inter

    return G_tot, mat_tot, G_global, node_com


def generate_sbm_multilayer_perturbation(n_nodes=100, n_layers=3, n_communities=3, 
                            p_intra=0.16, p_inter=0.05, mu=1000, beta=10, perturbation=[0.02, 0.02]):
    """
    Generate a perturbed multilayer Stochastic Block Model network.
    
    Extends the standard SBM by introducing controlled perturbations to specific layers,
    allowing for the study of network resilience and community detection under noise.

    Parameters
    ----------
    n_nodes : int, optional
        Number of nodes per layer. Default: 100
    n_layers : int, optional
        Number of network layers. Default: 3
    n_communities : int, optional
        Number of communities in the network. Default: 3
    p_intra : float, optional
        Base probability of intra-community edges. Default: 0.16
    p_inter : float, optional
        Base probability of inter-community edges. Default: 0.05
    mu : float, optional
        Self-connection strength between layers. Default: 1000
    beta : float, optional
        Inter-layer connectivity scaling. Default: 10
    perturbation : list of float, optional
        [intra_perturbation, inter_perturbation] for layer 2. Default: [0.02, 0.02]
        
    Returns
    -------
    tuple
        (G_tot, mat_tot, G_global, node_com) containing:
        - Network graphs for each layer
        - Adjacency matrices
        - Global connectivity matrix
        - Node-to-community mapping

    Mathematical Details
    ------------------
    - Layer 2 probabilities:
      p_intra_effective = p_intra + perturbation[0]
      p_inter_effective = p_inter + perturbation[1]
    - Inter-layer scaling:
      p_intra_layer = p_intra / beta
      p_inter_layer = p_inter / beta
    """
    # --- Validate Parameters ---
    if not isinstance(n_nodes, int) or n_nodes <= 0:
        raise ValueError(f"'n_nodes' must be a positive integer. Got {n_nodes}.")
    if not isinstance(n_layers, int) or n_layers <= 0:
        raise ValueError(f"'n_layers' must be a positive integer. Got {n_layers}.")
    if not isinstance(n_communities, int) or n_communities <= 0 or n_communities > n_nodes:
        raise ValueError(f"'n_communities' must be a positive integer <= n_nodes. Got {n_communities}.")
    if not (isinstance(p_intra, (int, float)) and 0 <= p_intra <= 1):
        raise ValueError(f"'p_intra' must be a float in [0, 1]. Got {p_intra}.")
    if not (isinstance(p_inter, (int, float)) and 0 <= p_inter <= 1):
        raise ValueError(f"'p_inter' must be a float in [0, 1]. Got {p_inter}.")
    if not (isinstance(mu, (int, float)) and mu >= 0):
        raise ValueError(f"'mu' must be a non-negative float. Got {mu}.")
    if not (isinstance(beta, (int, float)) and beta > 0):
        raise ValueError(f"'beta' must be a positive float. Got {beta}.")

    # --- Parameters ---
    p_intra_l = p_intra / beta
    p_inter_l = p_inter / beta

    # --- Define Communities ---
    nodes = np.arange(n_nodes)
    np.random.shuffle(nodes)  # Shuffle node order to randomize community assignments

    # Divide nodes into approximately equal-sized communities
    avg_group_size = len(nodes) // n_communities
    remainder = len(nodes) % n_communities
    groups = []
    start_idx = 0
    for i in range(n_communities):
        group_size = avg_group_size + (1 if i < remainder else 0)
        groups.append(nodes[start_idx:start_idx + group_size])
        start_idx += group_size

    # Create a dictionary mapping nodes to their community (block)
    node_com = {node: i_com for i_com, group in enumerate(groups) for node in group}

    # --- Intra-Layer Connections (SBM Structure) ---
    G_tot = []
    mat_tot = []
    for layer in range(n_layers):
        G = nx.Graph()
        G.add_nodes_from(nodes)

        if layer == 1 : 
            p_intra_loc = p_intra + perturbation[0]
            p_inter_loc = p_inter + perturbation[1]
        else : 
            p_intra_loc = p_intra
            p_inter_loc = p_inter

        # Add edges probabilistically based on community membership
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if node_com[i] == node_com[j]:  # Same community
                    if generate_binary_value(p_intra_loc) == 1:
                        G.add_edge(i, j)
                else:  # Different communities
                    if generate_binary_value(p_inter_loc) == 1:
                        G.add_edge(i, j)
        
        # Save the graph and its adjacency matrix
        matrix = nx.to_numpy_array(G)
        G_tot.append(G)
        mat_tot.append(matrix)

    # --- Inter-Layer Connections ---
    G_global = np.zeros((n_nodes * n_layers, n_nodes * n_layers))

    # Populate intra-layer connections in the global matrix
    for layer in range(n_layers):
        G_global[layer * n_nodes:(layer + 1) * n_nodes, 
                 layer * n_nodes:(layer + 1) * n_nodes] = mat_tot[layer]

    # Add inter-layer connections based on SBM rules
    for l1 in range(n_layers):
        for l2 in range(l1 + 1, n_layers):
            G_inter = np.identity(n_nodes) * mu  # Initialize inter-layer connections with self-connections
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if node_com[i] == node_com[j]:  # Same community
                        if generate_binary_value(p_intra_l) == 1:
                            G_inter[i, j] = G_inter[j, i] = 1
                    else:  # Different communities
                        if generate_binary_value(p_inter_l) == 1:
                            G_inter[i, j] = G_inter[j, i] = 1

            # Update the global matrix with inter-layer connections
            G_global[l1 * n_nodes:(l1 + 1) * n_nodes, 
                     l2 * n_nodes:(l2 + 1) * n_nodes] = G_inter
            G_global[l2 * n_nodes:(l2 + 1) * n_nodes, 
                     l1 * n_nodes:(l1 + 1) * n_nodes] = G_inter

    return G_tot, mat_tot, G_global, node_com

## MAIN 
if __name__ =="__main__" :
    
    pass

