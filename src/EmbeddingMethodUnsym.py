#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Asymmetric Multiplex Hyperbolic Embedding (MLNHypEmb-Unsym)

A Python implementation for embedding asymmetric multiplex networks into hyperbolic space.
Specifically designed for networks where different layers can have different numbers of nodes.

Key Features:
- Handles layers with varying numbers of nodes
- Supports multiple embedding methods (Isomap, Spectral)
- Implements various radius computation strategies
- Provides flexible edge weighting options

Mathematical Background:
- Uses Poincaré disk model for hyperbolic embedding
- Supports both local (per-layer) and global network structure
- Implements adaptive radius computation based on network topology

Author: Martin Guillemaud
Created: Dec 30, 2024

# Copyright (c) 2025, Martin Guillemaud. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file in the root directory of this source tree.
"""

## LIBRARY IMPORTS
from sklearn.metrics import pairwise_distances
import numpy as np 
import networkx as nx
from sklearn.manifold import spectral_embedding
from sklearn.manifold import Isomap


## FUNCTIONS DEFINITIONS

## CLASS DEFINITION 
class MlHypEmbUnsym:
    """
    Hyperbolic embedding for asymmetric multiplex networks.
    
    This class implements methods to embed multilayer networks with different numbers of nodes per layer 
    into the Poincaré disk model of hyperbolic space. It supports various embedding strategies and 
    radius computation methods.

    Key Features:
        - Handles layers with different numbers of nodes
        - Supports both weighted and unweighted networks
        - Implements multiple radius computation strategies
        - Offers pre-weighting mechanisms for edge importance
    """
    
    def __init__(self, preweight=True, weight='weight', beta=0.9, eta='auto', method='Isomap', beta_dist=1000, metric='precomputed', n_neighbors='auto', radius='order', message=True):
        """
       Initializes the class object with the specified parameters.
    
       Parameters
       ----------
       preweight : bool, optional
           Indicates whether to apply weights before the transformation. The default is True.
       weight : str oor None, optional
           The name of the weight to use for Degree calculation. The default is 'weight'. If None the degree is just the number of connexions of the node.
       beta : float or int, optional
           The beta parameter controlling the radius attribution of the method. The default is 0.9.
       eta : str, float, or int, optional
           The eta parameter for radius attribution, which can be a number or 'auto'. The default is 'auto'. 
       method : str, optional
           The  dimension reduction method to use, either 'Isomap' or 'Spectral'. The default is 'Isomap'.
       beta_dist : float or int, optional
           The beta_dist parameter controlling the distance normalisation. The default is 1000.
       metric : str, optional
           The metric used for distance calculation in Isomap method. Must be one of the valid metrics for Isomap. If 'precomputed', uses the given matrix as distance matrix. The default is 'precomputed'.
       n_neighbors : int, float, or str, optional
           The number of neighbors to consider for Isomap method, or 'auto' for automatic calculation. The default is 'auto'.
       radius : str, optional
           The method for calculating the radius, chosen from 'Order', 'Degree', or 'LogDegree'. The default is 'Order'.
       message : Bool, optional 
           Whether to print a message at the end of the initiation of the class. The default value is True 
       Raises
       ------
       TypeError
           If any of the parameters is not of the expected type (e.g., `preweight` is not a `bool`, `weight` is not a `str`).
       ValueError
           If any of the parameters contains an invalid value (e.g., `method` is neither 'Isomap' nor 'Spectral', `metric` is not valid).
    
       Returns
       -------
       None
           The function does not return anything; it initializes the object with the specified parameters.
       """

        ## Make sure parameters correspond
        
        # preweight
        if not isinstance(preweight, bool):
            raise TypeError(f"Expected a boolean for preweight variable, but got {type(preweight).__name__} instead.")
        
        # weight
        if not (isinstance(weight, str) or weight is None):
            raise TypeError(f"Expected a str or None for weight variable, but got {type(weight).__name__} instead.")
        
        # beta
        if not isinstance(beta, (int, float)):
            raise TypeError(f"Expected an int or a float for beta variable, but got {type(beta).__name__} instead.")
        
        # eta
        if not (isinstance(eta, (int, float)) or eta == 'auto'):
            raise TypeError(f"Expected an int, a float, or the string 'auto' for eta variable, but got {type(eta).__name__} with value {eta}.")
        
        # method
        allowed_methods = ['Isomap', 'Spectral']
        if not isinstance(method, str) or method not in allowed_methods:
            raise ValueError(f"Expected a string and one of {allowed_methods} for method variable, but got {type(method).__name__} with value {method}.")
        
        # beta_dist
        if not isinstance(beta_dist, (int, float)):
            raise TypeError(f"Expected an int or a float for beta_dist variable, but got {type(beta_dist).__name__} instead.")
        
        # metric
        #valid_metrics = pairwise_distances.__dict__['__all__']
        valid_metrics = ['euclidean', 'manhattan', 'cosine', 'hamming', 'jaccard', 'minkowski', 'chebyshev', 'seuclidean', 'sqeuclidean', 'cityblock', 'precomputed']
        if not isinstance(metric, str) or metric not in valid_metrics:
            raise ValueError(f"Invalid metric '{metric}'. Expected a string and one of {valid_metrics}.")
        
        # n_neighbors
        if not (isinstance(n_neighbors, (int, float)) or eta == 'auto'):
            raise TypeError(f"Expected an int, a float, or the string 'auto' for n_neighbors variable, but got {type(n_neighbors).__name__} with value {eta}.")
        
        # radius
        allowed_radius = ['order', 'degree', 'logDegree']
        if not isinstance(radius, str) or radius not in allowed_radius:
            raise ValueError(f"Expected a string and one of {allowed_radius} for method variable, but got {type(radius).__name__} with value {radius}.")
        
        ## Save parameters 
        self.preweight = preweight
        self.weight = weight 
        self.beta = beta 
        self.eta = eta
        self.method = method
        self.beta_dist = beta_dist
        self.metric = metric 
        self.n_neighbors = n_neighbors 
        self.radius = radius 
        
        # Print message 
        if message : 
            print('[I] Object of class MlHypEmb well initiated')
        
        return None 
        
        
    
    def Load_data(self, matrices, L_global): 
        """
        Loads and preprocesses network data for embedding.
    
        Processes input matrices representing different network layers and their global structure.
        Handles asymmetric cases where layers can have different numbers of nodes.
    
        Parameters
        ----------
        matrices : list of np.ndarray
            List of adjacency matrices, one per layer. Matrices can have different dimensions.
        L_global : np.ndarray 
            Global Laplacian matrix representing the complete multilayer structure.
            Must be symmetric and have dimensions matching the total number of nodes.

        Notes
        -----
        - Performs validation checks on input matrices
        - Constructs graph representations for each layer
        - Applies pre-weighting if enabled
        - Computes necessary indices for layer management
        """
        
        # Ensure matrices is a list of numpy arrays with matching shapes
        if not isinstance(matrices, list):
            raise TypeError(f"Expected a list for matrices, but got {type(matrices).__name__}.")
        
        # array_shape = matrices[0].shape
        # for arr in matrices:
        #     if not isinstance(arr, np.ndarray):
        #         raise TypeError(f"Expected a numpy.ndarray, but got {type(arr).__name__} in matrices list.")
        #     if arr.shape != array_shape or arr.shape[0] != arr.shape[1]:
        #         raise ValueError(f"Array with shape {arr.shape} is either not square or doesn't match the first array's shape in matrices list.")
        
        # Validate L_global if provided
        if L_global is not None:
            # Ensure L_global is a numpy ndarray
            if not isinstance(L_global, np.ndarray):
                raise TypeError(f"Expected a numpy.ndarray or None for L_global, but got {type(L_global).__name__}.")
            
            # Ensure L_global is square and matches the expected size
            expected_size = np.sum([len(matrices[i]) for i in range(len(matrices))])
            if L_global.shape[0] != L_global.shape[1] or L_global.shape[0] != expected_size:
                raise ValueError(f"L_global should be a square matrix with size {expected_size}, but got shape {L_global.shape}.")
            
            # Ensure L_global is symmetric
            if not np.array_equal(L_global, L_global.T):
                raise ValueError(f"L_global with shape {L_global.shape} is not symmetric.")
        
        # Initialize eta if 'auto' is given
        if self.eta == 'auto':
            self.eta = 2 * np.log(len(matrices[0])) / (1 - 1e-5)
            
        # Initialize Graphs_tot to store graph representations
        Graphs_tot = []
        for matrix in matrices:
            G = nx.from_numpy_array(matrix)
            Graphs_tot.append(G)
        
        # Store matrices in W_tot for later use
        W_tot = matrices
        
        # Pre-weight the edges if pre_weight is enabled
        if self.preweight:
            for i, G in enumerate(Graphs_tot):
                edges = G.edges()
                for edge in edges:
                    CN = len(list(nx.common_neighbors(G, edge[0], edge[1])))
                    d_1 = nx.degree(G, edge[0])
                    d_2 = nx.degree(G, edge[1])
                    G[edge[0]][edge[1]]['weight'] = (d_1 + d_2 + d_1 * d_2) / (1 + CN)
                W = np.log(nx.adjacency_matrix(G).toarray() + 1)
                Graphs_tot[i] = G
                W_tot[i] = W
        
        # Initialize Laplacian matrix
        n_tot = [len(W_tot[i]) for i in range(len(W_tot))]  # Number of nodes in each graph
        start_n = np.cumsum([0] + n_tot[:-1])  # Starting indices for each graph
        end_n = np.cumsum(n_tot)  # Ending indices for each graph
   
        # Add connectivity matrices in the diagonals of L_global
        for i, W in enumerate(W_tot):
            L_global[start_n[i]:end_n[i], start_n[i]:end_n[i]] = W
        
        # Save the variables
        self.matrices = matrices
        self.Graphs_tot = Graphs_tot
        self.W_tot = W_tot
        self.L_global = L_global
        self.n_nodes_tot = n_tot
        self.n_layers = len(W_tot)
        self.start_n = start_n
        self.end_n = end_n
        self.n_nodes = np.sum(n_tot)
        return None
    

    def Embedding(self, n_jobs=-1):
        """
        Performs hyperbolic embedding of the multiplex network.

        Embeds the global network structure into the Poincaré disk model using dimensionality 
        reduction techniques. Handles the asymmetric case where layers have different numbers of nodes.

        Parameters
        ----------
        n_jobs : int, default=-1
            Number of parallel jobs for computation. Use -1 for all available cores.

        Technical Details
        ----------------
        1. Performs dimensionality reduction using either:
           - Spectral embedding
           - Isomap embedding
        2. Extracts individual layer embeddings
        3. Computes node radii using selected strategy:
           - Order-based: based on node degree ordering
           - Degree-based: using hyperbolic tangent of degrees
           - LogDegree-based: using power law of degrees
        4. Projects embeddings onto the Poincaré disk

        Notes
        -----
        The embedding process preserves both the network structure and the 
        hierarchical relationships between nodes.
        """
        
        # Validate the `n_jobs` parameter
        if not isinstance(n_jobs, int) or (n_jobs != -1 and n_jobs <= 0):
            raise ValueError(f"Expected n_jobs to be -1 or a positive integer, but got {n_jobs} of type {type(n_jobs).__name__}.")

        # Dimension reduction step
        if self.method == 'Spectral':
            # Spectral embedding based on the global Laplacian matrix
            L_global_emb = spectral_embedding(self.L_global, n_components=2)
        elif self.method == 'Isomap':
            # Determine the number of neighbors if set to 'auto'
            n_neighbors = np.shape(self.L_global)[0] - 1 if self.n_neighbors == 'auto' else self.n_neighbors
            # Perform Isomap embedding
            embedding = Isomap(n_components=2, n_neighbors=n_neighbors, eigen_solver='dense',
                               n_jobs=n_jobs, metric=self.metric, neighbors_algorithm='brute')
            L_global_emb = embedding.fit_transform(np.exp(-self.L_global / self.beta_dist))

        # Extract embeddings for individual layers
        emb_tot = []  # List to store embeddings for all layers
        for i in range(self.n_layers):
            # Extract embedding for layer i
            emb_i = L_global_emb[self.start_n[i]:self.end_n[i]]
            emb_tot.append(emb_i)

        # Compute radii and normalized embeddings
        emb_r_tot = []  # List to store radius-scaled embeddings
        emb_n_tot = []  # List to store normalized embeddings

        for emb_i, G_i in zip(emb_tot, self.Graphs_tot):
            # Normalize the embeddings to unit radius
            emb_i_n = emb_i / np.sqrt(np.sum(emb_i ** 2, axis=1, keepdims=True))

            # Compute node degrees
            deg_i = list(dict(nx.degree(G_i, weight=self.weight)).values())
            deg_i_u = np.sort(np.unique(deg_i))[::-1]  # Unique degrees in descending order

            # Compute node order based on degree
            order_i = np.array([np.where(deg_i_u == deg)[0][0] + 1 for deg in deg_i])

            # Compute the radius for each node based on the chosen strategy
            if self.radius == 'order':
                radius_i = 2 / self.eta * (self.beta * np.log(order_i) + (1 - self.beta) * np.log(self.n_nodes))
            elif self.radius == 'degree':
                radius_i = 1 - np.tanh(np.array(deg_i) / self.beta)
            elif self.radius == 'logdegree':
                radius_i = 1 / (np.array(deg_i) + 1) ** self.beta

            # Scale the normalized embeddings by the computed radii
            emb_i_r = emb_i_n * radius_i[:, np.newaxis]

            # Store the results
            emb_n_tot.append(emb_i_n)
            emb_r_tot.append(emb_i_r)

        # Save results as class attributes
        self.n_jobs = n_jobs
        self.embeddings = emb_r_tot

        return None


## MAIN 
if __name__ =="__main__" :
    
    pass