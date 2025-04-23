#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:19:02 2025

@author: martin.guillemaud

Tools for hyperboling emebdded data exploitation 

"""


## LIBRAIRIES 
import numpy as np
import networkx as nx


## FUNCTIONS 

def hyp_dist(coord_1, coord_2):
    """
    Compute the hyperbolic 2D distance between two points given in Euclidean coordinates.

    Parameters
    ----------
    coord_1 : array of shape (1, 2)
        Coordinates of the first point [x, y].
    coord_2 : array of shape (1, 2)
        Coordinates of the second point [x, y].

    Returns
    -------
    dist : float
        Hyperbolic distance between the two points.
    """
    
    # Extract x and y coordinates from the arrays
    xi, yi = coord_1.flatten()
    xj, yj = coord_2.flatten()
    
    # Compute numerator and denominator for cosh(d_hyp)
    numerator = 2 * ((xi - xj)**2 + (yi - yj)**2)
    denominator = (1 - (xi**2 + yi**2)) * (1 - (xj**2 + yj**2))
    
    # Calculate cosh of the hyperbolic distance
    if denominator > 0 : 
        cosh_d = 1 + numerator / denominator
        
    else : 
        cosh_d = np.inf
    
    # Compute the hyperbolic distance if valid, otherwise set to 0
    if cosh_d >= 1:
        dist = np.arccosh(cosh_d)
    else:
        dist = 0  # Handle edge case where distance is undefined or invalid
    
    return dist



def hyp_disp_score(emb_1, emb_2):
    """
    Calculate hyperbolic distance scores for all nodes between two embeddings.

    Parameters
    ----------
    emb_1 : array of shape (n_nodes, 2)
        First embedding, where each row represents the coordinates [x, y] 
        of a node in the Poincaré disk.
    emb_2 : array of shape (n_nodes, 2)
        Second embedding, where each row represents the coordinates [x, y] 
        of the same nodes in the Poincaré disk.

    Returns
    -------
    nodes_score_tot : list of floats
        List of hyperbolic distances between corresponding nodes in `emb_1` and `emb_2`.
    """
    
    # Initialize list to store hyperbolic distances
    nodes_score_tot = []
    
    # Iterate over nodes
    for node in range(len(emb_1)):
        # Get coordinates for the current node in both embeddings
        coord_1 = emb_1[node, :]
        coord_2 = emb_2[node, :]
        
        # Compute the hyperbolic distance between the two points
        hyp_score = hyp_dist(coord_1, coord_2)
        
        # Store the distance
        nodes_score_tot.append(hyp_score)
    
    return nodes_score_tot



def ECO_filter(connectivity_tot, mean_degree=3, normalized_edges=True):
    
        """
        
        Filter the network by keeping the edges with the most important weight while reaching a fixed mean degree over the whole network 

        Parameters
        ----------
        connectivity_tot : list of connectivity matrices 
        mean_degree : float, optional
            MEAN DEGREE. The default is 3.
        normalized_edges : Bool, optional
            IF EDGES WEIGHT . The default is True.
            
        Returns
        -------
        None.

        """
        
        # Init 
        raw_connectivity_filtered_tot = []
        
        # Iteration on all the matrices 
        for matrix in connectivity_tot:
        
            # To graph
            G = nx.from_numpy_array(matrix)
        
            # Get nodes
            nodes = G.nodes()
        
            # Creat new graph with no edges 
            G_new = nx.Graph()
            G_new.add_nodes_from(nodes)
        
            # Ini
            edgelist = []
        
            ## Get edges weight 
        
            # Get edges
            edges = list(G.edges())

            # Dict of edges weight 
            edges_weight_dict = {}
        
            # Iteration on edges 
            for edge in edges :
                edges_weight_dict[edge] = matrix[edge[0], edge[1]]
                
            # Remove the edges from the spanning tree 
            for edge in edgelist :
                edges_weight_dict[edge]=0
                
            edges_dict_item = list(edges_weight_dict)
            edges_dict_values = list(edges_weight_dict.values())
            
            # Get edges order by weight 
            edges_order = np.argsort(edges_dict_values)
            
            # Get number of edges to add 
            n_edges_to_add = int(mean_degree*len(nodes)/2 - len(edgelist))
            
            # Iteration on the edges to add 
            for i_edge in edges_order[-n_edges_to_add:] :
                
                # Add the edges to the graph 
                G_new.add_edge(edges_dict_item[i_edge][0], edges_dict_item[i_edge][1], weight=edges_dict_values[i_edge])
                
            # Get adjacency matrix 
            A_filtered = nx.adjacency_matrix(G_new).toarray()
            
            # Save threshold
            mario_th = np.unique(A_filtered.flatten())[1]
            
            # Normalisation of all the edges 
            if normalized_edges==True :
                A_filtered[A_filtered>0]=1
            
            # Save 
            raw_connectivity_filtered_tot.append(A_filtered)
    
        return mario_th, raw_connectivity_filtered_tot


def Poincare_to_Klein(pos_p):
    """
    Convert Poincaré coordinates to Klein coordinates.
    Parameters
    ----------
    pos_p : array of shape (n_nodes, 2)
        Poincaré coordinates of the nodes.
    Returns
    -------
    pos_k : array of shape (n_nodes, 2)
        Klein coordinates of the nodes.
    """

    # Convert Poincaré coordinates to Klein coordinates using vectorized operations
    denom = 1 + np.sum(pos_p**2, axis=1, keepdims=True)
    pos_k = 2 * pos_p / denom

    # Return the Klein coordinates
    return pos_k

def Klein_to_Poincare(pos_k):
    """
    Convert Klein coordinates to Poincaré coordinates.
    Parameters
    ----------
    pos_k : array of shape (n_nodes, 2)
        Klein coordinates of the nodes.
    Returns
    -------
    pos_p : array of shape (n_nodes, 2)
        Poincaré coordinates of the nodes.
    """ 

    # Convert Klein coordinates to Poincaré coordinates using vectorized operations
    denom = 1 + np.sqrt(1 - np.sum(pos_k**2, axis=1, keepdims=True))
    pos_p = pos_k / denom

    # Return the Poincaré coordinates
    return pos_p

def Klein_baycenter(pos_k):
    """
    Compute the barycenter of a set of points in Klein coordinates.
    
    Parameters
    ----------
    pos_k : array of shape (n_nodes, 2)
        Klein coordinates of the nodes.

    Returns
    -------
    barycenter : array of shape (1, 2)
        Barycenter in Klein coordinates.
    """
    
    # Compute the barycenter in Klein coordinates
    gamma = np.sqrt(1 - np.sum(pos_k**2, axis=1, keepdims=True))
    barycenter = np.sum(gamma*pos_k, axis=0) / np.sum(gamma)

    # Return the barycenter
    return barycenter



def Mobius_addition(X, Y):
    """
    Perform Möbius addition of two points in Poincaré coordinates.
    
    Parameters
    ----------
    X : array of shape (1, 2)
        First point in Poincaré coordinates.
    Y : array of shape (1, 2)
        Second point in Poincaré coordinates.

    Returns
    -------
    m_addition : array of shape (1, 2)
        Result of the Möbius addition in Poincaré coordinates.
    """
    
    # Usfull values 
    norm_x2 = np.dot(X, X)
    norm_y2 = np.dot(Y, Y)
    xy_dot = np.dot(X, Y)
    
    # Get the numerator and denominator
    numerator = (1 + 2 * xy_dot + norm_y2) * X + (1 - norm_x2) * Y
    denominator = 1 + 2 * xy_dot + norm_x2 * norm_y2
    
    # Compute addition 
    if denominator > 0:
        m_addition = numerator / denominator
    else:
        # Handle the case where the denominator is zero or negative
        m_addition = np.nan  # or some other appropriate value
        # Alternatively, you could raise an exception or return NaN
        # raise ValueError("Denominator is zero or negative in Möbius addition.")

    return m_addition
    
def Logarithmic_map(pos_t, pos_p):
    """
    Compute the logarithmic map from Poincaré coordinates to tangent space coordinates.
    
    Parameters
    ----------
    pos_t : array of shape (1, 2)
        Tangent space coordinates.
    pos_p : array of shape (1, 2)
        Poincaré coordinates.

    Returns
    -------
    log_map : array of shape (1, 2)
        Logarithmic map in tangent space coordinates.
    """
    
    # Get lambda
    lambda_t = 2/(1 - np.dot(pos_t, pos_t))

    # Get log map 
    log_map = 2/lambda_t*np.arctanh(np.linalg.norm(Mobius_addition(-pos_t, pos_p)))*Mobius_addition(-pos_t, pos_p)/np.linalg.norm(Mobius_addition(-pos_t, pos_p))
    
    return log_map

def Exponential_map(pos_t, pos_e):
    """
    Compute the exponential map from tangent space coordinates to Poincaré coordinates.
    
    Parameters
    ----------
    pos_t : array of shape (1, 2)
        Tangent space coordinates.
    pos_e : array of shape (1, 2)
        Exponential coordinates.

    Returns
    -------
    exp_map : array of shape (1, 2)
        Exponential map in Poincaré coordinates.
    """
    
    # Get lambda
    lambda_t = 2/(1 - np.dot(pos_t, pos_t))

    # Get exp map 
    exp_map = Mobius_addition(pos_t, np.tanh(lambda_t*np.linalg.norm(pos_e)/2)*pos_e/np.linalg.norm(pos_e))
    
    return exp_map

def Get_barycenter_cov(data_p) :
    """
    Compute the barycenter and covariance matrix of a set of points in Poincaré coordinates.
    Parameters
    ----------
    data_p : array of shape (n_points, 2)
        Poincaré coordinates of the points.
    Returns
    -------
    barycenter_p : array of shape (1, 2)
        Barycenter in Poincaré coordinates.
    cov_matrix : array of shape (2, 2)
        Covariance matrix in tangent space coordinates.
    """
 
    # POincaré to Klein 
    data_k = Poincare_to_Klein(data_p)

    # Compute the barycenter
    barycenter_k = Klein_baycenter(data_k)

    # Barycenter to Poincaré
    barycenter_p = Klein_to_Poincare(np.array([barycenter_k]))[0]

    ## Get covariance matrix in tangeant space centred on the barycenter
    # Project all the points in the tangent space
    data_t = [Logarithmic_map(barycenter_p, x) for x in data_p]
    data_t = np.array(data_t)

    # Remove row where there is a nan of inf value (Points at radius r=1 on poincaré disk)
    data_t = data_t[~np.isnan(data_t).any(axis=1)]
    data_t = data_t[~np.isinf(data_t).any(axis=1)]

    # Compute the covariance matrix
    cov_matrix = np.cov(data_t, rowvar=False)  # rowvar=False to treat each column as a variable

    return barycenter_p, cov_matrix


def poincare_distance(x, y):
    """
    Calculate the hyperbolic distance between two points in the Poincaré disk.
    """
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    diff_norm = np.linalg.norm(x - y, axis=-1)
    return np.arccosh(1 + 2 * (diff_norm**2) / ((1 - norm_x**2) * (1 - norm_y**2)))

def rotationnal_alignement(emb_1, emb_2, n_angles=360):
    """
    Function to align two embeddings using rotation to minimize the hyperbolic distance.
    ARGUMENTS : 
        emb_1: np.array
            n*2 array of the first embedding, each row is the position of 1 node in the Poincaré disk.
        emb_2: np.array
            n*2 array of the second embedding, each row is the position of 1 node in the Poincaré disk.
    RETURN : 
        emb_2_aligned : np.array
            Aligned second embedding.
        optimal_angle : float
            Optimal rotation angle (in radians).
        min_error : float
            Minimum error (nanmean of hyperbolic distances).
    """
    def rotate(embedding, angle):
        """Apply a 2D rotation to the embedding."""
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        return np.dot(embedding, rotation_matrix.T)

    # Initialize variables
    angles = np.linspace(0, 2 * np.pi, n_angles)  # Test angles from 0 to 2π
    min_error = float('inf')
    optimal_angle = 0
    emb_2_aligned = emb_2

    # Iterate over angles to find the optimal rotation
    for angle in angles:
        rotated_emb_2 = rotate(emb_2, angle)
        distances = poincare_distance(emb_1, rotated_emb_2)
        error = np.nanmean(distances)  # Ignore NaN values
        if error < min_error:
            min_error = error
            optimal_angle = angle
            emb_2_aligned = rotated_emb_2

    return emb_2_aligned, optimal_angle, min_error

# Function that does the alignement but also give the score for all angles 
def rotationnal_alignement_all(emb_1, emb_2, n_angles=360):
    """
    Function to align two embeddings using rotation to minimize the hyperbolic distance.
    ARGUMENTS : 
        emb_1: np.array
            n*2 array of the first embedding, each row is the position of 1 node in the Poincaré disk.
        emb_2: np.array
            n*2 array of the second embedding, each row is the position of 1 node in the Poincaré disk.
        n_angles: int
            Number of angles to test for rotation.
    RETURN : 
        emb_2_aligned : np.array
            Aligned second embedding.
        optimal_angle : float
            Optimal rotation angle (in radians).
        min_error : float
            Minimum error (nanmean of hyperbolic distances).
    """
    def rotate(embedding, angle):
        """Apply a 2D rotation to the embedding."""
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        return np.dot(embedding, rotation_matrix.T)

    # Initialize variables
    angles = np.linspace(0, 2 * np.pi, n_angles)  # Test angles from 0 to 2π
    min_error = float('inf')
    optimal_angle = 0
    emb_2_aligned = emb_2

    # Initialize an array to store errors for each angle
    errors = []

    # Iterate over angles to find the optimal rotation
    for angle in angles:
        rotated_emb_2 = rotate(emb_2, angle)
        distances = poincare_distance(emb_1, rotated_emb_2)
        error = np.nanmean(distances)  # Ignore NaN values
        errors.append(error)  # Store error for this angle
        if error < min_error:
            min_error = error
            optimal_angle = angle
            emb_2_aligned = rotated_emb_2

    return emb_2_aligned, optimal_angle, min_error, errors, angles 

## MAIN 
if __name__ =="__main__" :
    
    pass
