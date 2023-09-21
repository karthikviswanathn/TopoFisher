#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 14:53:48 2023

@author: karthikviswanathan
"""

import numpy as np
import gudhi
from . import filtration_layers

class AlphaDTMLayer(filtration_layers.FiltrationLayer):
    def __init__(self, m, edge_filt = "mixDTMandL", hom_dim = 2, \
                 name = "AlphaDTM", show_tqdm = False):
        """
        Initialize an AlphaDTMLayer for computing persistent homology using 
        AlphaDTM filtration.

        Parameters
        ----------
        m : float
            The fraction of nearest neighbors to consider for the AlphaDTM 
            filtration.
        edge_filt : str, optional
            The method for computing edge filtration values. 
            Default is "mixDTMandL".
        hom_dim : int, optional
            The homological dimension for which to compute persistent homology. 
            Default is 2.
        name : str, optional
            The name of the filtration layer. Default is "AlphaDTM".
        show_tqdm : bool, optional
            A flag indicating whether to display a progress bar using tqdm. 
            Default is False.
        """
        self.m = m    
        self.edge_filt = edge_filt
        super().__init__(hom_dim, name, show_tqdm)
        
    """
        Returns the simplex tree for a given point cloud (ptCloud). The simplex tree
        corresponds to an alphaDTM filtration for a given value of 'm'. This uses
        p = 2, q = 1 from equation (2.1) and (2.2) in arxiv.org/pdf/2203.08262.pdf
        
    """
    def get_st(self, ptCloud):
        """
        Compute a simplex tree (ST) for a given point cloud (ptCloud) using 
        AlphaDTM filtration for a given value of 'm'. This uses
        p = 2, q = 1 from equation (2.1) and (2.2) in 
        arxiv.org/pdf/2203.08262.pdf.

        Parameters
        ----------
        ptCloud : numpy array
            The input point cloud.

        Raises
        ------
        ValueError
            If the edge filtration is not implemented.

        Returns
        -------
        st : gudhi.SimplexTree
            A SimplexTree object representing the AlphaDTM filtration.
        """
        alpha = gudhi.AlphaComplex(points = ptCloud) # creating an AlphaComplex
        st = alpha.create_simplex_tree() 
        
      
        # Collecting the edges and vertices of the alphaComplex to update their
        # filtration values using AlphaDTM.
        
        vert = np.array([s[0] for s in st.get_skeleton(0)])
        edges = np.array([s[0] for s in st.get_skeleton(1)\
                          if len(s[0]) == 2])
        
        num_nn = int(self.m * len(vert))
        
        # Finding vertex filtration values given by vfilt.
        vfilt = calculateDTMValues(ptCloud, num_nn) 
        
        # Finding the edge filtration values given by efilt.
        inda, indb = edges[:,0], edges[:,1]
        # d = length of the edges in the simplex tree.
        d = np.linalg.norm(ptCloud[inda] - ptCloud[indb], axis = -1)
        fx, fy = vfilt[inda], vfilt[indb]
        fmax = np.maximum(fx, fy)
        
        if(self.edge_filt == "mixDTMandL") : efilt = d + fmax
        else : raise ValueError("The pipeline doesn't support other edge \
                                filtration methods.")
        
        # Updating the simplex with the alphaDTM filtration values for vertices
        # and edges. 
        st = gudhi.SimplexTree()
        st.insert_batch(vert.T, vfilt.reshape(-1))
        st.insert_batch(edges.T, efilt)
        st.expansion(self.hom_dim) # hom_dim = 2 in this case
        st.make_filtration_non_decreasing()
        st.persistence()
        return st
    
def calculateDTMValues(ptCloud, num_nn):
    """
    Calculate DTM (Distance to Measure) values for a given point cloud.
    Here, DX is the euclidean distance matrix. To find the DTM values, we
    1. Take the 'num_nn' closest neighbors from the distance matrix.
    2. Find the "square root mean" (L2 norm) of these distances. 
    The output of this is vfilt = vertex filtration values of the simplex 
    tree.
    Parameters:
    ----------
    ptCloud : numpy array
        The input point cloud.
    num_nn : int
        The number of nearest neighbors to consider for DTM computation.

    Returns:
    -------
    vfilt : numpy array
        An array of vertex filtration values for the point cloud.
    """
    DX = np.linalg.norm(np.expand_dims(ptCloud, axis = -3) - \
                        np.expand_dims(ptCloud, axis = -2), axis = -1)
    DX = np.sort(DX, axis = -1)[:, :num_nn]
    vfilt = np.sqrt(np.mean(DX*DX, axis = -1))
    return vfilt
