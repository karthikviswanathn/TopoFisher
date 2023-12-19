#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 14:53:48 2023

@author: karthikviswanathan
"""

import numpy as np
import gudhi
from . import filtration_layers
from gudhi.point_cloud.dtm import DistanceToMeasure

class AlphaDTMLayer(filtration_layers.FiltrationLayer):
    def __init__(self, max_hom_dim, hom_dim_list, m = None, num_nn = None, 
                 implementation = "sklearn", edge_filt = "mixDTMandL", \
                 name = "AlphaDTM", show_tqdm = False):
        """
        Initialize an AlphaDTMLayer for computing persistent homology using 
        AlphaDTM filtration.

        Parameters
        ----------        
        max_hom_dim : int
            The homological dimension upto which to compute persistent homology.
        hom_dim_list : list 
            The list of homological dimensions to find the persistence
            diagrams for. 
        m : float, optional
            The fraction of nearest neighbors to consider for the AlphaDTM 
            filtration. Default is None
        num_nn : int, optional
                 The number of nearest neihbors to consider for the AlphaDTM 
                 filtration. If m is not provided, then this number is used.
                 Default is None.
        implementation : str, optional
            The method used to compute the DTM values. 
            Default is "keops".
            The possible options now are 
            1. keops - Constructs the full distance matrix and 
            2. sklearn - Uses sklearn to compute the k-nearest neighbors.
        edge_filt : str, optional
            The method for computing edge filtration values. 
            Default is "mixDTMandL".
        name : str, optional
            The name of the filtration layer. Default is "AlphaDTM".
        show_tqdm : bool, optional
            A flag indicating whether to display a progress bar using tqdm. 
            Default is False.
        """
        if m is None and num_nn is None : 
            raise ValueError("Either provide 'm' or 'num_nn'!")
        if m is not None and num_nn is not None:
            raise ValueError("Provide either 'm' or 'num_nn'!")
        self.num_nn = num_nn
        self.m = m    
        self.edge_filt = edge_filt
        self.implementation = implementation
        
        super().__init__(max_hom_dim, hom_dim_list, name, show_tqdm)
        
    def get_num_nn(self, N):
        """
        Finds the number of nearest neighbors to find the DTM.

        Parameters
        ----------
        N : int
            Total number of points in the point cloud.

        Returns
        -------
        int
            The number of nearest neighbors to find DTM.

        """
        return int(self.m * N) if self.num_nn is None else self.num_nn
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
        edges = np.array([s[0] for s in st.get_skeleton(1) if len(s[0]) == 2])
        
        num_nn = self.get_num_nn(N = len(vert))
        
        # Finding vertex filtration values given by vfilt.
        vfilt = self.calculateDTMValues(ptCloud, num_nn) 
        
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
        st.expansion(self.max_hom_dim) # max_hom_dim = 2 in this case
        st.make_filtration_non_decreasing()
        st.persistence()
        return st
    
    def calculateDTMValues(self, ptCloud, num_nn):
        """
        Calculate DTM (Distance to Measure) values for a given point cloud.
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
        dtm = DistanceToMeasure(num_nn, implementation = self.implementation)
        return dtm.fit_transform(ptCloud)
