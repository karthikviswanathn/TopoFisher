#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 13:36:06 2023

@author: karthikviswanathan
"""

import numpy as np

class VectorizationLayers:
    def __init__(self, vectorizations, hom_dims, name = "vecLayer"):
        """
        Initialize the VectorizationLayers.

        Parameters:
            vectorizations (list): List of vectorization methods for each 
                                   homological dimension present in hom_dims.
            hom_dims (list): List of homological dimensions.
            name (str, optional): The name of the vectorization layer. 
                                  Defaults to "vecLayer".
        """
        if len(vectorizations) != len(hom_dims): 
            raise ValueError(
        "Make sure that the vectorizations and the hom_dims are compatible.")
        self.vectorizations = vectorizations
        self.hom_dims = hom_dims
        self.num_hom_dim = len(hom_dims)
        self.name = name
        self.is_fitted = [False for idx in range(self.num_hom_dim)]
        
    def get_persistence_diagrams(self, sts, hom_dim):
        """
        Get persistence diagrams from a list of simplex trees.

        Parameters:
            sts (list): List of simplex trees.
            hom_dim (int): Homological dimension.

        Returns:
            list: List of persistence diagrams.
        """
        pds = []
        for st in sts :
            pd = st.persistence_intervals_in_dimension(hom_dim)
            pds.append(getFinitePairs(pd))
        return pds
    
    def vectorize_simplex_trees(self, sts):
        """
        Vectorize a list of simplex trees containing the persistence diagrams.
        
        Parameters:
            sts (list): List of simplex trees.
        
        Returns:
            numpy array: Vectorized data from the persistent diagrams 
            corresponding to the homology dimensions in 'hom_dims'.
        """
        hom_dims = self.hom_dims
        vecs = []
        for idx in range(self.num_hom_dim):
            hom_dim = hom_dims[idx]
            veclayer = self.vectorizations[idx]
            pds = self.get_persistence_diagrams(sts, hom_dim)
            if(self.is_fitted[idx] == False) :
                veclayer.fit(pds)
                self.is_fitted[idx] = True
            vecs.append(veclayer.transform(pds))
        return self.post_process(np.concatenate(vecs, axis = -1))
    
    def post_process(self, vecs):
        """
        Perform post-processing on vectorized data. This method can be 
        overwritten in case some post processing is necessary by instances that
        inherit this class. For example, in the persistent image vectorization
        layers, post-processing is done by reshaping a vector into
        a collection of images, i.e. a vector of length l is reshaped into 
        (res, res, num_hom_dims) to view them as images.

        Parameters:
            vecs (numpy array): Vectorized data.

        Returns:
            numpy array: Post-processed vectorized data.
        """
        return vecs
           
def getFinitePairs(diag):
    """
    Get finite pairs from a persistence diagram.

    Parameters:
        diag (numpy array): Persistence diagram.

    Returns:
        numpy array: Finite pairs from the persistence diagram.

    """
    return diag[diag[:,1] < np.inf]