#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 13:36:06 2023

@author: karthikviswanathan
"""

import numpy as np

class VectorizationLayers:
    def __init__(self, vectorizations, pds_idx_list = None, name = "vecLayer"):
        """
        Initialize the VectorizationLayers.

        Parameters:
            vectorizations (list): List of vectorization methods for each 
                                   index present in pds_idx_list.
            pds_idx_list (list, optional): List of indices of persistence 
                                          diagrams to vectorize. They generally
                                          stand for the hom_dims. If None, 
                                          vectorize all the persistence 
                                          diagrams.
                                          Defaults to None.
            name (str, optional): The name of the vectorization layer. 
                                  Defaults to "vecLayer".
        """
        if pds_idx_list is None :
            pds_idx_list = [i for i in range(len(vectorizations))]
        if len(vectorizations) != len(pds_idx_list) and \
            pds_idx_list is not None: 
            raise ValueError(
        "Make sure that the vectorizations and the hom_dims are compatible.")
            
        self.vectorizations = vectorizations
        self.pds_idx_list = pds_idx_list
        self.num_hom_dim = len(pds_idx_list)
        self.name = name
        self.is_fitted = [False for idx in range(self.num_hom_dim)]
            
    def vectorize_persistence_diagrams(self, pds_all_hom_dims):
        """
        Vectorize a list of persistence diagrams
        
        Parameters:
            pds_all_hom_dims(list of list): List of persistence diagrams for 
                                            all the homology dimensions.
        
        Returns:
            numpy array: Vectorized data from the persistent diagrams 
            corresponding to the homology dimensions in 'pds_idx_list'.
        """
        vecs = []
        for idx, veclayer in enumerate(self.vectorizations):
            pd_idx = self.pds_idx_list[idx]
            pds = pds_all_hom_dims[pd_idx]
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
           
