#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 13:36:06 2023

@author: karthikviswanathan
"""

import tensorflow as tf

class VectorizationLayers_TF(tf.keras.layers.Layer):
    def __init__(self, vectorizations, pds_idx_list = None, 
                 data_format = 'batch_first', **kwargs):
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
            data_format (str, optional): The format of the input data. It can
                                         either be 'batch_first' or '
                                         hom_dim_first', which refers to the
                                         first dimension of the input.
                                         Defaults to 'batch_first'.
            name (str, optional): The name of the vectorization layer. 
                                  Defaults to "vecLayer".
        """
        tf.keras.layers.Layer.__init__(self, **kwargs)
        if pds_idx_list is None :
            pds_idx_list = [i for i in range(len(vectorizations))]
        if len(vectorizations) != len(pds_idx_list) and \
            pds_idx_list is not None: 
            raise ValueError(
        "Make sure that the vectorizations and the hom_dims are compatible.")
            
        self.vectorizations = vectorizations
        self.pds_idx_list = pds_idx_list
        self.data_format = data_format
        self.num_hom_dim = len(pds_idx_list)
        self.is_fitted = [False for idx in range(self.num_hom_dim)]
            
    def call(self, pds_all_hom_dims):
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
        for idx, veclayer in zip(self.pds_idx_list, self.vectorizations):
            if self.data_format == 'batch_first': 
                pds = pds_all_hom_dims[:, idx]  
            elif self.data_format == 'hom_dim_first' : 
                pds = pds_all_hom_dims[idx, :] 
            else : raise ValueError("Check self.data_format!")
            vecs.append(veclayer(pds))
        return self.post_process(tf.concat(vecs, axis = -1))
    
    def post_process(self, vecs):
        return vecs