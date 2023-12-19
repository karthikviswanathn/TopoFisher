#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 20:23:33 2023

@author: karthikviswanathan
"""
import tensorflow as tf
import numpy as np
class TOPK_TF(tf.keras.layers.Layer):
    """
    Calculate top-k vectors from persistence diagrams.
    """
    def __init__(self, bdp_type, is_binned, topk=np.nan, reduce_frac=0.9, 
                  num_bins=np.nan, pad_value=0., **kwargs):
        """
        Initialize the TOPK class.

        Parameters:
            bdp_type (str): Types of persistence values to include
            (e.g., 'bdp' stands for birth, death and persistence).
            is_binned (bool): Whether to bin the values.
            topk (int, optional): The top-k value. Defaults to np.nan.
            reduce_frac (float, optional): The reduction fraction for 
            calculating top-k. Defaults to 0.9.
            num_bins (int, optional): The number of bins. Defaults to np.nan.
            pad_value (float, optional): The value used for padding.
                                         Defaults to 0.
        """
        tf.keras.layers.Layer.__init__(self, **kwargs)
        self.bdp_type = bdp_type
        self.is_binned = is_binned
        self.topk = topk
        self.reduce_frac = reduce_frac
        self.num_bins = num_bins
        self.pad_value = pad_value
        self._check_parameters()
        
    def _check_parameters(self):
        if(self.bdp_type not in ["b", "d", "p", "bd", "bp", "dp", "bdp"]):
            raise ValueError("Check bdp_tpye. Incorrect results are possible.")
        if(self.is_binned and self.num_bins == np.nan):
            raise ValueError("How many bins? Check inputs.")
            
    def _fit_topk_from_pds(self, inputs):
        self.topk = int(self.reduce_frac * np.min([item.shape[0] \
                                                  for item in inputs.numpy()]))
    def _choose_topk(self, ragged_tensor):
        tens = ragged_tensor.to_tensor(default_value = -np.inf)
        value, indices = tf.math.top_k(tens, self.topk)
        return tf.gather(ragged_tensor, indices, batch_dims = -1)

    def _bin_tensor(self, tensor):
        binx = self.num_bins
        div = self.topk // binx * binx
        arr = tf.reshape(tensor[:, :div], (tensor.shape[0], binx, \
                                           self.topk // binx))
        return tf.math.reduce_sum(arr, axis = -1)
    
    def call(self, inputs):
        if(np.isnan(self.topk)) : self._fit_topk_from_pds(inputs)
        birth, death = inputs[:, :, 0], inputs[:, :, 1]
        pers = death - birth
        topk_bdp = [self._choose_topk(item) for item in [birth, death, pers]]
        
        bdp_dic = {}
        for typ, arr in zip(["b", "d", "p"], topk_bdp):
            bdp_dic[typ] = arr
            
        topk_vecs = []
        # Fetching some or all of "bdp" from the bdp_dic.  
        for letter in self.bdp_type:
            vec = bdp_dic[letter]
            # Binning them if necessary.
            if(self.is_binned) : 
                vec = self._bin_tensor(vec)
            topk_vecs.append(vec)
        return tf.concat(topk_vecs, axis = -1)