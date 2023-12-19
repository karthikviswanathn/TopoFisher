#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 12:20:28 2023

@author: karthikviswanathan
"""

import tensorflow as tf
from gudhi.point_cloud.knn import KNearestNeighbors
from . import flag_layer


class DTMFiltLayer(flag_layer.FlagComplexLayer):
    def __init__(self, num_nn, autodiff = True, **kwargs):
        flag_layer.FlagComplexLayer.__init__(self, **kwargs)
        self.num_nn = num_nn
        self.autodiff = autodiff
        
    def find_knn_matrices(self, inputs):
        all_dists = [] 
        for pc in inputs:
            knn = KNearestNeighbors(self.num_nn, return_index = False, \
                                    return_distance = True, \
                                    enable_autodiff = self.autodiff)
            dists = knn.fit_transform(pc)
            all_dists.append(dists[:, 1:])
        return tf.stack(all_dists)
    
    def constructVfiltFromKNN(self, knn_matrices):
        return tf.math.sqrt(
            tf.math.reduce_sum(knn_matrices ** 2, axis = -1)/self.num_nn)
        
    def vFilt(self, inputs):
        knn_matrices = self.find_knn_matrices(inputs)
        return self.constructVfiltFromKNN(knn_matrices)

class NNFiltLayer(DTMFiltLayer):
    def __init__(self, phi, **kwargs):
        DTMFiltLayer.__init__(self, **kwargs)
        self.phi= phi
    @tf.function
    def constructVfiltFromKNN(self, knn_matrices):
        return tf.squeeze(self.phi(knn_matrices), -1)