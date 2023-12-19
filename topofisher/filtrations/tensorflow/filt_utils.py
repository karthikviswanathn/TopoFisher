#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:02:54 2023

@author: karthikviswanathan
"""
import tensorflow as tf

def stack_ragged(tensors, tensorize):
    if tensorize : tensors = [tf.constant(item) for item in tensors]
    values = tf.concat(tensors, axis = 0)
    lens = tf.stack([tf.shape(t, out_type=tf.int64)[0] for t in tensors])
    return tf.RaggedTensor.from_row_lengths(values, lens)

def transpose_dgms(all_dgms):
    """
    input shape = num_theta, num_hom_dims, num_sims, None, 2
    output_shape = num_sims, num_theta, num_hom_dims, None, 2

    Parameters
    ----------
    all_dgms : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    dgms_transpose = [list(map(list, zip(*dgms))) for dgms in all_dgms]
    return list(map(list, zip(*dgms_transpose)))
    
def list_to_ragged_tensor(input_dgms, num_theta, num_hom_dim, \
                          transpose = True, tensorize = True):
    """
    input shape = num_theta, num_hom_dims, num_sims, None, 2

    Parameters
    ----------
    input_dgms : TYPE
        DESCRIPTION.
    num_theta : TYPE
        DESCRIPTION.
    num_hom_dim : TYPE
        DESCRIPTION.
    transpose : TYPE, optional
        DESCRIPTION. The default is True.
    tensorize : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    ragged_dgms : TYPE
        DESCRIPTION.

    """
    # all_pds.shape = num_sims, num_theta, num_hom_dims, None, 2
    if transpose : all_pds = transpose_dgms(input_dgms) 
    all_pds_list = []
    print(len(all_pds))
    for pds in all_pds: # Iterating over simulations
        print(len(pds))
        for dgms in pds: # Iterating over num_theta
            print(len(dgms))
            ragged_dgm = stack_ragged(dgms, tensorize)
            all_pds_list.append(ragged_dgm)
    stacked = tf.concat(all_pds_list, axis = 0)
    t1 = tf.RaggedTensor.from_uniform_row_length(stacked, num_hom_dim)
    ragged_dgms = tf.RaggedTensor.from_uniform_row_length(t1, num_theta)
    return ragged_dgms

class ExtraDimFiltLayer(tf.keras.layers.Layer):
    def __init__(self, inp_layer, **kwargs):
        """
        Initialize an ExtraDimLayer.

        Parameters
        ----------
        inp_layer : tf.keras.layers.Layer
            Input layer.
        """
        super(ExtraDimFiltLayer, self).__init__(**kwargs)
        self.inp_layer = inp_layer
    def call(self, inputs):
        """
        Call method for the ExtraDimLayer.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.

        Returns
        -------
        output : tf.Tensor
            Output tensor.
        """
        lis = []
        for idx in range(inputs.shape[1]):
            op = self.inp_layer(inputs[:, idx])
            lis.append(op)
        num_hom_dim = lis[0].shape[0]
        all_pds_list = [[] for _ in range(num_hom_dim)]
        for pds in lis:
            for idx, dgms in enumerate(pds): 
                all_pds_list[idx].append(dgms)
        stacked_list = []
        for item in all_pds_list : stacked_list.extend(item)
        stacked = tf.concat(stacked_list, axis = 0)
        t1 = tf.RaggedTensor.from_uniform_row_length(stacked, inputs.shape[0])
        ragged_dgms = tf.RaggedTensor.from_uniform_row_length(t1, \
                                                              inputs.shape[1])
        return ragged_dgms