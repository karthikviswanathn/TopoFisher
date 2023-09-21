#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 11:23:14 2023

@author: karthikviswanathan
"""

import tensorflow as tf 
import tensorflow_probability as tfp
import sklearn
import numpy as np
from sklearn.feature_selection import VarianceThreshold

class baseFisher:
    """
        In this class, we do the Fisher analysis given the summaries at the
        fiducial values and the derivatives. 
    """
    def __init__(self, vecs_cov, derivatives, name = None, clean_data = True):
        """
        Initialize the base Fisher class.

        Parameters
        ----------
        vecs_cov : tf.Tensor of shape (num_sims, num_summaries)
            The summaries used to estimate the covariance.
        derivatives : tf.Tensor of shape (num_theta, num_sims, num_summaries)
            The calculated derivatives.
        name : str, optional
            Name for the Fisher class. The default is None.
        clean_data : bool, optional
            Flag to clean data. By 'clean', we remove the features that have 
            zero variance. This is to avoid non-invertible covariance matrices.
            The default is True.
        """
        if(clean_data) : 
            vecs_cov, derivatives = cleanData(vecs_cov, derivatives)
        self.name = name
        self.vecs_cov = vecs_cov
        self.derivatives = derivatives
        self.C = tfp.stats.covariance(vecs_cov)
        self.invC = tf.linalg.inv(self.C)  # n_summaries * n_summaries
        self.ders = tf.math.reduce_mean(derivatives, axis = 1)
        self.compmat = tf.linalg.solve(self.C, tf.transpose(self.ders))
        self.FMμ = tf.matmul(self.ders, self.compmat) 
        self.FM = self.FMμ # because final summary stat is compressed
        self.invFM = tf.linalg.inv(self.FM) # a.k.a fisher forecast
        self.lnDetF = self.logdet() # for the loss function.
        self.FI = tf.math.exp(self.lnDetF) 
        self.err = self.find_err()
        fm_casted = tf.cast(self.FM, dtype = self.err.dtype)
        self.forecast_bias = tf.linalg.solve(fm_casted, 
                            tf.transpose(tf.linalg.solve(fm_casted, self.err)))
        self.fractional_bias = tf.linalg.tensor_diag_part(
            self.forecast_bias/tf.cast(self.invFM, self.forecast_bias.dtype))
        
    def find_err(self):
        """
        Finds the fisher bias error based on 
        https://github.com/wcoulton/CompressedFisher.

        Returns
        -------
        bias : tf.Tensor
            The fisher bias error.

        """
        derivatives = self.derivatives
        invC = self.invC
        n_params = derivatives.shape[0]
        bias = tf.zeros((n_params, n_params), dtype=tf.float64)
        
        for i in range(n_params):
            for j in range(n_params):
                cov_ij = tfp.stats.covariance(derivatives[i], \
                                              derivatives[j])
                trace_result = tf.linalg.trace(tf.linalg.matmul(cov_ij, \
                                                                invC))
                bias = tf.tensor_scatter_nd_add(bias, indices=[[i, j]], \
                                                updates=[trace_result])
        
        return bias
        
    def logdet(self):
        """
        Calculates the signed log determinant of the Fisher matrix.
        
        Returns
        -------
        tf.Tensor
            The signed log determinant.

        """
        matrix = self.FM
        lndet = tf.linalg.slogdet(matrix)
        return lndet[0] * lndet[1] 
        

                
class fisherFromVecs(baseFisher):
    """
    Fisher analysis given the summaries calculated at 
    [theta_fid, ..., (theta_fid - delta_theta/2)_i, \
     (theta_fid + delta_theta/2)_i, ...]. 
    The first set of simulations is used to estimate the covariance 
    matrix and the rest are used to estimate the derivative mean.
    """
    def __init__(self, summaries, delta_theta, name = "fromVecs", \
                 clean_data = True):
        """
        Initialize a Fisher class based on summaries.
        
        Parameters
        ----------
        summaries : list of tf.Tensors
            The summaries calculated at 
            [theta_fid, ..., (theta_fid - delta_theta/2)_i, \
             (theta_fid + delta_theta/2)_i, ...]. The signature of this method 
            should match with the generateData() method in the input simulator.
            The first set of simulations is used to estimate the covariance 
            matrix and the rest are used to estimate the derivative mean.
        delta_theta : tf.Tensor
            The step size that is used to calculate the derivatives.
        name : str, optional
            Name for the Fisher class. The default is None.
        clean_data : bool, optional
            Flag to clean data. If True, removes columns with zero variance.
            The default is True.
        """

        self.summaries = summaries
        self.delta_theta = delta_theta
        summaries = self.summaries
        self.n_s = summaries[0].shape[0] 
        self.n_d = summaries[1].shape[0]
        der_vecs = summaries[1:]       
        diffs = computeDerivatives(der_vecs, delta_theta)
        super().__init__(summaries[0], diffs, name, clean_data)
    
   
class fisherMOPED(fisherFromVecs):
    """
    Fisher analysis after compression using MOPED given the summaries 
    calculated at 
    [theta_fid, ..., (theta_fid - delta_theta/2)_i, \
     (theta_fid + delta_theta/2)_i, ...]. One half of the simulations is used 
    to estimate the MOPED compression matrix and the other half is used to 
    carry out the Fisher analysis of the compressed summaries. 
    """
    def __init__(self, summaries, delta_theta, name = "MOPED", \
                 clean_data = True):
        """
        Initialize a MOPED Fisher class.
        
        Parameters
        ----------
        summaries : tf.Tensor
            Summaries for constructing the compression matrix (training) and
            for the Fisher analysis (testing).
        delta_theta : tf.Tensor
            The step size that is used to calculate the derivatives.
        name : str, optional
            Name for the Fisher class. The default is None.
        clean_data : bool, optional
            Flag to clean data. If True, removes columns with zero variance.
            The default is True.
        """
        n_s = summaries.shape[1]
        train_vecs, test_vecs = summaries[: , :n_s//2], summaries[:, n_s//2:]
        vecs_cov, der_vecs = train_vecs[0], tf.stack(train_vecs[1:])
        derivatives = computeDerivatives(der_vecs, delta_theta)
        ders = tf.math.reduce_mean(derivatives, axis = 1)
        moped_covariance = tfp.stats.covariance(vecs_cov)
        self.moped_compmat = \
            tf.linalg.solve(moped_covariance, tf.transpose(ders))
        compressed_vecs = tf.matmul(test_vecs, self.moped_compmat)
        super().__init__(compressed_vecs, delta_theta, name, clean_data)

        
def computeDerivatives(der_vecs, delta_theta):
    """
    Compute derivatives based on centered differentiaton. 

    Parameters
    ----------
    der_vecs : tf.Tensor
        The summaries evaluated at [..., (theta_fid - delta_theta/2)_i, \
         (theta_fid + delta_theta/2)_i, ...].
    delta_theta : tf.Tensor
        The step sizes used to calculate the derivatives.

    Returns
    -------
    tf.Tensor
        The corresponding derivatives of shape 
        [num(delta_theta), num_sims, num_summaries].

    """
    delta_theta_cast = tf.cast(delta_theta[:, tf.newaxis, tf.newaxis], \
                               der_vecs.dtype)
    return (der_vecs[1::2] - der_vecs[::2])/delta_theta_cast      


def cleanData(vecs_cov, derivatives):
    """
    Removes the columns with zero variance from vecs_cov and derivatives.

    Parameters
    ----------
    vecs_cov : tf.Tensor
        The summaries used to estimate the covariance matrix.
    derivatives : tf.Tensor
        The summaries used to estimate the derivative mean.

    Returns
    -------
    tf.Tensor, tf.Tensor
        The vecs_cov and derivatives after removing zero variance columns.

    """
    selector = sklearn.feature_selection.VarianceThreshold()
    selector.fit(vecs_cov)
    selInd = selector.get_support()
    ind = np.where(selInd)[0]
    if(any(~selInd)) : print("Indices = ", np.where(~selInd)[0], 
                          " are removed because they have zero variance.")
    return tf.gather(vecs_cov, ind, axis = -1), \
        tf.gather(derivatives, ind, axis = -1)

def show_fm_and_bias(fisher):
    """
    Prints the fisher matrix and bias.

    Parameters
    ----------
    fisher : baseFisher
        The Fisher analysis object.
    """
    print("FM = ", np.round(fisher.FM.numpy().flatten(), 2))
    print("Fractional bias = ", \
          np.round(fisher.fractional_bias.numpy().flatten(), 2))