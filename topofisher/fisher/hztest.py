#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 17:11:04 2023

@author: karthikviswanathan
"""
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

def lognorm_sf_tf(hz, psi, pmu):
    z = (tf.math.log(tf.cast(hz, pmu.dtype)) - pmu) / psi
    ccdf = 0.5 * (1.0 + tf.math.erf(z / np.sqrt(2)))
    sf = 1.0 - ccdf
    return sf

def multivariate_normality(X):
    # Check input and remove missing values
    # X = tf.convert_to_tensor(X, dtype=tf.float64)
    assert X.shape.rank == 2, "X must be of shape (n_samples, n_features)."
    X = X[~tf.reduce_any(tf.math.is_nan(X), axis=1)]
    n, p = tf.shape(X)[0], tf.shape(X)[1]

    # Covariance matrix
    S = tfp.stats.covariance(X)
    S_inv = tf.linalg.pinv(S)  
    difT = X - tf.reduce_mean(X, axis=0)
    # Squared-Mahalanobis distances
    Dj = tf.linalg.diag_part(tf.matmul(tf.matmul(difT, S_inv), tf.linalg.matrix_transpose(difT)))
    Y = tf.matmul(tf.matmul(X, S_inv), tf.linalg.matrix_transpose(X))
    
    diag_Y_T = tf.linalg.diag_part(tf.linalg.matrix_transpose(Y))
    # tiled_diag_Y_T = tf.tile(diag_Y_T[tf.newaxis, :], [n, 1])
    n = tf.shape(Y)[0]
    tiled_diag_Y_T = tf.tile(tf.expand_dims(diag_Y_T, axis=0), [tf.shape(Y)[0], 1])
    Djk = -2 * tf.transpose(Y)  + tf.reshape(tf.repeat(tf.linalg.diag_part(Y), n), (n , n)) + tiled_diag_Y_T 

    # Smoothing parameter
    # b = 1 / (np.sqrt(2)) * ((2 * p + 1) / 4) ** (1 / (p + 4)) * (n ** (1 / (p + 4)))
    # b = 1 / (np.sqrt(2)) * tf.math.pow(((2 * p + 1) / 4), (1 / (p + 4))) * tf.math.pow(n , (1 / (p + 4)))
    b = tf.pow(tf.cast(n, tf.float32), 1 / tf.cast((p + 4), dtype=tf.float32))
    
    b = b *  tf.cast(1 / (np.sqrt(2)) * tf.math.pow(((2 * p + 1) / 4), (1 / (p + 4))), tf.float32)
    # b =  1 / tf.sqrt(2) * tf.pow(((2 * p + 1.) / 4.), 1. / tf.cast((p + 4), dtype=tf.float32)) * tf.pow(tf.cast(n, tf.float32), 1 / tf.cast((p + 4), dtype=tf.float32))
    # Is matrix full-rank (columns are linearly independent)?
    if tf.linalg.matrix_rank(S) == p:
        n = tf.cast(n, tf.float32)
        p = tf.cast(p, tf.float32)
        hz = (
            1. / (n**2) * tf.reduce_sum(tf.reduce_sum(tf.exp(-(b**2) / 2. * Djk)))
            - 2.
            * ((1. + (b**2)) ** (-p / 2.))
            * (1. / n)
            * (tf.reduce_sum(tf.exp(-((b**2) / (2 * (1 + (b**2)))) * Dj)))
            + ((1. + (2 * (b**2))) ** (-p / 2))
        )
        
        hz = n * hz
    else:
        
        n = tf.cast(n, tf.float32)
        p = tf.cast(p, tf.float32)
        hz = n * 4.
    wb = (1 + b**2) * (1 + 3 * b**2)
    a = 1 + 2 * b**2
    # Mean and variance
    mu = 1 - a ** (-p / 2) * (1 + p * b**2 / a + (p * (p + 2) * (b**4)) / (2 * a**2))
    si2 = (
        2 * (1 + 4 * b**2) ** (-p / 2)
        + 2
        * a ** (-p)
        * (1 + (2 * p * b**4) / a**2 + (3 * p * (p + 2) * b**8) / (4 * a**4))
        - 4
        * wb ** (-p / 2)
        * (1 + (3 * p * b**4) / (2 * wb) + (p * (p + 2) * b**8) / (2 * wb**2))
    )

    # Lognormal mean and variance
    pmu = tf.math.log(tf.sqrt(mu**4 / (si2 + mu**2)))
    psi = tf.sqrt(tf.math.log1p(si2 / mu**2))  
    pval = lognorm_sf_tf(hz, psi, pmu)

    return tf.cast(pval, dtype = tf.float32)