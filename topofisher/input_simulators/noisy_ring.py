#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 20:08:55 2023

@author: karthikviswanathan
"""
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from . import input_simulator

tfd = tfp.distributions 

class CircleSimulator(input_simulator.FisherSimulator):
    def __init__(self, ncirc, nback, bgmAvg, name = 'circle'):
        """
        Initialize a CircleSimulator object to generate point clouds 
        corresponding to the noisy ring data.
        Example usage - 
        inp_sim = CircleSimulator(ncirc = 200, nback = 20, bgmAvg = 1)
        pts = inp_sim.generateData(theta_fid = tf.constant([1., 0.2]), \
                                   num_sims = 100)
        
        Parameters
        ----------
        nthet: Integer.
               The number of points drawn from the noisy ring in a point cloud.
        nback : Integer.
               The number of points drawn from the background in a point cloud.
        bgmAvg : Float tensor.
                 Specifies the mean distance to the center of the background
                 points.
        name : String, optional
              The default is 'circle'.
        """
        self.ncirc = ncirc
        self.nback = nback
        self.bgmAvg = bgmAvg
        self.ntot = ncirc + nback
        self.p = nback/self.ntot
        
        # self.dist is a lambda that takes in theta and outputs the 
        # distribution.
        # distribution.
        super().__init__(name)
       
    def dist(self, theta):
        """
        Generate the distribution given the input parameter.

        Parameters
        ----------
        theta : tf.Tensor
            The parameters to generate the distribution.

        Returns
        -------
        tfd.Mixture
            The generated distribution.

        """
        p = self.p
        bgmAvg = self.bgmAvg
        return tfd.Mixture(\
          cat=tfd.Categorical(probs=[p, 1 - p]),
          components=[
            tfd.Uniform(low=0, high = 2 * bgmAvg),
            tfd.Normal(loc=theta[0], scale=theta[1]
          ),
        ])

    def generateInstances(self, θ, num_sims, seed = None): 
        """
        Simulates 'num_sims' point clouds for a given fiducial parameter.

        Parameters
        ----------
        θ : tf.Tensor.
            The fiducial value for which the point cloud is simulated.
        num_sims : Integer.
            The number of point clouds to generate.
        seed : Integer, optional
            Used for seed matching to find the derivatives.

        Returns
        -------
        pts : A tf.Tensor of shape (num_sims, ncirc + nback, 2)
            The simulated point clouds. 
        """
        
        def radialToCartesian(rad, theta):
            circx = rad * tf.cos(theta); 
            circy = rad * tf.sin(theta);
            return tf.stack([circx, circy], axis = -1)
        
        if (seed != None) : tf.random.set_seed(seed)
        ntot = self.ntot
        rad = tfd.Sample(self.dist(θ), sample_shape = (num_sims, ntot)).sample()
        thetas = 2 * np.pi * tf.random.uniform((num_sims, ntot))
        pts = radialToCartesian(rad, thetas)
        return pts
    
    def TFM(self, θ, xr = tf.range(0, 2, 0.001)):
        """
        Calculates the theoretical Fisher information matrix by integrating
        the correlation of the score function.

        Parameters
        ----------
        θ : The fiducial value for which the theoretical fisher matrix is 
            calcualted.
        xr : tf.Tensor. The range to integrate the score function over.
             The default is tf.range(0, 2, 0.001).

        Returns
        -------
        tf.Tensor of shape [len(θ), len(θ)].
            The theoretical Fisher information matrix.

        """
        A = tf.eye(2)
        lis = []
        for item in A:
            with tf.autodiff.ForwardAccumulator(θ , tf.constant(item)) as acc:
                y = self.dist(θ).log_prob(xr)
            lis.append(acc.jvp(y))
        # TODO : Check whether it is required to close the forward accumulator.
        score = tf.transpose(tf.stack(lis))
        probs = tf.exp(y)/tf.reduce_sum(tf.exp(y))
        weighted_score = score * tf.expand_dims(probs, axis = 1)
        return tf.matmul(tf.transpose(score), weighted_score) * self.ntot
    

    def sortedDistSummary(self, allPts):
        """
        Find the distance of every point in a point cloud to the center and 
        sorts them in ascending order. Hence every point cloud will be 
        represented by 'n_tot' numbers that are sorted.  

        Parameters:
        ----------
        allPts : list
            A list of point clouds.

        Returns:
        -------
        list
            A list of sorted distances.
        """
        return [tf.sort(tf.linalg.norm(data, axis = -1), axis = -1) for \
                data in allPts]
    
    def meanDistSummary(self, allPts):
        """
        Finds the mean and variance of the distances to the center for each
        point cloud in the noisy ring data. Each opint cloud is hence 
        represented by two numbers.

        Parameters:
        ----------
        allPts : list
            A list of point clouds.

        Returns:
        -------
        list
            A list of mean and standard deviation values for distances.
        """
        def mean_and_sigma(pts):
            rads = tf.linalg.norm(pts, axis = -1)
            means = tf.math.reduce_mean(rads, axis = -1)
            sig = tf.sqrt(tf.math.reduce_variance(rads, axis = -1))
            return tf.stack([means, sig], axis = -1)
        return [mean_and_sigma(data) for data in allPts]
    

