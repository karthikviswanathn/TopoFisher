#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 09:15:31 2023

@author: karthikviswanathan
"""
import tensorflow as tf
import numpy as np

class FisherSimulator:
    def __init__(self, name):
        """
        Initialize a FisherSimulator object.
        
        Parameters:
        ----------
        name : str
            Name of the simulator.
        """
        self.name = name
    
    def generateData(self, theta_fid, delta_theta, n_s, n_d, find_derivative,\
                     seed_cov = None, seed_ders = None):
        """
        Generate data for Fisher analysis to estimate the covariance and 
        derivative means.
        
        This method assumes that we use centered differentiation and 
        seed matching to estimate derivative means. This means we need two sets
        of simulations for each parameter for which the Fisher forecasts are 
        calculated.
        
        For example if theta_fid = [1, 0.2] and delta_theta = [0.1, 0.02], 
        the simulations are calculated for the list of thetas = 
        [[1, 0.2], [0.95, 0.2], [1.05, 0.2], [1, 0.19], [1, 0.21]]
        Parameters:
        ----------
        theta_fid : tf.Tensor
            The fiducial parameter values.
        delta_theta : tf.Tensor
            Parameter step sizes to estimate the derivative.
        n_s : int
            Number of simulations to estimate covariance.
        n_d : int
            Number of simulations to estimate the derivative mean.
        find_derivative : list
            Boolean list indicating which parameters to find derivatives with
            respect to. The Fisher forecasts are only calculated for those
            parameters for which the value is True.
        seed_cov : int, optional
            Seed for random number generation used to generate the simulations 
            for covariance estimation.
            The default is None.
        seed_ders : int, optional
            Seed for random number generation used to generate the simulations
            for derivative means estimation.
            The default is None.

        Returns:
        -------
        all_pts : list
            A list containing simulations used for Fisher analysis:
            - all_pts[0]: The simulations evaluated at theta_fid used to estimate covariance.
            - all_pts[1:]: The set of simulations used to estimate the derivative.
        """
        num_params = delta_theta.shape[0]
        
        if seed_cov is None:
            seed_cov = np.random.randint(1e6)
        if seed_ders is None:
            seed_ders = np.random.randint(1e6, size=num_params)
        if len(seed_ders) != num_params:
            print("Check length of seed_ders!")

        # Generate the list of parameters to run the simulations for estimating
        # the dereivative.
        diff_theta = tf.linalg.diag(delta_theta)
        thetaList = (theta_fid + tf.einsum("i,jk->kij", tf.constant([-1., 1.]),
                                           diff_theta / 2.))
        thetaList = tf.reshape(thetaList, (-1, num_params))
        
        # Simulate data for covariance estimation
        ptsCov = self.generateInstances(theta_fid, n_s, seed=seed_cov)
        
        # Repeat seeds and find_derivative for derivative estimation
        seed_ders = np.repeat(seed_ders, 2)
        find_derivative_rep = np.repeat(find_derivative, 2)
        
        # Simulate data for derivative estimation
        pts_derivatives = [
            self.generateInstances(theta, n_d, seed=seed_ders[idx])
            for idx, theta in enumerate(thetaList) if find_derivative_rep[idx]
        ]
        
        # Combine all simulation data
        all_pts = [ptsCov, *pts_derivatives]
        return all_pts
