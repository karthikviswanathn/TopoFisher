#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 10:23:26 2023

@author: karthikviswanathan
"""

import tensorflow as tf
import numpy as np

class FisherPipeline:
    def __init__(self, inp_sim, n_s, n_d, \
                 theta_fid, delta_theta, \
                 filtLayer, vecLayer, fisherLayer, find_derivative):
        """
         Initialize a FisherPipeline.
    
         Parameters:
         ----------
         inp_sim : Object inheriting class FisherSimulator with the method 
                   generateData().
             An input simulator object responsible for generating simulated data.
         n_s : int
             Number of simulations to estimate covariance.
         n_d : int
             Number of simulations to estimate the derivative mean.
         theta_fid : tf.Tensor 
             Fiducial parameter values.
         delta_theta : tf.Tensor 
             Parameter step sizes to estimate the derivative mean. 
         filtLayer : Object inheriting class FiltrationLayer with the method
                     find_persistence_diagrams().
             Input - A list of data points from 'n' simulations.
             Output - A nested list with the first outer dimension correspoding
                     to the hom_dims and the second outer dimension 
                     corresponding to the 'n' simulations.                     
             A filtration layer that takes in a list of simulations,
             and outputs a list of corresponding persistence diagrams.
         vecLayer : Object inheriting class VectorizationLayer with the method
                    vectorize_persistence_diagrams().
             Input - A list of list of persistence diagrams
             Output - A list of 'n' persistent summaries.
             A vectorization layer that takes in a nested list of 
             persistence diagrams, and outputs a list of persistent summaries.
         fisherLayer : Object inheriting class FisherLayer with the method 
                        computeFisher().
             Input - A list of 'n' persistence summaries. 
             Output - A Fisher.baseFisher object.
             A fisher analysis layer that takes in a list of summaries, and 
             outputs an object that contains the Fisher matrix, Fisher forecast
             bias, etc..
         find_derivative : list
             Boolean list indicating which parameters to find derivatives with 
             respect to. The Fisher forecasts are only calculated for those
             parameters for which the value is True.
         """
        
        self.inp_sim = inp_sim
        self.n_s = n_s
        self.n_d = n_d

# =============================================================================
#         if(n_s != n_d) :
#             raise ValueError(
#                 "$n_s != n_d$. Currently the pipeline doesn't support this.")
# =============================================================================

        self.theta_fid = theta_fid

        self.filtLayer = filtLayer
        self.vecLayer = vecLayer
        self.fisherLayer = fisherLayer
        
        if(len(theta_fid) != len(delta_theta)) :
            raise ValueError("Check length of theta_fid and delta_theta")
        
        self.total_num_params = theta_fid.shape[0]
        self.set_delta_theta(delta_theta, find_derivative)

  
    def set_delta_theta(self, delta_theta, find_derivative):
        """
        Set the delta_theta and find_derivative attributes. 
        
        Parameters:
        ----------
        delta_theta : tf.Tensor
            Parameter step sizes.
        find_derivative : List
            Boolean array indicating which parameters to find derivatives 
            with respect to.
        """
        if(len(find_derivative) != len(delta_theta)) :
            raise ValueError("Check length of find_derivative and delta_theta")
        self.find_derivative = find_derivative
        self.delta_theta_all = delta_theta
        # Getting step sizes only for parameters that have find_derivative set
        # to True.
        delta_theta_list = [delta_theta[idx] 
                            for idx in range(self.total_num_params) \
                            if self.find_derivative[idx]]
        self.delta_theta = tf.convert_to_tensor(delta_theta_list)
        # Storing the number of parameters for which the Fisher forecast is
        # calculated for.
        self.fisher_num_params = len(self.delta_theta)  
        
    def generate_input_data(self, seed_cov = None, seed_ders = None):
        """
        Generate input data for Fisher analysis and store in the 'all_pts'. 
        'all_pts' is a list of length l = (1 + 2 * fisher_num_params)
        containing simulations to estimate the covariance and the derivative
        mean. We collect the covariance and derivative simulatons together and
        put them in a list because they undergo the same transformations 
        (like filtration and vectorization) until the Fisher analysis. 
        Refer to the documentation on InputSimulator for more details on the
        shape of all_pts. 
        Parameters 
        ----------
        seed_cov : int, optional
            The seed to generate samples for estimating 
            covariance.
        seed_ders : list of int of size len(theta_fid), optional. 
            The seeds to generate samples for estimating 
            derivate mean.
        """
        large_number = 1e10 # upper bound for random integer generation.
        
        # seed_cov and seed_ders are the seeds for seed matching. Setting them 
        # in case they are not provided them in the input.
        self.seed_cov = np.random.randint(large_number) if seed_cov is None \
                                                            else seed_cov
        self.seed_ders = \
            np.random.randint(large_number, size = len(self.theta_fid)) \
                            if seed_ders is None else seed_ders
                                
        # Generating and setting the 'all_pts'.
        self.all_pts = self.inp_sim.generateData(self.theta_fid, \
                        self.delta_theta_all, self.n_s, self.n_d, \
                        seed_cov = self.seed_cov, seed_ders = self.seed_ders,\
                            find_derivative = self.find_derivative)
     
    def filter_input_data(self):
        """
        Perform persistent homology computations on the input data stored in 
        the 'all_pts' and stores the corresponding persistence diagrams in 
        the 'all_persistence_diagrams'. The filtration procedure is  
        specifed by the 'filtLayer'.
        """
        self.all_persistence_diagrams = []; 
        for ptclouds in self.all_pts:
            pds = self.filtLayer.find_persistence_diagrams(ptclouds)
            self.all_persistence_diagrams.append(pds)
    
    def vectorize_all_persistence_diagrams(self):
        """
        Vectorizes the persistence diagrams from the 'all_persistence_diagrams' 
        and stores the persistent summaries in the 'all_vecs'. 
        The vectorization procedure specified by the 'vecLayer'.
        """
        self.all_vecs = []; 
        for pds in self.all_persistence_diagrams:
            vecs = self.vecLayer.vectorize_persistence_diagrams(pds)
            self.all_vecs.append(vecs)
    
    def fisher_analysis(self):
        """
        Performs the Fisher analysis on the 'all_vecs' according to 
        the specification given by the 'fisherLayer'.
        """
        self.fisher = self.fisherLayer.computeFisher(self.all_vecs, \
                                                     self.delta_theta)
        
    def run_pipeline(self, seed_cov = None, seed_ders = None):
        """
        Run the entire pipeline, including data generation, filtration, 
        vectorization, and Fisher analysis. If 'vecLayer' is None, the 
        vectorization is not performed. If 'fisherLayer' is None, the fisher
        analysis doesn't happen.
        
        Parameters
        ----------
        seed_cov : int, optional
            The seed to generate samples for estimating 
            covariance.
        seed_ders : list of int of size len(theta_fid), optional. 
            The seeds to generate samples for estimating 
            derivate mean.
    
        Raises
        ------
        ValueError
            Raises error if there is a pipeline incompatibility.
        """
        self.generate_input_data(seed_cov = seed_cov, seed_ders = seed_ders)
        self.filter_input_data()
        if self.vecLayer is not None : 
            self.vectorize_all_persistence_diagrams()
        if self.vecLayer is None and self.fisherLayer is not None:
            raise ValueError(\
                        'Cannot perform Fisher Analysis without vectorizing.')
        if self.fisherLayer is not None : self.fisher_analysis()
    def vectorize(self, inpVecLayer):
        """
        Vectorize the persistence diagrams in 'all_persistence_diagrams'

        Parameters:
        ----------
        inpVecLayer : object
            Custom vectorization layer object.

        Returns
        -------
        all_vecs : list
            List containing the vectorized representation of the persistence
            diagrams.
        """
        all_vecs = []; 
        for pds in self.all_persistence_diagrams:
            vecs = inpVecLayer.vectorize_persistence_diagrams(pds)
            all_vecs.append(vecs)
        return all_vecs
    
    def vectorize_and_fisher(self, inpVecLayer, inpFisherLayer, stack = True):
        """
        Vectorize the persistence diagrams in 'all_persistence_diagrams' and 
        perform Fisher analysis using custom vectorization and Fisher layers.

        Parameters:
        ----------
        inpVecLayer : object
            Custom vectorization layer object.
        inpFisherLayer : object
            Custom Fisher analysis layer object.

        Returns:
        ----------
        fisher : Object of base class Fisher.
            Fisher matrix or analysis result.
        """
        all_vecs = self.vectorize(inpVecLayer)
        stacked_vecs = tf.stack(all_vecs) if stack else all_vecs
        fisher = inpFisherLayer.computeFisher(stacked_vecs, self.delta_theta)
        return fisher