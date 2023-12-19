#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 20:05:09 2023

@author: karthikviswanathan
"""

from gudhi.representations import Landscape, PersistenceImage
from . import vectorization_layer
import numpy as np
import matplotlib.pyplot as plt

class PersistenceLandscapes(vectorization_layer.VectorizationLayers):
    def __init__(self, resolutions, num_landscapes, pds_idx_list = None, \
                 name = "Landscapes"):
        """
        Initialize PersistenceLandscapes vectorization.

        Parameters:
            resolutions (list): List of resolutions of the landscape for 
                               different homological dimensions.
            num_landscapes (list): List of numbers of landscapes for different
                                  homological dimensions.
            pds_idx_list (list, optional): List of indices of persistence 
                                          diagrams to vectorize. They generally
                                          stand for the hom_dims. If None, 
                                          vectorize all the persistence 
                                          diagrams.
                                          Defaults to None.
            name (str, optional): Name of the vectorization layer. 
                                 Defaults to "Landscapes".
        """
        if(len(resolutions) != len(num_landscapes)) : 
            raise ValueError(
                "Check dimensions of resolutions and num_landscapes")
        
        num_hom_dim = len(resolutions)
        vectorizations = [[] for idx in range(num_hom_dim)]
        for idx in range(num_hom_dim):
            vectorizations[idx] = Landscape(resolution = resolutions[idx], \
                                        num_landscapes = num_landscapes[idx])
        super().__init__(vectorizations, pds_idx_list, name)
                
class PersistenceImages(vectorization_layer.VectorizationLayers):
    def __init__(self, resolutions, bandwidths, pds_idx_list = None, 
                 reshape_images = True, name = "Images"):
        """
        Initialize PersistenceImages vectorization.

        Parameters:
            resolutions (list): List of resolutions of the persistence images 
                                for different homological dimensions.
            bandwidths (list): List of bandwidths of the persistence images for
                              different homological dimensions.
            pds_idx_list (list, optional): List of indices of persistence 
                                          diagrams to vectorize. They generally
                                          stand for the hom_dims. If None, 
                                          vectorize all the persistence 
                                          diagrams.
                                          Defaults to None.
            reshape_images (bool, optional): Whether to reshape images or not.
                                             Defaults to True.
            name (str, optional): Name of the vectorization layer. 
                                  Defaults to "Images".
        """
        self.resolutions = resolutions
        if(len(resolutions) != len(bandwidths)) : 
            raise ValueError(
                "Check dimensions of resolutions and bandwidths!")
        
        self.reshape_images = reshape_images
        num_layers = len(resolutions)
        vectorizations = [[] for idx in range(num_layers)]
        for idx in range(num_layers):
            vectorizations[idx] = PersistenceImage(
                bandwidth = bandwidths[idx], resolution = resolutions[idx], \
                    weight = lambda x: x[1]**2)
        super().__init__(vectorizations, pds_idx_list, name)
        
    def post_process(self, vecs):
        """
        Perform post-processing on vectorized data.

        Parameters:
            vecs (numpy array): Vectorized data.

        Returns:
            numpy array: Post-processed vectorized data.
        """
        if(self.reshape_images is True): 
            return self.reshape_vec_to_images(vecs)
        else : return vecs
    
    def check_resolutions_match(self):
        """
        Check if resolutions for the different homological dimensions match.

        Returns:
            bool: True if resolutions match, False otherwise.
        """
        res = np.array(self.resolutions).T
        num_distinct_res = [len(set(res[idx])) for idx in [0, 1]]
        if(num_distinct_res[0] == 1 and num_distinct_res[1] == 1) : 
            self.res = self.resolutions[0]
            return True
        else : return False
 
    def reshape_vec_to_images(self, vecs):
        """
        Checks if the resolutions match and reshapes flat vectorized data to 
        persistence images of shape (res, res, num_hom_dims).

        Parameters:
            vecs (numpy array): Vectorized data.

        Returns:
            numpy array: Reshaped data.
        """
        cur_idx = 0
        resolutions = np.array(self.resolutions)
        if (vecs.shape[-1] != np.sum(resolutions.T[0] * resolutions.T[1]) or
            len(vecs.shape) != 2) : 
            raise ValueError("Can't reshape. Check shape of vectors")
            return
        
        if (self.check_resolutions_match() == False) :
            raise ValueError(
                "Resolutions do not match. Change reshape_images attribute.")
            
        ret_vecs = []
        for idx in range(self.num_hom_dim):
            resx, resy = resolutions[idx]
            end_idx = int(cur_idx + resx*resy)
            cur_vecs = vecs[:, cur_idx : end_idx]
            num_sims = cur_vecs.shape[0]
            ret_vecs.append(np.reshape(cur_vecs, (num_sims, resx, resy)))
            cur_idx = end_idx
        return np.stack(ret_vecs, axis = -1)
    
    def plot_persistence_image(self, pds):
        """
        Plot persistence images for the persistence diagrams corresponding to 
        the pds_idx_list.

        Parameters:
            st: Simplex tree.
        """
        im = self.vectorize_persistence_diagrams([[item] for item in pds])
        ncols = self.num_hom_dim
        fig, axes = plt.subplots(nrows=1, ncols= ncols, \
                                 figsize=(3 * ncols + 2 , 3))
        for idx in range(ncols):
            ax = axes[idx]
            ax.imshow(np.flip(im[:, :, idx], axis = 0))
            ax.set_title("PI for hom_dim = " + str(self.hom_dims[idx]))
        plt.tight_layout()
        plt.show()