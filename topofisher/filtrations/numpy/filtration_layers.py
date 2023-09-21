#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 11:31:13 2023

@author: karthikviswanathan
"""
import numpy as np
import tqdm
from tqdm import tqdm

class FiltrationLayer:
    def __init__(self, hom_dim, name, show_tqdm):
        """
        Initialize a FiltrationLayer object for persistent homology analysis.

        Parameters:
        ----------
        hom_dim : int
            The homological dimension for which to compute persistent homology.
        name : str
            The name of the filtration layer.
        show_tqdm : bool
            A flag to indicate whether to display a progress bar using tqdm.
        """
        self.hom_dim = hom_dim
        self.show_tqdm = show_tqdm
        self.name = name
    
    def find_simplex_trees(self, pc_sims):
        """
        Find simplex trees for a list of point cloud simulations.

        Parameters:
        ----------
        pc_sims : list of numpy arrays
            A list of point cloud simulations.

        Returns:
        -------
        sts : list of simplex trees
            A list of simplex trees computed for each point cloud simulation.
        """
        sts = []
        if (self.show_tqdm == True) : 
            for pc in tqdm(pc_sims):
                sts.append(self.get_st(np.array(pc)))
        else :
            for pc in pc_sims:
                sts.append(self.get_st(np.array(pc)))
        
        return sts
        