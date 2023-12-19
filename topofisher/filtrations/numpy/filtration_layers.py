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
    def __init__(self, max_hom_dim, hom_dim_list, name, show_tqdm):
        """
        Initialize a FiltrationLayer object for persistent homology analysis.

        Parameters:
        ----------
        max_hom_dim : int
            The homological dimension upto which to compute persistent homology.
        hom_dim_list : list 
            The list of homological dimensions to find the persistence
            diagrams for.
        name : str
            The name of the filtration layer.
        show_tqdm : bool
            A flag to indicate whether to display a progress bar using tqdm.
        """
        self.max_hom_dim = max_hom_dim
        self.hom_dim_list = hom_dim_list
        self.name = name
        self.show_tqdm = show_tqdm
    
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
    
    def find_persistence_diagrams(self, pc_sims):
        """
        Find persistence diagrams corresponding to the hom_dim_list for a list 
        of point cloud simulations.

        Parameters:
        ----------
        pc_sims : list of numpy arrays
            A list of point cloud simulations.

        Returns:
            pds : list of list of persistence diagrams
            A nested list with the first outer dimension correspoding
            to the hom_dims and the second outer dimension 
            corresponding to the 'n' simulations. 
        """
        sts = self.find_simplex_trees(pc_sims)
        pds = []
        for hom_dim in self.hom_dim_list:
            pds_hom_dim = []
            for st in sts :
                pd = st.persistence_intervals_in_dimension(hom_dim)
                pds_hom_dim.append(getFinitePairs(pd))
            pds.append(pds_hom_dim)
        return pds

def getFinitePairs(diag):
    """
    Get finite pairs from a persistence diagram.

    Parameters:
        diag (numpy array): Persistence diagram.

    Returns:
        numpy array: Finite pairs from the persistence diagram.

    """
    return diag[diag[:,1] < np.inf]       