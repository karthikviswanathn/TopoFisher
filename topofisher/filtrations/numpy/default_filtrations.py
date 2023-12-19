#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 12:14:14 2023

@author: karthikviswanathan
"""

import numpy as np
import gudhi
from . import filtration_layers
from gudhi import AlphaComplex
from gudhi.dtm_rips_complex import DTMRipsComplex


class CubicalFiltration(filtration_layers.FiltrationLayer):
    def __init__(self, max_hom_dim, hom_dim_list, show_tqdm, name = "Cubical"):
        """
        Initialize a CubicalFiltration object for computing persistent homology
        using a cubical complex.

        Parameters:
        ----------
        max_hom_dim : int
            The homological dimension upto which to compute persistent homology.
        hom_dim_list : list 
            The list of homological dimensions to find the persistence
            diagrams for.
        show_tqdm : bool
            A flag to indicate whether to display a progress bar using tqdm.
        name : str, optional
            The name of the filtration layer. The default is "Cubical".
        """
        super().__init__(max_hom_dim, hom_dim_list, name, show_tqdm)
    
    def get_st(self, field):
        """
        Compute a simplex tree (ST) for a given field using a Cubical Complex.

        Parameters:
        ----------
        field : numpy array
            The top-dimensional cells of the cubical complex.

        Returns:
        -------
        cc : gudhi.CubicalComplex
            A CubicalComplex object.
        """
        cc = gudhi.CubicalComplex(top_dimensional_cells = field)
        cc.persistence()
        return cc

class AlphaFiltration(filtration_layers.FiltrationLayer):
    def __init__(self, max_hom_dim, hom_dim_list, show_tqdm, name = "Alpha"):
        """
        Initialize an AlphaFiltration object for computing persistent homology
        using an alpha complex.

        Parameters:
        ----------
        max_hom_dim : int
            The homological dimension upto which to compute persistent homology.
        hom_dim_list : list 
            The list of homological dimensions to find the persistence
            diagrams for.
        show_tqdm : bool
            A flag to indicate whether to display a progress bar using tqdm.
        name : str, optional
            The name of the filtration layer. The default is "Alpha".
        """
        super().__init__(max_hom_dim, hom_dim_list, name, show_tqdm)
    
    def get_st(self, points):
        """
        Compute a simplex tree (ST) for a given set of points using an 
        Alpha Complex.

        Parameters:
        ----------
        points : numpy array
            The input points for the Alpha Complex.

        Returns:
        -------
        st : gudhi.SimplexTree
            A SimplexTree object.
        """
        ac = AlphaComplex(points)
        st = ac.create_simplex_tree()
        st.persistence()
        return st
    
class RipsFiltration(filtration_layers.FiltrationLayer):
    def __init__(self, max_hom_dim, hom_dim_list, show_tqdm, max_edge = np.inf,\
                 sparse = None, collapse_edges = False, name = "Rips"):
        """
        Initialize a RipsFiltration object for computing persistent homology 
        using a Rips complex.

        Parameters:
        ----------
        max_hom_dim : int
            The homological dimension upto which to compute persistent homology.
        hom_dim_list : list 
            The list of homological dimensions to find the persistence
            diagrams for.
        show_tqdm : bool
            A flag to indicate whether to display a progress bar using tqdm.
        max_edge : float, optional
            The maximum edge length to be included in the Rips complex. 
            The default is np.inf.
        sparse : bool, optional
            A flag indicating whether the Rips complex should be sparse. 
            The default is None.
        collapse_edges : bool, optional
            A flag indicating whether to collapse edges in the complex to speed
            up the computations. 
            The default is False.
        name : str, optional
            The name of the filtration layer. The default is "Rips".
        """
        self.max_edge = max_edge
        self.sparse = sparse
        self.collapse_edges =collapse_edges
        super().__init__(max_hom_dim, hom_dim_list, name, show_tqdm)
    
    def get_st(self, points):
        """
        Compute a simplex tree (ST) for a given set of points using a Rips 
        Complex.
        
        Parameters:
        ----------
        points : numpy array
            The input points for the Rips Complex.
        
        Returns:
        -------
        st : gudhi.SimplexTree
            A SimplexTree object.
        """
        rips = gudhi.RipsComplex(points = points, \
                                 max_edge_length = self.max_edge, \
                                 sparse = self.sparse)
        if(self.collapse_edges) :
            st = rips.create_simplex_tree(max_dimension = 1)
            st.collapse_edges()
            st.expansion(self.max_hom_dim)
        else : 
            st = rips.create_simplex_tree(max_dimension = self.max_hom_dim)
        st.persistence()
        return st

class DTMRipsFiltration(filtration_layers.FiltrationLayer):
    def __init__(self, num_nn, max_hom_dim, hom_dim_list, show_tqdm, max_edge = np.inf,\
                 sparse = None, collapse_edges = False, name = "DTM Rips"):
        """
        Initialize a DTMRipsFiltration object for computing persistent 
        homology using a DTM Rips complex.

        Parameters:
        ----------
        num_nn : int
            The number of neighbours used to calculate the DTM function.
        max_hom_dim : int
            The homological dimension upto which to compute persistent homology.
        hom_dim_list : list 
            The list of homological dimensions to find the persistence
            diagrams for.
        show_tqdm : bool
            A flag to indicate whether to display a progress bar using tqdm.
        collapse_edges : bool, optional
            A flag indicating whether to collapse edges in the complex to speed
            up the computations. 
            The default is False.
        name : str, optional
            The name of the filtration layer. The default is "DTM Rips".
        """
        self.num_nn = num_nn
        self.max_edge = max_edge
        self.sparse = sparse
        self.collapse_edges = collapse_edges
        super().__init__(max_hom_dim, hom_dim_list, name, show_tqdm)
    
    def get_st(self, points):
        """
        Compute a simplex tree (ST) for a given set of points using a DTM Rips 
        Complex.
        
        Parameters:
        ----------
        points : numpy array
            The input points for the DTM Rips Complex.
        
        Returns:
        -------
        st : gudhi.SimplexTree
            A SimplexTree object.
        """
        rips = DTMRipsComplex(points = points, k = self.num_nn)
        if(self.collapse_edges) :
            st = rips.create_simplex_tree(max_dimension = 1)
            st.collapse_edges()
            st.expansion(self.max_hom_dim)
        else : 
            st = rips.create_simplex_tree(max_dimension = self.max_hom_dim)
        st.persistence()
        return st