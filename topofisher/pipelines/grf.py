#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 12:32:39 2023

@author: karthikviswanathan
"""
from . import pipeline
from imph.inputs.grf import GRFSimulator
from imph.fisher.imnn import MopedLayer
from imph.vectorizations.numpy.TOPK import MeanBirthDeathLayers


class GRFPipeline(pipeline.FisherPipeline):
    def __init__(self, N, dim, boxlength, ensure_physical, \
                 vol_normalised_power, **kwargs):
        """
        Initialize a GRFPipeline for doing Fisher analysis on Gaussian Random 
        Field (GRF) data.

        Parameters:
        ----------
        N : int
            Number of grid points along each dimension.
        dim : int
            Number of dimensions (e.g., 2 for 2D, 3 for 3D).
        boxlength : float
            Size of the box (physical size) for the GRF.
        ensure_physical : bool
            This parameter is to ensure that the GRFs generated are physical. 
            This is done by setting the ensure_physical parameter in powerbox. 
        vol_normalised_power : bool
            Whether the power spectrum of the GRF is volume normalized (True) 
            or not (False).
        **kwargs : dict
            Additional keyword arguments to pass to the base class constructor.
        """
        self.N = N;
        inp_sim = GRFSimulator(N = N, dim = dim, boxlength = boxlength, \
                               ensure_physical = ensure_physical, \
                               vol_normalised_power = vol_normalised_power)
        super().__init__(inp_sim, **kwargs)
        
    def collect_benchmarks(self):
        """
        Collect benchmarks using different summary statistics.
        """
        self.benchmarks = {}
        # Computing the theoretical fisher matrix (TFM).
        self.benchmarks['TFM'] = self.inp_sim.TFM(self.theta_fid)
        
        # Computing the Fisher forecast by computing the mean birth and death of
        # the persistence diagrams.
        vecLayer = MeanBirthDeathLayers(hom_dim_list = [0, 1])
        inpFisherLayer = MopedLayer()
        self.benchmarks['BDMean'] = self.vectorize_and_fisher(\
                                            vecLayer, inpFisherLayer) 