#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 12:28:52 2023

@author: karthikviswanathan
"""
import tensorflow as tf
from . import pipeline
from topofisher.input_simulators.noisy_ring import CircleSimulator
from topofisher.fisher.Fisher import fisherFromVecs, fisherMOPED

class CirclePipeline(pipeline.FisherPipeline):
    def __init__(self, ncirc, nback, bgmAvg, **kwargs):
        """
        Initialize a CirclePipeline for performing Fisher analysis on the 
        noisy ring data.

        Parameters:
        ----------
        ncirc : int
            Number of data points in the circle.
        nback : int
            Number of background data points.
        bgmAvg : float
            Average distance to the center of the background data points.
        **kwargs : dict
            Additional keyword arguments to pass to the base class constructor.
        """
        inp_sim = CircleSimulator(ncirc= ncirc, nback = nback, bgmAvg = bgmAvg)
        super().__init__(inp_sim, **kwargs)
        
    def collect_benchmarks(self):
        """
        Collects benchmarks using different summary statistics to compare the 
        Fisher analysis results.

        """
        self.benchmarks = {}
        # Computing the theoretical fisher matrix (TFM).
        self.benchmarks['TFM'] = self.inp_sim.TFM(self.theta_fid).numpy()
        # Computing the Fisher forecast using the mean and variance of the
        # distances to the center from the noisy ring data.
        mean_var_vecs = tf.stack(self.inp_sim.meanDistSummary(self.all_pts))
        self.benchmarks['MeanVariance'] = fisherFromVecs(mean_var_vecs, \
                                                         self.delta_theta)
        # Computing the Fisher forecast using by first sorting the distances to 
        # the center and compressing using MOPED. 
        sorted_dists = tf.stack(self.inp_sim.sortedDistSummary(self.all_pts))
        self.benchmarks['SortedMOPED'] = fisherMOPED(sorted_dists, \
                                                     self.delta_theta)
