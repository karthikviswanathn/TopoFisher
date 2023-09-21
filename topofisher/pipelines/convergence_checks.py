#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 15:24:06 2023

@author: karthikviswanathan
"""
import numpy as np
import tqdm
from tqdm import tqdm
import tensorflow as tf 
from copy import deepcopy
import matplotlib.pyplot as plt

def fisher_estimates_wrt_step_size(pipeline, param_idx, step_list,\
                                   seed_cov, seed_ders):
    """
    Calculate Fisher estimates for varying step sizes of a specific parameter
    given by the 'param_idx' index.
    
    # TODO : Optimize running time by only running the derivative simulations,
    and not calculating the covariance simulations because the covariance 
    simulations are going to be the same.

    Parameters:
    ----------
    pipeline : FisherPipeline
        The FisherPipeline object containing the simulation and analysis setup.
    step_list : list
        A list of step sizes for the parameter of interest.
    param_idx : int
        The index of the parameter for which step sizes are varied.
    seed_cov : int
        Seed for random number generation used in covariance estimation.
    seed_ders : int
        Seed for random number generation used in derivative mean estimation.

    Returns:
    ----------
    fishers_step_sizes : list
        A list of Fisher analysis results for each step size.
    """
    fishers_step_sizes = []
    vecLayer = deepcopy(pipeline.vecLayer)
    fisherLayer = deepcopy(pipeline.fisherLayer)
    
    # Preparing the 'delta_theta' and the 'find_deriative' attributes.
    find_derivative = [False for _ in range(pipeline.total_num_params)]        
    delta_theta_all = [0. for _ in range(pipeline.total_num_params)]
    find_derivative[param_idx] = True
    
    for step in tqdm(step_list): 
        # Setting the vecLayer, fisherLayer and the step size.
        pipeline.vecLayer = deepcopy(vecLayer)
        pipeline.fisherLayer = deepcopy(fisherLayer)
        delta_theta_all[param_idx] = step
        pipeline.set_delta_theta(
            tf.convert_to_tensor(delta_theta_all), find_derivative)
        # Running the pipeline with the same seeds.
        pipeline.run_pipeline(seed_cov = seed_cov, seed_ders = seed_ders)
        fishers_step_sizes.append(deepcopy(pipeline.fisher))
    plot_fisher_step_size_stats(fishers_step_sizes, step_list)
    return fishers_step_sizes

def plot_subplot(ax, arr, x_vals, title, den=None):
    """
    Plot subplots with given data.

    Parameters:
    ----------
    ax : object
        Matplotlib axis object.
    arr : numpy.ndarray
        Data array to plot.
    x_vals : list
        List of x-values.
    title : str
        Title for the subplot.
    den : float, optional
        Denominator value for normalization (default is None).
    """
    for idx in range(arr[0].shape[0]):
        den = np.median(arr[:, idx]) if den is None else den
        ax.plot(x_vals, arr[:, idx] / den, marker='.')
        ax.set_title(title)
        
def plot_fisher_step_size_stats(fishers_step, step_list):
    """
    Plot various statistics related to Fisher analysis with varying step sizes.

    Parameters:
    ----------
    fishers_step : list
        A list of Fisher analysis results for each step size.
    step_list : list
        A list of step sizes.
    """
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(14, 4))
    ax0, ax1, ax2, ax3 = axes[0], axes[1], axes[2], axes[3]
    all_matrices = tf.stack([item.FM.numpy().flatten()\
                             for item in fishers_step])
    plot_subplot(ax0, all_matrices, step_list, \
                 title="Fisher matrix plots wrt $\delta \\theta $")
    
    all_biases = tf.stack([item.fractional_bias.numpy().flatten()\
                           for item in fishers_step])
    plot_subplot(ax1, all_biases, step_list, \
                 title="Fractional bias plots wrt $\delta \\theta $", den=1.)
        
    all_ders = tf.stack([item.ders.numpy().flatten() for item in fishers_step])
    plot_subplot(ax2, all_ders, step_list, \
                 title="Compressed derivative plots wrt $\delta \\theta $")
        
    all_C = tf.stack([item.C.numpy().flatten() for item in fishers_step])
    plot_subplot(ax3, all_C, step_list, \
                 title="Compressed variance plots wrt $\delta \\theta $")
    
    plt.tight_layout()
    plt.show() 
