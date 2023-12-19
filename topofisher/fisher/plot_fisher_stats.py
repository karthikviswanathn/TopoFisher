#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:21:00 2023

@author: karthikviswanathan
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from numpy import meshgrid as coord
import matplotlib.colors as mcolors
from . import Fisher
from .Fisher import computeDerivatives, baseFisher


def plot_derivative_convergence(summaries, delta_theta, parnames = None, \
                                max_repeats = None):
    """
    Plot the convergence of Fisher forecasts keeping the covariance constant
    as a function of the number of splits to test the convergence of 
    derivatives.
    
    Parameters
    ----------
    summaries : list
        List of summary vectors.
    delta_theta : np.array
        Array of delta theta values.
    parnames : list, optional
        List of parameter names. The default is None.
    max_repeats : int, optional
        Maximum number of repeats for convergence analysis. 
        The default is None.
    """
    # List of sample fractions for data splitting
    sample_fractions = [1/10, 1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1]
    
    # Number of simulations in the summary vectors    
    n_s = summaries[0].shape[0]
    n_d = summaries[1].shape[0]
    # Number of parameters
    n_params = len(delta_theta)


    # Initialize arrays to store convergence results
    # mdns - medians of Fisher information
    # stds - standard deviation/error in the Fisher information.    
    mdns  = np.zeros([len(sample_fractions), n_params, n_params])
    stds = np.zeros([len(sample_fractions), n_params, n_params])
    
    # vecs_cov - the  summaries used to estimate the covariance.
    vecs_cov = summaries[0]


    # Iterate over different data split fractions    
    for i,s_frac in enumerate(sample_fractions):
        tmp = []

        # Calculate the number of repeats based on the sample fraction
        nRepeats  = int(1/s_frac)

        # Calculate the number of data points in each split
        n_split = s_frac * n_d

        # Limit the number of repeats if max_repeats is provided
        if max_repeats is not None: nRepeats = min(nRepeats, max_repeats)
        
        # Create an array of shuffled indices for data splitting
        ids_all = np.arange(0, n_d)
        np.random.shuffle(ids_all)
        
        # Perform the convergence analysis for the specified number of repeats
        for I in range(nRepeats):
            ids_fish = ids_all[int(I*n_split):int((I+1)*n_split)]
            
            # Create shuffled derivative vectors
            shuffled_ders = np.stack(summaries[1:])[:, ids_fish]
            
            # Calculate derivatives from shuffled data
            derivatives = computeDerivatives(shuffled_ders, delta_theta)
            
            # Calculate Fisher information matrix
            fisher = baseFisher(vecs_cov, derivatives)
            tmp.append(fisher.invFM.numpy())
        
        # Calculate median and standard deviation for each parameter
        mdns[i]  = np.median(tmp,axis=0)
        if nRepeats!=1:
            stds[i] = np.std(tmp,axis=0)*np.sqrt(1/(len(tmp)-1))
    
    # Extracting the diagonal elements for Fisher forecasting.
    mdns, dev = np.array([np.diag(item) for item in mdns]), \
                    np.array([np.diag(item) for item in stds])
                    
    # Plot convergence results for each parameter    
    for dim in range(n_params):
        label = "theta[" + str(dim) + "]" if parnames is None else parnames[dim]
        plt.errorbar(sample_fractions, mdns[:, dim]**.5/mdns[-1, dim]**.5,\
                yerr = dev[:, dim]**.5/mdns[-1, dim]**.5, label = label, \
                    alpha = 1./(dim + 1))

    # Set axis labels and title    
    plt.xlabel("num_split/num_sims")
    plt.ylabel("FI")
    
    # Display the legend and plot
    plt.legend()
    plt.show()


def plotContours2D(allArr, names, theta_fid, title = None, colors = None, \
            theta_arr = [2*np.arange(-0.1, 0.1, 2e-4), \
                         2*np.arange(-0.1, 0.1, 2e-4)], parameter_list = None, \
            file_loc = None) :
    """
    Plot 2D contours based on Fisher information matrices. 
    It visualizes the confidence contours for each Fisher matrix.
    
    Parameters:
    - allArr (list of 2D arrays): List of Fisher information matrices.
    - names (list of str): Names or labels for each Fisher matrix.
    - theta_fid (tuple): Tuple containing the fiducial parameter values (A, B).
    - title (str, optional): Title for the plot. Default is None.
    - colors (list of str, optional): List of color names for each contour. 
                                     Default is None.
    - theta_arr (list of 1D arrays, optional): Parameter grid for contour 
                                plotting. Default is a grid centered at (0, 0).
    - parameter_list (list of str, optional): Labels for x and y axes. 
                                              Default is None.
    - file_loc (str, optional): File location to save the plot as an image.
                                Default is None.
    """
    A, B = theta_fid
    A_array, B_array = theta_arr
    # Define default colors if not provided
    clr = list(mcolors.TABLEAU_COLORS) if colors is None else colors
    # Initialize a list to store legend elements
    cs = [[] for idx in range(len(allArr))]
    
    # Plot contours for each Fisher matrix
    for idx, item in enumerate(allArr) :
        G = item.flatten()
        x, y = coord(A_array, B_array)
        z_tot = G[0]*x**2 + 2.0*G[1]*(x*y) + G[3]*(y**2)
        # Set contour level for a 68% confidence ellipse
        f_1sig = 0.434 
        con = plt.contour(x + A, y + B, z_tot, [1/f_1sig] , colors=clr[idx])
        # Create legend elements
        l,_ = con.legend_elements()
        cs[idx] = l[0]
    # Add dashed lines for fiducial parameters
    plt.axhline(B, c = 'black', linestyle = 'dashed')
    plt.axvline(A, c = 'black', linestyle = 'dashed')
    # Set axis labels based on parameter_list or default labels
    if parameter_list is None : 
        xlabel = "theta[0]"
        ylabel = "theta[1]"
    else : xlabel, ylabel = parameter_list

    # Set x and y labels, title, and legend
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if(title is not None) : plt.title(title)
    plt.legend(cs, names, loc = 'best')    
    
    # Save the plot as an image if file_loc is provided
    if(file_loc is not None) : plt.savefig(file_loc)
    
    plt.show()
    
def plotHists(vecs, axes, titles = None):
    """
    Plot histograms for a list of vectors.

    Parameters
    ----------
    vecs : list of tf.Tensor
        List of vectors to plot histograms for.
    axes : list
        List of axes objects for plotting.
    titles : list, optional
        List of titles for subplots. The default is None.
    """
    if(titles == None):
        titles = ["dim = " + str(i) for i in range(len(axes))]
    
    for vec, ax, title in zip(vecs, axes, titles) :
        ax.hist(vec, bins = 100); 
        ax.axvline(vec.numpy().mean(), c = 'r')
        ax.set_title(title)

def plotSummaryDerivativeHists(fisher, file_loc = None):
    """
    Plot histograms for summaries and derivatives.
    
    Parameters
    ----------
    fisher : Fisher.baseFisher
        Fisher object containing summaries and derivatives.
    file_loc : string
            File location to save the plot as an image. Default is None.
    """
    n_summaries = fisher.summaries[0].shape[-1]
    n_sum_plots = n_summaries
    n_thet = fisher.derivatives.shape[0]
    
    if(n_summaries  > 4) : 
        print("Plotting only the first 4 dimensions. Can't plot so many \
              histograms for aesthetic purposes.")
        n_sum_plots = 4
    
    # Plotting summary histograms
    row_sz = 3 * n_sum_plots - 1
    fig, axes = plt.subplots(nrows = 1, ncols = n_sum_plots, \
                             figsize=(row_sz, 3))
    fig.suptitle('Summary histograms')
    vecs = tf.transpose(fisher.summaries[0])[:n_sum_plots]
    if n_sum_plots == 1: axes = [axes]
    plotHists(vecs, axes)        
    plt.tight_layout()
    if(file_loc is not None) : 
        fig.savefig(file_loc + "/summary.png")
    else : plt.show()

    # Plotting derivative histograms
    derivatives = fisher.derivatives
    
    # If the number of parameters is less than or equal to 2, plot the 
    # derivative histograms on the same row
    if (n_thet <= 2) :
        col_sz = 3 * n_thet - 1
        fig, axes = plt.subplots(nrows = 1, ncols = n_thet * n_sum_plots, \
                                 figsize=(2 * row_sz, 3))
        titles = []; vecs = []
        if n_thet == 1 and n_sum_plots == 1: axes = [axes]
        for der_unpack, idx in zip(derivatives, range(n_thet)) :
            vecs.extend(tf.transpose(der_unpack)[:n_sum_plots])
            titles.extend(["$\partial_" + str(idx) + "V^" + str(i)  + "$" \
                      for i in range(n_sum_plots)])
        plotHists(vecs, axes, titles)   

    # If the number of parameters is more than 2, plot the derivative
    # histograms on different rows.
    else :
        col_sz = 3 * n_thet - 1
        fig, axes = plt.subplots(nrows = n_thet, ncols = n_sum_plots, \
                                 figsize=(row_sz, col_sz))    
        for der_unpack, axs, idx in zip(derivatives, axes, range(n_thet)) :
            vecs = tf.transpose(der_unpack)[:n_sum_plots]
            titles = ["$\partial_" + str(idx) + "V^" + str(i)  + "$" \
                      for i in range(len(axes))]
            plotHists(vecs, axs, titles)   
    
    fig.suptitle('Derivative histograms')
    plt.tight_layout()
    # Save the plot as an image if file_loc is provided
    if(file_loc is not None) : fig.savefig(file_loc + "/derivative.png")
    else : plt.show()
