#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 20:21:00 2023

@author: karthikviswanathan
"""

import numpy as np
import math
from sklearn.neighbors import NearestNeighbors, KDTree
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from gudhi.point_cloud.dtm import DistanceToMeasure


def plotDTM(vals, ax, fig, title):
    im = ax.imshow(vals, interpolation='None')
    ax.title.set_text(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')


def getFiltrationGrid(pts, num_nn, model, grid, nx, ny) :
    knn = NearestNeighbors(n_neighbors =  num_nn)
    grid_dists = knn.fit(pts).kneighbors(grid)[0][:,1:]   
    vals = model(grid_dists).numpy().reshape(nx, ny)
    return np.flip(vals, axis = 0)
    
def visualizeFiltrations(initial_model, final_model, pts, num_nn, sgn = 1., 
                         m = None, extent = [-2, 2]):
    ntot = pts.shape[0]
    if m is None : m = num_nn/ntot
    lo, hi = extent
    nx = 100; ny = 100
    xR = np.linspace(lo, hi, nx); yR = np.linspace(lo, hi, ny)
    grid = np.array(list(itertools.product(xR, yR)))
    

    
    fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize=(8, 10))
    vals = sgn * getFiltrationGrid(pts, num_nn, initial_model, grid, nx, ny)
    plotDTM(vals = vals, ax = axes[0, 0], fig = fig, title = "Before learning")
    
    vals = sgn * getFiltrationGrid(pts, num_nn, final_model, grid, nx, ny)
    plotDTM(vals = vals, ax = axes[0, 1], fig = fig, title = "After learning")
    
    
    axes[1, 0].scatter(pts[:,0], pts[:,1])
    axes[1, 0].set_xlim([lo, hi])
    axes[1, 0].set_ylim([lo ,hi])
    axes[1, 0].set_title('Point cloud')
    
    dtm = DistanceToMeasure(k = math.floor(m * ntot)).fit(pts).transform(grid)
    vals = -np.flip(dtm.reshape(nx, ny), axis = 0)
    plotDTM(vals = vals, ax = axes[1, 1], fig = fig, \
            title = "DTM with m = " + str(np.round(m, 2)))
    plt.tight_layout()
    plt.show()