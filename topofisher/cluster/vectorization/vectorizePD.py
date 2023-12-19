#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 16:21:10 2023

@author: karthikviswanathan
"""

import numpy as np
import itertools
import sys
from topofisher.vectorizations.numpy.custom_vectorizations import TOPK
from topofisher.vectorizations.numpy.histogram_counts import Histogram, fit_histogram
from topofisher.vectorizations.numpy.vectorization_layer import VectorizationLayers
import pickle
import tqdm 
from tqdm import tqdm
import json



def cleanPDFast(pd):
    # Extract dimensions and conditions
    dimensions = pd[:, 0]
    cond_dim2 = (dimensions == 2) & (pd[:, 2] != pd[:, 1]) & (pd[:, 2] > 0)
    cond_dim1 = (dimensions == 1) & (pd[:, 2] != pd[:, 1]) & (pd[:, 2] > 0)
    cond_dim0 = (dimensions == 0) &(~np.isinf(pd[:, 2])) & (pd[:, 2] != pd[:, 1])

    # Filter data based on conditions
    p2 = pd[cond_dim2][:, 1:]
    p1 = pd[cond_dim1][:, 1:]
    p0 = pd[cond_dim0][:, 1:]
    return p0, p1, p2
    

def find_vec(baseDir, param_list, hod_list, rsd_list, realization_list, hyper_parameters, k) :
    print("Finding vectors for ", param_list)    
    pd_all = {}
    show_tqdm = hyper_parameters["show_tqdm"]
    for item in param_list : pd_all[item] = [[], [], []]
    vecType = hyper_parameters["vectorization"]
    
    if  vecType == "topk":
        print("Fitting  PD")
        num_bins = hyper_parameters["num_bins"]
        bdp_types = hyper_parameters["bdp_types"]
        topks = [85720, 33127, 31494]
        vectorizations = [TOPK(bdp_type = bdp_types[idx], \
            num_bins = num_bins[idx], is_binned = True, topk = topks[idx]) for  idx in range(3)]
        
    
    if  vecType == "histogram":
        num_bins = hyper_parameters["num_bins"]
        param_name, hod_num, rsd_num, realization_num = \
            [hyper_parameters[item] for item in ["param_name", "hod_num", \
                                                 "rsd_num", "realization_num"]]
        
        reference_pd = np.load("{}/{}_{}_{}_{}_{}.npy".format(baseDir, param_name, \
                                        hod_num, rsd_num, realization_num, k))
        bins = fit_histogram(reference_pd, num_bins)
        vectorizations = [Histogram(bdp_type = "bd", bins = bins[0], show_tqdm = show_tqdm), \
                          Histogram(bdp_type = "bd", bins = bins[1], show_tqdm = show_tqdm),\
                          Histogram(bdp_type = "bd", bins = bins[2], show_tqdm = show_tqdm)]
    vecLayer = VectorizationLayers(vectorizations = vectorizations) 

    # Loading the derivatives
    lis = list(itertools.product(param_list, hod_list, rsd_list, realization_list))
    for element in lis:
    # Specifying a PD file
	
        param, hod, rsd, realization = element  
        pd_full = np.load("{}/{}_{}_{}_{}_{}.npy".format(baseDir, param, hod, rsd, \
                                                        realization, k))  
        pds = cleanPDFast(pd_full)  
        for idx, pd in enumerate(pds):
            pd_all[param][idx].append(pd)
        
    vecs = {}  
    print("Vectorizing now!")
    for item in param_list :
        vecs[item] = \
            vecLayer.vectorize_persistence_diagrams(pd_all[item])
    return vecs

def writeToFile(lis, fileName):
    g = open(fileName, "wb")
    pickle.dump(lis, g)
    g.close()

def readFromFile(fileName) :
    f = open(fileName, "rb")
    PD = pickle.load(f)
    f.close()
    return PD

  
if len(sys.argv) != 9:
    print(len(sys.argv))
    print("Usage: python myscript.py <inputFolderName> <outputFileName> <jsonFile>\
          fid_start fid_end deriv_start deriv_end k")
    sys.exit(1)

# Access the arguments
"""
fid_start = 0 # indexed by 0
fid_end = 10000
deriv_start = 0
deriv_end = 500
"""
inputFolderName = sys.argv[1]
outputFolderName = sys.argv[2]
json_file_path = sys.argv[3]

fid_start = int(sys.argv[4]) # indexed by 0
fid_end = int(sys.argv[5])
deriv_start = int(sys.argv[6])
deriv_end = int(sys.argv[7])
k = int(sys.argv[8])

with open(json_file_path, "r") as json_file:
    hyper_parameters = json.load(json_file)

print(fid_start, fid_end, deriv_start, deriv_end, outputFolderName)  

param_list = ["fiducial"]
hod_list = [0]
rsd_list = [3]
realization_list = np.arange(500 + fid_start, 500 + fid_end, dtype = np.int32)
fid_vecs = find_vec(inputFolderName, param_list, hod_list, rsd_list, realization_list, hyper_parameters, k)
param_list = ["Om_m", "Om_p", "s8_m", "s8_p", "h_m", "h_p"]
hod_list = [0, 1, 2, 3, 4]
rsd_list = [1, 2, 3]
realization_list = np.arange(deriv_start, deriv_end, dtype = np.int32)
der_vecs = find_vec(inputFolderName, param_list, hod_list, rsd_list, realization_list, hyper_parameters, k)
fileName = hyper_parameters["vectorization"] + "_" + str(k) + \
    "/" + str(fid_start) + "_" + str(fid_end) + "_" + str(deriv_start) + "_"  + str(deriv_end) 
writeToFile([fid_vecs, der_vecs], outputFolderName + "/" + fileName + ".pkl")
print("Done for fiducials ", fid_start, " to ", fid_end)
print("Done for derivatives ", deriv_start, " to ", deriv_end)