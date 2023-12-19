#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 10:04:07 2023

@author: karthikviswanathan
"""

import numpy as np
import itertools
import sys, os
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import tqdm 
from tqdm import tqdm
from sklearn.neighbors import KernelDensity

def cleanPD(pd, cut, power=0.5):
    p0 = []
    p1 = []
    p2 = []
    for elm in pd:
        if elm[0] == 2 and elm[1][1] != elm[1][0] and elm[1][1] > (float(cut) / 2)**2:
            p2.append([np.power(elm[1][0], power), np.power(elm[1][1] - elm[1][0], power)])
        elif elm[0] == 1 and elm[1][1] != elm[1][0] and elm[1][1] > (float(cut) / 2)**2:
            p1.append([np.power(elm[1][0], power), np.power(elm[1][1] - elm[1][0], power)]) 
        elif (np.isinf(elm[1][1])) == False and elm[1][1] != elm[1][0] and elm[1][1] > (float(cut) / 2)**2:
            p0.append([np.power(elm[1][0], power), np.power(elm[1][1] - elm[1][0], power)])
    return np.array(p0), np.array(p1), np.array(p2)

def PI(sigma, pd, bounds, res):
    kde = KernelDensity(bandwidth=sigma, algorithm='kd_tree', \
                                          kernel='epanechnikov').fit(pd, sample_weight=[elm[1] for elm in pd])
    
    x = np.linspace(bounds[0], bounds[1], res[0])
    y = np.linspace(bounds[2], bounds[3], res[1])
    xx, yy = np.meshgrid(x, y)
    xx = xx.ravel()
    yy = yy.ravel()
    xy_sample = np.array([[xx[i], yy[i]] for i in range(len(xx))])
    d = np.exp(kde.score_samples(xy_sample))
    return np.reshape(d, (res[1], res[0])) * sum([elm[1] for elm in pd])

def calcSavePI(diag, res_len):       
    dimPow = 1
    power = float(dimPow)
    p0, p1, p2 = cleanPD(diag, 0, power)  # pX = [[birth, persistence], [item 2], [item 3]]

    bounds = np.array([[ 5.1021975, 0., 30.09040102,  9.8886027 ,  5.36960632,
        30.31527875],
       [ 6.90789521,  0.        , 33.95395657, 11.42243877,  6.95657454,
        34.399112  ],
       [ 7.82891043,  0.        , 46.0478407 , 10.19376076,  7.86568383,
        46.93917743]])
    bound0 = bounds[0]
    bound1 = bounds[1]
    bound2 = bounds[2]
        
    img0=PI(5*min([(bound0[2]-bound0[0])/res_len,(bound0[3])/res_len]),p0,[0.9*bound0[0],1.1*bound0[2],0,1.1*bound0[3]],res=[res_len,res_len]) 
    img1=PI(5*min([(bound1[2]-bound1[0])/res_len,(bound1[3])/res_len]),p1,[0.9*bound1[0],1.1*bound1[2],0,1.1*bound1[3]],res=[res_len,res_len])
    img2=PI(5*min([(bound2[2]-bound2[0])/res_len,(bound2[3])/res_len]),p2,[0.9*bound2[0],1.1*bound2[2],0,1.1*bound2[3]],res=[res_len,res_len]) 

    return [img0, img1, img2]



def find_vec(param, hod_list, rsd_list, realization_list, baseDir, res_len) :
    print("Finding vectors for ", param)

    
    vecs = []
    
    
    # Loading the derivatives
    lis = list(itertools.product(hod_list, rsd_list, realization_list))
    for element in tqdm(lis):
    # Specifying a PD file
	
        hod, rsd, realization = element  
        k = 15  
        diag = np.load("{}/{}_{}_{}_{}_{}.npy".format(baseDir, param, hod, rsd, \
                                                        realization, k))  
        diag = [[elm[0], [elm[1], elm[2]]] for elm in diag]  # [dimension, [birth, death]]
        imgs = calcSavePI(diag, res_len)
        vecs.append(imgs)
        
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

  
if len(sys.argv) != 7:
    print(len(sys.argv))
    print("Usage: python myscript.py <arg1> [<arg2>]")
    sys.exit(1)

# Access the arguments
param = sys.argv[1]
start = int(sys.argv[2]) # indexed by 0
end = int(sys.argv[3])
inputFolderName = sys.argv[4]
outputFolderName = sys.argv[5]
res_len = int(sys.argv[6])

# print(param, start, end, outputFolderName)  

der_param_list = ["Om_m", "Om_p", "s8_m", "s8_p", "h_m", "h_p"]

if param == "fiducial":
    hod_list = [0]
    rsd_list = [3]
    realization_list = np.arange(500 + start, 500 + end, dtype = np.int32)
elif param in der_param_list:
    hod_list = [0, 1, 2, 3, 4]
    rsd_list = [1, 2, 3]
    realization_list = np.arange(start, end, dtype = np.int32)

else : 
    print("Check input parameter. No such ", param)
    sys.exit(1)

vecs = find_vec(param, hod_list, rsd_list, realization_list, inputFolderName, res_len)
output_file_name = os.path.join(outputFolderName, \
                            param + "_" + str(start) + "_" + str(end) + ".pkl")
writeToFile(vecs, output_file_name)
print("Done for parameter = ", param, ", start = ", start, " end = ", end)