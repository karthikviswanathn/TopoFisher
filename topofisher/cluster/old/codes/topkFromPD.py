#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 10:04:07 2023

@author: karthikviswanathan
"""

import numpy as np
import itertools
import sys
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import tqdm 
from tqdm import tqdm

def _fit_topk_from_pds(self, X):
    """
    Calculate the top-k value based on the input persistence diagrams (pds). It
    is the minimum of the number of persistent points present in the list of
    persistent diagrams, scaled down by the reduce_frac, which is typically 
    between 0.9 and 1.
    
    Parameters:
        self: Instance of the TOPK class.
        X (list): List of persistence diagrams (pds).
    """
    lis = [np.array(item).shape[0] for item in X]
    self.topk = int(self.reduce_frac * np.array(lis).min())

def _pad_pd(pdx, topk, pad_value):
    """
    Pad a persistence diagram (pdx) with 'pad_value' up to a specified 
    top-k value.

    Parameters:
        pdx (numpy array): The input persistence diagram.
        topk (int): The desired top-k value.
        pad_value (float): The value used for padding.

    Returns:
        numpy array: The padded persistence diagram.
    """
    print("Decrease topk! Input shape = ", pdx.shape[0], " topk = ", topk);
    pdx = np.pad(pdx, (0, int(topk) - pdx.shape[0]), 'constant', \
                 constant_values= pad_value)
    return pdx

def _choose_topk(arr, topk):
    """
    Select the top-k values from an array.

    Parameters:
        arr (numpy array): The input array.
        topk (int): The number of top values to select.

    Returns:
        numpy array: The top-k values.
    """
    arr.sort()
    return arr[: -topk -1:-1]

def _bin_arr(arr, topk, binx):
    """
    Bin an array into a specified number of bins.

    Parameters:
        arr (numpy array): The input array.
        topk (int): The top-k value.
        binx (int): The number of bins.

    Returns:
        numpy array: The binned array.
    """
    div = len(arr) // binx * binx
    arr = arr[:div]
    return arr.reshape((binx, -1)).sum(axis=-1)

# reduce_frac is a misleading name
class TOPK(BaseEstimator, TransformerMixin):
    """
    Calculate top-k vectors from persistence diagrams.
    """
    def __init__(self, bdp_type, is_binned, topk=np.nan, reduce_frac=0.9, 
                  num_bins=np.nan, pad_value=0.):
        """
        Initialize the TOPK class.

        Parameters:
            bdp_type (str): Types of persistence values to include
            (e.g., 'bdp' stands for birth, death and persistence).
            is_binned (bool): Whether to bin the values.
            topk (int, optional): The top-k value. Defaults to np.nan.
            reduce_frac (float, optional): The reduction fraction for 
            calculating top-k. Defaults to 0.9.
            num_bins (int, optional): The number of bins. Defaults to np.nan.
            pad_value (float, optional): The value used for padding.
                                         Defaults to 0.
        """
        self.bdp_type = bdp_type
        self.is_binned = is_binned
        self.topk = topk
        self.reduce_frac = reduce_frac
        self.num_bins = num_bins
        self.pad_value = pad_value
        
    def fit(self, X, y=None):
        """
        Fit the TOPK class on a list of persistence diagram. This means setting 
        the topk attribute if it is np.nan.

        Parameters:
            X (list): List of persistence diagrams (pds).
            y: (unused)

        Returns:
            self
        """
        if(self.bdp_type not in ["b", "d", "p", "bd", "bp", "dp", "bdp"]):
            raise ValueError("Check bdp_tpye. Incorrect results are possible.")
        if(self.is_binned and self.num_bins == np.nan):
            raise ValueError("How many bins? Check inputs.")
            
        if(np.isnan(self.topk)) :
            _fit_topk_from_pds(self, X)
        
        return self

    def transform(self, X):
        """
        Compute the top-k vectors for each persistence diagram individually and 
        concatenate the results.

        Parameters:
            X (list): List of persistence diagrams (pds).

        Returns:
            numpy array: Transformed top-k vectors.
        """
        Xfit = []
        # Iterating over the list of persistence diagrams.
        for pdx in X:
            bdp_dic = {}
            pdx = np.array(pdx)
            if(pdx.shape[0] < self.topk) :
                pdx = _pad_pd(pdx, self.topk, self.pad_value)
            birth, death = pdx[:, 0], pdx[:, 1]
            pers = death - birth
            # Choosing topk births, deaths and persistences and storing in 
            # topk_bdp as a list. Converting it to a dictionary and storing in 
            # bdp_dic.
            topk_bdp = [_choose_topk(item, self.topk) \
                        for item in [birth, death, pers]]
            for typ, arr in zip(["b", "d", "p"], topk_bdp):
                bdp_dic[typ] = arr
                
            topk_vecs = []
            # Fetching some or all of "bdp" from the bdp_dic.  
            for letter in self.bdp_type:
                vec = bdp_dic[letter]
                # Binning them if necessary.
                if(self.is_binned) : 
                    vec = _bin_arr(vec, self.topk, self.num_bins)
                topk_vecs.extend(vec)
            topk_vecs = np.array(topk_vecs)
            # Concatenating the topk summary statistic for this 
            # persistence diagram.
            Xfit.append(topk_vecs)

        return np.stack(Xfit, axis=0)

    def __call__(self, diag):
        """
        Apply TOPK on a single persistence diagram and output the result.

        Parameters:
            diag (n x 2 numpy array): Input persistence diagram.

        Returns:
            numpy array : Output top-k vectors.
        """
        return self.fit_transform([diag])[0, :]

class VectorizationLayers:
    def __init__(self, vectorizations, pds_idx_list = None, name = "vecLayer"):
        """
        Initialize the VectorizationLayers.

        Parameters:
            vectorizations (list): List of vectorization methods for each 
                                   index present in pds_idx_list.
            pds_idx_list (list, optional): List of indices of persistence 
                                          diagrams to vectorize. They generally
                                          stand for the hom_dims. If None, 
                                          vectorize all the persistence 
                                          diagrams.
                                          Defaults to None.
            name (str, optional): The name of the vectorization layer. 
                                  Defaults to "vecLayer".
        """
        if pds_idx_list is None :
            pds_idx_list = [i for i in range(len(vectorizations))]
        if len(vectorizations) != len(pds_idx_list) and \
            pds_idx_list is not None: 
            raise ValueError(
        "Make sure that the vectorizations and the hom_dims are compatible.")
            
        self.vectorizations = vectorizations
        self.pds_idx_list = pds_idx_list
        self.num_hom_dim = len(pds_idx_list)
        self.name = name
        self.is_fitted = [False for idx in range(self.num_hom_dim)]
            
    def vectorize_persistence_diagrams(self, pds_all_hom_dims):
        """
        Vectorize a list of persistence diagrams
        
        Parameters:
            pds_all_hom_dims(list of list): List of persistence diagrams for 
                                            all the homology dimensions.
        
        Returns:
            numpy array: Vectorized data from the persistent diagrams 
            corresponding to the homology dimensions in 'pds_idx_list'.
        """
        vecs = []
        for idx, veclayer in zip(self.pds_idx_list, self.vectorizations):
            pds = pds_all_hom_dims[idx]
            if(self.is_fitted[idx] == False) :
                veclayer.fit(pds)
                self.is_fitted[idx] = True
            vecs.append(veclayer.transform(pds))
        return self.post_process(np.concatenate(vecs, axis = -1))
    
    def post_process(self, vecs):
        """
        Perform post-processing on vectorized data. This method can be 
        overwritten in case some post processing is necessary by instances that
        inherit this class. For example, in the persistent image vectorization
        layers, post-processing is done by reshaping a vector into
        a collection of images, i.e. a vector of length l is reshaped into 
        (res, res, num_hom_dims) to view them as images.

        Parameters:
            vecs (numpy array): Vectorized data.

        Returns:
            numpy array: Post-processed vectorized data.
        """
        return vecs
"""
def cleanPD(pd, cut, power=0.5):
    p0 = []
    p1 = []
    p2 = []
"""
"""
    for elm in pd:
        if elm[0] == 2 and elm[1][1] != elm[1][0] and elm[1][1] > (float(cut)/2)**2:
            p2.append([np.power(elm[1][0], power), np.power(elm[1][1]-elm[1][0], power)])
        
        elif elm[0] == 1 and elm[1][1] != elm[1][0] and elm[1][1] > (float(cut)/2)**2:
            p1.append([np.power(elm[1][0], power), np.power(elm[1][1]-elm[1][0], power)])
        
        elif (np.isinf(elm[1][1])) == False and elm[1][1] != elm[1][0] and elm[1][1] > (float(cut)/2)**2:
            p0.append([np.power(elm[1][0], power), np.power(elm[1][1]-elm[1][0], power)])
"""
"""
    
    for elm in pd:
        if elm[0] == 2 and elm[1][1] != elm[1][0] and elm[1][1] > (float(cut)/2)**2:
            p2.append([elm[1][0], elm[1][1]])
        
        elif elm[0] == 1 and elm[1][1] != elm[1][0] and elm[1][1] > (float(cut)/2)**2:
            p1.append([elm[1][0], elm[1][1]])
        
        elif (np.isinf(elm[1][1])) == False and elm[1][1] != elm[1][0] and elm[1][1] > (float(cut)/2)**2:
            p0.append([elm[1][0], elm[1][1]])
    return np.array(p0), np.array(p1), np.array(p2)
"""

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

def find_vec(param_list, hod_list, rsd_list, realization_list) :
    print("Finding vectors for ", param_list)

    baseDir = "/projects/0/gusr0688/vk/pd_sancho/"  # Where the .npy files are on Snellius
    
    pd_all = {}
    
    for item in param_list : pd_all[item] = [[], [], []]
    vectorizations = [TOPK(bdp_type = "bdp", is_binned = True, num_bins = 100), \
                      TOPK(bdp_type = "bdp", is_binned = True, num_bins = 100),\
                      TOPK(bdp_type = "bdp", is_binned = True, num_bins = 100)]
    vecLayer = VectorizationLayers(vectorizations = vectorizations) 
    
    
    # Loading the derivatives
    lis = list(itertools.product(param_list, hod_list, rsd_list, realization_list))
    for element in lis:
    # Specifying a PD file
	
        param, hod, rsd, realization = element  
        k = 15  
        pd_full = np.load("{}{}_{}_{}_{}_{}.npy".format(baseDir, param, hod, rsd, \
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

  
if len(sys.argv) != 6:
    print(len(sys.argv))
    print("Usage: python myscript.py <arg1> [<arg2>]")
    sys.exit(1)

# Access the arguments
fid_start = int(sys.argv[1]) # indexed by 0
fid_end = int(sys.argv[2])
deriv_start = int(sys.argv[3])
deriv_end = int(sys.argv[4])
outputFileName = sys.argv[5]

print(fid_start, fid_end, deriv_start, deriv_end, outputFileName)  

param_list = ["fiducial"]
hod_list = [0]
rsd_list = [3]
realization_list = np.arange(500 + fid_start, 500 + fid_end, dtype = np.int32)
fid_vecs = find_vec(param_list, hod_list, rsd_list, realization_list)
 
param_list = ["Om_m", "Om_p", "s8_m", "s8_p"]
hod_list = [0, 1, 2, 3, 4]
rsd_list = [1, 2, 3]
realization_list = np.arange(deriv_start, deriv_end, dtype = np.int32)
der_vecs = find_vec(param_list, hod_list, rsd_list, realization_list)

writeToFile([fid_vecs, der_vecs], outputFileName)
print("Done for fiducials ", fid_start, " to ", fid_end)
print("Done for derivatives ", deriv_start, " to ", deriv_end)