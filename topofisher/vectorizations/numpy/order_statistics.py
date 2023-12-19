#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:11:36 2023

@author: karthikviswanathan
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import tqdm 
from tqdm import tqdm

def _fit_topk_lowk_from_pds(self, X):
    """
    Calculate the topk and the lowk value based on the input persistence 
    diagrams (pds). We assign topk to be the minimum of the number of persistent 
    points present in the list of persistent diagrams, scaled down by the
    reduce_frac, which is typically between 0.9 and 1. We would like lowk to
    capture the persistent features that are missed out by the topk. For this
    reason, we assign lowk to be the difference between the maximum number of 
    points in the persistence diagrams and topk. 
    
    Parameters:
        self: Instance of the TOPK class.
        X (list): List of persistence diagrams (pds).
    """
    lis = np.array([np.array(item).shape[0] for item in X])
    if np.isnan(self.topk):
        self.topk = int(self.reduce_frac * lis.min())
        
    if np.isnan(self.lowk):
        self.lowk = np.minimum(int(np.quantile(lis, 0.9) - self.topk), self.topk)
        # print(self.lowk, self.topk, int(np.quantile(lis, 0.9)))    

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

def _choose_order_stats(arr, num, order):
    """
    Select the top-k values from an array.

    Parameters:
        arr (numpy array): The input array.
        topk (int): The number of top values to select.

    Returns:
        numpy array: The top-k values.
    """
    arr.sort()
    if order == "topk" : return arr[: -num - 1:-1]
    elif order == "lowk" : return arr[: num]
    else: raise ValueError("Incorrect order provided. Order is topk or lowk.")
    

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
class OrderStatistics(BaseEstimator, TransformerMixin):
    """
    Calculate top-k vectors from persistence diagrams.
    """
    def __init__(self, bdp_type, is_binned, topk=np.nan, lowk=np.nan, \
                 reduce_frac=0.9, num_bins=None, pad_value=0.,\
                 show_tqdm = False):
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
        self.lowk = lowk
        self.reduce_frac = reduce_frac
        self.num_bins = num_bins
        self.pad_value = pad_value
        self.show_tqdm = show_tqdm
        
    def fit(self, X, y=None):
        """
        Fit the TOPK class on a list of persistence diagram. This means setting 
        the topk attribute if itis np.nan.

        Parameters:
            X (list): List of persistence diagrams (pds).
            y: (unused)

        Returns:
            self
        """
        if(self.bdp_type not in ["b", "d", "p", "bd", "bp", "dp", "bdp"]):
            raise ValueError("Check bdp_tpye. Incorrect results are possible.")
        if(self.is_binned and self.num_bins is None):
            raise ValueError("How many bins? Check inputs.")
            
        if(self.is_binned and len(self.num_bins) != 2):
            raise ValueError("Assign num_bins for topk and lowk.")
            
        if(np.isnan(self.topk) or np.isnan(self.topk)) :
            _fit_topk_lowk_from_pds(self, X)
        
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
        iter_obj = tqdm(X) if self.show_tqdm else X
        for pdx in iter_obj:
            top_bdp_dic = {}
            low_bdp_dic = {}
            pdx = np.array(pdx)
            if(pdx.shape[0] < self.topk) :
                pdx = _pad_pd(pdx, self.topk, self.pad_value)
            birth, death = pdx[:, 0], pdx[:, 1]
            pers = death - birth
            # Choosing topk births, deaths and persistences and storing in 
            # topk_bdp as a list. Converting it to a dictionary and storing in 
            # bdp_dic.
            topk_bdp = [_choose_order_stats(item, self.topk, "topk") \
                        for item in [birth, death, pers]]
            
            lowk_bdp = [_choose_order_stats(item, self.lowk, "lowk") \
                        for item in [birth, death, pers]]
                
            for typ, arr in zip(["b", "d", "p"], topk_bdp):
                top_bdp_dic[typ] = arr
            for typ, arr in zip(["b", "d", "p"], lowk_bdp):
                low_bdp_dic[typ] = arr
                
            topk_vecs = []
            # Fetching some or all of "bdp" from the bdp_dic.  
            for letter in self.bdp_type:
                vec_top = top_bdp_dic[letter]
                vec_low = low_bdp_dic[letter]
                
                # Binning them if necessary.
                if(self.is_binned) : 
                    vec_top = _bin_arr(vec_top, self.topk, self.num_bins[0])
                    vec_low = _bin_arr(vec_low, self.lowk, self.num_bins[1])
                topk_vecs.extend(vec_top)
                topk_vecs.extend(vec_low)
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