#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 16:49:38 2023

@author: karthikviswanathan
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import tqdm
from tqdm import tqdm
    
# reduce_frac is a misleading name
class Histogram(BaseEstimator, TransformerMixin):
    """
    Calculate histogram of counts from persistence diagrams.
    """
    def __init__(self, bins, bdp_type, show_tqdm = False):
        """
        Initialize the Histogram class.

        Parameters:
            bins (int): The bins to compute the histogram for. 
            bdp_type (str): Types of persistence values to include
            (e.g., 'bdp' stands for birth, death and persistence).
        """
        self.bins = bins
        self.bdp_type = bdp_type
        self.show_tqdm = show_tqdm
        
    def fit(self, X, y=None):
        """
        Fit the hist class on a list of persistence diagram.\

        Parameters:
            X (list): List of persistence diagrams (pds).
            y: (unused)

        Returns:
            self
        """
        if(self.bdp_type not in ["b", "d", "p", "bd", "bp", "dp", "bdp"]):
            raise ValueError("Check bdp_tpye. Incorrect results are possible.")

        if not all(item in self.bins.keys() for item in ["b", "d", "p"]) :
            print("Check keys of bins. The keys of the bins are ", self.bins.keys())
        
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
            pdx = np.array(pdx)
            birth, death = pdx[:, 0], pdx[:, 1]
            pers = death - birth
            # Choosing topk births, deaths and persistences and storing in 
            # topk_bdp as a list. Converting it to a dictionary and storing in 
            # bdp_dic.
            arr_dict = {"b": birth, "d": death, "p": pers}
            hist_vecs = []
            # Fetching some or all of "bdp" from the bdp_dic.  
            for letter in self.bdp_type:
                vec = np.histogram(arr_dict[letter], bins = self.bins[letter])[0]
                hist_vecs.extend(vec)
            hist_vecs = np.array(hist_vecs)
            # Concatenating the hist summary statistic for this 
            # persistence diagram.
            Xfit.append(hist_vecs)

        return np.stack(Xfit, axis=0)

    def __call__(self, diag):
        """
        Apply hist on a single persistence diagram and output the result.

        Parameters:
            diag (n x 2 numpy array): Input persistence diagram.

        Returns:
            numpy array : Output top-k vectors.
        """
        return self.fit_transform([diag])[0, :]

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

def fit_histogram(diag, binNum):
    diag = [[elm[0], [elm[1], elm[2]]] for elm in diag]  # [dimension, [birth, death]]
    a0, a1, a2 = cleanPD(diag, 0, float(1))  # aX = [[birth, persistence], [item 2], [item 3]]
    datab0 = a0[:, 0]
    datab1 = a1[:, 0]
    datab2 = a2[:, 0]
    datap0 = a0[:, 1]
    datap1 = a1[:, 1]
    datap2 = a2[:, 1]
    datad0 = a0[:, 0] + a0[:, 1]
    datad1 = a1[:, 0] + a1[:, 1]
    datad2 = a2[:, 0] + a2[:, 1]
    
    datab0_up = np.percentile(datab0, 99.9)
    datab1_up = np.percentile(datab1, 99.9)
    datab2_up = np.percentile(datab2, 99.9)
    datab0_do = np.percentile(datab0, 0.01)
    datab1_do = np.percentile(datab1, 0.01)
    datab2_do = np.percentile(datab2, 0.01)
    datap0_up = np.percentile(datap0, 99.9)
    datap1_up = np.percentile(datap1, 99.9)
    datap2_up = np.percentile(datap2, 99.9)
    datap0_do = np.percentile(datap0, 0.01)
    datap1_do = np.percentile(datap1, 0.01)
    datap2_do = np.percentile(datap2, 0.01)
    datad0_up = np.percentile(datad0, 99.9)
    datad1_up = np.percentile(datad1, 99.9)
    datad2_up = np.percentile(datad2, 99.9)
    datad0_do = np.percentile(datad0, 0.01)
    datad1_do = np.percentile(datad1, 0.01)
    datad2_do = np.percentile(datad2, 0.01)
    
    datab0 = datab0[np.where(datab0<=datab0_up)]
    datab1 = datab1[np.where(datab1<=datab1_up)]
    datab2 = datab2[np.where(datab2<=datab2_up)]
    datab0 = datab0[np.where(datab0>=datab0_do)]
    datab1 = datab1[np.where(datab1>=datab1_do)]
    datab2 = datab2[np.where(datab2>=datab2_do)]
    datap0 = datap0[np.where(datap0<=datap0_up)]
    datap1 = datap1[np.where(datap1<=datap1_up)]
    datap2 = datap2[np.where(datap2<=datap2_up)]
    datap0 = datap0[np.where(datap0>=datap0_do)]
    datap1 = datap1[np.where(datap1>=datap1_do)]
    datap2 = datap2[np.where(datap2>=datap2_do)]
    datad0 = datad0[np.where(datad0<=datad0_up)]
    datad1 = datad1[np.where(datad1<=datad1_up)]
    datad2 = datad2[np.where(datad2<=datad2_up)]
    datad0 = datad0[np.where(datad0>=datad0_do)]
    datad1 = datad1[np.where(datad1>=datad1_do)]
    datad2 = datad2[np.where(datad2>=datad2_do)]
    
    
    bins0 = {"b": np.histogram(datab0, bins=binNum)[1], \
             "d": np.histogram(datad0, bins=binNum)[1], \
                 "p" : np.histogram(datap0, bins=binNum)[1]}
    bins1 = {"b": np.histogram(datab1, bins=binNum)[1], \
             "d": np.histogram(datad1, bins=binNum)[1], \
                 "p": np.histogram(datap1, bins=binNum)[1]}
    bins2 = {"b": np.histogram(datab2, bins=binNum)[1], \
             "d": np.histogram(datad2, bins=binNum)[1], \
                 "p": np.histogram(datap2, bins=binNum)[1]}

    return [bins0, bins1, bins2]