#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 11:23:14 2023

@author: karthikviswanathan
"""

import tensorflow as tf 
import tensorflow_probability as tfp
import sklearn
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import pingouin as pg
from scipy.stats import kstest, norm

# TODO : Add Gaussian tests
class baseFisher:
    """
        In this class, we do the Fisher analysis given the summaries at the
        fiducial values and the derivatives. 
    """
    def __init__(self, vecs_cov, derivatives, name = None, clean_data = True):
        """
        Initialize the base Fisher class.

        Parameters
        ----------
        vecs_cov : tf.Tensor of shape (num_sims, num_summaries)
            The summaries used to estimate the covariance.
        derivatives : tf.Tensor of shape (num_theta, num_sims, num_summaries)
            The calculated derivatives.
        name : str, optional
            Name for the Fisher class. The default is None.
        clean_data : bool, optional
            Flag to clean data. By 'clean', we remove the features that have 
            zero variance. This is to avoid non-invertible covariance matrices.
            The default is True.
        """
        if(clean_data) : 
            vecs_cov, derivatives = cleanData(vecs_cov, derivatives)
        self.name = name
        self.vecs_cov = vecs_cov
        self.derivatives = derivatives
        self.fisherAnalysis()
    
    def fisherAnalysis(self):
        self.C = computeCovariance(self.vecs_cov)
        self.invC = tf.linalg.inv(self.C)  # n_summaries * n_summaries
        self.ders = tf.math.reduce_mean(self.derivatives, axis = 1)
        self.compmat = tf.linalg.solve(self.C, tf.transpose(self.ders))
        self.FMμ = tf.matmul(self.ders, self.compmat) 
        self.FM = self.FMμ # because final summary stat is compressed
        self.invFM = tf.linalg.inv(self.FM) # a.k.a fisher forecast
        self.lnDetF = self.logdet() # for the loss function.
        self.FI = tf.math.exp(self.lnDetF) 
        self.err = self.find_err()
        fm_casted = tf.cast(self.FM, dtype = self.err.dtype)
        self.forecast_bias = tf.linalg.solve(fm_casted, 
                            tf.transpose(tf.linalg.solve(fm_casted, self.err)))
        self.fractional_bias = tf.linalg.diag_part(
            self.forecast_bias/tf.cast(self.invFM, self.forecast_bias.dtype))
        
    def find_err(self):
        """
        Finds the fisher bias error based on 
        https://github.com/wcoulton/CompressedFisher.

        Returns
        -------
        bias : tf.Tensor
            The fisher bias error.

        """
        derivatives = self.derivatives
        invC = self.invC
        n_params = derivatives.shape[0]
        num_ders = tf.cast(tf.shape(derivatives)[1], dtype = derivatives.dtype)
        bias = tf.zeros((n_params, n_params), dtype=tf.float64)
        
        for i in range(n_params):
            for j in range(n_params):
                cov_ij = tfp.stats.covariance(derivatives[i], \
                                              derivatives[j])/num_ders
                trace_result = tf.linalg.trace(tf.linalg.matmul(cov_ij, \
                                                                invC))
                bias = tf.tensor_scatter_nd_add(bias, indices=[[i, j]], \
                                                updates=[trace_result])
        
        return bias
    
    def logdet(self):
        """
        Calculates the signed log determinant of the Fisher matrix.
        
        Returns
        -------
        tf.Tensor
            The signed log determinant.

        """
        matrix = self.FM
        lndet = tf.linalg.slogdet(matrix)
        return lndet[0] * lndet[1]
    
    def show_fm_and_bias(self):
        """
        Prints the log Fisher information, fisher matrix and bias.
        """
        print("log FI = ", np.round(self.lnDetF.numpy(), 2))
        print("FM = ", np.round(self.FM.numpy(), 2).tolist())
        print("Fractional bias = ", \
              np.round(self.fractional_bias.numpy().flatten(), 2))
            
        print("Constraints = ", np.sqrt(np.diag(self.invFM.numpy())))
                
class fisherFromVecs(baseFisher):
    """
    Fisher analysis given the summaries calculated at 
    [theta_fid, ..., (theta_fid - delta_theta/2)_i, \
     (theta_fid + delta_theta/2)_i, ...]. 
    The first set of simulations is used to estimate the covariance 
    matrix and the rest are used to estimate the derivative mean.
    """
    def __init__(self, summaries, delta_theta, name = "fromVecs", \
                 clean_data = True):
        """
        Initialize a Fisher class based on summaries.
        
        Parameters
        ----------
        summaries : list of tf.Tensors
            The summaries calculated at 
            [theta_fid, ..., (theta_fid - delta_theta/2)_i, \
             (theta_fid + delta_theta/2)_i, ...]. The signature of this method 
            should match with the generateData() method in the input simulator.
            The first set of simulations is used to estimate the covariance 
            matrix and the rest are used to estimate the derivative mean.
        delta_theta : tf.Tensor
            The step size that is used to calculate the derivatives.
        name : str, optional
            Name for the Fisher class. The default is None.
        clean_data : bool, optional
            Flag to clean data. If True, removes columns with zero variance.
            The default is True.
        """

        self.summaries = summaries
        self.delta_theta = delta_theta
        summaries = self.summaries
        self.n_s = summaries[0].shape[0] 
        self.n_d = summaries[1].shape[0]
        der_vecs = tf.stack(summaries[1:])       
        diffs = computeDerivatives(der_vecs, delta_theta)
        super().__init__(summaries[0], diffs, name, clean_data)
        pert_cov = tf.map_fn(computeCovariance, tf.stack(self.summaries[1:]))
        der_cov = computeDerivatives(pert_cov, delta_theta)
        self.FMSig = 0.5 * tf.einsum('ij,mjk,kl,nli->mn', self.invC, der_cov, self.invC, der_cov)
        
    def gaussian_mean_only_test(self, ks_threshold = 5e-2, \
                                cov_threshold = 0.15):
        """
        Checks the assumption if the data vectors satisfy the 
        "Gaussian Mean Only" condition - i.e., they are drawn from a 
        multivariate normal where the covariance is independent of the 
        parameters, i.e. there is a "weak" dependence of the covariance matrix 
        on the input parameters. This function checks for i) multivariate 
        gaussianity using the Henze-Zirkler test, ii) univariate Gaussianity 
        for each dimension of the summary statistic, iii) dependence of the 
        covariance matrix on the parameters. 
        

        Parameters
        ----------
        ks_threshold : float, optional
            The p-value. 
            The default is 5e-2.
        cov_threshold : float, optional
            The tolerance to check the covariance independence assumption. 
            The default is 1e-1.

        Returns
        -------
        bool
            A boolean indicating whether the data vectors satsify the 
            "Gaussian Mean Only" assumption.

        """
        summaries = self.summaries
        if summaries[0].shape[-1] == 1:
            is_multivariate_gaussian = [True for _ in summaries]
        else :
            pgs = [pg.multivariate_normality(item) for item in summaries]
            is_multivariate_gaussian = [item.normal for item in pgs] 
            for idx, item in enumerate(is_multivariate_gaussian): 
                if not item: 
                    print(f"summaries[{idx}] is not multivariate Gaussian " \
                          f"with p-value = {pgs[idx].pval:.4f}.")
        
        is_ks_test = []
        for idx, item in enumerate(summaries):
            p_vals = []
            for dim in range(tf.shape(summaries[0])[-1]):
                sample = item[:, dim] 
                sample = (sample - np.mean(sample)) / np.std(sample, ddof=1)
                ks_statistic, ks_p_value = kstest(sample, 'norm')
                p_vals.append(ks_p_value > ks_threshold)
                if ks_p_value < ks_threshold:
                    print(f"summaries[{idx}][:, {dim}] is not Gaussian with "\
                          f"p_val = {ks_p_value:.4f}.")
            is_ks_test.append(all(p_vals))
            
        cov_matrices = [tfp.stats.covariance(item) for item in self.summaries]
        diag_cov = np.array([np.diag(item) for item in cov_matrices])
        abs_diff_cov = diag_cov[2::2, :] - diag_cov[1::2, :]
        rel_diff_cov = np.abs(abs_diff_cov/ diag_cov[1::2, :])
        is_cov_param_ind = [all(item < cov_threshold) for item in rel_diff_cov]
        
        for idx, item in enumerate(is_cov_param_ind): 
            if not item:
                op = f'Covariance matrix at {idx} could have parameter ' \
                f'dependence with score = {np.round(rel_diff_cov[idx], 4)}' \
                f' with diag_cov_m[{idx}] = {np.round(diag_cov[2*idx + 1], 2)}'\
                f' and diag_cov_p[{idx}] = {np.round(diag_cov[2*idx + 2], 2)}.' 
                          
                print(op)
        return all(is_multivariate_gaussian + is_ks_test)
    
    def show_and_test(self):
        """
        Prints the log Fisher information, fisher matrix and bias. 
        It also performs the Gaussianity Mean Only check.
        """
        self.show_fm_and_bias()
        if not self.gaussian_mean_only_test():
            print("The summaries don't satisfy the Gaussian Mean Only "\
                  "assumption.")
        else : 
            print("The summaries satisfy the Gaussian Mean Only assumption.")
        

class fisherMOPED(fisherFromVecs):
    """
    Fisher analysis after compression using MOPED given the summaries 
    calculated at 
    [theta_fid, ..., (theta_fid - delta_theta/2)_i, \
     (theta_fid + delta_theta/2)_i, ...]. One half of the simulations is used 
    to estimate the MOPED compression matrix and the other half is used to 
    carry out the Fisher analysis of the compressed summaries. 
    """
    def __init__(self, summaries, delta_theta, name = "MOPED", \
                 compress_frac = 0.5, clean_data = True):
        """
        Initialize a MOPED Fisher class.
        
        Parameters
        ----------
        summaries : tf.Tensor
            Summaries for constructing the compression matrix (training) and
            for the Fisher analysis (testing).
        delta_theta : tf.Tensor
            The step size that is used to calculate the derivatives.
        name : str, optional
            Name for the Fisher class. The default is None.
        clean_data : bool, optional
            Flag to clean data. If True, removes columns with zero variance.
            The default is True.
        """
        if(clean_data) : 
            vecs_cov, vecs_perturbed = cleanData(summaries[0], summaries[1:])
            summaries = [vecs_cov, *vecs_perturbed]
        compress_frac_cov, compress_frac_ders = \
            self.assign_compress_fracs(compress_frac)
        
        self.delta_theta = delta_theta
        self._n_params = (len(summaries) - 1)//2 # Subtracting 1 for fiducial
        shuffled_summaries = self.shuffle_data_vectors(summaries)
        
        comp_cov, fisher_cov = \
            self.apply_covmat_split(shuffled_summaries[0], compress_frac_cov)
        
        comp_ders, fisher_ders = \
            self.apply_derivative_split(shuffled_summaries[1:], \
                                        compress_frac_ders)
        
        self.moped_compmat = self.compute_compression_matrix(comp_cov, \
                                                             comp_ders)
        
        fisher_vecs = [fisher_cov, *fisher_ders]

        compressed_vecs = [tf.matmul(item, self.moped_compmat) \
                           for item in fisher_vecs] 
        super().__init__(compressed_vecs, delta_theta, name, clean_data)
    
    def assign_compress_fracs(self, compress_frac):
        if type(compress_frac) is list:
            compress_frac_cov, compress_frac_ders = compress_frac[0], \
                                                        compress_frac[1]
        else : 
            compress_frac_cov = compress_frac
            compress_frac_ders = compress_frac
        
        return compress_frac_cov, compress_frac_ders
    
    def shuffle_data_vectors(self, summaries):
        seeds = np.random.randint(1e6, size = 1 + self._n_params)
        shuffled_summaries = []
        for idx in range(1 + 2*self._n_params):
            arr = shuffle_with_seed(summaries[idx], seeds[(1 + idx)//2])
            shuffled_summaries.append(arr)
        return shuffled_summaries
    
    def apply_covmat_split(self, fid_sims, compress_frac_cov):
        n_s = fid_sims.shape[0]
        cov_comp_sims = int(compress_frac_cov * n_s)
        return fid_sims[:cov_comp_sims], fid_sims[cov_comp_sims:]
    
    def apply_derivative_split(self, der_sims, compress_frac_ders):
        n_d = der_sims[0].shape[0]
        n_sims_comp = int(compress_frac_ders * n_d)
        comp_ders = []; fisher_ders = []
        for item in der_sims:
            comp_ders.append(item[:n_sims_comp])
            fisher_ders.append(item[n_sims_comp:])
        return comp_ders, fisher_ders
    
    def compute_compression_matrix(self, vecs_cov, der_vecs):
        derivatives = computeDerivatives(tf.stack(der_vecs), self.delta_theta)    
        ders = tf.math.reduce_mean(derivatives, axis = 1)
        moped_covariance = computeCovariance(vecs_cov)
        moped_compmat = \
            tf.linalg.solve(moped_covariance, tf.transpose(ders))
        return moped_compmat
    
def shuffle_with_seed(arr, seed):
    np.random.seed(seed)
    indices = np.random.permutation(arr.shape[0])
    return tf.gather(arr, indices)

def computeCovariance(vecs_cov):
    cov = tfp.stats.covariance(vecs_cov)
    n_sims_covmat, dim = tf.cast(tf.shape(vecs_cov)[0], cov.dtype), \
        tf.cast(tf.shape(vecs_cov)[1], cov.dtype)
    hartlap_fisher = tf.cast((n_sims_covmat - dim - 2.)/(n_sims_covmat - 1.),\
                             dtype = cov.dtype)
    return cov/hartlap_fisher   


def computeDerivatives(der_vecs, delta_theta):
    """
    Compute derivatives based on centered differentiaton. 

    Parameters
    ----------
    der_vecs : tf.Tensor
        The summaries evaluated at [..., (theta_fid - delta_theta/2)_i, \
         (theta_fid + delta_theta/2)_i, ...].
    delta_theta : tf.Tensor
        The step sizes used to calculate the derivatives.

    Returns
    -------
    tf.Tensor
        The corresponding derivatives of shape 
        [num(delta_theta), num_sims, num_summaries].

    """
    delta_theta_cast = tf.cast(delta_theta[:, tf.newaxis, tf.newaxis], \
                               der_vecs.dtype)
    return (der_vecs[1::2] - der_vecs[::2])/delta_theta_cast      


def cleanData(vecs_cov, derivatives):
    """
    Removes the columns with zero variance from vecs_cov and derivatives.

    Parameters
    ----------
    vecs_cov : tf.Tensor
        The summaries used to estimate the covariance matrix.
    derivatives : tf.Tensor
        The summaries used to estimate the derivative mean.

    Returns
    -------
    tf.Tensor, tf.Tensor
        The vecs_cov and derivatives after removing zero variance columns.

    """
    selector = sklearn.feature_selection.VarianceThreshold()
    selector.fit(vecs_cov)
    selInd = selector.get_support()
    ind = np.where(selInd)[0]
    if(any(~selInd)) : print("Indices = ", np.where(~selInd)[0], 
                          " are removed because they have zero variance.")
    return tf.gather(vecs_cov, ind, axis = -1), \
        tf.gather(derivatives, ind, axis = -1)