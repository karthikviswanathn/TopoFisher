#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 23:56:03 2023

@author: karthikviswanathan
"""

import matplotlib.pyplot as plt
import CompressedFisher

def standard_fisher(fid_arr, der_arr, params_fid, delta_params, parameter_names):
    dict_param_steps = {parameter_names[i]:delta_params[i] for i in range(len(params_fid))}
    dict_deriv_sims = {parameter_names[i]:der_arr[2*i: 2*i + 2] for i in range(len(params_fid))}
    nSims_deriv = der_arr.shape[1]
    cFisher = CompressedFisher.gaussianFisher(parameter_names, nSims_deriv, \
                                              include_covmat_param_depedence=False,
                                          deriv_finite_dif_accuracy=2)
    compress_frac_split_ders = 0.001
    covmat_sims = fid_arr
    cFisher.initailize_covmat(covmat_sims, True)
    cFisher.initailize_mean(covmat_sims)
    cFisher.initailize_deriv_sims(dic_deriv_sims=dict_deriv_sims, dict_param_steps=dict_param_steps)
    cFisher.generate_deriv_sim_splits(compress_fraction=compress_frac_split_ders)
    stnd_constraint    = cFisher.compute_fisher_forecast(parameter_names)
    stnd_constraint_bias = cFisher.est_fisher_forecast_bias(parameter_names)
    # print(cFisher._compute_fisher_matrix(parameter_names))
    print('Parameter  \t standard Fisher \t Est. Fractional bias ')
    for i,name in enumerate(parameter_names):
        print(f'{name}  \t\t  {stnd_constraint[i,i]**.5:.3f} \t\t     {(stnd_constraint_bias/stnd_constraint)[i,i]:.3f} ')
    
    std_nsim, std_mns,std_stds = cFisher.run_fisher_deriv_stablity_test(parameter_names,)
    for i,p in enumerate(parameter_names):
        plt.errorbar(std_nsim/std_nsim[-1],std_mns[:,i,i]**.5/std_mns[-1,i,i]**.5,yerr=std_stds[:,i,i]**.5/std_mns[-1,i,i]**.5,label=parameter_names[i])
#plt.plot(geom_nsim/geom_nsim[-1],(geom_nsim/geom_nsim[-1])**.5,scaley=False,color='k',alpha=.5,linestyle=':')
        plt.title("Convergence of standard forecasts")
    plt.legend()
    plt.show()
    return cFisher._compute_fisher_matrix(parameter_names)

def compressed_and_combined_fisher(fid_arr, der_arr, params_fid, delta_params, parameter_names):

    dict_param_steps = {parameter_names[i]:delta_params[i] for i in range(len(params_fid))}
    dict_deriv_sims = {parameter_names[i]:der_arr[2*i: 2*i + 2] for i in range(len(params_fid))}
    nSims_deriv = der_arr.shape[1]
    cFisher = CompressedFisher.gaussianFisher(parameter_names, nSims_deriv, \
                                              include_covmat_param_depedence=False,
                                          deriv_finite_dif_accuracy=2)
    compress_frac_split_ders = 0.5
    compress_frac_split_cov = 0.5
    covmat_sims = fid_arr
    
    cFisher.initailize_covmat(covmat_sims, True)
    cFisher.generate_covmat_sim_splits(compress_fraction=compress_frac_split_cov)
    cFisher.initailize_mean(covmat_sims)
    cFisher.initailize_deriv_sims(dic_deriv_sims=dict_deriv_sims, dict_param_steps=dict_param_steps)
    cFisher.generate_deriv_sim_splits(compress_fraction=compress_frac_split_ders)
    
    compressed_constraint = cFisher.compute_compressed_fisher_forecast(parameter_names)
    compressed_constraint_bias = cFisher.est_compressed_fisher_forecast_bias(parameter_names)
    
    print('Parameter \t Compressed Fisher \t Est. Fractional bias ')
    for i,name in enumerate(parameter_names):
        print(f'{name}   \t\t  {compressed_constraint[i,i]**.5:.3f} \t\t   \
        {(compressed_constraint_bias/compressed_constraint)[i,i]:.3f} ')
    
    
    combined_constraint = cFisher.compute_combined_fisher_forecast(parameter_names)
    # print(f'Parameter  standard Fisher  Bias compressed comp Bias combined ')
    print('Parameter \t Combined Fisher  ')
    for i,name in enumerate(parameter_names):
        print(f'{name}   \t\t  {combined_constraint[i,i]**.5:.3f}')
    
    comp_nsim, comp_mns,comp_stds = cFisher.run_compressed_fisher_deriv_stablity_test(parameter_names,compress_frac_split_ders)
    geom_nsim, geom_mns,geom_stds = cFisher.run_combined_fisher_deriv_stablity_test(parameter_names,compress_frac_split_ders)
    
    for i,p in enumerate(parameter_names):
        plt.errorbar(comp_nsim/comp_nsim[-1],comp_mns[:,i,i]**.5/comp_mns[-1,i,i]**.5,yerr=comp_stds[:,i,i]**.5/comp_mns[-1,i,i]**.5,label=parameter_names[i])
    plt.title("Convergence of compressed forecasts")
    plt.legend()
    plt.ylim([.5,2])
    plt.show()
    
    for i,p in enumerate(parameter_names):
        plt.errorbar(geom_nsim/geom_nsim[-1],geom_mns[:,i,i]**.5/geom_mns[-1,i,i]**.5,yerr=geom_stds[:,i,i]**.5/geom_mns[-1,i,i]**.5,label=parameter_names[i])
#plt.plot(geom_nsim/geom_nsim[-1],(geom_nsim/geom_nsim[-1])**.5,scaley=False,color='k',alpha=.5,linestyle=':')
    plt.title("Convergence of compbined forecasts")    
    plt.legend()
    plt.ylim([.5,2])
    plt.show()
    
    return cFisher._compute_compressed_fisher_matrix(parameter_names), \
        cFisher._compute_combined_fisher_matrix(parameter_names)
        