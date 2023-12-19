#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 13:06:00 2023

@author: karthikviswanathan
"""
import sys, os, pickle
import glob
import numpy as np
import tqdm
from tqdm import tqdm
import tensorflow as tf
import topofisher
from topofisher.fisher.Fisher import show_fm_and_bias
from topofisher.fisher.imnn import IMNNLayer, ExtraDimLayer
from topofisher.pipelines.utils import writeToFile
import matplotlib.pyplot as plt
from topofisher.fisher.plot_fisher_stats import plotContours2D, plotSummaryDerivativeHists, plot_derivative_convergence

inputFileName = sys.argv[1]
outputFolderName = sys.argv[2]
numIterations = int(sys.argv[3])

pickle_files = glob.glob(os.path.join(inputFileName, '*/*.pkl'))
all_vecs = {}
der_params = ["Om", "s8"]

param_list = ["fiducial"]
delta_theta = []
if "Om" in der_params:
   param_list.extend(["Om_m", "Om_p"]) 
   delta_theta.append(0.02)

if "s8" in der_params:
    param_list.extend(["s8_m", "s8_p"])
    delta_theta.append(0.03)

delta_theta = np.array(delta_theta)
    
for item in param_list : all_vecs[item] = []
print(param_list)
for file_path in pickle_files:
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        param = file_path.split("/")[-2][12:]
        if param in param_list:
            all_vecs[param].append(data)

all_arr = [np.concatenate(all_vecs[item], axis = 0) for item in tqdm(param_list)]
all_arr = [np.transpose(item, (0, 2, 3, 1)) for item in all_arr]
all_arr[0] = all_arr[0][:7500]
all_arr = tf.stack(all_arr, axis = 1)


def plotLearningGraphs(history, fileName):
    ncols = 4
    fig, axes = plt.subplots(nrows=1, ncols= ncols, figsize=(3 * ncols + 2 , 3))
    lis = ['loss', 'lnfi', 'bias0', 'bias1']
    for idx in range(4) :
        axes[idx].plot(history[lis[idx]][1:], label = "Training")
        axes[idx].plot(history['val_' + lis[idx]][1:], label = "Validation")
        if "bias" in lis[idx]: axes[idx].axhline(0.2, linestyle = 'dotted', c = 'black')
        axes[idx].set_title(lis[idx])
    fig.legend(labels=['Training', 'Validation']) 
    fig.savefig(fileName)    
        
def run_pi_imnn(all_arr, outputFolder, reg = 0):
    all_vecs = tf.random.shuffle(all_arr, seed = np.random.randint(1e3))
    print(all_vecs.shape)
    res = all_vecs.shape[-2]
    num_input_filters = all_vecs.shape[-1]
    if res < 50:
        model = tf.keras.Sequential(
            [
                ExtraDimLayer(tf.keras.layers.Conv2D(32, (3,3), padding='same', \
                                                     activation = "relu", \
                                        input_shape=(res, res, num_input_filters))),
                ExtraDimLayer(tf.keras.layers.MaxPooling2D((2, 2), strides = 2)),        
                ExtraDimLayer(tf.keras.layers.Flatten()),
                ExtraDimLayer(tf.keras.layers.Dense(256, activation="relu")),
                ExtraDimLayer(tf.keras.layers.Dense(len(delta_theta)))
            ]
        )
    else : 
        model = tf.keras.Sequential(
            [
                ExtraDimLayer(tf.keras.layers.Conv2D(res, (3,3), padding='same', \
                                                     activation = "relu", \
                                        input_shape=(res, res, num_input_filters))),
                ExtraDimLayer(tf.keras.layers.MaxPooling2D((2, 2), strides = 2)),       
                
                
                ExtraDimLayer(tf.keras.layers.Conv2D(res, (3,3), padding='same', \
                                                     activation = "relu")),
                ExtraDimLayer(tf.keras.layers.MaxPooling2D((2, 2), strides = 2)), 
                ExtraDimLayer(tf.keras.layers.Flatten()),
                tf.keras.layers.Dense(2*res, activation="relu"),
                tf.keras.layers.Dense(len(delta_theta))
                ]
            )
                
    """
                ExtraDimLayer(tf.keras.layers.Conv2D(2 * res, (3,3), padding='same', \
                                                     activation = "relu")),
                ExtraDimLayer(tf.keras.layers.MaxPooling2D((2, 2), strides = 2)), 
                
                
                ExtraDimLayer(tf.keras.layers.Conv2D(2 * res, (3,3), padding='same', \
                                                     activation = "relu")),
                ExtraDimLayer(tf.keras.layers.MaxPooling2D((2, 2), strides = 2)), 
                
    """

            

    # 40 for \omega_m, 15 for \sigma_8
    pi_imnn_layer = IMNNLayer(model, run_eagerly = False, verbose = 1, epochs = 80,\
                              data_splits = [0.4, 0.2, 0.4], \
                                callbacks = [tf.keras.callbacks.EarlyStopping(
                                patience = 3, restore_best_weights = True)],\
                                stack = False, show_bias = True, transpose = False, \
                                batch_size = 512, reg = tf.constant([reg, 2.*reg]), show_fi = True)
    fisher = pi_imnn_layer.computeFisher(all_vecs, delta_theta)
    show_fm_and_bias(fisher)
    plotSummaryDerivativeHists(fisher, outputFolder + "/plots") 
    plotLearningGraphs(model.history.history, outputFolder + "/plots/learning.png")

lis = []
reg_list = 3. * np.ones (numIterations)
for idx in range(len(reg_list)):
    try: run_pi_imnn(all_arr, outputFolderName + "/run" + str(idx), reg_list[idx])
    except Exception as e:
        print("An error occured: ", e)
