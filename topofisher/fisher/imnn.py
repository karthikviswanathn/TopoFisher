#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 15:41:02 2023

@author: karthikviswanathan
"""
from . import Fisher, hztest
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

relu = tf.keras.layers.ReLU()

class FisherLayer:
    def __init__(self, name = "Fisher"):
        """
        Initialize a FisherLayer.
        
        Parameters
        ----------
        name : str, optional
            Name for the FisherLayer. The default is Fisher.
        """
        self.name = name

    def computeFisher(self, all_vecs, delta_theta):
        """
        Compute Fisher information using the provided vectors and delta_theta.
        
        Parameters
        ----------
        all_vecs : tf.Tensor
            Input summary vectors.
        delta_theta : tf.Tensor
            The step sizes for the Fisher analysis.
        
        Returns
        -------
        Fisher.fisherFromVecs
            Fisher information object.
        """
        fisher = Fisher.fisherFromVecs(all_vecs, delta_theta)
        return fisher

class MopedLayer(FisherLayer):
    def __init__(self, compression_matrix = None, compression_frac = 0.5, name = "MOPEDFisher"):
        """
        Initialize a MOPEDLayer.
        
        Parameters
        ----------
        compression_matrix : tf.Tensor, optional
                             The matrix to perform compression. If None, the 
                             MOPED compression matrix is estimated from a part 
                             of the dataset and the Fisher analysis is carried
                             out by compressing the rest of the dataset.
        name : str, optional
            Name for the MOPEDLayer. The default is "MOPEDFisher".
        """
        self.compression_matrix = compression_matrix
        self.compression_frac = compression_frac
        super().__init__(name)
    
    def computeFisher(self, all_vecs, delta_theta):
        """
        Compute Fisher information using MOPED compression from the provided
        vectors and step size. 

        Parameters
        ----------
        all_vecs : tf.Tensor
            Input summary vectors.
        delta_theta : tf.Tensor
            The step sizes for the Fisher analysis.

        Returns
        -------
        Fisher.fisherMOPED
            MOPED compression based Fisher information object.
        """
        if self.compression_matrix is None :
            fisher = Fisher.fisherMOPED(all_vecs, delta_theta, compress_frac = self.compression_frac)
            self.compression_matrix = fisher.moped_compmat
            return fisher
        else :
            compressed_vecs = tf.einsum("...j, jk->...k", all_vecs, 
                                        self.compression_matrix)
            fisher = Fisher.fisherFromVecs(compressed_vecs, delta_theta)
            return fisher

class IMNNLayer(FisherLayer):
    def __init__(self, comp, 
                 data_splits = [0.5, 0.25, 0.25], \
                 optimizer = tf.keras.optimizers.Adam(learning_rate= 1e-3), \
                 epochs = 100, batch_size = 512, verbose = 0, \
                 callbacks = None, transpose = True, moped = False, \
                 run_eagerly = False, show_bias = False, stack = True, \
                 reg = tf.constant(0.), hz_strength = tf.constant(0.), show_fi = False, \
                 hz_threshold = 0.1,
                 show_hz = False, name = "Compression Layer"):
        """
        Initialize an IMNNLayer.
        
        Parameters
        ----------
        comp : tf.keras.Model
            Compression model to extract information from the persistent 
            summaries.
        data_splits : list, optional
            Data split percentages for training, validation, and testing.
            The default is [0.5, 0.25, 0.25].
        optimizer : tf.keras.optimizers.Optimizer, optional
            Optimizer for training. The default is Adam with learning rate 1e-3.
        epochs : int, optional
            Number of training epochs. The default is 100.
        batch_size : int, optional
            Batch size for training. The default is 512.
        verbose : int, optional
            Verbosity level for training. The default is 0.
        callbacks : list, optional
            List of callbacks for training. The default is None.
        transpose : boolean, optional
            Indicates if the dimensions correspoding to the batch and
            the different thetas should be transposed. The default is True.
        moped : boolean, optional
                Indicates whether to use MOPED compression to calculate the 
                Fisher information. The default is False.
        run_eagerly : boolean, optional
                Indicates whether to compile the model in 'eager' mode.
                The default is False.
        show_bias : boolean, optional
                Indicates whether to show the biases during training.
                The default is False.
        stack : boolean, optional
                Indicates whether to stack the vectors or not.
                The default is False.
        name : str, optional
            Name for the IMNNLayer. The default is "Compression Layer".
        """
        self.comp = comp
        self.data_splits = data_splits
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.callbacks = callbacks
        self.transpose = transpose
        self.moped = moped
        self.run_eagerly = run_eagerly
        self.show_bias = show_bias
        self.is_trained = False
        self.stack = stack
        self.reg = reg
        self.hz_strength = hz_strength
        self.hz_threshold = hz_threshold
        self.show_fi = show_fi
        self.show_hz = show_hz
        if moped is False: self.fisher_func = Fisher.fisherFromVecs
        else : self.fisher_func = Fisher.fisherMOPED
        super().__init__(name)
    
    def split_data(self, all_vecs):
        """
        Split input vectors into training, validation, and test sets.
        
        Parameters
        ----------
        all_vecs : tf.Tensor
            Input vectors.
        
        Returns
        -------
        train_vecs : tf.Tensor
            Training vectors.
        valid_vecs : tf.Tensor
            Validation vectors.
        test_vecs : tf.Tensor
            Test vectors.
        """
        # Transposing the first and the second index
        # Old first index - indexing covariance and derivative simulations.
        # Old second index - indexing simulation number.
        vecs = tf.stack(all_vecs) if self.stack else all_vecs
        inp_vecs = tf.einsum("ij...->ji...", vecs) \
            if self.transpose else all_vecs
        div = (np.array(self.data_splits) * inp_vecs.shape[0]).astype(int)
        div = np.cumsum(div)
        train_vecs, valid_vecs, test_vecs = \
            inp_vecs[:div[0]], inp_vecs[div[0]:div[1]], inp_vecs[div[1]:div[2]]
        return train_vecs, valid_vecs, test_vecs
    
    def train_imnn(self, train_vecs, valid_vecs, delta_theta):
        """
        Train and validate the IMNN model.

        Parameters
        ----------
        train_vecs : tf.Tensor
            Training vectors.
        valid_vecs : tf.Tensor
            Validation vectors.
        delta_theta : tf.Tensor
            The step sizes for Fisher Analysis.
        """
        custom_loss = self.fisher_loss(delta_theta)
        model = self.comp
        metrics = []
        if(self.show_fi):
            metrics.append(self.lnfi_metrics(delta_theta))
        if(self.show_bias):
            metrics.extend([self.bias_metrics(delta_theta, idx) \
                       for idx in range(len(delta_theta))])
        
        if(self.show_hz):
            metrics.extend([self.hz_metrics(delta_theta)])
        
        model.compile(optimizer = self.optimizer, loss = custom_loss, 
                      run_eagerly = self.run_eagerly, metrics = metrics)
        # Creating "fake" training data since this optimization is unsupervized.
        train_y = tf.zeros(shape = (train_vecs.shape[0], 1))
        validation_data = (valid_vecs, \
                           tf.zeros(shape = (valid_vecs.shape[0], 1)))

        # Fitting the model
        self.history = model.fit(train_vecs, train_y,\
                                 epochs = self.epochs, \
                                  batch_size = self.batch_size, \
                                     validation_data = validation_data, \
                                         verbose = self.verbose, \
                                             callbacks = self.callbacks)
        self.is_trained = True
    
    def computeFisher(self, all_vecs, delta_theta):
        """
        Train and compute Fisher information using the trained IMNN model.

        Parameters
        ----------
        all_vecs : tf.Tensor
            Input vectors.
        delta_theta : tf.Tensor
           Step sizes for Fisher analysis.

        Returns
        -------
        Fisher.fisherFromVecs
            Fisher information object.
        """
        if(not self.is_trained) :
            train_vecs, valid_vecs, test_vecs = self.split_data(all_vecs)
            self.train_imnn(train_vecs, valid_vecs, delta_theta)
            # Transposing data so that it is compatible for the Fisher 
            # analysis. The input format is given by the fisherFromVecs() init
            # function.
            if self.transpose == True : 
                test_vecs = tf.einsum("ij...->ji...", test_vecs) 
        else : test_vecs = all_vecs
        if (self.transpose == True) :
            fisher = self.fisher_func(self.comp(test_vecs), delta_theta)
        else : 
            final_vecs = self.comp(test_vecs)
            transposed_vecs = tf.einsum("ij...->ji...", final_vecs)
            fisher = self.fisher_func(transposed_vecs, delta_theta)
        return fisher
      
    def fisher_loss(self, delta_theta):
        """
        Defines the Fisher Information based loss function.
        
        Parameters
        ----------
        delta_theta : tf.Tensor
            The step sizes for Fisher analysis.
        
        Returns
        -------
        loss_fn
            The negative log determinant of the Fisher matrix.
        """
        def loss_fn(y_true, y_pred):
            # Transposing data so that it is compatible for the Fisher 
            # analysis. The input format is given by the fisherFromVecs() init
            # function.
            if len(y_pred.shape) == 2 : 
                y_pred = tf.expand_dims(y_pred, axis = -1)
            y_t = tf.einsum("ij...->ji...", y_pred)
            fish = self.fisher_func(y_t, delta_theta, clean_data = False) 
            lnDetF = fish.lnDetF
            
            hz_score = tf.constant(0.)
            if y_t[0].shape[-1] > 1:
               hz_score = tf.reduce_sum(relu(self.hz_threshold - tf.map_fn(hztest.multivariate_normality, y_t)))
            
            reg_term = tf.cast(
                           tf.reduce_sum(self.reg * \
                            tf.cast(fish.fractional_bias, dtype = self.reg.dtype)),\
                             dtype = lnDetF.dtype) + self.hz_strength * hz_score  
            loss = -lnDetF +  reg_term
            return loss
        return loss_fn
    
    def bias_metrics(self, delta_theta, idx):
        def bias(y_true, y_pred):
            if len(y_pred.shape) == 2 : 
                y_pred = tf.expand_dims(y_pred, axis = -1)
            y_t = tf.einsum("ij...->ji...", y_pred)
            fish = self.fisher_func(y_t, delta_theta, clean_data = False)
            return fish.fractional_bias[idx]
        bias.__name__ = 'bias' + str(idx)
        return bias

    def fm_metrics(self, delta_theta, idx):
        def fm(y_true, y_pred):
            if len(y_pred.shape) == 2 : 
                y_pred = tf.expand_dims(y_pred, axis = -1)
            y_t = tf.einsum("ij...->ji...", y_pred)
            fish = self.fisher_func(y_t, delta_theta, clean_data = False)
            return np.diag(fish.FM.numpy())[idx]
        fm.__name__ = 'fm' + str(idx)
        return fm
    
    def lnfi_metrics(self, delta_theta):
        def lnfi(y_true, y_pred):
            if len(y_pred.shape) == 2 : 
                y_pred = tf.expand_dims(y_pred, axis = -1)
            y_t = tf.einsum("ij...->ji...", y_pred)
            fish = self.fisher_func(y_t, delta_theta, clean_data = False)
            return fish.lnDetF
        lnfi.__name__ = 'lnfi' 
        return lnfi
    
    def hz_metrics(self, delta_theta):
        def hz(y_true, y_pred):
            if len(y_pred.shape) == 2 : 
                y_pred = tf.expand_dims(y_pred, axis = -1)
            y_t = tf.einsum("ij...->ji...", y_pred)
            hz_score = tf.constant(0.)
            if y_t[0].shape[-1] > 1:
               hz_score = tf.reduce_sum(relu(self.hz_threshold - tf.map_fn(hztest.multivariate_normality, y_t)))
            return hz_score
        hz.__name__ = 'hz' 
        return hz

class ExtraDimLayer(tf.keras.layers.Layer):
    def __init__(self, inp_layer, **kwargs):
        """
        Initialize an ExtraDimLayer.

        Parameters
        ----------
        inp_layer : tf.keras.layers.Layer
            Input layer.
        """
        super(ExtraDimLayer, self).__init__(**kwargs)
        self.inp_layer = inp_layer
    def call(self, inputs):
        """
        Call method for the ExtraDimLayer.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.

        Returns
        -------
        output : tf.Tensor
            Output tensor.
        """
        lis = []
        for idx in range(inputs.shape[1]):
            op = self.inp_layer(inputs[:, idx])
            lis.append(op)
        output = tf.stack(lis, axis = 1)
        return output

def plot_loss(history, file_loc = None, log_tfi = None):
    """
    Plot training and validation loss.
    
    Parameters
    ----------
    history : tf.keras.callbacks.History
        Training history.
    file_loc : str, optional
        File location to save the plot. The default is None.
    log_tfi : float, optional
        Logarithm of the target Fisher information for reference. 
        The default is None.
    """
    loss_values = history.history['loss']
    val_loss_values = history.history['val_loss']
    epochs = range(1, len(loss_values)+1)
    
    plt.plot(epochs, loss_values, label='Training Loss')
    plt.plot(epochs, val_loss_values, label='Validation Loss')
    if (log_tfi is not None) : 
        plt.axhline(-log_tfi, c = 'r', label = 'log_tfi', \
                    linestyle = 'dotted')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if (file_loc is not None) : plt.savefig(file_loc)
    plt.show()
    
def plotLearningGraphs(history, metric_list = ['loss', 'lnfi', 'bias0', 'bias1'],\
                       fileName = None, tfi_dict = {}):
    ncols = len(metric_list)
    fig, axes = plt.subplots(nrows=1, ncols= ncols, figsize=(3 * ncols + 2 , 3))
    for idx in range(ncols) :
        ax = axes[idx]
        m_key = metric_list[idx]
        ax.plot(history[m_key][1:], label = "Training")
        ax.plot(history['val_' + m_key][1:], label = "Validation")
        if "bias" in m_key: 
            ax.axhline(0.2, linestyle = 'dotted', c = 'black')
        if m_key in tfi_dict.keys(): 
            ax.axhline(tfi_dict[m_key], linestyle = 'dotted', c = 'black')
        ax.set_title(metric_list[idx])
    fig.legend(labels=['Training', 'Validation']) 
    if fileName is not None : fig.savefig(fileName)
    else : plt.show()
