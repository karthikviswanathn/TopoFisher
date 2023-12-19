#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 12:13:18 2023

@author: karthikviswanathan
"""
import tensorflow as tf
import tqdm
from tqdm import tqdm
import numpy as np
import gudhi
from . import filt_utils

class MixDTMAndL:
    def __init__(self, p):
        self.p = p
        self.name = "Mix DTML"
    
    def ef_tf(self, pts, vf, inds):
        p = self.p
        fmax = tf.math.maximum(tf.gather(vf, inds[:,0]), \
                               tf.gather(vf, inds[:,1]))
         
        edge_coords = tf.gather(pts, inds)
        start, end = edge_coords[:, 0, :], edge_coords[:, 1, :] 
        d = tf.linalg.norm(start - end, axis = -1)
        return (d**p + fmax**p)**(1/p)
    @tf.function
    def ef_tf_batched(self, all_pts, vFilts, inds):
        """
        

        Parameters
        ----------
        all_pts : TYPE
            DESCRIPTION.
        vFilts : TYPE
            DESCRIPTION.
        inds : list of shape (num_sims, None, 2)
            List of outer length = num_sims containing the edges that are part 
            of the flag persistence generators. The inner dimensions stand for 
            the number of points on the persistence diagrams and the edges from 
            the point cloud that constitute either the birth or the death 
            simplex.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        p = self.p
        inds_ragged = filt_utils.stack_ragged(inds, False)
        end_pts = tf.gather(all_pts, inds_ragged, axis = 1, batch_dims = 1)
        diff_vector = end_pts[:, :, :, 1] - end_pts[:, :, :, 0]
        ragged_tensor = diff_vector
        squared_norm = tf.reduce_sum(ragged_tensor**2, axis=-1)
        
        # Compute the Euclidean norm by taking the square root
        d = tf.sqrt(squared_norm)
        vfilts_end_pts = tf.gather(vFilts, inds_ragged, axis = 1, \
                                   batch_dims = 1)
        fmax = tf.math.reduce_max(vfilts_end_pts, axis = -1)
        return (d**p + fmax**p)**(1/p)
    
    # TODO : Note that the input signatures are different with the "tf" version.
    def ef_np(self, pts, filt, inds):
        p = self.p
        pc = p.numpy() if(tf.is_tensor(p)) else p 
        inda, indb = inds[:,0], inds[:,1]    
        d = np.linalg.norm(pts[inda] - pts[indb], axis = -1)
        fx, fy = filt[inda], filt[indb] 
        fmax = np.maximum(fx, fy)
        return (d**pc+fmax**pc)**(1/pc)

class FlagComplexLayer(tf.keras.layers.Layer):
    def __init__(self, max_hom_dim, complex_args, efilt = MixDTMAndL(p = 1), 
                 tqdm = False, is_ragged = True, data_format = 'batch_first',
                 batched_gather = True, **kwargs):
        tf.keras.layers.Layer.__init__(self, **kwargs)
        self.complex_args = complex_args
        self.max_hom_dim = max_hom_dim
        self.efilt = efilt
        self.tqdm = tqdm
        self.is_ragged = is_ragged
        self.data_format = data_format
        self.batched_gather = batched_gather
    
    def computePP(self, allPts, allFilts):
        pp = []        
        iter_obs = tqdm(zip(allPts, allFilts), total = len(allPts)) \
            if self.tqdm else zip(allPts, allFilts)
        for pts, vfilt in iter_obs:
            st = self.updateFiltration(pts, vfilt)
            pp.append(st.flag_persistence_generators())     
        return pp
    
    def updateFiltration(self, pts, vfilt):
        st =  self.getSimplexTree(pts)
        vert = np.array([s[0] for s in st.get_skeleton(0)])
        edges = np.array([s[0] for s in st.get_skeleton(1)\
                          if len(s[0]) == 2])
        
        efilt = self.efilt.ef_np(pts, vfilt, edges)
        st = gudhi.SimplexTree()
        st.insert_batch(vert.T, vfilt.reshape(-1))
        st.insert_batch(edges.T, efilt)
        st.expansion(self.max_hom_dim)
        st.make_filtration_non_decreasing()
        st.persistence()
        return st;
    
    def getSimplexTree(self, pts, default = False):
        complex_args = self.complex_args
        complex_type = complex_args['complex_type']
        
        if(complex_type  == "alpha") :
            alpha = gudhi.AlphaComplex(points = pts)
            st = alpha.create_simplex_tree(\
                                        default_filtration_value = not default)
            
        if(complex_type  == "rips") :
            max_edge = complex_args['max_edge'] \
                if 'max_edge' in complex_args.keys() else np.inf
            sparse = complex_args['sparse'] if 'sparse' in complex_args.keys() \
                                            else None
            rips = gudhi.RipsComplex(points = pts, max_edge_length = max_edge, \
                                     sparse = sparse)
            st = rips.create_simplex_tree(max_dimension = 2)
        
        return st

    def getPDFromPairs(self, pts, pers_pairs, vFilts):                     
        efilt = self.efilt
        pd0, pd1 = [], []
        ppb0 = []; ppd0 = []; ppb1 = []; ppd1 = []; 
        if self.batched_gather :
            for item in pers_pairs:
                
                ppb0.append(np.array(item[0][:, 0]))
                
                ppd0.append(np.array(item[0][:, 1:]))
                ppb1.append(np.array(item[1][0][:, :2]))
                ppd1.append(np.array(item[1][0][:, 2:]))
            b0 = tf.gather(vFilts, filt_utils.stack_ragged(ppb0, False),  \
                           axis = 1, batch_dims = 1)
            
            d0 = efilt.ef_tf_batched(pts, vFilts, ppd0)
            b1 = efilt.ef_tf_batched(pts, vFilts, ppb1)
            d1 = efilt.ef_tf_batched(pts, vFilts, ppd1)
            pd0 = tf.stack([b0, d0], axis = -1)
            pd1 = tf.stack([b1, d1], axis = -1)
            return [pd0, pd1]
             
        else :
            for pt, pers, vf  in zip(pts, pers_pairs, vFilts) :
                ind0, ind1 = pers[0], pers[1][0]
                # print(len(ind0), len(ind1))
                # print(len(pers[1][0]))
                b0 = tf.gather(vf, ind0[:,0])
                d0 = efilt.ef_tf(pt, vf, ind0[:, 1:])
                b1 = efilt.ef_tf(pt, vf, ind1[:, :2])
                d1 = efilt.ef_tf(pt, vf, ind1[:, 2:])
                """
                d0 = tf.gather_nd(dx, ind0[:, 1:]) 
                b1 = tf.gather_nd(dx, ind1[:, :2])
                d1 = tf.gather_nd(dx, ind1[:, 2:])
                """
                diag0, diag1 = tf.stack([b0, d0], axis = -1), tf.stack([b1, d1], \
                                                                       axis = -1)
                pd0.append(diag0)
                pd1.append(diag1)
        return [pd0, pd1]
    
    def post_process(self, pds_all_hom_dims):            
        if self.data_format == 'batch_first': 
            transposed_diags = [list(map(list, zip(*dgms))) for dgms in \
                              pds_all_hom_dims]
            if not self.is_ragged : return transposed_diags
            else:
                all_pds_list = []
                for dgms in pds_all_hom_dims: # Iterating over hom_dim
                    ragged_dgm = dgms if self.batched_gather \
                        else filt_utils.stack_ragged(dgms, tensorize = False)        
                    all_pds_list.append(ragged_dgm)
                    
                stacked = tf.concat(all_pds_list, axis = 0)
                ragged_dgms = tf.RaggedTensor.from_uniform_row_length(
                                stacked, len(pds_all_hom_dims))
                return ragged_dgms
            
                
        if self.data_format == 'hom_dim_first' : 
            if self.is_ragged :
                all_pds_list = []
                for dgms in pds_all_hom_dims: # Iterating over hom_dim
                    ragged_dgm = dgms if self.batched_gather \
                        else filt_utils.stack_ragged(dgms, tensorize = False)        
                    all_pds_list.append(ragged_dgm)
                    
                stacked = tf.concat(all_pds_list, axis = 0)
                ragged_dgms = tf.RaggedTensor.from_uniform_row_length(
                    stacked, stacked.shape[0]//len(pds_all_hom_dims))
                return ragged_dgms
            else : return pds_all_hom_dims
        else : raise ValueError("Check self.data_format!")
            
    
    def call(self, inputs):
        vFilts = self.vFilt(inputs)
        pps = self.computePP(inputs.numpy(), vFilts.numpy())
        pds = self.getPDFromPairs(inputs, pps, vFilts)
        return self.post_process(pds)
        

