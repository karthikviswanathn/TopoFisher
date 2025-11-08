"""
MMA (Multiparameter Module Approximation) filtration as a PyTorch module.
"""
from typing import List, Optional
import torch
import torch.nn as nn
import numpy as np
import multipers as mp


class MMALayer(nn.Module):
    """
    PyTorch module for computing MMA (Multiparameter Module Approximation).
    
    This layer takes grid data (field) and a second parameter (gradient) and 
    computes multiparameter persistent homology using Freudenthal triangulation
    and MMA approximation. Handles both single samples and batches.
    
    Unlike standard persistent homology which produces persistence diagrams,
    MMA produces module approximation objects that require specialized vectorization.
    """

    def __init__(self, nlines=None, max_error=None, homology_dimensions=None):
        """
        Initialize MMA layer.

        Args:
            nlines: Number of lines to use for MMA approximation (default: 500)
                   Controls the trade-off between approximation quality and 
                   computational cost. Higher values = better approximation but slower.
                   If max_error is specified, nlines is ignored.
            max_error: Maximum error threshold for MMA approximation (default: None)
                      If specified, this takes precedence over nlines.
                      Used for testing error bounds.
            homology_dimensions: List of homology dimensions (e.g., [0, 1])
                                Kept for compatibility with pipeline, but MMA computes
                                all dimensions together.
        """
        super().__init__()
        
        # Handle nlines vs max_error
        if max_error is not None:
            self.max_error = max_error
            self.nlines = -1  # Will be ignored by multipers
        elif nlines is not None:
            self.nlines = nlines
            self.max_error = -1  # Disable max_error
        else:
            # Default: use nlines=500
            self.nlines = 500
            self.max_error = -1
            
        self.homology_dimensions = homology_dimensions if homology_dimensions is not None else [0, 1]

    def forward(self, field, gradient):
        """
        Compute MMA from field and gradient using Freudenthal triangulation.

        Args:
            field: Input tensor of shape (H, W) for single 2D sample or
                   (n_samples, H, W) for batch of 2D samples
            gradient: Second parameter tensor, same shape as field

        Returns:
            List of module approximation objects, one per sample.
            Each object contains the multiparameter module approximation
            and can be used with specialized MMA vectorization layers.
        """
        # Handle single sample vs batch
        if field.ndim == 2:
            field = field.unsqueeze(0)
            gradient = gradient.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False

        assert field.shape == gradient.shape, \
            f"field and gradient must have same shape, got {field.shape} vs {gradient.shape}"

        n_samples = field.shape[0]
        
        # MMA computation is CPU-only (multipers uses GUDHI which is CPU-only)
        # Convert to CPU if needed
        field_cpu = field.detach().cpu()
        gradient_cpu = gradient.detach().cpu()

        # Compute MMA for each sample
        mma_modules = []
        for i in range(n_samples):
            mma_obj = self._compute_single_mma(field_cpu[i], gradient_cpu[i])
            mma_modules.append(mma_obj)

        # If single sample, unwrap batch dimension
        if single_sample:
            return [mma_modules[0]]
        
        return mma_modules

    def _compute_single_mma(self, field_sample, gradient_sample):
        """
        Compute MMA for a single sample.

        Args:
            field_sample: Single field sample (H, W)
            gradient_sample: Single gradient sample (H, W)

        Returns:
            Module approximation object
        """
        # Step 1: Freudenthal triangulation
        simplices = self._freudenthal_triangulation(field_sample)
        
        # Step 2: Create SimplexTreeMulti with 2 parameters
        st_multi = mp.SimplexTreeMulti(num_parameters=2)
        height, width = field_sample.shape
        
        # Step 3: Insert simplices with 2-parameter filtration
        for simplex, filt_field in simplices:
            # Get gradient values at vertices of this simplex
            grad_values = [
                gradient_sample[v // width, v % width].item() 
                for v in simplex
            ]
            # Filtration: [field_value, max_gradient_value]
            st_multi.insert(simplex, filtration=[filt_field, max(grad_values)])
        
        # Step 4: Compute MMA approximation
        mma_obj = mp.module_approximation(
            st_multi,
            nlines=self.nlines,
            max_error=self.max_error
        )
        
        return mma_obj

    def _freudenthal_triangulation(self, field_2d):
        """
        Freudenthal triangulation of 2D grid.
        
        Creates a simplicial complex from a 2D grid where:
        - Vertices are grid points
        - Edges connect adjacent vertices (horizontal, vertical, diagonal)
        - Triangles fill in the squares
        
        Filtration values are set as max of vertex values (upper-star filtration).

        Args:
            field_2d: 2D tensor of shape (H, W)

        Returns:
            List of tuples (simplex, filtration_value) where:
            - simplex is a list of vertex indices
            - filtration_value is the max value at vertices
        """
        height, width = field_2d.shape
        simplices = []
        
        # Index function: 2D grid position -> 1D vertex index
        idx = lambda i, j: i * width + j
        
        # Vertices (0-simplices)
        for i in range(height):
            for j in range(width):
                simplices.append(([idx(i, j)], field_2d[i, j].item()))
        
        # Horizontal edges (1-simplices)
        for i in range(height):
            for j in range(width - 1):
                v1, v2 = idx(i, j), idx(i, j + 1)
                filt = max(field_2d[i, j].item(), field_2d[i, j + 1].item())
                simplices.append(([v1, v2], filt))
        
        # Vertical edges (1-simplices)
        for i in range(height - 1):
            for j in range(width):
                v1, v2 = idx(i, j), idx(i + 1, j)
                filt = max(field_2d[i, j].item(), field_2d[i + 1, j].item())
                simplices.append(([v1, v2], filt))
        
        # Diagonal edges and triangles (1-simplices and 2-simplices)
        for i in range(height - 1):
            for j in range(width - 1):
                # Vertices of the square
                v_bl = idx(i, j)
                v_br = idx(i, j + 1)
                v_tl = idx(i + 1, j)
                v_tr = idx(i + 1, j + 1)
                
                # Diagonal edge (bottom-left to top-right)
                filt_diag = max(field_2d[i, j].item(), field_2d[i + 1, j + 1].item())
                simplices.append(([v_bl, v_tr], filt_diag))
                
                # Triangle 1: bottom-left, bottom-right, top-right
                filt_tri1 = max(
                    field_2d[i, j].item(), 
                    field_2d[i, j + 1].item(), 
                    field_2d[i + 1, j + 1].item()
                )
                simplices.append(([v_bl, v_br, v_tr], filt_tri1))
                
                # Triangle 2: bottom-left, top-left, top-right
                filt_tri2 = max(
                    field_2d[i, j].item(), 
                    field_2d[i + 1, j].item(), 
                    field_2d[i + 1, j + 1].item()
                )
                simplices.append(([v_bl, v_tl, v_tr], filt_tri2))
        
        return simplices
