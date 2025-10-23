"""
MMA Kernel vectorization layer - Version that uses original mma_vectorization functions.

This implementation wraps the kernel-based vectorization methods from mma_vectorization.py
to work with the TopoFisher pipeline architecture.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Literal
import sys
import os

# Import the original mma_vectorization functions
# You'll need to adjust this path or add mma_vectorization.py to the topofisher package
try:
    from ..mma_vectorization import (
        interleaving_weights,
        distance_to,
        linear_kernel,
        gaussian_kernel,
        exponential_kernel
    )
    MMA_VECTORIZATION_AVAILABLE = True
except ImportError:
    # Fallback if import fails - you can copy the functions here
    MMA_VECTORIZATION_AVAILABLE = False
    print("Warning: Could not import mma_vectorization functions")

import multipers as mp
from multipers import grids


class MMAKernelLayer(nn.Module):
    """
    MMA Kernel vectorization for a single homology dimension.
    
    This version uses the original distance and kernel functions from mma_vectorization.py
    for accurate computation of interleaving weights and distances.
    
    Use with CombinedVectorization to handle multiple homology dimensions:
        vectorization = CombinedVectorization([
            MMAKernelLayer(kernel='gaussian', resolution=30, bandwidth=0.05),  # H0
            MMAKernelLayer(kernel='gaussian', resolution=30, bandwidth=0.05)   # H1
        ])
    """
    
    def __init__(
        self,
        kernel: Literal['gaussian', 'linear', 'exponential'] = 'gaussian',
        resolution: int = 30,
        bandwidth: float = 0.05,
        p: float = 2,
        signed: bool = False,
        return_flat: bool = True
    ):
        """
        Initialize MMA Kernel layer.
        
        Args:
            kernel: Type of kernel ('gaussian', 'linear', 'exponential')
            resolution: Grid resolution for evaluation
            bandwidth: Kernel bandwidth
            p: Power for weights
            signed: Whether to use signed kernel
            return_flat: If True, returns flattened vector (required for pipeline)
        """
        super().__init__()
        self.kernel = kernel
        self.resolution = resolution
        self.bandwidth = bandwidth
        self.p = p
        self.signed = signed
        self.return_flat = return_flat
        
        # Select kernel function
        if MMA_VECTORIZATION_AVAILABLE:
            self.kernel_func = {
                'linear': linear_kernel,
                'gaussian': gaussian_kernel,
                'exponential': exponential_kernel
            }[kernel]
        else:
            self.kernel_func = None
        
        # Features: resolution^2 for flat
        self.n_features = resolution * resolution if return_flat else None
        
    def forward(self, corner_data: List[List[tuple]]) -> torch.Tensor:
        """
        Vectorize MMA corners using kernel methods.
        
        Args:
            corner_data: List[sample][interval] -> (births, deaths)
                        where births and deaths are tensors of shape (n_corners, 2)
                        This is the corner data for a SINGLE homology dimension.
                        
        Returns:
            Tensor of shape (n_samples, resolution^2) if return_flat=True
            Tensor of shape (n_samples, resolution, resolution) if return_flat=False
        """
        n_samples = len(corner_data)
        features = []
        
        for sample_idx in range(n_samples):
            # Get intervals for this sample (already for single homology dimension)
            intervals = corner_data[sample_idx]
            
            if len(intervals) == 0:
                # No intervals - return zeros
                if self.return_flat:
                    vec = torch.zeros(self.resolution * self.resolution)
                else:
                    vec = torch.zeros(self.resolution, self.resolution)
                features.append(vec)
                continue
            
            # Convert to format expected by original functions
            # intervals is List[(births, deaths)]
            mma_diff = intervals  # Already in the right format!
            
            # Get bounding box from all corners for grid creation
            all_corners = []
            for births, deaths in intervals:
                if births.numel() > 0:
                    all_corners.append(births)
                if deaths.numel() > 0:
                    all_corners.append(deaths)
            
            if len(all_corners) == 0:
                # No corners - return zeros
                if self.return_flat:
                    vec = torch.zeros(self.resolution * self.resolution)
                else:
                    vec = torch.zeros(self.resolution, self.resolution)
                features.append(vec)
                continue
                
            all_corners_cat = torch.cat(all_corners, dim=0)
            
            # Create bounding box with small margin
            box_min = all_corners_cat.min(dim=0).values.cpu().numpy()
            box_max = all_corners_cat.max(dim=0).values.cpu().numpy()
            margin = 0.1 * (box_max - box_min)
            box = np.stack([box_min - margin, box_max + margin]).T
            
            # Create evaluation grid
            R = grids.compute_grid(box, strategy="regular", resolution=self.resolution)
            R_dense = grids.todense(R)
            R_dense = torch.from_numpy(R_dense).to(torch.float32)
            
            # Move to same device as data
            if all_corners_cat.is_cuda:
                R_dense = R_dense.cuda()
            
            # Compute using original functions if available
            if MMA_VECTORIZATION_AVAILABLE:
                # Compute weights using original function
                w = interleaving_weights(mma_diff)
                
                # Compute distances using original function
                SD = distance_to(mma_diff, R_dense)
                
                # Apply kernel using original function
                img = self.kernel_func(SD, w, self.bandwidth, p=self.p, signed=self.signed)
            else:
                # Fallback implementation
                img = self._fallback_compute(mma_diff, R_dense)
            
            # Reshape based on return_flat
            if not self.return_flat:
                img = img.reshape(self.resolution, self.resolution)
            
            features.append(img)
        
        return torch.stack(features)
    
    def _fallback_compute(self, intervals: List[tuple], grid_points: torch.Tensor) -> torch.Tensor:
        """
        Fallback computation if original functions not available.
        This is a simplified version.
        """
        n_points = grid_points.shape[0]
        result = torch.zeros(n_points, device=grid_points.device)
        
        for births, deaths in intervals:
            if births.numel() == 0 or deaths.numel() == 0:
                continue
                
            # Simple weight (average interval size)
            weight = (deaths - births).abs().max(dim=1).values.mean()
            
            # Simple distance (minimum distance to interval)
            birth_dist = torch.cdist(grid_points, births).min(dim=1).values
            death_dist = torch.cdist(grid_points, deaths).min(dim=1).values
            dist = torch.maximum(birth_dist, death_dist)
            
            # Apply kernel
            if self.kernel == 'gaussian':
                kernel_val = torch.exp(-0.5 * ((dist / self.bandwidth) ** 2))
            elif self.kernel == 'linear':
                kernel_val = torch.where(
                    dist < self.bandwidth,
                    (self.bandwidth - dist) / self.bandwidth,
                    torch.zeros_like(dist)
                )
            else:  # exponential
                kernel_val = torch.exp(-(dist / self.bandwidth))
            
            result += kernel_val * (weight ** self.p)
        
        return result
    
    def __repr__(self):
        return (f"MMAKernelLayer(kernel='{self.kernel}', resolution={self.resolution}, "
                f"bandwidth={self.bandwidth}, p={self.p}, signed={self.signed})")


# Additional convenience classes for specific kernels
class MMAGaussianLayer(MMAKernelLayer):
    """Gaussian kernel vectorization for MMA."""
    def __init__(self, resolution=30, bandwidth=0.05, p=2, signed=False, return_flat=True):
        super().__init__('gaussian', resolution, bandwidth, p, signed, return_flat)

class MMALinearLayer(MMAKernelLayer):
    """Linear kernel vectorization for MMA."""
    def __init__(self, resolution=30, bandwidth=0.05, p=2, signed=False, return_flat=True):
        super().__init__('linear', resolution, bandwidth, p, signed, return_flat)

class MMAExponentialLayer(MMAKernelLayer):
    """Exponential kernel vectorization for MMA."""
    def __init__(self, resolution=30, bandwidth=0.05, p=2, signed=False, return_flat=True):
        super().__init__('exponential', resolution, bandwidth, p, signed, return_flat)
