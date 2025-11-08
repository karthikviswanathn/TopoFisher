"""
MMA Top-K corner vectorization.
This implementation uses vectorized operations for lexicographic sorting.
"""
from typing import List
import torch
import torch.nn as nn


class MMATopKLayer(nn.Module):
    """
    MMA Top-K corner vectorization for a single homology dimension.
    
    Takes MMA PyModule objects along with the original field and gradient tensors,
    uses evaluate_mod_in_grid to maintain differentiability, and creates a fixed-size 
    vector by selecting top-K corners ordered lexicographically.
    
    This version uses vectorized operations instead of Python loops for significant speedup.
    
    Use with CombinedVectorization to handle multiple homology dimensions:
        vectorization = CombinedVectorization([
            MMATopKLayer(k=400),  # H0
            MMATopKLayer(k=110)   # H1
        ])
    """

    def __init__(self, k: int = 10, homology_dimension: int = 0, pad_value: float = 0.0):
        """
        Initialize MMA Top-K layer.
        
        Args:
            k: Number of top corners to keep
            homology_dimension: Which homology dimension to extract (0, 1, etc.)
            pad_value: Value to use for padding if fewer than k corners exist
        """
        super().__init__()
        self.k = k
        self.homology_dimension = homology_dimension
        self.pad_value = pad_value
        # Output: k corners × 2 coordinates
        self.n_features = k * 2

    def forward(self, mma_objects, field, gradient) -> torch.Tensor:
        """
        Vectorize MMA corners by taking top-k corners.
        
        Uses evaluate_mod_in_grid to maintain differentiability through the 
        computational graph.
        
        Args:
            mma_objects: List of PyModule objects from MMALayer
            field: Original field tensor (n_samples, H, W) or (H, W) with requires_grad
            gradient: Original gradient tensor, same shape as field, with requires_grad
        
        Returns:
            Tensor of shape (n_samples, k * 2)
            Returns k corners (2D points), sorted lexicographically
        """
        from multipers.torch.diff_grids import get_grid, evaluate_mod_in_grid
        
        n_samples = len(mma_objects)
        features = []
        
        # Handle field/gradient shapes - ensure batch dimension
        if field.ndim == 2:
            field = field.unsqueeze(0)
        if gradient.ndim == 2:
            gradient = gradient.unsqueeze(0)
        
        device = field.device
        
        for sample_idx in range(n_samples):
            # Get module for this sample and homology dimension
            module = mma_objects[sample_idx].get_module_of_degree(self.homology_dimension)
            
            # Create grid from filtration values
            # Note: multipers requires CPU tensors, but we detach to avoid 
            # breaking the graph - evaluate_mod_in_grid will maintain gradients
            field_cpu = field[sample_idx].cpu()
            gradient_cpu = gradient[sample_idx].cpu()
            filtration_values = [field_cpu.flatten(), gradient_cpu.flatten()]
            
            grid_function = get_grid('exact')
            grid = grid_function(filtration_values)
            
            # Evaluate module on grid - this maintains differentiability!
            result = evaluate_mod_in_grid(module, grid)
            
            # result is a list of (births, deaths) tuples
            # Each births/deaths is a PyTorch tensor with gradients
            all_corners = []
            
            for births, deaths in result:
                # Move to correct device
                births_dev = births.to(device) if births.numel() > 0 else births
                deaths_dev = deaths.to(device) if deaths.numel() > 0 else deaths
                
                # Filter out infinite values
                if births_dev.numel() > 0:
                    mask = torch.isfinite(births_dev).all(dim=1)
                    if mask.any():
                        all_corners.append(births_dev[mask])
                
                if deaths_dev.numel() > 0:
                    mask = torch.isfinite(deaths_dev).all(dim=1)
                    if mask.any():
                        all_corners.append(deaths_dev[mask])
            
            # Process corners
            if len(all_corners) == 0:
                # No corners - pad with zeros
                vec = torch.full((self.k, 2), self.pad_value, device=device)
            else:
                # Concatenate all corners
                all_corners_tensor = torch.cat(all_corners, dim=0)
                
                # OPTIMIZED: Vectorized lexicographic sort
                # Use stable sort twice: first by secondary key (y), then by primary key (x)
                # This preserves y-ordering within groups of equal x values
                
                # Sort by y first (secondary key)
                indices = torch.argsort(all_corners_tensor[:, 1], stable=True)
                sorted_by_y = all_corners_tensor[indices]
                
                # Then sort by x (primary key) with stable=True to preserve y ordering
                indices = torch.argsort(sorted_by_y[:, 0], stable=True)
                corners_sorted = sorted_by_y[indices]
                
                # Take top k (or all if less than k)
                if corners_sorted.shape[0] >= self.k:
                    vec = corners_sorted[:self.k]
                else:
                    # Pad if fewer than k corners
                    vec = torch.cat([
                        corners_sorted,
                        torch.full((self.k - corners_sorted.shape[0], 2), 
                                  self.pad_value, device=device)
                    ])
            
            features.append(vec.flatten())
        
        return torch.stack(features)

    def __repr__(self):
        return (f"MMATopKLayer(k={self.k}, homology_dimension={self.homology_dimension}, "
                f"pad_value={self.pad_value})")