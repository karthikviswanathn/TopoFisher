# mma_vectorization.py
import torch
import numpy as np
import multipers as mp
from multipers.filtrations import Cubical
from multipers.torch.diff_grids import get_grid, evaluate_mod_in_grid
from multipers import grids

# ============= Core MMA Functions =============

def create_grid_from_data(filtration_values, strategy="regular_closest", resolution=50):
    """Creates a grid based on filtration values."""
    grid_function = get_grid(strategy)
    if strategy == "exact":
        grid = grid_function(filtration_values)
    else:
        if isinstance(resolution, int):
            resolution = [resolution] * len(filtration_values)
        grid = grid_function(filtration_values, resolution)
    return grid

def compute_mma_descriptor(field, derivative, degree=None, grid=None):
    """Computes the MMA descriptor for a scalar field and its derivative."""
    if not isinstance(field, torch.Tensor):
        field = torch.from_numpy(field)
    if not isinstance(derivative, torch.Tensor):
        derivative = torch.from_numpy(derivative)
    
    bifiltration = torch.stack([field, derivative], dim=-1)
    temp = Cubical(bifiltration)
    mma = mp.module_approximation(temp)
    
    if grid is None:
        filtration_values = [field.flatten(), derivative.flatten()]
        grid = create_grid_from_data(filtration_values, strategy='exact')

    max_degree = mma.max_degree
    result = []

    if degree is None:
        for i in range(max_degree+1):
            result.append(evaluate_mod_in_grid(mma.get_module_of_degree(i), grid))
    else:
        result = evaluate_mod_in_grid(mma.get_module_of_degree(degree), grid)
    
    return result, mma

# ============= Weight and Distance Functions =============

def is_finite(I):
    """Check if interval is finite"""
    inf_birth_idx = torch.any(I[0] == torch.inf, axis=-1)
    return ~(inf_birth_idx.all())

def dI0(I):
    """Interleaving to 0 of an interval (max diag in support)"""
    inf_birth_idx = torch.any(I[0] == torch.inf, axis=-1)
    B = I[0][~inf_birth_idx, None, :]
    D = I[1][None, :, :]
    pairwise_d = (torch.nn.functional.relu(D - B)).min(dim=2).values
    interleaving_to_0 = pairwise_d.max(0).values.max(0).values
    return torch.tensor(0.) if interleaving_to_0 == torch.inf else interleaving_to_0

def interleaving_weights(mma_diff):
    """Compute interleaving weights for all intervals"""
    return torch.cat([dI0(I)[None] for I in mma_diff])

def distance_birth(B, x):
    """Compute distance from points to birth corners"""
    B_exp = B.unsqueeze(1)
    x_exp = x.unsqueeze(0).unsqueeze(2)
    dist = (B_exp - x_exp).max(-1).values.min(-1).values
    return dist

def distance_death(D, x):
    """Compute distance from points to death corners"""
    D_exp = D.unsqueeze(1)
    x_exp = x.unsqueeze(0).unsqueeze(2)
    dist = (x_exp - D_exp).max(-1).values.min(-1).values
    return dist

def distance_to(mma_diff, x):
    """Compute distance from points to all intervals"""
    if x.ndim == 1:
        x = x[None]
    assert x.ndim == 2
    
    nB = np.max([I[0].shape[0] for I in mma_diff])
    nD = np.max([I[1].shape[0] for I in mma_diff])
    
    dtype = x.dtype
    Bs = torch.zeros(size=(len(mma_diff), nB, 2), dtype=dtype) + torch.inf
    Ds = torch.zeros(size=(len(mma_diff), nD, 2), dtype=dtype) - torch.inf
    
    for i, (B, D) in enumerate(mma_diff):
        Bs[i, :len(B), :] = B.to(dtype)
        Ds[i, :len(D), :] = D.to(dtype)
    
    birth_dist = distance_birth(Bs, x)
    death_dist = distance_death(Ds, x)
    
    return torch.maximum(birth_dist, death_dist)

# ============= Kernel Functions =============

def linear_kernel(dist, weights, bandwidth, p=2, signed=False):
    """Linear kernel for MMA vectorization"""
    s = torch.where(dist >= 0, 1, -1) if signed else 1
    x = torch.abs(dist) if signed else torch.relu(dist)
    return (s * torch.where(x < bandwidth, (bandwidth - x) / bandwidth, 0) * (weights[:, None] ** p)).sum(0)

def gaussian_kernel(dist, weights, bandwidth, p=2, signed=False):
    """Gaussian kernel for MMA vectorization"""
    s = torch.where(dist >= 0, 1, -1) if signed else 1
    x = torch.abs(dist) if signed else torch.relu(dist)
    return (s * torch.exp(-0.5 * ((x / bandwidth) ** 2)) * (weights[:, None] ** p)).sum(0)

def exponential_kernel(dist, weights, bandwidth, p=2, signed=False):
    """Exponential kernel for MMA vectorization"""
    s = torch.where(dist >= 0, 1, -1) if signed else 1
    x = torch.abs(dist) if signed else torch.relu(dist)
    return (s * torch.exp(-(x / bandwidth)) * (weights[:, None] ** p)).sum(0)

# ============= Vectorization Methods =============

def vectorize_corners(field, derivative, k=10, grid=None, homology_degree=None):
    """
    Returns first k corners ordered lexicographically.
    
    Args:
        field, derivative: Input tensors
        k: Number of corners to return
        grid: Optional precomputed grid
        homology_degree: Strategy for handling homological dimensions:
            - None (default): first k corners mixed from all dimensions
            - 0: only H0 corners
            - 1: only H1 corners
            - 'concat': k from H0 + k from H1 (output shape: (2k, 2))
    
    Returns:
        torch.Tensor: 
            - Shape (k, 2) if homology_degree in [None, 0, 1]
            - Shape (2k, 2) if homology_degree='concat'
    """
    result, _ = compute_mma_descriptor(field, derivative, grid=grid)
    
    if homology_degree is None:
        # Mix all corners from all dimensions
        all_corners = []
        for degree_result in result:
            for births, deaths in degree_result:
                if births.numel() > 0:
                    all_corners.append(births)
                if deaths.numel() > 0:
                    all_corners.append(deaths)
        
        output = torch.zeros(k, 2)
        if all_corners:
            all_pts = torch.cat(all_corners, dim=0)
            sorted_indices = torch.argsort(all_pts[:, 0], stable=True)
            all_sorted = all_pts[sorted_indices]
            n = min(k, all_sorted.shape[0])
            output[:n] = all_sorted[:n]
        
        return output
    
    elif homology_degree in [0, 1]:
        # Only specified degree
        output = torch.zeros(k, 2)
        
        if homology_degree < len(result):
            degree_corners = []
            for births, deaths in result[homology_degree]:
                if births.numel() > 0:
                    degree_corners.append(births)
                if deaths.numel() > 0:
                    degree_corners.append(deaths)
            
            if degree_corners:
                degree_all = torch.cat(degree_corners, dim=0)
                sorted_indices = torch.argsort(degree_all[:, 0], stable=True)
                degree_sorted = degree_all[sorted_indices]
                n = min(k, degree_sorted.shape[0])
                output[:n] = degree_sorted[:n]
        
        return output
    
    elif homology_degree == 'concat':
        # Original behavior: k from H0 + k from H1
        output = torch.zeros(2*k, 2)
        
        # Process H0
        if len(result) > 0:
            h0_corners = []
            for births, deaths in result[0]:
                if births.numel() > 0:
                    h0_corners.append(births)
                if deaths.numel() > 0:
                    h0_corners.append(deaths)
            
            if h0_corners:
                h0_all = torch.cat(h0_corners, dim=0)
                sorted_indices = torch.argsort(h0_all[:, 0], stable=True)
                h0_sorted = h0_all[sorted_indices]
                
                for i in range(len(h0_sorted)):
                    if i == 0 or h0_sorted[i, 0] != h0_sorted[i-1, 0]:
                        j = i + 1
                        while j < len(h0_sorted) and h0_sorted[j, 0] == h0_sorted[i, 0]:
                            j += 1
                        if j > i + 1:
                            h0_sorted[i:j] = h0_sorted[i:j][torch.argsort(h0_sorted[i:j, 1])]
                
                n_h0 = min(k, h0_sorted.shape[0])
                output[:n_h0] = h0_sorted[:n_h0]
        
        # Process H1
        if len(result) > 1:
            h1_corners = []
            for births, deaths in result[1]:
                if births.numel() > 0:
                    h1_corners.append(births)
                if deaths.numel() > 0:
                    h1_corners.append(deaths)
            
            if h1_corners:
                h1_all = torch.cat(h1_corners, dim=0)
                sorted_indices = torch.argsort(h1_all[:, 0], stable=True)
                h1_sorted = h1_all[sorted_indices]
                
                for i in range(len(h1_sorted)):
                    if i == 0 or h1_sorted[i, 0] != h1_sorted[i-1, 0]:
                        j = i + 1
                        while j < len(h1_sorted) and h1_sorted[j, 0] == h1_sorted[i, 0]:
                            j += 1
                        if j > i + 1:
                            h1_sorted[i:j] = h1_sorted[i:j][torch.argsort(h1_sorted[i:j, 1])]
                
                n_h1 = min(k, h1_sorted.shape[0])
                output[k:k+n_h1] = h1_sorted[:n_h1]
        
        return output
    
    else:
        raise ValueError(f"Invalid homology_degree: {homology_degree}. Choose from None, 0, 1, or 'concat'")

def vectorize_kernel(field, derivative, kernel="gaussian", resolution=30, 
                     bandwidth=0.05, p=2, signed=False, return_flat=False,
                     homology_degree=None):
    """
    Vectorize MMA using kernel methods on a regular grid.
    
    Args:
        kernel: "linear", "gaussian", or "exponential"
        resolution: Grid resolution for evaluation
        bandwidth: Kernel bandwidth
        p: Power for weights
        signed: Whether to use signed kernel
        return_flat: If True, returns flattened vector; if False, returns 2D image
        homology_degree: Strategy for handling homological dimensions:
            - None (default): mix all intervals together
            - 0: use only H0 intervals
            - 1: use only H1 intervals  
            - 'concat': compute H0 and H1 separately, then concatenate
    
    Returns:
        Vectorized representation:
            - If homology_degree in [None, 0, 1]: shape (resolution, resolution) or (resolution^2,) if flat
            - If homology_degree='concat': shape (2, resolution, resolution) or (2*resolution^2,) if flat
    """
    # Ensure float32 and gradients
    if not isinstance(field, torch.Tensor):
        field = torch.tensor(field, dtype=torch.float32, requires_grad=True)
    else:
        field = field.to(torch.float32)
    
    if not isinstance(derivative, torch.Tensor):
        derivative = torch.tensor(derivative, dtype=torch.float32, requires_grad=True)
    else:
        derivative = derivative.to(torch.float32)
    
    # Compute MMA
    result, mma = compute_mma_descriptor(field, derivative)
    
    # Create evaluation grid
    box = mma.get_box()
    R = grids.compute_grid(box.T, strategy="regular", resolution=resolution)
    R_dense = grids.todense(R)
    R_dense = torch.from_numpy(R_dense).to(torch.float32)
    
    # Apply kernel
    kernel_funcs = {
        "linear": linear_kernel,
        "gaussian": gaussian_kernel,
        "exponential": exponential_kernel
    }
    
    def compute_for_degree(degree_intervals):
        """Helper to compute kernel for a specific set of intervals"""
        if len(degree_intervals) == 0:
            # Return zeros if no intervals
            return torch.zeros(len(R_dense))
        
        w = interleaving_weights(degree_intervals)
        SD = distance_to(degree_intervals, R_dense)
        return kernel_funcs[kernel](SD, w, bandwidth, p=p, signed=signed)
    
    # Handle different homology strategies
    if homology_degree is None:
        # Default: mix all intervals
        all_intervals = []
        for degree_result in result:
            all_intervals.extend(degree_result)
        
        img = compute_for_degree(all_intervals)
        
        if return_flat:
            return img
        else:
            return img.reshape(*[len(g) for g in R])
    
    elif homology_degree in [0, 1]:
        # Use only specified degree
        if homology_degree < len(result):
            degree_intervals = result[homology_degree]
        else:
            degree_intervals = []
        
        img = compute_for_degree(degree_intervals)
        
        if return_flat:
            return img
        else:
            return img.reshape(*[len(g) for g in R])
    
    elif homology_degree == 'concat':
        # Compute H0 and H1 separately, then concatenate
        imgs = []
        
        # H0
        h0_intervals = result[0] if len(result) > 0 else []
        imgs.append(compute_for_degree(h0_intervals))
        
        # H1
        h1_intervals = result[1] if len(result) > 1 else []
        imgs.append(compute_for_degree(h1_intervals))
        
        if return_flat:
            # Stack as (2, resolution^2)
            return torch.stack(imgs)
        else:
            # Stack as (2, resolution, resolution)
            img_h0 = imgs[0].reshape(*[len(g) for g in R])
            img_h1 = imgs[1].reshape(*[len(g) for g in R])
            return torch.stack([img_h0, img_h1])
    
    else:
        raise ValueError(f"Invalid homology_degree: {homology_degree}. Choose from None, 0, 1, or 'concat'")

# ============= Unified Interface =============

def mma_vectorize(field, derivative, method="gaussian", **kwargs):
    """
    Unified interface for MMA vectorization.
    
    Args:
        field: Input field (2D array)
        derivative: Field derivative (2D array)
        method: Vectorization method
        **kwargs: Method-specific parameters
            For all methods: homology_degree (None, 0, 1, or 'concat')
    """
    if method == "corners":
        k = kwargs.get('k', 10)
        grid = kwargs.get('grid', None)
        homology_degree = kwargs.get('homology_degree', None)
        return vectorize_corners(field, derivative, k=k, grid=grid, 
                               homology_degree=homology_degree)
    
    elif method in ["linear", "gaussian", "exponential"]:
        resolution = kwargs.get('resolution', 30)
        bandwidth = kwargs.get('bandwidth', 0.05)
        p = kwargs.get('p', 2)
        signed = kwargs.get('signed', False)
        return_flat = kwargs.get('return_flat', False)
        homology_degree = kwargs.get('homology_degree', None)  # Nuovo parametro
        
        return vectorize_kernel(field, derivative, kernel=method, 
                              resolution=resolution, bandwidth=bandwidth,
                              p=p, signed=signed, return_flat=return_flat,
                              homology_degree=homology_degree)
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'corners', 'linear', 'gaussian', 'exponential'")