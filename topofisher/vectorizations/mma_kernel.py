"""
MMA Kernel vectorization layer.

Self-contained implementation with all kernel and distance functions included.
Fully GPU-compatible by preserving tensor devices throughout computation.

Updated to accept MMA PyModule objects directly and maintain differentiability.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Literal, Optional
from multipers import grids


# ============= Distance and Weight Functions (GPU-compatible) =============

def is_finite(I):
    """Check if interval is finite."""
    inf_birth_idx = torch.any(I[0] == torch.inf, axis=-1)
    return ~(inf_birth_idx.all())


def dI0(I):
    """Interleaving to 0 of an interval (max diag in support)."""
    inf_birth_idx = torch.any(I[0] == torch.inf, axis=-1)
    B = I[0][~inf_birth_idx, None, :]
    D = I[1][None, :, :]
    pairwise_d = (torch.nn.functional.relu(D - B)).min(dim=2).values
    interleaving_to_0 = pairwise_d.max(0).values.max(0).values

    # GPU-compatible: create tensor on same device as input
    if interleaving_to_0 == torch.inf:
        return torch.tensor(0., device=I[0].device, dtype=I[0].dtype)
    return interleaving_to_0


def interleaving_weights(mma_diff):
    """Compute interleaving weights for all intervals."""
    if len(mma_diff) == 0:
        return torch.tensor([], device=mma_diff[0][0].device if mma_diff else torch.device('cpu'))

    weights = []
    for I in mma_diff:
        weights.append(dI0(I).unsqueeze(0))

    return torch.cat(weights)


def distance_birth(B, x):
    """Compute distance from points to birth corners."""
    B_exp = B.unsqueeze(1)
    x_exp = x.unsqueeze(0).unsqueeze(2)
    dist = (B_exp - x_exp).max(-1).values.min(-1).values
    return dist


def distance_death(D, x):
    """Compute distance from points to death corners."""
    D_exp = D.unsqueeze(1)
    x_exp = x.unsqueeze(0).unsqueeze(2)
    dist = (x_exp - D_exp).max(-1).values.min(-1).values
    return dist


def distance_to(mma_diff, x):
    """Compute distance from points to all intervals."""
    if x.ndim == 1:
        x = x.unsqueeze(0)
    assert x.ndim == 2

    if len(mma_diff) == 0:
        return torch.zeros(0, x.shape[0], device=x.device, dtype=x.dtype)

    nB = max([I[0].shape[0] for I in mma_diff])
    nD = max([I[1].shape[0] for I in mma_diff])

    # Create tensors on same device as x
    device = x.device
    dtype = x.dtype

    Bs = torch.zeros(size=(len(mma_diff), nB, 2), dtype=dtype, device=device) + torch.inf
    Ds = torch.zeros(size=(len(mma_diff), nD, 2), dtype=dtype, device=device) - torch.inf

    for i, (B, D) in enumerate(mma_diff):
        Bs[i, :len(B), :] = B.to(dtype).to(device)
        Ds[i, :len(D), :] = D.to(dtype).to(device)

    birth_dist = distance_birth(Bs, x)
    death_dist = distance_death(Ds, x)

    return torch.maximum(birth_dist, death_dist)


# ============= Kernel Functions (GPU-compatible) =============

def linear_kernel(dist, weights, bandwidth, p=2, signed=False):
    """Linear kernel for MMA vectorization."""
    s = torch.where(dist >= 0, 1, -1) if signed else 1
    x = torch.abs(dist) if signed else torch.relu(dist)
    return (s * torch.where(x < bandwidth, (bandwidth - x) / bandwidth, torch.zeros_like(x)) * (weights[:, None] ** p)).sum(0)


def gaussian_kernel(dist, weights, bandwidth, p=2, signed=False):
    """Gaussian kernel for MMA vectorization."""
    s = torch.where(dist >= 0, 1, -1) if signed else 1
    x = torch.abs(dist) if signed else torch.relu(dist)
    return (s * torch.exp(-0.5 * ((x / bandwidth) ** 2)) * (weights[:, None] ** p)).sum(0)


def exponential_kernel(dist, weights, bandwidth, p=2, signed=False):
    """Exponential kernel for MMA vectorization."""
    s = torch.where(dist >= 0, 1, -1) if signed else 1
    x = torch.abs(dist) if signed else torch.relu(dist)
    return (s * torch.exp(-(x / bandwidth)) * (weights[:, None] ** p)).sum(0)


# ============= PyTorch Module =============

class MMAKernelLayer(nn.Module):
    """
    MMA Kernel vectorization for a single homology dimension.

    Converts MMA into a fixed-size vector by evaluating a kernel function
    on a regular grid in the 2-parameter space. Fully GPU-compatible.
    
    Can accept either:
    1. MMA PyModule objects + field + gradient (new, maintains differentiability)
    2. Corner data directly (old format, for backwards compatibility)

    Use with CombinedVectorization to handle multiple homology dimensions:
        vectorization = CombinedVectorization([
            MMAKernelLayer(resolution=30, bandwidth=0.05),  # H0
            MMAKernelLayer(resolution=30, bandwidth=0.05)   # H1
        ])
    """

    def __init__(
        self,
        kernel: Literal['gaussian', 'linear', 'exponential'] = 'gaussian',
        resolution: int = 30,
        bandwidth: float = 0.05,
        p: float = 2,
        signed: bool = False,
        return_flat: bool = True,
        fixed_box: Optional[np.ndarray] = None,
        homology_dimension: int = 0
    ):
        """
        Initialize MMA Kernel layer.

        Args:
            kernel: Type of kernel function
            resolution: Grid resolution
            bandwidth: Kernel bandwidth parameter
            p: Power for interleaving weights
            signed: Whether to use signed kernel
            return_flat: If True, returns flattened vector
            fixed_box: Optional fixed bounding box
            homology_dimension: Which homology dimension (for MMA objects)
        """
        super().__init__()

        self.kernel = kernel
        self.resolution = resolution
        self.bandwidth = bandwidth
        self.p = p
        self.signed = signed
        self.return_flat = return_flat
        self.fixed_box = fixed_box
        self.homology_dimension = homology_dimension

        # Select kernel function
        self.kernel_func = {
            'linear': linear_kernel,
            'gaussian': gaussian_kernel,
            'exponential': exponential_kernel
        }[kernel]

        # Output features
        self.n_features = resolution * resolution if return_flat else None

    def forward(self, data, field=None, gradient=None) -> torch.Tensor:
        """
        Vectorize MMA using kernel methods.
        
        Can be called in two ways:
        1. forward(mma_objects, field, gradient) - NEW
        2. forward(corner_data) - OLD

        Args:
            data: MMA objects or corner data
            field: Original field tensor (if MMA objects)
            gradient: Original gradient tensor (if MMA objects)

        Returns:
            Kernel vectorization tensor
        """
        # Detect input type
        if len(data) > 0 and hasattr(data[0], 'get_module_of_degree'):
            # New format: MMA PyModule objects
            if field is None or gradient is None:
                raise ValueError("field and gradient required for MMA objects")
            return self._forward_mma_objects(data, field, gradient)
        else:
            # Old format: corner data
            return self._forward_corner_data(data)

    def _forward_mma_objects(self, mma_objects, field, gradient) -> torch.Tensor:
        """Process MMA PyModule objects (maintains differentiability)."""
        from multipers.torch.diff_grids import get_grid, evaluate_mod_in_grid
        
        n_samples = len(mma_objects)
        features = []
        
        # Ensure batch dimension
        if field.ndim == 2:
            field = field.unsqueeze(0)
        if gradient.ndim == 2:
            gradient = gradient.unsqueeze(0)
        
        device = field.device
        
        for sample_idx in range(n_samples):
            module = mma_objects[sample_idx].get_module_of_degree(self.homology_dimension)
            
            # Create grid - NO detach() to maintain gradients!
            field_cpu = field[sample_idx].cpu()
            gradient_cpu = gradient[sample_idx].cpu()
            filtration_values = [field_cpu.flatten(), gradient_cpu.flatten()]
            
            grid_function = get_grid('exact')
            grid = grid_function(filtration_values)
            
            # Evaluate module - maintains differentiability!
            result = evaluate_mod_in_grid(module, grid)
            
            # Extract intervals
            intervals = []
            for births, deaths in result:
                births_dev = births.to(device) if births.numel() > 0 else births
                deaths_dev = deaths.to(device) if deaths.numel() > 0 else deaths
                
                # Filter infinites
                if births_dev.numel() > 0:
                    mask = torch.isfinite(births_dev).all(dim=1)
                    births_dev = births_dev[mask]
                
                if deaths_dev.numel() > 0:
                    mask = torch.isfinite(deaths_dev).all(dim=1)
                    deaths_dev = deaths_dev[mask]
                
                if births_dev.numel() > 0 or deaths_dev.numel() > 0:
                    intervals.append((births_dev, deaths_dev))
            
            # Process with kernel
            vec = self._process_intervals(intervals, device)
            features.append(vec)
        
        return torch.stack(features)

    def _forward_corner_data(self, corner_data: List[List[tuple]]) -> torch.Tensor:
        """Process corner data (old format, backwards compatible)."""
        n_samples = len(corner_data)
        features = []

        for sample_idx in range(n_samples):
            intervals = corner_data[sample_idx]
            device = intervals[0][0].device if len(intervals) > 0 and intervals[0][0].numel() > 0 else torch.device('cpu')
            vec = self._process_intervals(intervals, device)
            features.append(vec)

        return torch.stack(features)

    def _process_intervals(self, intervals, device):
        """Common kernel processing logic."""
        if len(intervals) == 0:
            # No intervals - return zeros
            if self.return_flat:
                return torch.zeros(self.resolution * self.resolution, device=device)
            else:
                return torch.zeros(self.resolution, self.resolution, device=device)

        # Collect corners for bounding box
        all_corners = []
        for births, deaths in intervals:
            if births.numel() > 0:
                all_corners.append(births)
            if deaths.numel() > 0:
                all_corners.append(deaths)

        if len(all_corners) == 0:
            if self.return_flat:
                return torch.zeros(self.resolution * self.resolution, device=device)
            else:
                return torch.zeros(self.resolution, self.resolution, device=device)

        all_corners_cat = torch.cat(all_corners, dim=0)

        # Use fixed box if available
        if self.fixed_box is not None:
            box = self.fixed_box
        else:
            box_min = all_corners_cat.min(dim=0).values.detach().cpu().numpy()
            box_max = all_corners_cat.max(dim=0).values.detach().cpu().numpy()
            box = np.stack([box_min, box_max]).T

        # Create evaluation grid
        R = grids.compute_grid(box, strategy="regular", resolution=self.resolution)
        R_dense = grids.todense(R)
        R_dense = torch.from_numpy(R_dense).to(torch.float32).to(device)

        # Compute weights and distances
        w = interleaving_weights(intervals)
        SD = distance_to(intervals, R_dense)

        # Apply kernel
        img = self.kernel_func(SD, w, self.bandwidth, p=self.p, signed=self.signed)

        if not self.return_flat:
            img = img.reshape(self.resolution, self.resolution)

        return img

    def __repr__(self):
        return (f"MMAKernelLayer(kernel='{self.kernel}', resolution={self.resolution}, "
                f"bandwidth={self.bandwidth}, homology_dimension={self.homology_dimension})")


# Convenience classes
class MMAGaussianLayer(MMAKernelLayer):
    """Gaussian kernel vectorization."""
    def __init__(self, resolution=30, bandwidth=0.05, p=2, signed=False, return_flat=True, fixed_box=None, homology_dimension=0):
        super().__init__('gaussian', resolution, bandwidth, p, signed, return_flat, fixed_box, homology_dimension)


class MMALinearLayer(MMAKernelLayer):
    """Linear kernel vectorization."""
    def __init__(self, resolution=30, bandwidth=0.05, p=2, signed=False, return_flat=True, fixed_box=None, homology_dimension=0):
        super().__init__('linear', resolution, bandwidth, p, signed, return_flat, fixed_box, homology_dimension)


class MMAExponentialLayer(MMAKernelLayer):
    """Exponential kernel vectorization."""
    def __init__(self, resolution=30, bandwidth=0.05, p=2, signed=False, return_flat=True, fixed_box=None, homology_dimension=0):
        super().__init__('exponential', resolution, bandwidth, p, signed, return_flat, fixed_box, homology_dimension)
