"""
Cubical complex filtration as a PyTorch module.
"""
from typing import List, Optional
import torch
import torch.nn as nn
import numpy as np
import gudhi
from tqdm import tqdm


class CubicalLayer(nn.Module):
    """
    PyTorch module for computing persistent homology of cubical complexes.

    This layer takes grid data (e.g., GRF) and computes persistence diagrams
    using cubical complex filtration. Handles both single samples and batches.
    """

    def __init__(
        self,
        homology_dimensions: List[int],
        min_persistence: Optional[List[float]] = None,
        show_progress: bool = False
    ):
        """
        Initialize cubical complex layer.

        Args:
            homology_dimensions: List of homology dimensions to compute (e.g., [0, 1])
            min_persistence: Minimum persistence threshold for each dimension
                           (default: 0 for all dimensions)
            show_progress: Whether to show progress bar during computation (default: False)
        """
        super().__init__()
        self.dimensions = homology_dimensions
        self.min_persistence = min_persistence if min_persistence is not None else [0.0] * len(self.dimensions)
        self.show_progress = show_progress

        assert len(self.min_persistence) == len(self.dimensions), \
            "min_persistence must have same length as homology_dimensions"

    def __repr__(self):
        """String representation showing configuration."""
        if all(p == 0.0 for p in self.min_persistence):
            return f"CubicalLayer(homology_dimensions={self.dimensions})"
        else:
            return f"CubicalLayer(homology_dimensions={self.dimensions}, min_persistence={self.min_persistence})"

    def get_config(self) -> dict:
        """Return configuration dictionary for serialization."""
        return {
            'homology_dimensions': self.dimensions,
            'min_persistence': self.min_persistence,
        }

    def forward(self, X: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Compute persistence diagrams from cubical complex.

        Args:
            X: Input tensor of shape (H, W) for single 2D sample,
               (n_samples, H, W) for batch of 2D samples,
               (D, H, W) for single 3D sample, or
               (n_samples, D, H, W) for batch of 3D samples

        Returns:
            List of lists of persistence diagrams.
            Outer list: homology dimensions (in order of self.dimensions)
            Inner list: diagrams for each sample
            Each diagram: tensor of shape (n_pairs, 2) with (birth, death) pairs
        """
        # Handle single sample vs batch
        if X.ndim == 2:  # Single 2D sample
            X = X.unsqueeze(0)
            single_sample = True
        elif X.ndim == 3 and len(self.dimensions) == 1:  # Single 3D sample or batch of 2D
            # Assume batch of 2D if we get (n, h, w)
            single_sample = False
        else:
            single_sample = False

        n_samples = X.shape[0]
        device = X.device

        # Initialize output: [hom_dim][sample_idx] -> diagram
        all_diagrams = [[] for _ in self.dimensions]

        # Compute diagrams for each sample
        iterator = tqdm(range(n_samples), desc="Computing Cubical Complex") if self.show_progress else range(n_samples)
        for i in iterator:
            sample_diagrams = self._compute_single_diagram(X[i], device)

            # Organize by homology dimension
            for dim_idx, dgm in enumerate(sample_diagrams):
                all_diagrams[dim_idx].append(dgm)

        # If single sample, unwrap batch dimension
        if single_sample:
            all_diagrams = [[dgms[0]] for dgms in all_diagrams]

        return all_diagrams

    def _compute_single_diagram(
        self,
        X_sample: torch.Tensor,
        device: torch.device
    ) -> List[torch.Tensor]:
        """
        Compute persistence diagram for a single sample.

        Args:
            X_sample: Single grid sample
            device: Device to place output tensors

        Returns:
            List of diagrams (one per homology dimension)
        """
        # Convert to numpy for GUDHI
        X_shape = X_sample.shape
        X_numpy = X_sample.detach().cpu().numpy().flatten()

        # Create cubical complex
        cubical_complex = gudhi.CubicalComplex(
            dimensions=X_shape,
            top_dimensional_cells=X_numpy
        )

        # Compute persistence
        cubical_complex.compute_persistence()

        # Extract diagrams for each homology dimension
        diagrams = []
        for idx_dim, dimension in enumerate(self.dimensions):
            # Get persistence intervals
            persistence_pairs = cubical_complex.persistence_intervals_in_dimension(dimension)

            # Filter out infinite death times
            finite_pairs = persistence_pairs[persistence_pairs[:, 1] < np.inf]

            # Apply minimum persistence threshold
            min_pers = self.min_persistence[idx_dim]
            if min_pers > 0 and len(finite_pairs) > 0:
                persistence = np.abs(finite_pairs[:, 1] - finite_pairs[:, 0])
                finite_pairs = finite_pairs[persistence > min_pers]

            # Convert to torch tensor
            diagram = torch.from_numpy(finite_pairs).float().to(device)
            diagrams.append(diagram)

        return diagrams
