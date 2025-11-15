"""
Differentiable cubical complex filtration using torch.gather approach.

This module implements a differentiable cubical persistence layer inspired by
GUDHI's TensorFlow implementation. The key idea:

1. Compute persistence topology using GUDHI (non-differentiable)
2. Extract critical cell indices from persistence pairs
3. Use torch.gather to get actual values from input tensor (differentiable!)
4. Gradients flow back through gather operation to input

This enables end-to-end training of networks that transform fields before
persistence computation, optimizing topological features for downstream tasks.
"""
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
import gudhi


class DifferentiableCubicalLayer(nn.Module):
    """
    Differentiable cubical persistence layer using torch.gather.

    Unlike standard cubical persistence which breaks the computational graph,
    this layer preserves differentiability by:

    1. Using GUDHI to find which pixels are topologically critical
    2. Gathering their values from the input tensor via torch.gather
    3. Constructing diagrams from gathered values (preserves gradients)

    The topology computation is non-differentiable, but gathering pixel values
    from the original tensor allows gradients to flow back to the input.

    Example:
        >>> layer = DifferentiableCubicalLayer(homology_dimensions=[0, 1])
        >>> field = torch.randn(10, 16, 16, requires_grad=True)
        >>> diagrams = layer(field)  # Gradients preserved!
        >>> loss = some_loss_function(diagrams)
        >>> loss.backward()  # Gradients flow back to field
    """

    def __init__(
        self,
        homology_dimensions: List[int],
        min_persistence: Optional[List[float]] = None
    ):
        """
        Initialize differentiable cubical layer.

        Args:
            homology_dimensions: List of homology dimensions to compute
            min_persistence: Minimum persistence threshold for each dimension
        """
        super().__init__()
        self.dimensions = homology_dimensions
        self.min_persistence = min_persistence if min_persistence is not None else [0.0] * len(self.dimensions)

        assert len(self.min_persistence) == len(self.dimensions), \
            "min_persistence must have same length as homology_dimensions"

    def forward(self, X: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Compute differentiable persistence diagrams.

        Args:
            X: Input tensor of shape (H, W) for single sample or (n_samples, H, W) for batch

        Returns:
            List of lists of persistence diagrams.
            Outer list: homology dimensions
            Inner list: diagrams for each sample
            Each diagram: tensor of shape (n_pairs, 2) with (birth, death) values
        """
        # Handle single sample vs batch
        if X.ndim == 2:
            X = X.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False

        n_samples = X.shape[0]
        device = X.device

        # Initialize output: [hom_dim][sample_idx] -> diagram
        all_diagrams = [[] for _ in self.dimensions]

        # Compute diagrams for each sample
        for i in range(n_samples):
            sample_diagrams = self._compute_differentiable_diagram(X[i], device)

            # Organize by homology dimension
            for dim_idx, dgm in enumerate(sample_diagrams):
                all_diagrams[dim_idx].append(dgm)

        # If single sample, unwrap batch dimension
        if single_sample:
            all_diagrams = [[dgms[0]] for dgms in all_diagrams]

        return all_diagrams

    def _compute_differentiable_diagram(
        self,
        X_sample: torch.Tensor,
        device: torch.device
    ) -> List[torch.Tensor]:
        """
        Compute differentiable persistence diagram for a single sample.

        Strategy (noise-based indexing):
        1. Add tiny unique noise to make pixel values unique: X' = X + ε * arange
        2. Compute persistence on X' to get unique birth/death values
        3. Map values back to pixel indices
        4. Gather from ORIGINAL X using torch.gather (preserves gradients!)

        The noise is tiny (ε=1e-6) so it doesn't affect topology, but makes
        all pixel values unique for reliable indexing.

        Args:
            X_sample: Single grid sample of shape (H, W)
            device: Device for output tensors

        Returns:
            List of differentiable diagrams (one per homology dimension)
        """
        H, W = X_sample.shape

        # Step 1: Add tiny unique noise for indexing
        epsilon = 1e-6
        noise = torch.arange(H * W, dtype=X_sample.dtype, device=device).reshape(H, W) * epsilon
        X_with_noise = X_sample.detach() + noise

        # Step 2: Compute persistence on noised version
        X_numpy = X_with_noise.cpu().numpy().flatten()

        cubical_complex = gudhi.CubicalComplex(
            dimensions=[H, W],
            top_dimensional_cells=X_numpy
        )
        cubical_complex.compute_persistence()

        # Step 3: For each homology dimension, map values to indices and gather
        diagrams = []
        for idx_dim, dimension in enumerate(self.dimensions):
            # Get persistence intervals (birth, death) values
            persistence_intervals = cubical_complex.persistence_intervals_in_dimension(dimension)

            # Filter out infinite death times
            finite_pairs = persistence_intervals[persistence_intervals[:, 1] < np.inf]

            if len(finite_pairs) == 0:
                diagrams.append(torch.empty((0, 2), device=device))
                continue

            # Step 4: Map birth/death values to pixel indices
            X_flat_noised = X_with_noise.flatten().detach().cpu().numpy()

            birth_indices = []
            death_indices = []

            for birth_val, death_val in finite_pairs:
                # Find pixel with closest value (should be exact due to noise)
                birth_idx = np.argmin(np.abs(X_flat_noised - birth_val))
                death_idx = np.argmin(np.abs(X_flat_noised - death_val))

                birth_indices.append(birth_idx)
                death_indices.append(death_idx)

            # Step 5: Gather values from ORIGINAL (non-noised) tensor
            birth_indices_t = torch.tensor(birth_indices, device=device, dtype=torch.long)
            death_indices_t = torch.tensor(death_indices, device=device, dtype=torch.long)

            # Ensure X_sample is on the correct device before indexing
            X_flat_original = X_sample.to(device).flatten()
            birth_values = X_flat_original[birth_indices_t]
            death_values = X_flat_original[death_indices_t]

            # Stack into diagram
            diagram = torch.stack([birth_values, death_values], dim=1)

            # Apply minimum persistence threshold
            min_pers = self.min_persistence[idx_dim]
            if min_pers > 0:
                persistence = torch.abs(diagram[:, 1] - diagram[:, 0])
                diagram = diagram[persistence > min_pers]

            diagrams.append(diagram)

        return diagrams
