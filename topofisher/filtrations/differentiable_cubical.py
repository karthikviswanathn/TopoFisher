"""
Differentiable cubical complex filtration using GUDHI cofaces approach.

This module implements a differentiable cubical persistence layer following
GUDHI's TensorFlow implementation approach. The key idea:

1. Compute persistence topology using GUDHI (non-differentiable)
2. Extract critical cell indices using cofaces_of_persistence_pairs()
3. Gather values from input tensor using indices (differentiable!)
4. Gradients flow back through gather operation to input

This enables end-to-end training of networks that transform fields before
persistence computation, optimizing topological features for downstream tasks.

Key improvement: Uses GUDHI's cofaces API to directly get cell indices,
eliminating the need for noise-based value mapping.
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

        Direct translation of GUDHI TensorFlow CubicalLayer implementation:
        https://gudhi.inria.fr/python/latest/_modules/gudhi/tensorflow/cubical_layer.html

        Args:
            X_sample: Single grid sample of shape (H, W)
            device: Device for output tensors

        Returns:
            List of differentiable diagrams (one per homology dimension)
        """
        H, W = X_sample.shape

        # Flatten input (same as TensorFlow version)
        Xflat = X_sample.flatten()
        Xflat_numpy = X_sample.detach().cpu().numpy().flatten()

        # Compute cubical persistence
        cc = gudhi.CubicalComplex(
            dimensions=[H, W],
            top_dimensional_cells=Xflat_numpy
        )
        cc.compute_persistence()

        # Get cofaces of persistence pairs
        cof_pp = cc.cofaces_of_persistence_pairs()

        # Process each homology dimension
        diagrams = []
        for idx_dim, dim in enumerate(self.dimensions):
            # Get cofaces for this dimension
            # cof_pp[0] contains the finite pairs
            if len(cof_pp[0]) > dim and len(cof_pp[0][dim]) > 0:
                cof = cof_pp[0][dim]

                # Convert to torch tensor (tf.constant -> torch.tensor)
                cof = torch.tensor(cof, device=device, dtype=torch.long)

                # Gather values (tf.gather -> torch indexing)
                # tf.gather(Xflat, cof) -> Xflat[cof.flatten()]
                gathered = Xflat[cof.flatten()]

                # Reshape to pairs (tf.reshape -> torch.reshape)
                finite_dgm = gathered.reshape(-1, 2)

                # Apply minimum persistence threshold
                min_pers = self.min_persistence[idx_dim]
                if min_pers > 0:
                    persistence = torch.abs(finite_dgm[:, 1] - finite_dgm[:, 0])
                    finite_dgm = finite_dgm[persistence > min_pers]

                diagrams.append(finite_dgm)
            else:
                # Empty diagram for this dimension
                diagrams.append(torch.empty((0, 2), device=device))

        return diagrams
