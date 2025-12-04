"""
Persistence Image vectorization using GUDHI.
"""
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from gudhi.representations import PersistenceImage


class PersistenceImageLayer(nn.Module):
    """
    Persistence Image vectorization using GUDHI.

    Converts persistence diagrams to persistence images using Gaussian kernels.
    Based on Adams et al. 2017 (https://jmlr.org/papers/v18/16-337.html).
    """

    def __init__(
        self,
        n_pixels: int = 16,
        bandwidth: float = 1.0,
        homology_dimensions: List[int] = [0, 1],
        weighting: str = "persistence"
    ):
        """
        Initialize Persistence Image layer.

        Args:
            n_pixels: Resolution of the image (n_pixels x n_pixels)
            bandwidth: Bandwidth for Gaussian kernel (default 1.0, tune as hyperparameter)
            homology_dimensions: Which homology dimensions to include (default: [0, 1])
            weighting: Weighting scheme - "persistence" uses sqrt(persistence), "uniform" uses 1.0
        """
        super().__init__()
        self.n_pixels = n_pixels
        self.bandwidth = bandwidth
        self.homology_dimensions = homology_dimensions
        self.weighting = weighting
        self.n_features = len(homology_dimensions) * n_pixels * n_pixels

        # Image bounds will be computed from ALL data (fiducial + all derivatives)
        self.im_range = None
        self.pi_layers = {}  # One PersistenceImage per homology dimension

        # Create weight function
        # GUDHI passes the full point (birth, persistence) to the weight function after BirthPersistenceTransform
        # x is a 1D array [birth, persistence]
        if weighting == "persistence":
            self.weight_fn = lambda x: np.sqrt(x[1]) if len(x.shape) == 1 else np.sqrt(x[:, 1])
        elif weighting == "uniform":
            self.weight_fn = lambda x: 1.0
        else:
            raise ValueError(f"Unknown weighting: {weighting}. Use 'persistence' or 'uniform'")

    def fit(self, all_diagrams_list: List[List[List[torch.Tensor]]]):
        """
        Compute fixed bounds across ALL diagrams (fiducial, theta_minus, theta_plus for all params).

        Args:
            all_diagrams_list: List of [fiducial_diagrams, theta_minus_0, theta_plus_0, ...],
                              where each element is a list of diagrams for different simulations,
                              and each diagram list contains [dgm_H0, dgm_H1, dgm_H2, ...]
        """
        # Collect all birth-persistence values for each homology dimension
        all_data = {h_dim: {'births': [], 'pers': []} for h_dim in self.homology_dimensions}

        for diagrams_set in all_diagrams_list:
            for diagrams in diagrams_set:
                for h_dim in self.homology_dimensions:
                    if h_dim < len(diagrams):
                        dgm = diagrams[h_dim]
                        if dgm.shape[0] > 0:
                            dgm_np = dgm.cpu().numpy()
                            births = dgm_np[:, 0]
                            pers = dgm_np[:, 1] - dgm_np[:, 0]
                            # Filter positive persistence
                            valid = pers > 0
                            if valid.any():
                                all_data[h_dim]['births'].extend(births[valid])
                                all_data[h_dim]['pers'].extend(pers[valid])

        # Compute bounds for each homology dimension
        for h_dim in self.homology_dimensions:
            if len(all_data[h_dim]['births']) == 0:
                # No valid points, use default bounds
                im_range = [0.0, 1.0, 0.0, 1.0]
                raise RuntimeWarning(f"No valid points found for homology dimension {h_dim}. Using default bounds {im_range}.")
            else:
                births = np.array(all_data[h_dim]['births'])
                pers = np.array(all_data[h_dim]['pers'])

                # Compute bounds with 10% margin
                birth_min, birth_max = births.min(), births.max()
                pers_min, pers_max = pers.min(), pers.max()

                birth_range = birth_max - birth_min
                pers_range = pers_max - pers_min

                if birth_range < 1e-6:
                    birth_min, birth_max = 0.0, 1.0
                else:
                    margin = 0.1 * birth_range
                    birth_min -= margin
                    birth_max += margin

                if pers_range < 1e-6:
                    pers_min, pers_max = 0.0, 1.0
                else:
                    margin = 0.1 * pers_range
                    pers_min = max(0.0, pers_min - margin)
                    pers_max += margin

                # GUDHI expects im_range as [x_min, x_max, y_min, y_max]
                im_range = [birth_min, birth_max, pers_min, pers_max]

            # Create GUDHI PersistenceImage for this homology dimension
            self.pi_layers[h_dim] = PersistenceImage(
                bandwidth=self.bandwidth,
                resolution=[self.n_pixels, self.n_pixels],
                weight=self.weight_fn,
                im_range=im_range
            )

        print(f"Fitted PersistenceImageLayer with bounds:")
        for h_dim in self.homology_dimensions:
            bounds = self.pi_layers[h_dim].im_range
            # im_range is [birth_min, birth_max, pers_min, pers_max]
            print(f"  H{h_dim}: birth [{bounds[0]:.3f}, {bounds[1]:.3f}], "
                  f"pers [{bounds[2]:.3f}, {bounds[3]:.3f}]")

    def forward(self, diagrams_list: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Convert persistence diagrams to persistence images.

        Args:
            diagrams_list: List of diagram sets, where each set contains
                          [dgm_H0, dgm_H1, dgm_H2, ...] for one simulation.
                          Each dgm has shape (n_points, 2) with (birth, death) pairs.

        Returns:
            Tensor of shape (batch_size, n_channels, n_pixels, n_pixels)
            where n_channels = len(homology_dimensions)
        """
        if not self.pi_layers:
            raise RuntimeError("PersistenceImageLayer must be fitted before use. Call .fit() first.")

        batch_size = len(diagrams_list)
        n_channels = len(self.homology_dimensions)

        # Determine device
        device = diagrams_list[0][0].device if len(diagrams_list[0]) > 0 and diagrams_list[0][0].numel() > 0 else torch.device('cpu')

        images = torch.zeros(
            batch_size, n_channels, self.n_pixels, self.n_pixels,
            device=device
        )

        for i, diagrams in enumerate(diagrams_list):
            for ch_idx, h_dim in enumerate(self.homology_dimensions):
                if h_dim < len(diagrams):
                    dgm = diagrams[h_dim]
                    if dgm.shape[0] > 0:
                        images[i, ch_idx] = self._diagram_to_image(dgm, h_dim, device)

        return images

    def _diagram_to_image(self, dgm: torch.Tensor, h_dim: int, device: torch.device) -> torch.Tensor:
        """
        Convert a single persistence diagram to an image using GUDHI.

        Args:
            dgm: Diagram of shape (n_points, 2) with (birth, death) pairs
            h_dim: Homology dimension
            device: Target device for output tensor

        Returns:
            Image of shape (n_pixels, n_pixels)
        """
        if dgm.shape[0] == 0:
            return torch.zeros(self.n_pixels, self.n_pixels, device=device)

        # Convert to numpy for GUDHI
        dgm_np = dgm.cpu().numpy()

        # Filter out zero/negative persistence
        persistence = dgm_np[:, 1] - dgm_np[:, 0]
        valid_mask = persistence > 0
        if not valid_mask.any():
            return torch.zeros(self.n_pixels, self.n_pixels, device=device)

        dgm_np = dgm_np[valid_mask]

        # GUDHI expects (birth, death) coordinates
        # It will internally transform to (birth, persistence) using BirthPersistenceTransform
        birth_death = dgm_np  # Already in (birth, death) format

        # Generate persistence image using GUDHI
        # fit_transform expects a list of diagrams
        image_flat = self.pi_layers[h_dim].fit_transform([birth_death])[0]

        # Reshape to 2D image
        image = image_flat.reshape(self.n_pixels, self.n_pixels)

        # Convert back to torch tensor
        return torch.from_numpy(image).float().to(device)
