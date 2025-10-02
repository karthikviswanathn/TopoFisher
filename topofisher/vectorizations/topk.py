"""
Top-K persistence vectorization.
"""
from typing import List
import torch
import torch.nn as nn


class TopKLayer(nn.Module):
    """
    Top-K persistence vectorization.

    Selects the K most persistent points and flattens them.
    """

    def __init__(self, k: int = 10, pad_value: float = 0.0):
        """
        Initialize Top-K layer.

        Args:
            k: Number of top persistent points to keep
            pad_value: Value to use for padding if fewer than k points exist
        """
        super().__init__()
        self.k = k
        self.pad_value = pad_value
        self.n_features = k * 2  # k points × 2 coordinates

    def forward(self, diagrams: List[torch.Tensor]) -> torch.Tensor:
        """
        Vectorize by taking top-k persistent points.

        Args:
            diagrams: List of diagrams, each of shape (n_points, 2)

        Returns:
            Tensor of shape (len(diagrams), k * 2)
        """
        features = []

        for dgm in diagrams:
            if dgm.shape[0] == 0:
                # Empty diagram - pad with zeros
                vec = torch.full((self.k, 2), self.pad_value)
            else:
                # Compute persistence
                persistence = dgm[:, 1] - dgm[:, 0]

                # Get top k indices
                if dgm.shape[0] >= self.k:
                    _, top_idx = torch.topk(persistence, self.k)
                    vec = dgm[top_idx]
                else:
                    # Pad if fewer than k points
                    vec = torch.cat([
                        dgm,
                        torch.full((self.k - dgm.shape[0], 2), self.pad_value, device=dgm.device)
                    ])

            features.append(vec.flatten())

        return torch.stack(features)
