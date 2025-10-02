"""
Combined vectorization across multiple homology dimensions.
"""
from typing import List
import torch
import torch.nn as nn


class CombinedVectorization(nn.Module):
    """
    Combines multiple vectorization layers across multiple homology dimensions.

    Takes a list of vectorization modules (one per homology dimension) and
    concatenates their outputs into a single feature vector.
    """

    def __init__(self, vectorization_layers: List[nn.Module]):
        """
        Initialize combined vectorization.

        Args:
            vectorization_layers: List of vectorization modules, one per homology dimension
                                 (e.g., [TopKLayer(k=10), TopKLayer(k=10)] for H0 and H1)
        """
        super().__init__()
        self.layers = nn.ModuleList(vectorization_layers)

    def forward(self, all_diagrams: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Vectorize diagrams from multiple homology dimensions.

        Args:
            all_diagrams: List[hom_dim][sample_idx] of diagrams
                         e.g., [[H0_sample0, H0_sample1, ...], [H1_sample0, H1_sample1, ...]]

        Returns:
            Tensor of shape (n_samples, total_features) where features are
            concatenated across homology dimensions
        """
        all_features = []

        for hom_dim_idx, layer in enumerate(self.layers):
            diagrams_for_dim = all_diagrams[hom_dim_idx]
            features = layer(diagrams_for_dim)
            all_features.append(features)

        return torch.cat(all_features, dim=-1)
