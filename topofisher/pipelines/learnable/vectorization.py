"""Pipeline for training learnable vectorization."""

from typing import List
import torch
import torch.nn as nn

from .base import LearnablePipeline


class LearnableVectorizationPipeline(LearnablePipeline):
    """
    Pipeline for training vectorization from persistence diagrams.

    This pipeline takes persistence diagrams and trains a vectorization
    component (e.g., learnable persistence images) to maximize Fisher information.
    PyTorch automatically tracks and trains any learnable parameters.
    """

    def forward_pass(
        self,
        data: List,
        delta_theta: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for vectorization training.

        Args:
            data: Persistence diagrams [fid, minus_0, plus_0, ...]
            delta_theta: Parameter step sizes (for finite differences in FisherAnalyzer)

        Returns:
            Loss (negative log Fisher determinant)
        """
        # Vectorize diagrams
        summaries = self.vectorization(data)

        # Apply compression (no delta_theta needed - Fisher matrix invariant to scaling)
        compressed = self.compression(summaries)

        # Compute Fisher using finite differences
        fisher_result = self.fisher_analyzer.compute_fisher(compressed, delta_theta)

        # Return loss
        return -fisher_result.log_det_fisher

    def generate_diagrams(self, config):
        """
        Helper to generate persistence diagrams from scratch.

        Args:
            config: PipelineConfig with parameters

        Returns:
            List of persistence diagrams [fid, minus_0, plus_0, ...]
        """
        # Generate raw data
        all_data = self.generate_data(config)

        # Compute persistence diagrams
        all_diagrams = self.compute_diagrams(all_data)

        return all_diagrams