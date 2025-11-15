"""Pipeline for training learnable compression."""

from typing import List
import torch
import torch.nn as nn

from .base import LearnablePipeline


class LearnableCompressionPipeline(LearnablePipeline):
    """
    Pipeline for training compression from summaries.

    This pipeline takes vectorized summaries and trains a compression
    component to maximize the Fisher information determinant.
    PyTorch automatically tracks and trains any learnable parameters.
    """

    def forward_pass(
        self,
        data: List[torch.Tensor],
        delta_theta: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for compression training.

        Args:
            data: Vectorized summaries [fid, minus_0, plus_0, ...]
            delta_theta: Parameter step sizes (for finite differences in FisherAnalyzer)

        Returns:
            Loss (negative log Fisher determinant)
        """
        # Apply compression (no delta_theta needed - Fisher matrix invariant to scaling)
        compressed = self.compression(data)

        # Compute Fisher using finite differences
        fisher_result = self.fisher_analyzer.compute_fisher(compressed, delta_theta)

        # Return loss
        return -fisher_result.log_det_fisher

    def generate_summaries(self, config):
        """
        Helper to generate summaries from scratch.

        Args:
            config: PipelineConfig with parameters

        Returns:
            List of summaries [fid, minus_0, plus_0, ...]
        """
        # Generate raw data
        all_data = self.generate_data(config)

        # Compute persistence diagrams
        all_diagrams = self.compute_diagrams(all_data)

        # Vectorize to summaries
        all_summaries = self.vectorize(all_diagrams)

        return all_summaries