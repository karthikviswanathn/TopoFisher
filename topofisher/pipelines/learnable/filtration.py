"""Pipeline for training learnable filtration."""

from typing import List
import torch
import torch.nn as nn

from .base import LearnablePipeline


class LearnableFiltrationPipeline(LearnablePipeline):
    """
    Pipeline for training filtration from raw data.

    This pipeline takes raw data (e.g., images, point clouds) and trains a
    learnable filtration function to maximize Fisher information. This is
    the most end-to-end learnable approach. PyTorch automatically tracks
    and trains all learnable parameters in the pipeline.
    """

    def forward_pass(
        self,
        data: List[torch.Tensor],
        delta_theta: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for filtration training.

        Args:
            data: Raw data [fid, minus_0, plus_0, ...]
            delta_theta: Parameter step sizes (for finite differences in FisherAnalyzer)

        Returns:
            Loss (negative log Fisher determinant)
        """
        # Full pipeline: raw → filtration → vectorization → compression → Fisher
        diagrams = self.compute_diagrams(data)  # Uses self.filtration
        summaries = self.vectorize(diagrams)
        compressed = self.compression(summaries)  # No delta_theta needed - Fisher matrix invariant to scaling
        fisher_result = self.fisher_analyzer.compute_fisher(compressed, delta_theta)  # Finite differences

        # Return loss
        return -fisher_result.log_det_fisher