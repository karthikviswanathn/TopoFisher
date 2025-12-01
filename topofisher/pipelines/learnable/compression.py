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

    def _compute_summaries(self, data: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Compute compressed summaries from vectorized summaries.

        Pipeline: summaries → compression

        Args:
            data: Vectorized summaries [fid, minus_0, plus_0, ...]

        Returns:
            List of compressed summary tensors [fid, minus_0, plus_0, ...]
        """
        compressed = self.compression(data)
        return compressed

    def generate_summaries(self, config):
        """
        Helper to generate summaries from scratch.

        Args:
            config: AnalysisConfig with parameters

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