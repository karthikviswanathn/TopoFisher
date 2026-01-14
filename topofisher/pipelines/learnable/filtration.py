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

    def _compute_summaries(self, data: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Compute compressed summaries from raw data.

        Full pipeline: data → filtration → vectorization → compression

        Args:
            data: Raw data [fid, minus_0, plus_0, ...]

        Returns:
            List of compressed summary tensors [fid, minus_0, plus_0, ...]
        """
        diagrams = self.compute_diagrams(data)  # Uses self.filtration
        summaries = self.vectorize(diagrams)
        compressed = self.compression(summaries)
        return compressed