"""Cached pipeline for training learnable compression."""

from typing import List
import torch

from .base import CachedPipeline
from ..learnable.compression import LearnableCompressionPipeline
from ..configs.data_types import PipelineConfig


class CachedCompressionPipeline(CachedPipeline, LearnableCompressionPipeline):
    """
    Cached pipeline for training compression from summaries.

    This pipeline combines:
    - LearnableCompressionPipeline: training compression from summaries
    - CachedPipeline: saving/loading persistence diagrams and summaries to disk

    The key insight is that we only override generate_data() to return summaries
    instead of raw data, maintaining compatibility with the existing pipeline flow.
    """

    def generate_data(self, config: PipelineConfig) -> List[torch.Tensor]:
        """
        Override to return summaries instead of raw data.

        This is the ONLY method that needs to change! The rest of the pipeline
        expects summaries as input for compression training.

        Args:
            config: PipelineConfig with cache configuration

        Returns:
            List of summary tensors [fid, minus_0, plus_0, ...]
        """
        # Check if cache configuration exists
        if not config.cache:
            # No cache config, fall back to normal generation
            raw_data = super(LearnableCompressionPipeline, self).generate_data(config)
            diagrams = self.compute_diagrams(raw_data)
            return self.vectorize(diagrams)

        cache_config = config.cache

        # Load or generate based on source
        if cache_config.source == "summaries":
            print(f"\nLoading summaries from cache: {cache_config.source_path}")
            summaries = self.load_summaries(cache_config.source_path)

        elif cache_config.source == "diagrams":
            print(f"\nLoading diagrams from cache: {cache_config.source_path}")
            diagrams = self.load_diagrams(cache_config.source_path)

            print("Vectorizing diagrams to summaries...")
            summaries = self.vectorize(diagrams)

        else:  # "generate"
            print("\nGenerating data from scratch...")
            # Use the parent's generate_data to get raw simulation data
            raw_data = super(LearnableCompressionPipeline, self).generate_data(config)

            print("Computing persistence diagrams...")
            diagrams = self.compute_diagrams(raw_data)

            # Save diagrams if requested
            if cache_config.save_diagrams:
                print(f"Saving diagrams to: {cache_config.save_diagrams}")
                self.save_diagrams(diagrams, cache_config.save_diagrams, config)

            print("Vectorizing diagrams to summaries...")
            summaries = self.vectorize(diagrams)

        # Save summaries if requested
        if cache_config.save_summaries:
            print(f"Saving summaries to: {cache_config.save_summaries}")
            self.save_summaries(summaries, cache_config.save_summaries, config)

        return summaries  # Return summaries, not raw data!