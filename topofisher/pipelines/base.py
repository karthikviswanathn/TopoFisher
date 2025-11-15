"""
Base pipeline for Fisher information analysis.

Forward-only pipeline with no training capabilities.
"""
from typing import List
import torch
import torch.nn as nn

from .configs.data_types import PipelineConfig, FisherResult


class BasePipeline(nn.Module):
    """
    Base pipeline for Fisher information analysis.

    This pipeline orchestrates the full workflow:
        simulator → filtration → vectorization → compression → fisher_analyzer

    For non-learnable components only (e.g., MOPED compression).
    Use learnable pipelines for training neural network components.
    """

    def __init__(
        self,
        simulator,
        filtration: nn.Module,
        vectorization: nn.Module,
        compression: nn.Module,
        fisher_analyzer
    ):
        """
        Initialize pipeline.

        Args:
            simulator: Data simulator (must have generate() method)
            filtration: Persistence diagram computation (nn.Module with forward())
            vectorization: Diagram vectorization (nn.Module with forward())
            compression: Summary compression (nn.Module with forward())
            fisher_analyzer: Fisher matrix computation
        """
        super().__init__()
        self.simulator = simulator
        self.filtration = filtration
        self.vectorization = vectorization
        self.compression = compression
        self.fisher_analyzer = fisher_analyzer

    def generate_data(self, config: PipelineConfig) -> List[torch.Tensor]:
        """
        Generate raw data at fiducial and perturbed parameter values.

        Args:
            config: Pipeline configuration

        Returns:
            List of data tensors: [fid, minus_0, plus_0, minus_1, plus_1, ...]
        """
        n_params = len(config.delta_theta)
        all_data = []

        # Fiducial samples (for covariance)
        fid_data = self.simulator.generate(
            theta=config.theta_fid,
            n_samples=config.n_s,
            seed=config.seed_cov
        )
        all_data.append(fid_data)

        # Derivative samples (theta ± delta_theta/2 for each parameter)
        for i in range(n_params):
            # Get seed for this parameter's derivatives
            seed_der = config.seed_ders[i] if config.seed_ders is not None else None

            # theta - delta_theta/2
            theta_minus = config.theta_fid.clone()
            theta_minus[i] -= config.delta_theta[i] / 2
            data_minus = self.simulator.generate(
                theta=theta_minus,
                n_samples=config.n_d,
                seed=seed_der
            )
            all_data.append(data_minus)

            # theta + delta_theta/2
            theta_plus = config.theta_fid.clone()
            theta_plus[i] += config.delta_theta[i] / 2
            data_plus = self.simulator.generate(
                theta=theta_plus,
                n_samples=config.n_d,
                seed=seed_der  # Same seed for theta+ and theta- to reduce variance
            )
            all_data.append(data_plus)

        return all_data

    def compute_diagrams(self, all_data: List[torch.Tensor]) -> List[List[List[torch.Tensor]]]:
        """
        Compute persistence diagrams from raw data.

        Args:
            all_data: List of data tensors

        Returns:
            List of diagram sets: [fid_diagrams, minus_diagrams_0, plus_diagrams_0, ...]
            Each set has structure: List[hom_dim][sample] -> diagram
        """
        all_diagrams = []
        for data in all_data:
            diagrams = self.filtration(data)
            all_diagrams.append(diagrams)

        return all_diagrams

    def vectorize(self, all_diagrams: List[List[List[torch.Tensor]]]) -> List[torch.Tensor]:
        """
        Vectorize persistence diagrams to feature summaries.

        IMPORTANT: Vectorization hyperparameters (bounds, ranges, etc.) must be
        CONSISTENT across all sets. Use fit() on ALL diagrams before transform.

        Args:
            all_diagrams: List of diagram sets

        Returns:
            List of summary tensors: [fid_summaries, minus_summaries_0, plus_summaries_0, ...]
        """
        # Fit vectorization on ALL diagrams to ensure consistent hyperparameters
        if hasattr(self.vectorization, 'fit'):
            self.vectorization.fit(all_diagrams)

        # Transform each set using the SAME fitted parameters
        all_summaries = []
        for diagrams in all_diagrams:
            summaries = self.vectorization(diagrams)
            all_summaries.append(summaries)

        return all_summaries

    def compress(
        self,
        all_summaries: List[torch.Tensor],
        delta_theta: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Apply compression to summaries.

        Note: delta_theta is kept in signature for backward compatibility but not
        passed to compression. The Fisher matrix is invariant to the delta_theta
        values used internally by compression methods like MOPED.

        Args:
            all_summaries: List of summary tensors
            delta_theta: Parameter step sizes (kept for interface compatibility)

        Returns:
            Compressed summaries (may be smaller if compression splits data)
        """
        return self.compression(all_summaries)

    def compute_fisher(
        self,
        compressed_summaries: List[torch.Tensor],
        delta_theta: torch.Tensor
    ) -> FisherResult:
        """
        Compute Fisher information matrix.

        Args:
            compressed_summaries: Compressed summary tensors
            delta_theta: Parameter step sizes

        Returns:
            Fisher analysis results
        """
        return self.fisher_analyzer(compressed_summaries, delta_theta)

    def forward(self, config: PipelineConfig) -> FisherResult:
        """
        Run full pipeline: simulator → filtration → vectorization → compression → fisher.

        Args:
            config: Pipeline configuration

        Returns:
            Fisher information results
        """
        # 1. Generate data
        all_data = self.generate_data(config)

        # 2. Compute persistence diagrams
        all_diagrams = self.compute_diagrams(all_data)

        # 3. Vectorize diagrams
        all_summaries = self.vectorize(all_diagrams)

        # 4. Compress summaries
        compressed_summaries = self.compress(all_summaries, config.delta_theta)

        # 5. Compute Fisher matrix
        result = self.compute_fisher(compressed_summaries, config.delta_theta)

        return result