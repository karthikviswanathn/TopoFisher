"""
Fisher pipeline orchestrator.
"""
from typing import List
import torch
import torch.nn as nn
import numpy as np

from .data_types import FisherConfig, FisherResult


class FisherPipeline(nn.Module):
    """
    End-to-end pipeline for Fisher information analysis.

    Orchestrates: Simulation -> Filtration -> Vectorization -> Fisher Analysis

    TODO: Add support for selective derivative computation (find_derivative flag)
    """

    def __init__(
        self,
        simulator: nn.Module,
        filtration: nn.Module,
        vectorization: nn.Module,
        fisher_analyzer: nn.Module
    ):
        """
        Initialize Fisher pipeline.

        Args:
            simulator: Simulator module (e.g., GRFSimulator)
            filtration: Filtration module (e.g., CubicalLayer)
            vectorization: Vectorization module (e.g., CombinedVectorization)
            fisher_analyzer: Fisher analyzer module
        """
        super().__init__()
        self.simulator = simulator
        self.filtration = filtration
        self.vectorization = vectorization
        self.fisher_analyzer = fisher_analyzer

    def forward(self, config: FisherConfig) -> FisherResult:
        """
        Run the full pipeline.

        Args:
            config: Fisher configuration with parameters and settings

        Returns:
            FisherResult with Fisher matrix and analysis
        """
        # Step 1: Generate simulations
        all_data = self._generate_data(config)

        # Step 2: Compute persistence diagrams
        all_diagrams = []
        for data in all_data:
            diagrams = self.filtration(data)
            all_diagrams.append(diagrams)

        # Step 3: Vectorize persistence diagrams
        # diagrams is List[List[Tensor]] where outer=hom_dims, inner=samples
        # We need to concatenate features across homology dimensions
        all_summaries = []
        for diagrams in all_diagrams:  # For each simulation set
            features_per_hom = []
            for hom_diagrams in diagrams:  # For each homology dimension
                features = self.vectorization(hom_diagrams)
                features_per_hom.append(features)
            # Concatenate features from all homology dimensions
            summary = torch.cat(features_per_hom, dim=-1)
            all_summaries.append(summary)

        # Step 4: Fisher analysis
        result = self.fisher_analyzer(all_summaries, config.delta_theta)

        return result

    def _generate_data(self, config: FisherConfig) -> List[torch.Tensor]:
        """
        Generate all required simulations.

        Args:
            config: Fisher configuration

        Returns:
            List of data tensors:
                [0]: fiducial simulations (for covariance)
                [1:]: perturbed simulations (for derivatives)
                      Ordered as [theta_minus_0, theta_plus_0, theta_minus_1, theta_plus_1, ...]
        """
        # Set seeds
        seed_cov = config.seed_cov if config.seed_cov is not None else np.random.randint(1e10)
        n_params = len(config.theta_fid)
        seed_ders = config.seed_ders if config.seed_ders is not None else \
            [np.random.randint(1e10) for _ in range(n_params)]

        # Generate fiducial simulations
        data_fid = self.simulator.generate(
            config.theta_fid,
            config.n_s,
            seed=seed_cov
        )

        all_data = [data_fid]

        # Generate perturbed simulations for derivatives
        # TODO: Respect find_derivative flag to skip inactive parameters
        for i in range(n_params):
            # theta - delta/2
            theta_minus = config.theta_fid.clone()
            theta_minus[i] -= config.delta_theta[i] / 2.0

            data_minus = self.simulator.generate(
                theta_minus,
                config.n_d,
                seed=seed_ders[i]
            )

            # theta + delta/2
            theta_plus = config.theta_fid.clone()
            theta_plus[i] += config.delta_theta[i] / 2.0

            data_plus = self.simulator.generate(
                theta_plus,
                config.n_d,
                seed=seed_ders[i]
            )

            all_data.extend([data_minus, data_plus])

        return all_data
