"""
Fisher pipeline orchestrator.
"""
from typing import List, Optional
import torch
import torch.nn as nn
import numpy as np

from .data_types import FisherConfig, FisherResult, TrainingConfig


class FisherPipeline(nn.Module):
    """
    End-to-end pipeline for Fisher information analysis.

    Orchestrates: Simulation -> Filtration -> Vectorization -> Compression -> Fisher Analysis

    TODO: Add support for selective derivative computation (find_derivative flag)
    """

    def __init__(
        self,
        simulator: nn.Module,
        filtration: nn.Module,
        vectorization: nn.Module,
        compression: nn.Module,
        fisher_analyzer: nn.Module
    ):
        """
        Initialize Fisher pipeline.

        Args:
            simulator: Simulator module (e.g., GRFSimulator)
            filtration: Filtration module (e.g., CubicalLayer)
            vectorization: Vectorization module (e.g., CombinedVectorization)
            compression: Compression module (e.g., MOPEDCompression, IdentityCompression)
            fisher_analyzer: Fisher analyzer module
        """
        super().__init__()
        self.simulator = simulator
        self.filtration = filtration
        self.vectorization = vectorization
        self.compression = compression
        self.fisher_analyzer = fisher_analyzer

    def run(self, config: FisherConfig, training_config: Optional[TrainingConfig] = None,
            verbose_gaussianity: bool = False) -> FisherResult:
        """
        Run the full pipeline.

        Args:
            config: Fisher configuration with parameters and settings
            training_config: Optional training config for learned compressions
            verbose_gaussianity: If True, print detailed Gaussianity check results (default False)

        Returns:
            FisherResult with Fisher matrix and analysis
        """
        # Step 0: Move models to simulator's device
        device = getattr(self.simulator, 'device', 'cpu')
        if isinstance(device, str):
            device = torch.device(device)
        self.compression.to(device)

        # Step 1: Generate simulations
        all_data = self._generate_data(config)

        # Step 2: Compute persistence diagrams
        all_diagrams = []
        for data in all_data:
            diagrams = self.filtration(data)
            all_diagrams.append(diagrams)

        # Step 3: Vectorize persistence diagrams
        all_summaries = []
        for diagrams in all_diagrams:  # For each simulation set
            summary = self.vectorization(diagrams)
            all_summaries.append(summary)

        # Step 4: Train compression if needed
        training_history = None
        if training_config is not None:
            training_history = self.compression.train_compression(all_summaries, config.delta_theta, training_config)

        # Step 5: Apply compression
        compressed_summaries = self.compression(all_summaries, config.delta_theta)

        # Check if compression returns test set only or all data
        if self.compression.returns_test_only():
            train_frac = getattr(self.compression, 'train_frac', 0.5)
            print(f"\nCompression splits data internally (train_frac={train_frac:.1f})")
            print(f"  Compression matrix learned on train set ({int(train_frac*100)}% of data)")
            print(f"  Compressed test set shape: {compressed_summaries[0].shape}")
            print(f"  Fisher information will be computed on test set only")
        else:
            print(f"\nCompressed summary statistics shape: {compressed_summaries[0].shape}")
            if compressed_summaries[0].shape[0] == config.n_s and compressed_summaries[1].shape[0] == config.n_d:
                print("WARNING: Using all data for Fisher analysis (no train/test split)")

        # Step 6: Gaussianity check (before Fisher analysis)
        from ..fisher.gaussianity import test_gaussianity
        if verbose_gaussianity:
            print("\n" + "=" * 80)
            print("Gaussianity Check (After Compression)")
            print("=" * 80)
        gauss_results, is_gaussian = test_gaussianity(compressed_summaries, alpha=0.05, mode="summary", verbose=verbose_gaussianity)

        # Step 7: Fisher analysis
        result = self.fisher_analyzer(compressed_summaries, config.delta_theta)

        # Add Gaussianity flag and details to result
        result.is_gaussian = is_gaussian
        result.gaussianity_details = gauss_results

        return result

    def forward(self, config: FisherConfig) -> FisherResult:
        """
        PyTorch nn.Module forward pass (calls run with no training).

        Args:
            config: Fisher configuration with parameters and settings

        Returns:
            FisherResult with Fisher matrix and analysis
        """
        return self.run(config, training_config=None)

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
        # Set seeds (numpy seed must be between 0 and 2**32 - 1)
        seed_cov = config.seed_cov if config.seed_cov is not None else np.random.randint(2**32)
        n_params = len(config.theta_fid)
        seed_ders = config.seed_ders if config.seed_ders is not None else \
            [np.random.randint(2**32) for _ in range(n_params)]

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
