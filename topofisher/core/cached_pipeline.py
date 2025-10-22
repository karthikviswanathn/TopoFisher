"""
Cached Fisher pipeline that loads pre-computed diagrams.
"""
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
import pickle
import os

from .pipeline import FisherPipeline
from .data_types import FisherConfig, FisherResult


def save_diagrams(
    filepath: str,
    diagrams: List[List[List[torch.Tensor]]],
    config: FisherConfig,
    simulations: Optional[List[torch.Tensor]] = None
) -> None:
    """
    Save persistence diagrams and metadata to pickle file.

    Args:
        filepath: Path to save pickle file
        diagrams: List of diagram sets, each is [H0_diagrams, H1_diagrams, ...]
        config: FisherConfig with metadata
        simulations: Optional list of simulation tensors
    """
    # Convert diagrams to numpy for storage
    diagrams_np = []
    for diagram_set in diagrams:
        diagram_set_np = []
        for hom_dim_diagrams in diagram_set:
            # Handle both Cubical (list of tensors) and MMA (list of lists of (births, deaths))
            sample_diagrams_np = []
            for d in hom_dim_diagrams:
                if isinstance(d, list):
                    # MMA case: list of (births, deaths) tuples
                    d_np = [(births.cpu().numpy(), deaths.cpu().numpy()) for births, deaths in d]
                else:
                    # Cubical case: tensor
                    d_np = d.cpu().numpy()
                sample_diagrams_np.append(d_np)
            diagram_set_np.append(sample_diagrams_np)
        diagrams_np.append(diagram_set_np)

    # Convert simulations to numpy if provided
    simulations_np = None
    if simulations is not None:
        simulations_np = [s.cpu().numpy() for s in simulations]

    # Prepare metadata
    metadata = {
        'theta_fid': config.theta_fid.cpu().numpy(),
        'delta_theta': config.delta_theta.cpu().numpy(),
        'n_s': config.n_s,
        'n_d': config.n_d,
        'find_derivative': config.find_derivative,
        'seed_cov': config.seed_cov,
        'seed_ders': config.seed_ders,
    }

    # Save everything
    data = {
        'metadata': metadata,
        'diagrams': diagrams_np,
        'simulations': simulations_np,
    }

    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

    print(f"Saved diagrams to {filepath}")


def load_diagrams(filepath: str) -> Tuple[List[List[List[torch.Tensor]]], dict]:
    """
    Load persistence diagrams and metadata from pickle file.

    Args:
        filepath: Path to pickle file

    Returns:
        Tuple of (diagrams, metadata) where diagrams are converted to torch tensors
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    # Convert numpy diagrams back to torch tensors
    diagrams = []
    for diagram_set_np in data['diagrams']:
        diagram_set = []
        for hom_dim_np in diagram_set_np:
            # Handle both Cubical (numpy arrays) and MMA (lists of (births, deaths) tuples)
            sample_diagrams = []
            for d in hom_dim_np:
                if isinstance(d, list):
                    # MMA case: list of (births_np, deaths_np) tuples
                    d_torch = [(torch.from_numpy(births).float(), torch.from_numpy(deaths).float())
                               for births, deaths in d]
                else:
                    # Cubical case: numpy array
                    d_torch = torch.from_numpy(d).float()
                sample_diagrams.append(d_torch)
            diagram_set.append(sample_diagrams)
        diagrams.append(diagram_set)

    print(f"Loaded diagrams from {filepath}")
    return diagrams, data['metadata']


def generate_and_save_diagrams(
    config: FisherConfig,
    simulator: nn.Module,
    filtration: nn.Module,
    cache_path: str,
    save_simulations: bool = False
) -> None:
    """
    Generate simulations, compute persistence diagrams, and save to cache.

    Args:
        config: FisherConfig
        simulator: Simulator module
        filtration: Filtration module
        cache_path: Path to save diagrams
        save_simulations: Whether to save raw simulations (can be large)
    """
    # Generate simulations
    all_data = _generate_data_from_config(config, simulator)

    # Compute persistence diagrams
    print("Computing persistence diagrams...")
    all_diagrams = []
    for i, data in enumerate(all_data):
        diagrams = filtration(data)
        all_diagrams.append(diagrams)
        if (i + 1) % 1 == 0:
            print(f"  Processed {i + 1}/{len(all_data)} simulation sets")

    # Save to cache
    save_diagrams(
        cache_path,
        all_diagrams,
        config,
        simulations=all_data if save_simulations else None
    )


def _generate_data_from_config(config: FisherConfig, simulator: nn.Module) -> List[torch.Tensor]:
    """
    Generate all required simulations from config.

    Args:
        config: FisherConfig
        simulator: Simulator module

    Returns:
        List of data tensors [fiducial, theta_minus_0, theta_plus_0, ...]
    """
    # Set seeds (numpy seed must be between 0 and 2**32 - 1)
    seed_cov = config.seed_cov if config.seed_cov is not None else np.random.randint(2**32)
    n_params = len(config.theta_fid)
    seed_ders = config.seed_ders if config.seed_ders is not None else \
        [np.random.randint(2**32) for _ in range(n_params)]

    # Generate fiducial simulations
    print(f"Generating fiducial simulations (n={config.n_s})...")
    data_fid = simulator.generate(
        config.theta_fid,
        config.n_s,
        seed=seed_cov
    )

    all_data = [data_fid]

    # Generate perturbed simulations for derivatives
    for i in range(n_params):
        print(f"Generating derivative simulations for parameter {i} (n={config.n_d})...")

        # theta - delta/2
        theta_minus = config.theta_fid.clone()
        theta_minus[i] -= config.delta_theta[i] / 2.0

        data_minus = simulator.generate(
            theta_minus,
            config.n_d,
            seed=seed_ders[i]
        )

        # theta + delta/2
        theta_plus = config.theta_fid.clone()
        theta_plus[i] += config.delta_theta[i] / 2.0

        data_plus = simulator.generate(
            theta_plus,
            config.n_d,
            seed=seed_ders[i]
        )

        all_data.extend([data_minus, data_plus])

    return all_data


class CachedFisherPipeline(FisherPipeline):
    """
    Fisher pipeline that uses cached persistence diagrams.

    Overrides diagram computation to load from cache instead of computing.
    """

    def __init__(
        self,
        simulator: nn.Module,
        filtration: nn.Module,
        vectorization: nn.Module,
        compression: nn.Module,
        fisher_analyzer: nn.Module,
        cache_path: str,
        auto_generate: bool = True,
        save_simulations: bool = False
    ):
        """
        Initialize cached Fisher pipeline.

        Args:
            simulator: Simulator module (used if cache doesn't exist)
            filtration: Filtration module (used if cache doesn't exist)
            vectorization: Vectorization module
            compression: Compression module (e.g., MOPEDCompression, IdentityCompression)
            fisher_analyzer: Fisher analyzer module
            cache_path: Path to diagram cache file
            auto_generate: If True, automatically generate cache if it doesn't exist
            save_simulations: Whether to save raw simulations when generating cache
        """
        super().__init__(simulator, filtration, vectorization, compression, fisher_analyzer)
        self.cache_path = cache_path
        self.auto_generate = auto_generate
        self.save_simulations = save_simulations
        self._cached_diagrams = None

    def forward(self, config: FisherConfig) -> FisherResult:
        """
        Run pipeline using cached diagrams.

        Args:
            config: Fisher configuration

        Returns:
            FisherResult with Fisher matrix and analysis
        """
        # Load or generate diagrams
        if not os.path.exists(self.cache_path):
            if self.auto_generate:
                print(f"Cache not found at {self.cache_path}, generating...")
                generate_and_save_diagrams(
                    config,
                    self.simulator,
                    self.filtration,
                    self.cache_path,
                    save_simulations=self.save_simulations
                )
            else:
                raise FileNotFoundError(f"Cache not found at {self.cache_path}. Set auto_generate=True to generate.")

        # Load diagrams
        all_diagrams, metadata = load_diagrams(self.cache_path)

        # Vectorize persistence diagrams
        all_summaries = []
        for diagrams in all_diagrams:
            summary = self.vectorization(diagrams)
            all_summaries.append(summary)

        # Apply compression
        all_summaries = self.compression(all_summaries, config.delta_theta)

        # Fisher analysis
        result = self.fisher_analyzer(all_summaries, config.delta_theta)

        return result
