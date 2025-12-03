"""Simulators for TopoFisher pipeline."""

from abc import ABC, abstractmethod
from typing import List, Optional
import torch
import numpy as np


class Simulator(ABC):
    """Base class for simulators."""

    @abstractmethod
    def generate_single(self, theta: torch.Tensor, seed: int) -> np.ndarray:
        """
        Generate a single sample at given parameters.

        Args:
            theta: Parameter values
            seed: Random seed for reproducibility

        Returns:
            Generated data sample
        """
        pass

    def generate(
        self,
        theta: torch.Tensor,
        n_samples: int,
        seed_start: int = 0
    ) -> torch.Tensor:
        """
        Generate multiple samples at given parameters.

        Args:
            theta: Parameter values
            n_samples: Number of samples to generate
            seed_start: Starting seed value

        Returns:
            Tensor of shape (n_samples, ...)
        """
        samples = []
        for i in range(n_samples):
            sample = self.generate_single(theta, seed=seed_start + i)
            samples.append(torch.from_numpy(sample).float())
        return torch.stack(samples)


# Import concrete implementations
from .grf import GRFSimulator
from .gaussian_vector import GaussianVectorSimulator
from .noisy_ring import NoisyRingSimulator

__all__ = [
    'Simulator',
    'GRFSimulator',
    'GaussianVectorSimulator',
    'NoisyRingSimulator',
]