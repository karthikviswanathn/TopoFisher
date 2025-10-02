"""
Abstract base classes (interfaces) for TopoFisher components.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import torch


class Simulator(ABC):
    """Base class for data simulators."""

    @abstractmethod
    def generate(self, theta: torch.Tensor, n_samples: int, seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate n_samples at parameter value theta.

        Args:
            theta: Parameter values
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility

        Returns:
            Tensor of shape (n_samples, *data_shape)
        """
        pass


class Filtration(ABC):
    """Base class for computing persistence diagrams."""

    @abstractmethod
    def compute_diagrams(self, data: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Compute persistence diagrams from input data.

        Args:
            data: Input data of shape (n_samples, *data_shape)

        Returns:
            List of lists of persistence diagrams.
            Outer list: homology dimensions
            Inner list: diagrams for each sample
            Each diagram: tensor of shape (n_points, 2) with (birth, death) pairs
        """
        pass


class Vectorization(ABC):
    """Base class for vectorizing persistence diagrams."""

    @abstractmethod
    def fit(self, diagrams: List[torch.Tensor]) -> None:
        """
        Fit the vectorization to a set of diagrams (e.g., determine ranges).

        Args:
            diagrams: List of persistence diagrams
        """
        pass

    @abstractmethod
    def transform(self, diagrams: List[torch.Tensor]) -> torch.Tensor:
        """
        Transform persistence diagrams to feature vectors.

        Args:
            diagrams: List of persistence diagrams

        Returns:
            Tensor of shape (n_diagrams, n_features)
        """
        pass

    def fit_transform(self, diagrams: List[torch.Tensor]) -> torch.Tensor:
        """Convenience method to fit and transform in one call."""
        self.fit(diagrams)
        return self.transform(diagrams)


class FisherAnalyzer(ABC):
    """Base class for Fisher information analysis."""

    @abstractmethod
    def compute_fisher(
        self,
        summaries: List[torch.Tensor],
        delta_theta: torch.Tensor
    ) -> 'FisherResult':
        """
        Compute Fisher information from summaries.

        Args:
            summaries: List of summary statistics.
                summaries[0]: at theta_fid (for covariance)
                summaries[1:]: at perturbed values (for derivatives)
            delta_theta: Step sizes for derivative estimation

        Returns:
            FisherResult object containing Fisher matrix and related quantities
        """
        pass
