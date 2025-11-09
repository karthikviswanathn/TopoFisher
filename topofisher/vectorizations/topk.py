"""
Top-K persistence vectorization.
"""
from typing import List
import torch
import torch.nn as nn


class TopKBirthsDeathsLayer(nn.Module):
    """
    Top-K births and deaths vectorization (independent selection).

    Selects the K largest births and K largest deaths independently,
    ignoring the persistence pairing structure.
    """

    def __init__(self, k: int = 10, pad_value: float = 0.0):
        """
        Initialize Top-K births/deaths layer.

        Args:
            k: Number of top births and top deaths to keep
            pad_value: Value to use for padding if fewer than k points exist
        """
        super().__init__()
        self.k = k
        self.pad_value = pad_value
        self.n_features = k * 2  # k births + k deaths

    def forward(self, diagrams: List[torch.Tensor]) -> torch.Tensor:
        """
        Vectorize by taking top-k births and top-k deaths independently.

        Args:
            diagrams: List of diagrams, each of shape (n_points, 2)
                     where [:, 0] are births and [:, 1] are deaths

        Returns:
            Tensor of shape (len(diagrams), k * 2)
            Format: [birth₁, birth₂, ..., birthₖ, death₁, death₂, ..., deathₖ]
            Both births and deaths sorted in descending order
        """
        features = []

        for dgm in diagrams:
            if dgm.shape[0] == 0:
                # Empty diagram - pad with pad_value (preserve device!)
                vec = torch.full((self.k * 2,), self.pad_value, device=dgm.device)
            else:
                births = dgm[:, 0]
                deaths = dgm[:, 1]

                # Get top k births (descending order)
                if births.shape[0] >= self.k:
                    top_births, _ = torch.topk(births, self.k, largest=True, sorted=True)
                else:
                    # Pad if fewer than k points
                    top_births = torch.cat([
                        births.sort(descending=True)[0],  # All births sorted
                        torch.full((self.k - births.shape[0],), self.pad_value, device=dgm.device)
                    ])

                # Get top k deaths (descending order)
                if deaths.shape[0] >= self.k:
                    top_deaths, _ = torch.topk(deaths, self.k, largest=True, sorted=True)
                else:
                    # Pad if fewer than k points
                    top_deaths = torch.cat([
                        deaths.sort(descending=True)[0],  # All deaths sorted
                        torch.full((self.k - deaths.shape[0],), self.pad_value, device=dgm.device)
                    ])

                # Concatenate: [births, deaths]
                vec = torch.cat([top_births, top_deaths])

            features.append(vec)

        return torch.stack(features)


class TopKLayer(nn.Module):
    """
    Top-K persistence vectorization.

    Selects the K most persistent points and flattens them.
    """

    def __init__(self, k: int = 10, pad_value: float = 0.0):
        """
        Initialize Top-K layer.

        Args:
            k: Number of top persistent points to keep
            pad_value: Value to use for padding if fewer than k points exist
        """
        super().__init__()
        self.k = k
        self.pad_value = pad_value
        self.n_features = k * 2  # k points × 2 coordinates

    def forward(self, diagrams: List[torch.Tensor]) -> torch.Tensor:
        """
        Vectorize by taking top-k persistent points.

        Args:
            diagrams: List of diagrams, each of shape (n_points, 2)

        Returns:
            Tensor of shape (len(diagrams), k * 2)
        """
        features = []

        for dgm in diagrams:
            if dgm.shape[0] == 0:
                # Empty diagram - pad with zeros (preserve device!)
                vec = torch.full((self.k, 2), self.pad_value, device=dgm.device)
            else:
                # Compute persistence
                persistence = dgm[:, 1] - dgm[:, 0]

                # Get top k indices
                if dgm.shape[0] >= self.k:
                    _, top_idx = torch.topk(persistence, self.k)
                    vec = dgm[top_idx]
                else:
                    # Pad if fewer than k points
                    vec = torch.cat([
                        dgm,
                        torch.full((self.k - dgm.shape[0], 2), self.pad_value, device=dgm.device)
                    ])

            features.append(vec.flatten())

        return torch.stack(features)
