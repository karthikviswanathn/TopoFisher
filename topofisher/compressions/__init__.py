"""
Compression methods for TopoFisher pipeline.

This module provides the explicit compression stage in the pipeline:
    simulator → filtration → vectorization → COMPRESSION → fisher analysis

Compression methods:
- IdentityCompression: No-op pass-through (no compression)
- MOPEDCompression: Maximum a posteriori with Exponential Distribution
- MLPCompression: Multi-layer Perceptron learned compression
- CNNCompression: Convolutional Neural Network for persistence images
- InceptBlockCompression: IMNN-style Inception network for persistence images
"""
from typing import List, Optional
import torch

# Import base class from core
from ..core.interfaces import Compression


class IdentityCompression(Compression):
    """
    No-op compression that passes through summaries unchanged.

    Useful for:
    - Baseline comparisons
    - Backward compatibility with pipelines that don't need compression
    """

    def forward(
        self,
        summaries: List[torch.Tensor],
        delta_theta: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Pass through summaries unchanged.

        Args:
            summaries: List of summary tensors
            delta_theta: Ignored (kept for interface compatibility)

        Returns:
            Same summaries unchanged
        """
        return summaries


# Import concrete implementations
from .moped import MOPEDCompression
from .mlp import MLPCompression
from .cnn import CNNCompression
from .inception import InceptBlockCompression

__all__ = [
    "Compression",
    "IdentityCompression",
    "MOPEDCompression",
    "MLPCompression",
    "CNNCompression",
    "InceptBlockCompression",
]
