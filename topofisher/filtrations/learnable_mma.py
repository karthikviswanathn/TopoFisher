"""
Learnable MMA filtration layer with CNN-based second parameter.

This module implements a learnable MMA filtration where a CNN produces
the second filtration parameter instead of the gradient magnitude.
The CNN is trained to maximize Fisher information.

Pipeline:
    Input (N×N) → CNN → Second param (N×N)
    (Input, CNN output) → MMA → Module approximation

The CNN learns to produce a second filtration that enhances
topologically discriminative features for parameter inference.
"""
from typing import List, Optional
import torch
import torch.nn as nn

from topofisher.filtrations.mma_simplextree import MMALayer


class PreFiltrationCNN(nn.Module):
    """
    CNN that produces the second filtration parameter for MMA.

    Architecture:
        Input (N×N) → Conv layers → Output (N×N)

    The network learns to produce a second parameter that, combined with
    the original field in a bifiltration, maximizes Fisher information.
    Fully convolutional - works with any input size and preserves dimensions.
    """

    def __init__(
        self,
        hidden_channels: List[int] = [32, 64, 32],
        kernel_size: int = 3,
        activation: str = 'relu'
    ):
        """
        Initialize CNN.

        Args:
            hidden_channels: List of hidden channel dimensions
            kernel_size: Convolution kernel size (must be odd)
            activation: Activation function ('relu', 'leaky_relu', 'tanh')
        """
        super().__init__()

        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.2)
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        padding = kernel_size // 2

        layers = []
        in_channels = 1

        for out_channels in hidden_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
            layers.append(act_fn)
            in_channels = out_channels

        layers.append(nn.Conv2d(in_channels, 1, kernel_size, padding=padding))

        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Produce second filtration parameter.

        Args:
            x: Input tensor of shape (batch_size, H, W) or (H, W)

        Returns:
            Second parameter tensor of same shape
        """
        single_sample = (x.ndim == 2)
        if single_sample:
            x = x.unsqueeze(0)

        x = x.unsqueeze(1)  # Add channel dim
        x = self.conv_layers(x)
        x = x.squeeze(1)  # Remove channel dim

        if single_sample:
            x = x.squeeze(0)

        return x


class LearnableMMAFiltration(nn.Module):
    """
    Learnable MMA filtration with CNN-produced second parameter.

    Instead of using gradient magnitude as the second filtration parameter,
    this layer uses a learnable CNN to produce it. The CNN is trained
    end-to-end to maximize Fisher information.

    Pipeline:
        field → CNN(field) → second_param
        (field, second_param) → MMA → module approximation

    The field and CNN output (with gradients) are stored for use in
    vectorization, where evaluate_mod_in_grid maintains differentiability.

    Example:
        >>> filtration = LearnableMMAFiltration(nlines=500)
        >>> mma_objects = filtration(field)
        >>> # Access stored tensors for vectorization:
        >>> field_for_vec = filtration.last_field
        >>> second_param = filtration.last_second_param
    """

    def __init__(
        self,
        nlines: int = 500,
        max_error: float = None,
        hidden_channels: List[int] = [32, 64, 32],
        kernel_size: int = 3,
        activation: str = 'relu'
    ):
        """
        Initialize learnable MMA filtration.

        Args:
            nlines: Number of lines for MMA approximation
            max_error: Max error for MMA (alternative to nlines)
            hidden_channels: CNN hidden channel dimensions
            kernel_size: CNN convolution kernel size (must be odd)
            activation: CNN activation function
        """
        super().__init__()

        self.cnn = PreFiltrationCNN(
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            activation=activation
        )

        self.mma = MMALayer(nlines=nlines, max_error=max_error)

        # Store for vectorization (need originals with grad for differentiability)
        self.last_field = None
        self.last_second_param = None

    def forward(self, field: torch.Tensor, gradient: torch.Tensor = None) -> list:
        """
        Apply learnable MMA filtration.

        Args:
            field: Input tensor of shape (batch_size, H, W) or (H, W)
            gradient: Ignored - kept for API compatibility. The CNN output
                     is used as second parameter instead.

        Returns:
            List of MMA module approximation objects
        """
        # CNN produces second filtration parameter (differentiable)
        second_param = self.cnn(field)

        # Store for vectorization (with gradients intact)
        self.last_field = field
        self.last_second_param = second_param

        # MMA computation (not differentiable, but vectorization will be)
        mma_objects = self.mma(field, second_param)

        return mma_objects

    def get_cnn_parameters(self):
        """Return CNN parameters for training."""
        return self.cnn.parameters()

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_stored_tensors(self):
        """
        Get stored field and second parameter for vectorization.
        
        Returns:
            (field, second_param) tuple with gradients intact
        """
        return self.last_field, self.last_second_param
