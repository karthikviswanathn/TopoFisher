"""
Learnable filtration layer with CNN-based field transformation.

This module implements a learnable filtration that transforms input fields
using a CNN before computing persistence diagrams. The CNN is trained to
maximize Fisher information by upscaling and transforming the input field
in a way that enhances topological features relevant to parameter inference.

Training Strategy:
    - CNN transforms field: N×N → 2N×2N (learnable)
    - Differentiable cubical persistence on transformed field
    - Gradient flows through CNN via Fisher loss: minimize -log|F|
    - Data regenerated each epoch since filtration changes

Pipeline:
    Input (N×N) → CNN → Transformed (2N×2N) → Persistence → Diagrams

The CNN learns to enhance topologically discriminative features.
"""
from typing import List, Optional
import torch
import torch.nn as nn

from topofisher.filtrations.differentiable_cubical import DifferentiableCubicalLayer


class CNNUpsampler(nn.Module):
    """
    CNN that upscales and transforms input fields for learnable filtration.

    Architecture:
        Input (N×N) → Conv layers → Upsampling → Output (2N×2N)

    The network learns to transform the input field in a way that enhances
    topological features relevant to Fisher information maximization.
    """

    def __init__(
        self,
        input_size: int,
        hidden_channels: List[int] = [32, 64, 32],
        kernel_size: int = 3,
        activation: str = 'relu',
        upscale_factor: int = 2
    ):
        """
        Initialize CNN upsampler.

        Args:
            input_size: Size of input grid (N for N×N input)
            hidden_channels: List of hidden channel dimensions
            kernel_size: Convolution kernel size (must be odd)
            activation: Activation function ('relu', 'leaky_relu', 'tanh')
            upscale_factor: Upscaling factor (1=no upscale, 2=double size)
        """
        super().__init__()

        self.input_size = input_size
        self.upscale_factor = upscale_factor
        self.output_size = upscale_factor * input_size

        # Build activation function
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.2)
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Ensure kernel size is odd for proper padding
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        padding = kernel_size // 2

        # Build convolutional layers
        layers = []
        in_channels = 1  # Single-channel input field

        for out_channels in hidden_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
            layers.append(act_fn)
            in_channels = out_channels

        # Final conv to single channel
        layers.append(nn.Conv2d(in_channels, 1, kernel_size, padding=padding))

        self.conv_layers = nn.Sequential(*layers)

        # Upsampling layer (configurable scale factor)
        if upscale_factor > 1:
            self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='bilinear', align_corners=False)
        else:
            self.upsample = nn.Identity()  # No upsampling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform and upscale input field.

        Args:
            x: Input tensor of shape (batch_size, H, W) or (H, W)

        Returns:
            Upscaled tensor of shape (batch_size, 2H, 2W) or (2H, 2W)
        """
        # Handle single sample
        single_sample = (x.ndim == 2)
        if single_sample:
            x = x.unsqueeze(0)  # Add batch dimension

        # Add channel dimension: (batch, H, W) → (batch, 1, H, W)
        x = x.unsqueeze(1)

        # Apply convolutions
        x = self.conv_layers(x)

        # Upsample to 2× resolution
        x = self.upsample(x)

        # Remove channel dimension: (batch, 1, 2H, 2W) → (batch, 2H, 2W)
        x = x.squeeze(1)

        # Remove batch dimension if single sample
        if single_sample:
            x = x.squeeze(0)

        return x


class LearnableFiltration(nn.Module):
    """
    Learnable filtration layer combining CNN transformation and cubical persistence.

    Pipeline:
        Input field (N×N) → CNN → Transformed field (2N×2N) → Cubical persistence → Diagrams

    The CNN is trained end-to-end to maximize Fisher information by transforming
    the input field in a way that enhances topologically discriminative features.

    Training:
        - Generate fresh samples each epoch (since filtration changes)
        - Compute persistence on CNN-transformed fields
        - Maximize Fisher determinant: minimize -log|F|
        - Gradient flows through CNN and gather operations

    Example:
        >>> # Create learnable filtration
        >>> filtration = LearnableFiltration(
        ...     input_size=16,
        ...     homology_dimensions=[0, 1],
        ...     hidden_channels=[32, 64, 32]
        ... )
        >>>
        >>> # Forward pass
        >>> field = torch.randn(10, 16, 16, requires_grad=True)
        >>> diagrams = filtration(field)  # List[List[Tensor]]
        >>>
        >>> # diagrams[0] = H0 diagrams (10 diagrams)
        >>> # diagrams[1] = H1 diagrams (10 diagrams)
        >>>
        >>> # Compute loss and train
        >>> loss = some_fisher_loss(diagrams)
        >>> loss.backward()  # Gradients flow to CNN!
    """

    def __init__(
        self,
        input_size: int,
        homology_dimensions: List[int] = [0, 1],
        hidden_channels: List[int] = [32, 64, 32],
        kernel_size: int = 3,
        activation: str = 'relu',
        upscale_factor: int = 2,
        min_persistence: Optional[List[float]] = None
    ):
        """
        Initialize learnable filtration.

        Args:
            input_size: Size of input grid (N for N×N input)
            homology_dimensions: List of homology dimensions to compute
            hidden_channels: CNN hidden channel dimensions
            kernel_size: CNN convolution kernel size (must be odd)
            activation: CNN activation function
            upscale_factor: CNN upscaling factor (1=no upscale, 2=double size)
            min_persistence: Minimum persistence threshold for each dimension

        Note:
            BatchNorm is NOT used as it's incompatible with Fisher information
            estimation (removes mean structure which Fisher info measures).
        """
        super().__init__()

        self.input_size = input_size
        self.upscale_factor = upscale_factor
        self.output_size = upscale_factor * input_size
        self.homology_dimensions = homology_dimensions

        # CNN upsampler (learnable)
        self.cnn = CNNUpsampler(
            input_size=input_size,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            activation=activation,
            upscale_factor=upscale_factor
        )

        # Differentiable cubical persistence layer
        self.cubical = DifferentiableCubicalLayer(
            homology_dimensions=homology_dimensions,
            min_persistence=min_persistence
        )

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Apply learnable filtration to input fields.

        Args:
            x: Input tensor of shape (batch_size, H, W) or (H, W)

        Returns:
            List of lists of persistence diagrams.
            Outer list: homology dimensions
            Inner list: diagrams for each sample
            Each diagram: tensor of shape (n_pairs, 2)
        """
        # Transform field with CNN (differentiable)
        x_transformed = self.cnn(x)

        # Compute persistence on transformed field (differentiable via torch.gather)
        diagrams = self.cubical(x_transformed)

        return diagrams

    def get_output_size(self) -> int:
        """Return the size of the transformed field."""
        return self.output_size

    def get_cnn_parameters(self):
        """Return CNN parameters for training."""
        return self.cnn.parameters()

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
