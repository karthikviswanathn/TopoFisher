"""
Inception-based CNN compression for persistence images.

Based on IMNN (Information Maximizing Neural Networks):
https://www.aquila-consortium.org/doc/imnn/
"""
from typing import List, Optional, Union
from pathlib import Path
import torch
import torch.nn as nn

from . import Compression


class InceptionBlock(nn.Module):
    """
    Inception block with parallel convolution paths.

    Combines multiple kernel sizes (1x1, 3x3, 5x5) and max pooling
    in parallel to capture multi-scale features.
    """
    def __init__(self, in_channels, n_filters, stride=2, use_3x3=True, use_5x5=True):
        super().__init__()

        # 1x1 convolution path
        self.conv1x1 = nn.Conv2d(in_channels, n_filters, kernel_size=1, stride=stride)

        # 3x3 convolution path (optional)
        self.use_3x3 = use_3x3
        if use_3x3:
            self.conv3x3 = nn.Conv2d(in_channels, n_filters, kernel_size=3, stride=stride, padding=1)

        # 5x5 convolution path (optional)
        self.use_5x5 = use_5x5
        if use_5x5:
            self.conv5x5 = nn.Conv2d(in_channels, n_filters, kernel_size=5, stride=stride, padding=2)

        # Max pooling + 1x1 conv path
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
        self.conv_after_pool = nn.Conv2d(in_channels, n_filters, kernel_size=1)

        # Calculate output channels
        n_paths = 2  # 1x1 and maxpool+1x1 are always present
        if use_3x3:
            n_paths += 1
        if use_5x5:
            n_paths += 1
        self.out_channels = n_filters * n_paths

    def forward(self, x):
        # Collect outputs from all paths
        outputs = [self.conv1x1(x)]

        if self.use_3x3:
            outputs.append(self.conv3x3(x))

        if self.use_5x5:
            outputs.append(self.conv5x5(x))

        # Maxpool path
        pooled = self.maxpool(x)
        outputs.append(self.conv_after_pool(pooled))

        # Concatenate along channel dimension
        out = torch.cat(outputs, dim=1)

        return out


class InceptBlockCompression(Compression):
    """
    IMNN-style Inception network for compressing persistence images.

    Based on Information Maximizing Neural Networks (IMNN):
    https://www.aquila-consortium.org/doc/imnn/main/pages/examples/2d_field_inference/

    Uses parallel inception blocks with multiple kernel sizes to capture
    multi-scale features from persistence images.

    Architecture:
    - 4 Inception blocks with stride=2 downsampling
    - First 3 blocks use all paths (1x1, 3x3, 5x5, maxpool+1x1)
    - Last block uses only 1x1 and maxpool paths
    - Final 1x1 convolution + global average pooling
    """

    def __init__(
        self,
        n_channels: int,
        n_pixels: int,
        output_dim: int,
        n_filters: int = 16
    ):
        """
        Initialize InceptionBlock compression.

        Args:
            n_channels: Number of input channels (homology dimensions, e.g., 2 for H0+H1)
            n_pixels: Resolution of persistence images (n_pixels x n_pixels)
            output_dim: Output dimension (n_params)
            n_filters: Number of filters per inception path (default 16)
        """
        super().__init__()

        self.n_channels = n_channels
        self.n_pixels = n_pixels
        self.output_dim = output_dim
        self.n_filters = n_filters

        # Inception blocks (4 blocks total)
        # Downsampling sequence: n_pixels -> n_pixels/2 -> n_pixels/4 -> n_pixels/8 -> n_pixels/16

        # Block 1: stride=2, all paths enabled
        self.block1 = InceptionBlock(n_channels, n_filters, stride=2, use_3x3=True, use_5x5=True)
        self.act1 = nn.LeakyReLU()

        # Block 2: stride=2, all paths enabled
        self.block2 = InceptionBlock(self.block1.out_channels, n_filters, stride=2, use_3x3=True, use_5x5=True)
        self.act2 = nn.LeakyReLU()

        # Block 3: stride=2, all paths enabled
        self.block3 = InceptionBlock(self.block2.out_channels, n_filters, stride=2, use_3x3=True, use_5x5=True)
        self.act3 = nn.LeakyReLU()

        # Block 4: stride=2, only 1x1 and maxpool paths
        self.block4 = InceptionBlock(self.block3.out_channels, n_filters, stride=2, use_3x3=False, use_5x5=False)
        self.act4 = nn.LeakyReLU()

        # Final 1x1 convolution to output dimension
        self.final_conv = nn.Conv2d(self.block4.out_channels, output_dim, kernel_size=1)

    def forward(
        self,
        summaries: List[torch.Tensor],
        delta_theta: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Apply Inception compression to persistence image summaries.

        Args:
            summaries: List of persistence image tensors.
                      Each tensor has shape (batch, n_channels, n_pixels, n_pixels)
            delta_theta: Ignored (kept for interface compatibility)

        Returns:
            Compressed summaries with shape (batch, output_dim)
        """
        compressed = []
        for s in summaries:
            # Pass through inception blocks
            x = self.act1(self.block1(s))
            x = self.act2(self.block2(x))
            x = self.act3(self.block3(x))
            x = self.act4(self.block4(x))

            # Final convolution
            x = self.final_conv(x)

            # Global average pooling to get (batch, output_dim)
            x = x.mean(dim=[2, 3])

            compressed.append(x)

        return compressed

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]) -> 'InceptBlockCompression':
        """
        Load a pre-trained InceptBlock compression from disk.

        Args:
            path: Path to saved model (.pth or .pt file)

        Returns:
            Loaded InceptBlockCompression instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Load checkpoint
        checkpoint = torch.load(path, map_location='cpu')

        # Extract architecture from checkpoint
        if 'architecture' not in checkpoint:
            raise ValueError(f"Checkpoint missing 'architecture' field: {path}")

        arch = checkpoint['architecture']

        # Create instance
        model = cls(
            n_channels=arch['n_channels'],
            n_pixels=arch['n_pixels'],
            output_dim=arch['output_dim'],
            n_filters=arch.get('n_filters', 16)
        )

        # Load state dict
        model.load_state_dict(checkpoint['state_dict'])

        return model

    def save(self, path: Union[str, Path]):
        """
        Save the model to disk.

        Args:
            path: Path to save model (.pth or .pt file)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'state_dict': self.state_dict(),
            'architecture': {
                'n_channels': self.n_channels,
                'n_pixels': self.n_pixels,
                'output_dim': self.output_dim,
                'n_filters': self.n_filters
            }
        }

        torch.save(checkpoint, path)

    def print_layer_shapes(self, input_shape: tuple):
        """
        Print how tensor shapes change through the network.

        Args:
            input_shape: Tuple (n_channels, height, width)
        """
        print("\nIMNN Inception Network Layer Shapes:")
        print("=" * 60)

        device = next(self.parameters()).device
        x = torch.zeros(1, *input_shape, device=device)
        print(f"Input: {tuple(x.shape)}")

        x = self.act1(self.block1(x))
        print(f"  Block 1 (stride=2, 4 paths): {tuple(x.shape)}")

        x = self.act2(self.block2(x))
        print(f"  Block 2 (stride=2, 4 paths): {tuple(x.shape)}")

        x = self.act3(self.block3(x))
        print(f"  Block 3 (stride=2, 4 paths): {tuple(x.shape)}")

        x = self.act4(self.block4(x))
        print(f"  Block 4 (stride=2, 2 paths): {tuple(x.shape)}")

        x = self.final_conv(x)
        print(f"  Final 1x1 Conv: {tuple(x.shape)}")

        x = x.mean(dim=[2, 3])
        print(f"  Global Avg Pool: {tuple(x.shape)}")

        print("=" * 60)

    def __repr__(self):
        return (f"InceptBlockCompression(IMNN, {self.n_channels} channels, "
                f"{self.n_pixels}x{self.n_pixels} → {self.output_dim}, "
                f"n_filters={self.n_filters})")
