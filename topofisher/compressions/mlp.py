"""
MLP-based learned compression.

Multi-layer Perceptron trained to maximize Fisher information.
Typically trained on Top-K vectorizations of persistence diagrams.
"""
from typing import List, Optional, Union
from pathlib import Path
import torch
import torch.nn as nn

from . import Compression


class MLPCompression(Compression):
    """
    Multi-layer Perceptron for compressing summaries to maximize Fisher information.

    This is a learned compression that can be trained using pipeline.fit().
    Supports lazy initialization: dimensions can be inferred from data.
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2
    ):
        """
        Initialize MLP compression.

        Args:
            input_dim: Input feature dimension (None for lazy initialization)
            output_dim: Output dimension, typically n_params (None for lazy initialization)
            hidden_dims: List of hidden layer dimensions.
                        None or [] for linear compression (no hidden layers)
                        [h1] for 1 hidden layer, [h1, h2] for 2 hidden layers, etc.
            dropout: Dropout probability (default 0.2)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else []
        self.dropout_prob = dropout
        self._initialized = False
        self.network = None
        self._device = None  # Track device for lazy initialization

        # If dimensions provided, initialize network now
        if input_dim is not None and output_dim is not None:
            self._build_network(input_dim, output_dim)

    def _build_network(self, input_dim: int, output_dim: int):
        """Build the MLP network with given dimensions."""
        layers = []
        if not self.hidden_dims:
            # Linear compression (no hidden layers)
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # Add hidden layers
            prev_dim = input_dim
            for h_dim in self.hidden_dims:
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(nn.LeakyReLU(negative_slope=0.01))
                layers.append(nn.Dropout(self.dropout_prob))
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Move network to stored device if available
        if self._device is not None:
            self.network.to(self._device)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self._initialized = True

    def is_trainable(self) -> bool:
        """Return True since MLP compression requires training."""
        return True

    def is_initialized(self) -> bool:
        """Return True if network has been initialized."""
        return self._initialized

    def to(self, device):
        """Override to() to track device for lazy initialization."""
        self._device = device
        return super().to(device)

    def initialize(self, input_dim: int, output_dim: int) -> None:
        """
        Initialize network with inferred dimensions.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (typically n_params)
        """
        if self._initialized:
            raise RuntimeError("MLPCompression already initialized")
        self._build_network(input_dim, output_dim)

    def forward(
        self,
        summaries: List[torch.Tensor],
        delta_theta: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Apply MLP compression to summaries.

        Args:
            summaries: List of summary tensors
            delta_theta: Ignored (kept for interface compatibility)

        Returns:
            Compressed summaries
        """
        if not self._initialized:
            raise RuntimeError(
                "MLPCompression not initialized. Call initialize(input_dim, output_dim) "
                "or use pipeline.fit() to automatically initialize."
            )

        # Apply network to each summary tensor
        compressed = [self.network(s) for s in summaries]
        return compressed

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]) -> 'MLPCompression':
        """
        Load a pre-trained MLP compression from disk.

        Args:
            path: Path to saved model (.pth or .pt file)

        Returns:
            Loaded MLPCompression instance
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
            input_dim=arch['input_dim'],
            output_dim=arch['output_dim'],
            hidden_dims=arch.get('hidden_dims'),
            dropout=arch.get('dropout', 0.2)
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
        if not self._initialized:
            raise RuntimeError("Cannot save uninitialized MLPCompression")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'state_dict': self.state_dict(),
            'architecture': {
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'hidden_dims': self.hidden_dims,
                'dropout': self.dropout_prob
            }
        }

        torch.save(checkpoint, path)

    def __repr__(self):
        if not self._initialized:
            if self.hidden_dims:
                return f"MLPCompression(uninitialized, hidden_dims={self.hidden_dims}, dropout={self.dropout_prob})"
            else:
                return f"MLPCompression(uninitialized, linear)"
        elif not self.hidden_dims:
            return f"MLPCompression(Linear: {self.input_dim} → {self.output_dim})"
        else:
            dims_str = f"{self.input_dim} → " + " → ".join(map(str, self.hidden_dims)) + f" → {self.output_dim}"
            return f"MLPCompression({dims_str}, dropout={self.dropout_prob})"