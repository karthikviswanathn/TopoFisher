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

    This is a learned compression that must be trained externally (see examples/grf/train_mlp.py).
    Once trained, it provides a simple forward pass transformation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2
    ):
        """
        Initialize MLP compression.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (typically n_params)
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

        # Build network
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
                layers.append(nn.Dropout(dropout))
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

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
        if not self.hidden_dims:
            return f"MLPCompression(Linear: {self.input_dim} → {self.output_dim})"
        else:
            dims_str = f"{self.input_dim} → " + " → ".join(map(str, self.hidden_dims)) + f" → {self.output_dim}"
            return f"MLPCompression({dims_str}, dropout={self.dropout_prob})"
