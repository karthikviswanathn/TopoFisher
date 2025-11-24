"""
Data types for TopoFisher pipelines.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    theta_fid: torch.Tensor          # Fiducial parameter values
    delta_theta: torch.Tensor        # Parameter step sizes for derivatives (±Δθ/2)
    n_s: int                         # Number of samples for covariance estimation
    n_d: int                         # Number of samples for derivative estimation
    seed_cov: Optional[int] = None   # Seed for covariance simulations
    seed_ders: Optional[list] = None # Seeds for derivative simulations (one per parameter)


@dataclass
class TrainingConfig:
    """Configuration for training learned components."""
    n_epochs: int = 1000             # Number of training epochs
    lr: float = 1e-3                 # Learning rate
    batch_size: int = 500            # Batch size for training
    weight_decay: float = 1e-4       # L2 regularization
    train_frac: float = 0.5          # Fraction for training
    val_frac: float = 0.25           # Fraction for validation
    validate_every: int = 10         # Validation interval (epochs)
    verbose: bool = True             # Print training progress
    check_gaussianity: bool = True   # Enforce Gaussianity constraint


@dataclass
class FisherResult:
    """Results from Fisher information analysis."""
    fisher_matrix: torch.Tensor      # Fisher information matrix (n_params, n_params)
    inverse_fisher: torch.Tensor     # Covariance matrix = F^-1
    derivatives: torch.Tensor        # Parameter derivatives (n_params, n_d, n_features)
    covariance: torch.Tensor         # Summary covariance matrix (n_features, n_features)
    log_det_fisher: torch.Tensor     # log|F| (scalar)
    constraints: torch.Tensor        # 1-sigma constraints = sqrt(diag(F^-1))

    # Optional diagnostic information
    bias_error: Optional[torch.Tensor] = None
    fractional_bias: Optional[torch.Tensor] = None
    is_gaussian: Optional[bool] = None
    gaussianity_details: Optional[Dict[str, Any]] = None

    def print_gaussianity(self):
        """Print Gaussianity check result."""
        if self.is_gaussian is None:
            print("\nGaussianity Check: Not performed")
        else:
            gauss_mark = "✓ PASS" if self.is_gaussian else "✗ FAIL"
            print(f"\nGaussianity Check: {gauss_mark}")
