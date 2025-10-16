"""
Core data types for TopoFisher.
"""
from dataclasses import dataclass
from typing import List, Optional
import torch


@dataclass
class FisherConfig:
    """Configuration for Fisher analysis pipeline."""
    theta_fid: torch.Tensor          # Fiducial parameter values
    delta_theta: torch.Tensor        # Parameter step sizes for derivatives
    n_s: int                         # Number of simulations for covariance
    n_d: int                         # Number of simulations for derivatives
    find_derivative: List[bool]      # Which parameters to compute derivatives for
    seed_cov: Optional[int] = None   # Seed for covariance simulations
    seed_ders: Optional[List[int]] = None  # Seeds for derivative simulations


@dataclass
class TrainingConfig:
    """Configuration for training learned compressions."""
    n_epochs: int = 1000             # Number of training epochs
    lr: float = 1e-3                 # Learning rate
    batch_size: int = 500            # Batch size for training
    weight_decay: float = 1e-4       # L2 regularization weight
    train_frac: float = 0.5          # Fraction of data for training
    val_frac: float = 0.25           # Fraction of data for validation
    validate_every: int = 10         # Validation interval (every N epochs)
    verbose: bool = True             # Print training progress
    check_gaussianity: bool = True   # Enforce Gaussianity constraint during training


@dataclass
class FisherResult:
    """Results from Fisher analysis."""
    fisher_matrix: torch.Tensor      # Fisher information matrix
    inverse_fisher: torch.Tensor     # Covariance matrix (Fisher forecast)
    derivatives: torch.Tensor        # Computed derivatives
    covariance: torch.Tensor         # Covariance matrix
    log_det_fisher: torch.Tensor     # Log determinant of Fisher matrix
    constraints: torch.Tensor        # 1-sigma parameter constraints
    bias_error: Optional[torch.Tensor] = None  # Fisher bias error
    fractional_bias: Optional[torch.Tensor] = None  # Fractional bias
