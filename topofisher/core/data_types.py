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
    fisher_matrix_moped: Optional[torch.Tensor] = None  # MOPED Fisher matrix (if MOPED used)
    inverse_fisher_moped: Optional[torch.Tensor] = None  # MOPED inverse Fisher
    log_det_fisher_moped: Optional[torch.Tensor] = None  # MOPED log det
    constraints_moped: Optional[torch.Tensor] = None  # MOPED constraints
