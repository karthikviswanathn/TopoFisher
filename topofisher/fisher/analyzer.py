"""
Fisher information analysis.
"""
from typing import List
import torch
import torch.nn as nn
from ..core.data_types import FisherResult


class FisherAnalyzer(nn.Module):
    """
    Compute Fisher information matrix from summary statistics.

    Uses centered finite differences for derivatives and
    estimates covariance from simulations at fiducial parameters.
    """

    def __init__(self, clean_data: bool = True):
        """
        Initialize Fisher analyzer.

        Args:
            clean_data: If True, remove zero-variance features
        """
        super().__init__()
        self.clean_data = clean_data

    def forward(
        self,
        summaries: List[torch.Tensor],
        delta_theta: torch.Tensor
    ) -> FisherResult:
        """
        Compute Fisher information from summaries.

        Args:
            summaries: List of summary tensors.
                summaries[0]: shape (n_s, n_features) at theta_fid (for covariance)
                summaries[1:]: shape (n_d, n_features) at perturbed values
                    Ordered as [..., theta_minus_i, theta_plus_i, ...]
            delta_theta: Tensor of shape (n_params,) with step sizes

        Returns:
            FisherResult containing Fisher matrix and related quantities
        """
        # Clean data if requested
        if self.clean_data:
            summaries = self._clean_summaries(summaries)

        # Extract covariance samples
        vecs_cov = summaries[0]  # (n_s, n_features)

        # Compute covariance matrix with Hartlap correction
        C = self._compute_covariance(vecs_cov)
        inv_C = torch.linalg.inv(C)

        # Compute derivatives using centered differences
        derivatives = self._compute_derivatives(summaries[1:], delta_theta)  # (n_params, n_d, n_features)

        # Mean derivatives
        mean_derivatives = derivatives.mean(dim=1)  # (n_params, n_features)

        # Fisher matrix: F_ij = dμ_i^T C^{-1} dμ_j
        fisher_matrix = mean_derivatives @ inv_C @ mean_derivatives.T

        # Fisher forecast (inverse Fisher matrix = covariance of parameters)
        inv_fisher = torch.linalg.inv(fisher_matrix)

        # Constraints (1-sigma errors)
        constraints = torch.sqrt(torch.diag(inv_fisher))

        # Log determinant
        sign, logdet = torch.linalg.slogdet(fisher_matrix)
        log_det_fisher = sign * logdet

        return FisherResult(
            fisher_matrix=fisher_matrix,
            inverse_fisher=inv_fisher,
            derivatives=mean_derivatives,
            covariance=C,
            log_det_fisher=log_det_fisher,
            constraints=constraints
        )

    def _clean_summaries(self, summaries: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Remove zero-variance features.

        Args:
            summaries: List of summary tensors

        Returns:
            Cleaned summaries
        """
        vecs_cov = summaries[0]

        # Compute variance
        var = vecs_cov.var(dim=0)
        valid_idx = var > 1e-10

        if (~valid_idx).any():
            print(f"Removing {(~valid_idx).sum().item()} zero-variance features")

        # Filter all summaries
        return [s[:, valid_idx] for s in summaries]

    def _compute_covariance(self, vecs: torch.Tensor) -> torch.Tensor:
        """
        Compute covariance matrix with Hartlap correction.

        Args:
            vecs: Tensor of shape (n_samples, n_features)

        Returns:
            Covariance matrix of shape (n_features, n_features)
        """
        n_samples, n_features = vecs.shape

        # Compute covariance
        vecs_centered = vecs - vecs.mean(dim=0, keepdim=True)
        cov = (vecs_centered.T @ vecs_centered) / (n_samples - 1)

        # Hartlap correction factor
        hartlap_factor = (n_samples - n_features - 2.0) / (n_samples - 1.0)

        return cov / hartlap_factor

    def _compute_derivatives(
        self,
        perturbed_vecs: List[torch.Tensor],
        delta_theta: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute derivatives using centered finite differences.

        Args:
            perturbed_vecs: List of tensors ordered as [theta_minus_0, theta_plus_0, theta_minus_1, theta_plus_1, ...]
            delta_theta: Step sizes

        Returns:
            Tensor of shape (n_params, n_d, n_features)
        """
        n_params = len(delta_theta)
        derivatives = []

        for i in range(n_params):
            vecs_minus = perturbed_vecs[2 * i]      # theta - delta/2
            vecs_plus = perturbed_vecs[2 * i + 1]   # theta + delta/2

            # Centered difference
            deriv = (vecs_plus - vecs_minus) / delta_theta[i]
            derivatives.append(deriv)

        return torch.stack(derivatives)  # (n_params, n_d, n_features)
