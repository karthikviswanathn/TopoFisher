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

    def __init__(self, clean_data: bool = True, use_moped: bool = False, moped_compress_frac: float = 0.5):
        """
        Initialize Fisher analyzer.

        Args:
            clean_data: If True, remove zero-variance features
            use_moped: If True, use MOPED compression before Fisher analysis
            moped_compress_frac: Fraction of data to use for computing MOPED compression matrix
        """
        super().__init__()
        self.clean_data = clean_data
        self.use_moped = use_moped
        self.moped_compress_frac = moped_compress_frac

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

        # Compute full Fisher matrix
        fisher_matrix, inv_fisher, mean_derivatives, C, log_det_fisher, constraints = \
            self._compute_fisher(summaries, delta_theta)

        # Compute MOPED Fisher matrix if requested
        fisher_matrix_moped = None
        inverse_fisher_moped = None
        log_det_fisher_moped = None
        constraints_moped = None

        if self.use_moped:
            summaries_moped = self._apply_moped_compression(summaries, delta_theta)
            fisher_matrix_moped, inverse_fisher_moped, _, _, log_det_fisher_moped, constraints_moped = \
                self._compute_fisher(summaries_moped, delta_theta)

        return FisherResult(
            fisher_matrix=fisher_matrix,
            inverse_fisher=inv_fisher,
            derivatives=mean_derivatives,
            covariance=C,
            log_det_fisher=log_det_fisher,
            constraints=constraints,
            fisher_matrix_moped=fisher_matrix_moped,
            inverse_fisher_moped=inverse_fisher_moped,
            log_det_fisher_moped=log_det_fisher_moped,
            constraints_moped=constraints_moped
        )

    def _compute_fisher(
        self,
        summaries: List[torch.Tensor],
        delta_theta: torch.Tensor
    ):
        """
        Compute Fisher matrix and related quantities from summaries.

        Args:
            summaries: List of summary tensors
            delta_theta: Step sizes for derivatives

        Returns:
            Tuple of (fisher_matrix, inverse_fisher, mean_derivatives, covariance, log_det_fisher, constraints)
        """
        # Extract covariance samples
        vecs_cov = summaries[0]

        # Compute covariance matrix with Hartlap correction
        C = self._compute_covariance(vecs_cov)
        inv_C = torch.linalg.inv(C)

        # Compute derivatives using centered differences
        derivatives = self._compute_derivatives(summaries[1:], delta_theta)

        # Mean derivatives
        mean_derivatives = derivatives.mean(dim=1)

        # Fisher matrix: F_ij = dμ_i^T C^{-1} dμ_j
        fisher_matrix = mean_derivatives @ inv_C @ mean_derivatives.T

        # Fisher forecast (inverse Fisher matrix = covariance of parameters)
        inv_fisher = torch.linalg.inv(fisher_matrix)

        # Constraints (1-sigma errors)
        constraints = torch.sqrt(torch.diag(inv_fisher))

        # Log determinant
        sign, logdet = torch.linalg.slogdet(fisher_matrix)
        log_det_fisher = sign * logdet

        return fisher_matrix, inv_fisher, mean_derivatives, C, log_det_fisher, constraints

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

    def _apply_moped_compression(
        self,
        summaries: List[torch.Tensor],
        delta_theta: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Apply MOPED compression to summaries.

        Args:
            summaries: List of summary tensors
            delta_theta: Step sizes for derivatives

        Returns:
            Compressed summaries
        """
        # Split data for compression
        vecs_cov = summaries[0]
        n_s = vecs_cov.shape[0]
        n_comp = int(self.moped_compress_frac * n_s)

        # Use first fraction for computing compression matrix
        comp_cov = vecs_cov[:n_comp]
        comp_ders = [s[:int(self.moped_compress_frac * s.shape[0])] for s in summaries[1:]]

        # Compute derivatives for compression
        derivatives = self._compute_derivatives(comp_ders, delta_theta)
        mean_derivatives = derivatives.mean(dim=1)  # (n_params, n_features)

        # Compute covariance
        C = self._compute_covariance(comp_cov)

        # MOPED compression: B = C^{-1} dμ
        compression_matrix = torch.linalg.solve(C, mean_derivatives.T)  # (n_features, n_params)

        # Apply compression to all summaries
        compressed_summaries = [s @ compression_matrix for s in summaries]

        return compressed_summaries

    def compute_moped_compression(
        self,
        summaries: List[torch.Tensor],
        delta_theta: torch.Tensor,
        compress_frac: float = 0.5
    ) -> torch.Tensor:
        """
        Compute MOPED (Multiple Optimized Parameter Estimation and Data compression) compression matrix.

        Args:
            summaries: List of summary tensors (same format as forward())
            delta_theta: Step sizes for derivatives
            compress_frac: Fraction of data to use for computing compression (rest for Fisher)

        Returns:
            Compression matrix of shape (n_features, n_params)
        """
        # Clean data if requested
        if self.clean_data:
            summaries = self._clean_summaries(summaries)

        # Split data for compression
        vecs_cov = summaries[0]
        n_s = vecs_cov.shape[0]
        n_comp = int(compress_frac * n_s)

        # Use first fraction for computing compression matrix
        comp_cov = vecs_cov[:n_comp]
        comp_ders = [s[:int(compress_frac * s.shape[0])] for s in summaries[1:]]

        # Compute derivatives for compression
        derivatives = self._compute_derivatives(comp_ders, delta_theta)
        mean_derivatives = derivatives.mean(dim=1)  # (n_params, n_features)

        # Compute covariance
        C = self._compute_covariance(comp_cov)

        # MOPED compression: B = C^{-1} dμ
        compression_matrix = torch.linalg.solve(C, mean_derivatives.T)  # (n_features, n_params)

        return compression_matrix
