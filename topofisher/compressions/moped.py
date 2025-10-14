"""
MOPED (Multiple Optimized Parameter Estimation and Data compression) compression.

Reference: Heavens et al. 2000 (https://arxiv.org/abs/astro-ph/9911102)
"""
from typing import List, Optional
import torch

from . import Compression


class MOPEDCompression(Compression):
    """
    MOPED compression: B = C^{-1} dμ

    Where:
    - C is the covariance matrix of summaries
    - dμ are the derivatives of mean summaries w.r.t. parameters
    - B is the compression matrix: (n_features, n_params)

    The compressed summaries are: compressed = summaries @ B

    This compression is optimal for Gaussian likelihoods and maximizes
    the Fisher information preserved for a given compression dimension.
    """

    def __init__(
        self,
        compress_frac: float = 0.5,
        clean_data: bool = True
    ):
        """
        Initialize MOPED compression.

        Args:
            compress_frac: Fraction of data to use for computing compression matrix
                          (remaining data can be used for Fisher analysis)
            clean_data: If True, remove zero-variance features before compression
        """
        super().__init__()
        self.compress_frac = compress_frac
        self.clean_data = clean_data

    def forward(
        self,
        summaries: List[torch.Tensor],
        delta_theta: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Compute and apply MOPED compression to summaries.

        Args:
            summaries: List of summary tensors
            delta_theta: Step sizes for derivative estimation (required for MOPED)

        Returns:
            Compressed summaries with shape (n_samples, n_params)
        """
        if delta_theta is None:
            raise ValueError("MOPED compression requires delta_theta for derivative computation")

        # Clean data if requested
        if self.clean_data:
            summaries = self._clean_summaries(summaries)

        # Compute compression matrix using a fraction of the data
        compression_matrix = self._compute_moped_matrix(summaries, delta_theta)

        # Apply compression to ALL summaries
        compressed_summaries = [s @ compression_matrix for s in summaries]

        return compressed_summaries

    def _compute_moped_matrix(
        self,
        summaries: List[torch.Tensor],
        delta_theta: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MOPED compression matrix: B = C^{-1} dμ

        Args:
            summaries: List of summary tensors
            delta_theta: Step sizes for derivatives

        Returns:
            Compression matrix of shape (n_features, n_params)
        """
        # Split data for compression computation
        vecs_cov = summaries[0]
        n_s = vecs_cov.shape[0]
        n_comp = int(self.compress_frac * n_s)

        # Use first fraction for computing compression matrix
        comp_cov = vecs_cov[:n_comp]
        comp_ders = [s[:int(self.compress_frac * s.shape[0])] for s in summaries[1:]]

        # Compute derivatives for compression
        derivatives = self._compute_derivatives(comp_ders, delta_theta)
        mean_derivatives = derivatives.mean(dim=1)  # (n_params, n_features)

        # Compute covariance
        C = self._compute_covariance(comp_cov)

        # MOPED compression: B = C^{-1} dμ
        compression_matrix = torch.linalg.solve(C, mean_derivatives.T)  # (n_features, n_params)

        return compression_matrix

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
            n_removed = (~valid_idx).sum().item()
            print(f"MOPEDCompression: Removing {n_removed} zero-variance features")

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
