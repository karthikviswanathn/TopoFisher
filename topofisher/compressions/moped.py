"""
MOPED (Multiple Optimized Parameter Estimation and Data compression) compression.

Reference: Heavens et al. 2000 (https://arxiv.org/abs/astro-ph/9911102)
"""
from typing import List, Optional, Tuple
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
        train_frac: float = 0.5,
        clean_data: bool = True,
        reg: float = 1e-6
    ):
        """
        Initialize MOPED compression.

        Args:
            train_frac: Fraction of data for training set
                       Compression matrix is learned on train set, applied to test set
            clean_data: If True, remove zero-variance features before compression
            reg: Regularization parameter for covariance matrix (C + reg*I) to avoid singularity
        """
        super().__init__()
        self.train_frac = train_frac
        self.clean_data = clean_data
        self.reg = reg

    def returns_test_only(self) -> bool:
        """MOPED splits data and returns only test set."""
        return True

    def split_data(self, summaries: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Split summaries into train/test sets.

        Important: theta_minus and theta_plus pairs use the same random seed during generation,
        so they must use the same permutation to maintain pairing.

        Args:
            summaries: List of tensors [fiducial, theta_minus_0, theta_plus_0, theta_minus_1, theta_plus_1, ...]

        Returns:
            (train_summaries, test_summaries)
        """

        def split_with_perm(tensor, perm, train_frac):
            """Split a tensor using a given permutation."""
            n = tensor.shape[0]
            shuffled = tensor[perm]
            n_train = int(n * train_frac)
            return shuffled[:n_train], shuffled[n_train:]

        # Generate permutations with fixed seed for reproducibility
        torch.manual_seed(42)

        # Fiducial gets its own permutation
        perm_fid = torch.randperm(summaries[0].shape[0])

        # Each parameter pair (theta_minus, theta_plus) shares a permutation
        n_params = (len(summaries) - 1) // 2
        perms_deriv = [torch.randperm(summaries[1 + 2*i].shape[0]) for i in range(n_params)]

        # Apply permutations and split
        split_summaries_list = []

        # Fiducial
        split_summaries_list.append(split_with_perm(summaries[0], perm_fid, self.train_frac))

        # Derivatives (pairs share permutation to maintain seed pairing)
        for i in range(n_params):
            perm = perms_deriv[i]
            split_summaries_list.append(split_with_perm(summaries[1 + 2*i], perm, self.train_frac))
            split_summaries_list.append(split_with_perm(summaries[2 + 2*i], perm, self.train_frac))

        # Reorganize into separate lists
        train_summaries = [s[0] for s in split_summaries_list]
        test_summaries = [s[1] for s in split_summaries_list]

        return train_summaries, test_summaries

    def forward(
        self,
        summaries: List[torch.Tensor],
        delta_theta: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Compute and apply MOPED compression to summaries.

        The compression matrix is learned using the train set,
        then applied to the test set only.

        Args:
            summaries: List of summary tensors
            delta_theta: Step sizes for derivative estimation (required for MOPED)

        Returns:
            Compressed test summaries with shape (n_test_samples, n_params)
        """
        if delta_theta is None:
            raise ValueError("MOPED compression requires delta_theta for derivative computation")

        # Clean data if requested
        if self.clean_data:
            summaries = self._clean_summaries(summaries)

        # Split into train/test
        train_summaries, test_summaries = self.split_data(summaries)

        # Compute compression matrix using ONLY train set
        compression_matrix = self._compute_moped_matrix(train_summaries, delta_theta)

        # Apply compression to test summaries only
        compressed_summaries = [s @ compression_matrix for s in test_summaries]

        return compressed_summaries

    def _compute_moped_matrix(
        self,
        summaries: List[torch.Tensor],
        delta_theta: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MOPED compression matrix: B = C^{-1} dμ

        Args:
            summaries: List of summary tensors (already split, e.g., test set only)
            delta_theta: Step sizes for derivatives

        Returns:
            Compression matrix of shape (n_features, n_params)
        """
        # Extract fiducial and derivative summaries
        vecs_cov = summaries[0]
        perturbed_vecs = summaries[1:]

        # Compute derivatives
        derivatives = self._compute_derivatives(perturbed_vecs, delta_theta)
        mean_derivatives = derivatives.mean(dim=1)  # (n_params, n_features)

        # Compute covariance
        C = self._compute_covariance(vecs_cov)

        # Add regularization to avoid singularity: C → C + reg*I
        if self.reg > 0:
            C = C + self.reg * torch.eye(C.shape[0], device=C.device, dtype=C.dtype)

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
