"""
MOPED (Multiple Optimized Parameter Estimation and Data compression) compression.

Reference: Heavens et al. 2000 (https://arxiv.org/abs/astro-ph/9911102)

Note: While the original paper uses actual parameter step sizes (delta_theta) in
the compression matrix computation, we use unit values internally. This is simpler
and produces the same Fisher information matrix due to MOPED's scale-equivariance
property. The FisherAnalyzer will apply the correct finite differences regardless
of what scaling MOPED uses internally.
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

    Note: MOPED always splits data internally and returns ONLY the test set.
    This is because it computes the compression matrix on train data and
    evaluates on test data to avoid overfitting.
    """

    def __init__(
        self,
        train_frac: float = 0.5,
        clean_data: bool = True,
        reg: float = 1e-6,
        seed: int = 42
    ):
        """
        Initialize MOPED compression.

        Args:
            train_frac: Fraction of data for training set
                       Compression matrix is learned on train set, applied to test set
            clean_data: If True, remove zero-variance features before compression
            reg: Regularization parameter for covariance matrix (C + reg*I) to avoid singularity
            seed: Random seed for reproducible train/test splits
        """
        super().__init__()
        self.train_frac = train_frac
        self.clean_data = clean_data
        self.reg = reg
        self.seed = seed

    def split_data(self, summaries: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Split summaries into train/test sets using a local RNG.

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

        # Create local generator for reproducible splits without affecting global state
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        # Fiducial gets its own permutation
        perm_fid = torch.randperm(summaries[0].shape[0], generator=generator)

        # Each parameter pair (theta_minus, theta_plus) shares a permutation
        n_params = (len(summaries) - 1) // 2
        perms_deriv = [torch.randperm(summaries[1 + 2*i].shape[0], generator=generator) for i in range(n_params)]

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
        summaries: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Compute and apply MOPED compression to summaries.

        The compression matrix is learned using the train set,
        then applied to the test set only.

        Unlike the original paper (Heavens et al. 2000), we don't require actual
        parameter step sizes. We use unit values internally because:
        1. It's simpler - no need to pass extra parameters
        2. The Fisher information matrix is invariant to this choice due to
           MOPED's scale-equivariance property
        3. FisherAnalyzer applies the correct finite differences regardless

        Args:
            summaries: List of summary tensors [fid, minus_0, plus_0, minus_1, plus_1, ...]

        Returns:
            Compressed test summaries with shape (n_test_samples, n_params)
        """
        # Clean data if requested
        if self.clean_data:
            summaries = self._clean_summaries(summaries)

        # Split into train/test
        train_summaries, test_summaries = self.split_data(summaries)

        # Compute compression matrix using ONLY train set
        compression_matrix = self._compute_moped_matrix(train_summaries)

        # Apply compression to test summaries only
        compressed_summaries = [s @ compression_matrix for s in test_summaries]

        return compressed_summaries

    def _compute_moped_matrix(
        self,
        summaries: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute MOPED compression matrix: B = C^{-1} dμ

        Args:
            summaries: List of summary tensors (already split, e.g., test set only)

        Returns:
            Compression matrix of shape (n_features, n_params)
        """
        # Extract fiducial and finite difference summaries
        vecs_cov = summaries[0]
        perturbed_vecs = summaries[1:]

        # Compute finite differences
        finite_diffs = self._compute_finite_differences(perturbed_vecs)
        mean_derivatives = finite_diffs.mean(dim=1)  # (n_params, n_features)

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

    def _compute_finite_differences(
        self,
        perturbed_vecs: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute finite differences between plus and minus perturbations.

        We simply compute (plus - minus) without any scaling factor.
        The actual scaling doesn't matter because the Fisher matrix is invariant
        to this choice - FisherAnalyzer will apply the correct scaling based on
        the actual parameter step sizes used in data generation.

        Args:
            perturbed_vecs: List of tensors ordered as [minus_0, plus_0, minus_1, plus_1, ...]

        Returns:
            Tensor of shape (n_params, n_d, n_features)
        """
        n_params = len(perturbed_vecs) // 2
        finite_diffs = []

        for i in range(n_params):
            vecs_minus = perturbed_vecs[2 * i]      # theta minus perturbation
            vecs_plus = perturbed_vecs[2 * i + 1]   # theta plus perturbation

            # Simple difference - no scaling needed
            # Fisher matrix invariant to this choice
            diff = vecs_plus - vecs_minus
            finite_diffs.append(diff)

        return torch.stack(finite_diffs)  # (n_params, n_d, n_features)
