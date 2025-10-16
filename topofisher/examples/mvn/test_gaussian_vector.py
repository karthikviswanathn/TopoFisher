"""
Unit tests for the refactored pipeline using Gaussian vectors.

This test verifies that the new compression stage works correctly by:
1. Generating d-dimensional Gaussian vectors
2. Using identity transformations for filtration/vectorization (pass-through)
3. Testing different compression methods
4. Comparing to theoretical Fisher matrix
"""
import unittest
import torch

from topofisher import (
    FisherConfig,
    FisherPipeline,
    GaussianVectorSimulator,
    IdentityFiltration,
    IdentityVectorization,
    IdentityCompression,
    MOPEDCompression,
    FisherAnalyzer
)


class TestGaussianVectorPipeline(unittest.TestCase):
    """Test suite for Gaussian vector pipeline with identity covariance."""

    def setUp(self):
        """Set up test fixtures."""
        self.d = 5  # Small dimension for fast tests
        self.device = "cpu"
        self.simulator = GaussianVectorSimulator(d=self.d, covariance=None, device=self.device)
        self.theta_fid = torch.zeros(self.d)
        self.delta_theta = 0.1 * torch.ones(self.d)

    def test_theoretical_fisher_is_identity(self):
        """Test that theoretical Fisher matrix is identity for unit covariance."""
        F_theory = self.simulator.theoretical_fisher_matrix(self.theta_fid)

        # Check shape
        self.assertEqual(F_theory.shape, (self.d, self.d))

        # Check it's close to identity
        expected = torch.eye(self.d)
        self.assertTrue(torch.allclose(F_theory, expected, atol=1e-6))

    def test_pipeline_with_identity_compression(self):
        """Test full pipeline with identity compression."""
        # Setup
        config = FisherConfig(
            theta_fid=self.theta_fid,
            delta_theta=self.delta_theta,
            n_s=500,
            n_d=500,
            find_derivative=[True] * self.d,
            seed_cov=42,
            seed_ders=list(range(100, 100 + self.d))
        )

        pipeline = FisherPipeline(
            simulator=self.simulator,
            filtration=IdentityFiltration(),
            vectorization=IdentityVectorization(),
            compression=IdentityCompression(),
            fisher_analyzer=FisherAnalyzer(clean_data=True)
        )

        # Run
        result = pipeline(config)

        # Check outputs exist
        self.assertIsNotNone(result.fisher_matrix)
        self.assertIsNotNone(result.constraints)
        self.assertIsNotNone(result.log_det_fisher)

        # Check shape
        self.assertEqual(result.fisher_matrix.shape, (self.d, self.d))
        self.assertEqual(result.constraints.shape, (self.d,))

        # Check Fisher matrix is positive definite
        eigenvalues = torch.linalg.eigvalsh(result.fisher_matrix)
        self.assertTrue(torch.all(eigenvalues > 0))

    def test_pipeline_with_moped_compression(self):
        """Test full pipeline with MOPED compression."""
        config = FisherConfig(
            theta_fid=self.theta_fid,
            delta_theta=self.delta_theta,
            n_s=500,
            n_d=500,
            find_derivative=[True] * self.d,
            seed_cov=42,
            seed_ders=list(range(100, 100 + self.d))
        )

        pipeline = FisherPipeline(
            simulator=self.simulator,
            filtration=IdentityFiltration(),
            vectorization=IdentityVectorization(),
            compression=MOPEDCompression(compress_frac=0.5),
            fisher_analyzer=FisherAnalyzer(clean_data=True)
        )

        # Run
        result = pipeline(config)

        # Check outputs
        self.assertEqual(result.fisher_matrix.shape, (self.d, self.d))

        # MOPED should preserve Fisher information reasonably well
        F_theory = self.simulator.theoretical_fisher_matrix(self.theta_fid)
        log_det_theory = torch.logdet(F_theory).item()

        # Allow some degradation from MOPED compression
        self.assertGreater(result.log_det_fisher.item(), log_det_theory - 2.0)

    def test_identity_vs_theoretical(self):
        """Test that identity compression recovers theoretical Fisher matrix approximately."""
        config = FisherConfig(
            theta_fid=self.theta_fid,
            delta_theta=self.delta_theta,
            n_s=2000,  # More samples for better accuracy
            n_d=2000,
            find_derivative=[True] * self.d,
            seed_cov=42,
            seed_ders=list(range(100, 100 + self.d))
        )

        pipeline = FisherPipeline(
            simulator=self.simulator,
            filtration=IdentityFiltration(),
            vectorization=IdentityVectorization(),
            compression=IdentityCompression(),
            fisher_analyzer=FisherAnalyzer(clean_data=True)
        )

        result = pipeline(config)
        F_theory = self.simulator.theoretical_fisher_matrix(self.theta_fid)

        # Check diagonal elements are close to 1
        fisher_diag = torch.diag(result.fisher_matrix)
        theo_diag = torch.diag(F_theory)

        # Allow 20% error on diagonal
        for i in range(self.d):
            self.assertAlmostEqual(fisher_diag[i].item(), theo_diag[i].item(), delta=0.2)


class TestGaussianVectorRandomCovariance(unittest.TestCase):
    """Test suite for Gaussian vector pipeline with random covariance."""

    def setUp(self):
        """Set up test fixtures with random covariance."""
        self.d = 5
        self.device = "cpu"

        # Generate random positive definite covariance
        torch.manual_seed(42)  # For reproducibility in tests
        A = torch.randn(self.d, self.d)
        self.covariance = A @ A.T + 0.1 * torch.eye(self.d)  # Ensure positive definite

        self.simulator = GaussianVectorSimulator(
            d=self.d,
            covariance=self.covariance,
            device=self.device
        )
        self.theta_fid = torch.randn(self.d)
        self.delta_theta = 0.1 * torch.ones(self.d)

    def test_theoretical_fisher_is_inverse_covariance(self):
        """Test that theoretical Fisher matrix is inverse of covariance."""
        F_theory = self.simulator.theoretical_fisher_matrix(self.theta_fid)
        expected = torch.linalg.inv(self.covariance)

        self.assertTrue(torch.allclose(F_theory, expected, atol=1e-5))

    def test_pipeline_with_random_covariance(self):
        """Test pipeline works with random covariance matrix."""
        config = FisherConfig(
            theta_fid=self.theta_fid,
            delta_theta=self.delta_theta,
            n_s=2000,
            n_d=1000,
            find_derivative=[True] * self.d,
            seed_cov=42,
            seed_ders=list(range(43, 43 + self.d))
        )

        pipeline = FisherPipeline(
            simulator=self.simulator,
            filtration=IdentityFiltration(),
            vectorization=IdentityVectorization(),
            compression=IdentityCompression(),
            fisher_analyzer=FisherAnalyzer(clean_data=True)
        )

        result = pipeline(config)

        # Check positive definiteness
        eigenvalues = torch.linalg.eigvalsh(result.fisher_matrix)
        self.assertTrue(torch.all(eigenvalues > 0))

        # Check log determinant is reasonable
        F_theory = self.simulator.theoretical_fisher_matrix(self.theta_fid)
        log_det_theory = torch.logdet(F_theory).item()

        # Allow some error in log determinant
        self.assertAlmostEqual(
            result.log_det_fisher.item(),
            log_det_theory,
            delta=abs(log_det_theory) * 0.1  # 10% tolerance
        )


if __name__ == "__main__":
    unittest.main()
