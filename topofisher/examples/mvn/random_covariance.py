"""
Example: Computing Fisher information for multivariate normal vectors with random covariance.

This example demonstrates the TopoFisher pipeline with GaussianVectorSimulator
using a randomly generated positive definite covariance matrix.

For d-dimensional Gaussian vectors with mean θ and fixed covariance Σ,
the theoretical Fisher information matrix is F = Σ^{-1}.
"""
import torch

from topofisher import (
    GaussianVectorSimulator,
    IdentityFiltration,
    IdentityVectorization,
    IdentityCompression,
    FisherAnalyzer,
    FisherPipeline,
    FisherConfig
)


def generate_random_covariance(d, condition_number=10.0):
    """
    Generate a random positive definite covariance matrix.

    Args:
        d: Dimension
        condition_number: Condition number (ratio of largest to smallest eigenvalue)

    Returns:
        Random covariance matrix (d x d)
    """
    # Generate random orthogonal matrix
    A = torch.randn(d, d)
    Q, _ = torch.linalg.qr(A)

    # Generate eigenvalues with specified condition number
    # Evenly spaced between 1/condition_number and 1
    eigenvalues = torch.linspace(1.0 / condition_number, 1.0, d)

    # Construct covariance: Σ = Q Λ Q^T
    Lambda = torch.diag(eigenvalues)
    covariance = Q @ Lambda @ Q.T

    return covariance


def main():
    print("=" * 70)
    print("Computing Fisher Information for MVN with Random Covariance")
    print("=" * 70)

    # Configuration
    d = 10  # Dimension of vectors
    theta_fid = torch.randn(d)  # Random mean vector
    delta_theta = 0.1 * torch.ones(d)  # Small perturbations

    n_s = 5000  # Samples for covariance
    n_d = 2000  # Samples for derivatives

    device = "cpu"

    # Generate random covariance matrix
    covariance = generate_random_covariance(d, condition_number=10.0)

    print(f"\nConfiguration:")
    print(f"  Dimension: d = {d}")
    print(f"  Fiducial mean: θ = [{', '.join([f'{x:.2f}' for x in theta_fid.numpy()])}]")
    print(f"  Step sizes: Δθ = {delta_theta[0]:.2f} (same for all params)")
    print(f"  Samples for covariance: n_s = {n_s}")
    print(f"  Samples for derivatives: n_d = {n_d}")
    print(f"  Device: {device}")

    print(f"\nCovariance Matrix (first 5x5 block):")
    import numpy as np
    np.set_printoptions(precision=2, suppress=True)
    print(covariance[:5, :5].numpy())

    # Show condition number
    eigenvalues = torch.linalg.eigvalsh(covariance)
    cond_num = eigenvalues.max() / eigenvalues.min()
    print(f"  Condition number: {cond_num:.2f}")

    # 1. Set up simulator with random covariance
    print("\n" + "-" * 70)
    print("Setting up pipeline...")
    print("-" * 70)

    simulator = GaussianVectorSimulator(
        d=d,
        covariance=covariance,
        device=device
    )
    print(f"✓ Simulator: {simulator}")

    # 2. Identity filtration and vectorization (pass-through for vectors)
    filtration = IdentityFiltration()
    vectorization = IdentityVectorization()
    print(f"✓ Filtration: {filtration}")
    print(f"✓ Vectorization: {vectorization}")

    # 3. No compression
    compression = IdentityCompression()
    print(f"✓ Compression: IdentityCompression (no compression)")

    # 4. Fisher analyzer
    fisher = FisherAnalyzer(clean_data=True)
    print(f"✓ Fisher Analyzer: clean_data=True")

    # 5. Create pipeline
    pipeline = FisherPipeline(
        simulator=simulator,
        filtration=filtration,
        vectorization=vectorization,
        compression=compression,
        fisher_analyzer=fisher
    )
    print(f"✓ Pipeline created")

    # 6. Configure analysis
    print("\n" + "-" * 70)
    print("Running Fisher information analysis...")
    print("-" * 70)

    config = FisherConfig(
        theta_fid=theta_fid,
        delta_theta=delta_theta,
        n_s=n_s,
        n_d=n_d,
        find_derivative=[True] * d  # Compute derivatives for all parameters
    )

    # 7. Run pipeline
    result = pipeline(config)

    # 8. Get theoretical Fisher matrix
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    theoretical_fisher = simulator.theoretical_fisher_matrix(theta_fid)

    print("\nTheoretical Fisher Matrix (first 5x5 block):")
    print(theoretical_fisher[:5, :5].numpy())
    print(f"  Shape: {theoretical_fisher.shape}")
    print(f"  Note: F_theoretical = Σ^{-1}")

    print("\nEmpirical Fisher Matrix (first 5x5 block):")
    print(result.fisher_matrix[:5, :5].numpy())
    print(f"  Shape: {result.fisher_matrix.shape}")

    # 9. Compare with theoretical
    print("\n" + "-" * 70)
    print("Comparison with Theoretical Fisher Matrix")
    print("-" * 70)

    # Diagonal elements
    fisher_diag = torch.diag(result.fisher_matrix)
    theo_diag = torch.diag(theoretical_fisher)

    print(f"\nDiagonal elements (first 5):")
    print(f"  Empirical: [{', '.join([f'{x:.2f}' for x in fisher_diag[:5].numpy()])}]")
    print(f"  Theoretical: [{', '.join([f'{x:.2f}' for x in theo_diag[:5].numpy()])}]")

    # Frobenius norm of difference
    diff = result.fisher_matrix - theoretical_fisher
    frobenius_norm = torch.norm(diff, p='fro').item()
    frobenius_norm_rel = frobenius_norm / torch.norm(theoretical_fisher, p='fro').item()

    print(f"\nFrobenius norm of difference: {frobenius_norm:.4f}")
    print(f"Relative Frobenius norm: {frobenius_norm_rel:.4f}")

    # 10. Parameter constraints
    print("\n" + "-" * 70)
    print("Parameter Constraints (1-sigma)")
    print("-" * 70)

    print(f"\nEmpirical constraints (first 5):")
    print(f"  [{', '.join([f'{x:.2f}' for x in result.constraints[:5].numpy()])}]")

    theoretical_constraints = torch.sqrt(torch.diag(torch.linalg.inv(theoretical_fisher)))
    print(f"\nTheoretical constraints (first 5):")
    print(f"  [{', '.join([f'{x:.2f}' for x in theoretical_constraints[:5].numpy()])}]")

    # 11. Log determinants
    print("\n" + "-" * 70)
    print("Fisher Information (log determinant)")
    print("-" * 70)

    print(f"\nEmpirical: log|F| = {result.log_det_fisher:.4f}")
    theo_log_det = torch.logdet(theoretical_fisher).item()
    print(f"Theoretical: log|F| = {theo_log_det:.4f}")
    print(f"Difference: {abs(result.log_det_fisher - theo_log_det):.4f}")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
