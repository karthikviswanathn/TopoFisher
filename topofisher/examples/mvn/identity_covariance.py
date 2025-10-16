"""
Simple example: Computing Fisher information for multivariate normal vectors.

This example demonstrates the TopoFisher pipeline with GaussianVectorSimulator.
For d-dimensional Gaussian vectors with mean θ and fixed covariance Σ,
the theoretical Fisher information matrix is F = Σ^{-1}.

With unit covariance (Σ = I), we expect F = I (identity matrix).
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


def main():
    print("=" * 70)
    print("Computing Fisher Information for Multivariate Normal Vectors")
    print("=" * 70)

    # Configuration
    d = 10  # Dimension of vectors
    theta_fid = torch.randn(d)  # Random mean vector
    delta_theta = 0.1 * torch.ones(d)  # Small perturbations

    n_s = 5000  # Samples for covariance
    n_d = 2000 # Samples for derivatives

    device = "cpu"

    print(f"\nConfiguration:")
    print(f"  Dimension: d = {d}")
    print(f"  Fiducial mean: θ = [{', '.join([f'{x:.2f}' for x in theta_fid.numpy()])}]")
    print(f"  Step sizes: Δθ = {delta_theta[0]:.2f} (same for all params)")
    print(f"  Samples for covariance: n_s = {n_s}")
    print(f"  Samples for derivatives: n_d = {n_d}")
    print(f"  Device: {device}")

    # 1. Set up simulator with unit covariance
    print("\n" + "-" * 70)
    print("Setting up pipeline...")
    print("-" * 70)

    simulator = GaussianVectorSimulator(
        d=d,
        covariance=None,  # None = unit covariance (identity matrix)
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
    print(f"  Note: For unit covariance, F_theoretical = I (identity matrix)")

    print("\nEmpirical Fisher Matrix (first 5x5 block):")
    import numpy as np
    np.set_printoptions(precision=2, suppress=True)
    print(result.fisher_matrix[:5, :5].numpy())
    print(f"  Shape: {result.fisher_matrix.shape}")

    # 9. Compare with theoretical
    print("\n" + "-" * 70)
    print("Comparison with Theoretical Fisher Matrix")
    print("-" * 70)

    # Diagonal elements (should be ~1)
    fisher_diag = torch.diag(result.fisher_matrix)
    theo_diag = torch.diag(theoretical_fisher)

    print(f"\nDiagonal elements (first 5):")
    print(f"  Empirical: [{', '.join([f'{x:.2f}' for x in fisher_diag[:5].numpy()])}]")
    print(f"  Theoretical: [{', '.join([f'{x:.2f}' for x in theo_diag[:5].numpy()])}]")

    # 10. Parameter constraints
    print("\n" + "-" * 70)
    print("Parameter Constraints (1-sigma)")
    print("-" * 70)

    print(f"\nEmpirical constraints (first 5):")
    print(f"  [{', '.join([f'{x:.2f}' for x in result.constraints[:5].numpy()])}]")

    theoretical_constraints = torch.sqrt(torch.diag(torch.linalg.inv(theoretical_fisher)))
    print(f"\nTheoretical constraints (first 5):")
    print(f"  [{', '.join([f'{x:.2f}' for x in theoretical_constraints[:5].numpy()])}]")
    print(f"  Note: For unit covariance, σ_i = 1 for all parameters")

    # 11. Log determinants
    print("\n" + "-" * 70)
    print("Fisher Information (log determinant)")
    print("-" * 70)

    print(f"\nEmpirical: log|F| = {result.log_det_fisher:.4f}")
    theo_log_det = torch.logdet(theoretical_fisher).item()
    print(f"Theoretical: log|F| = {theo_log_det:.4f}")
    print(f"  Note: For unit covariance, log|F| = log|I| = 0")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()