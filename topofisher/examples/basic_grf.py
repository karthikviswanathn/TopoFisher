"""
Basic example: Fisher analysis on Gaussian Random Fields.
"""
import torch
from topofisher import (
    GRFSimulator,
    CubicalLayer,
    TopKLayer,
    CombinedVectorization,
    FisherAnalyzer,
    FisherPipeline,
    FisherConfig
)


def main():
    print("=" * 60)
    print("TopoFisher: Basic GRF Example")
    print("=" * 60)

    # 1. Set up components
    print("\n1. Initializing components...")

    simulator = GRFSimulator(
        N=32,               # 32x32 grid
        dim=2,              # 2D fields
        boxlength=1.0,
        device="cpu"
    )
    print("   ✓ GRF Simulator (32x32 grid)")

    filtration = CubicalLayer(
        homology_dimensions=[0, 1],
        min_persistence=[0.0, 0.0]
    )
    print("   ✓ Cubical Filtration (H0, H1)")

    vectorization = CombinedVectorization([
        TopKLayer(k=10),  # Top-10 for H0
        TopKLayer(k=10),  # Top-10 for H1
    ])
    print("   ✓ Top-K Vectorization (k=10 per dimension)")

    fisher = FisherAnalyzer(clean_data=True)
    print("   ✓ Fisher Analyzer")

    # 2. Create pipeline
    print("\n2. Building pipeline...")
    pipeline = FisherPipeline(
        simulator=simulator,
        filtration=filtration,
        vectorization=vectorization,
        fisher_analyzer=fisher
    )
    print("   ✓ Pipeline assembled")

    # 3. Configure analysis
    print("\n3. Configuring Fisher analysis...")
    config = FisherConfig(
        theta_fid=torch.tensor([1.0, 2.0]),      # Fiducial: A=1.0, B=2.0
        delta_theta=torch.tensor([0.1, 0.2]),    # Step sizes
        n_s=50,                                   # Simulations for covariance
        n_d=50,                                   # Simulations for derivatives
        find_derivative=[True, True],             # Derivatives for both params
        seed_cov=42,                              # Reproducibility
        seed_ders=[43, 44]
    )
    print(f"   Fiducial parameters: A={config.theta_fid[0]}, B={config.theta_fid[1]}")
    print(f"   Step sizes: δA={config.delta_theta[0]}, δB={config.delta_theta[1]}")
    print(f"   Simulations: {config.n_s} (covariance), {config.n_d} (derivatives)")

    # 4. Run pipeline
    print("\n4. Running pipeline...")
    print("   This may take a few minutes...")
    result = pipeline(config)
    print("   ✓ Analysis complete")

    # 5. Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\nFisher Matrix:")
    print(result.fisher_matrix.numpy())

    print("\nInverse Fisher Matrix (Parameter Covariance):")
    print(result.inverse_fisher.numpy())

    print("\nParameter Constraints (1-sigma):")
    print(f"   σ(A) = {result.constraints[0]:.4f}")
    print(f"   σ(B) = {result.constraints[1]:.4f}")

    print(f"\nLog Fisher Information: {result.log_det_fisher:.4f}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
