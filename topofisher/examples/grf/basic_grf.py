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
    CachedFisherPipeline,
    FisherConfig
)


def main(use_cache=True, cache_path='data/diagrams_basic.pkl'):
    print("=" * 60)
    print("TopoFisher: Basic GRF Example")
    if use_cache:
        print(f"Using cached diagrams: {cache_path}")
    print("=" * 60)

    # 1. Set up components
    print("\n1. Initializing components...")

    N = 32  # Grid size
    simulator = GRFSimulator(
        N=N,               # 32x32 grid
        dim=2,              # 2D fields
        boxlength=N,
        device="cpu"
    )
    print("   ✓ GRF Simulator (32x32 grid)")

    filtration = CubicalLayer(
        homology_dimensions=[0, 1],
        min_persistence=[0.0, 0.0]
    )
    print("   ✓ Cubical Filtration (H0, H1)")

    vectorization = CombinedVectorization([
        TopKLayer(k=50),  # Top-k for H0
        TopKLayer(k=80),  # Top-k for H1
    ])
    print("   ✓ Top-K Vectorization (k=10 per dimension)")

    fisher = FisherAnalyzer(clean_data=True, use_moped=True, moped_compress_frac=0.5)
    print("   ✓ Fisher Analyzer (with MOPED)")

    # 2. Create pipeline
    print("\n2. Building pipeline...")
    if use_cache:
        pipeline = CachedFisherPipeline(
            simulator=simulator,
            filtration=filtration,
            vectorization=vectorization,
            fisher_analyzer=fisher,
            cache_path=cache_path,
            auto_generate=True
        )
        print(f"   ✓ Cached pipeline assembled (cache: {cache_path})")
    else:
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
        n_s=20000,                                   # Simulations for covariance
        n_d=20000,                                   # Simulations for derivatives
        find_derivative=[True, True],             # Derivatives for both params
        seed_cov=42,                              # Reproducibility
        seed_ders=[43, 44]
    )
    print(f"   Fiducial parameters: A={config.theta_fid[0]}, B={config.theta_fid[1]}")
    print(f"   Step sizes: δA={config.delta_theta[0]:.4f}, δB={config.delta_theta[1]:.4f}")
    print(f"   Simulations: {config.n_s} (covariance), {config.n_d} (derivatives)")

    # 4. Generate test data and print PD statistics
    print("\n4. Persistence Diagram Statistics...")
    test_data = simulator.generate(config.theta_fid, n_samples=100, seed=999)
    test_diagrams = filtration(test_data)

    # Count points in each diagram
    n_points_h0 = torch.tensor([len(pd) for pd in test_diagrams[0]], dtype=torch.float32)
    n_points_h1 = torch.tensor([len(pd) for pd in test_diagrams[1]], dtype=torch.float32)

    print(f"   Number of Points in PD0: 1-percentile: {n_points_h0.quantile(0.01):.0f}, "
          f"5-percentile: {n_points_h0.quantile(0.05):.0f}, "
          f"Median: {n_points_h0.median():.0f}")
    print(f"   Number of Points in PD1: 1-percentile: {n_points_h1.quantile(0.01):.0f}, "
          f"5-percentile: {n_points_h1.quantile(0.05):.0f}, "
          f"Median: {n_points_h1.median():.0f}")

    # 5. Run pipeline
    print("\n5. Running pipeline...")
    print("   This may take a few minutes...")
    result = pipeline(config)
    print("   ✓ Analysis complete")

    # 6. Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print("\nTheoretical Fisher Matrix (from power spectrum):")
    F_theory = simulator.theoretical_fisher_matrix(config.theta_fid)
    print(F_theory.numpy())

    print("\nEmpirical Fisher Matrix (from topological summaries):")
    print(result.fisher_matrix.numpy())

    print("\nMOPED Fisher Matrix (compressed summaries):")
    print(result.fisher_matrix_moped.numpy())

    print("\nParameter Constraints (1-sigma):")
    inv_F_theory = torch.linalg.inv(F_theory)
    constraints_theory = torch.sqrt(torch.diag(inv_F_theory))
    print(f"  Theory: σ(A) = {constraints_theory[0]:.4f}, σ(B) = {constraints_theory[1]:.4f}")
    print(f"   Full:  σ(A) = {result.constraints[0]:.4f}, σ(B) = {result.constraints[1]:.4f}")
    print(f"   MOPED: σ(A) = {result.constraints_moped[0]:.4f}, σ(B) = {result.constraints_moped[1]:.4f}")

    print(f"\nLog Fisher Information:")
    sign, logdet = torch.linalg.slogdet(F_theory)
    log_det_theory = sign * logdet
    print(f" Theory:  {log_det_theory:.2f}")
    print(f"   Full:  {result.log_det_fisher:.2f}")
    print(f"  MOPED:  {result.log_det_fisher_moped:.2f}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
