"""
Fisher analysis on noisy circle point clouds using alpha complex.

This example computes Fisher information for the noisy ring distribution,
where points are sampled from:
- A noisy circle with radius ~ Normal(theta[0], theta[1])
- Uniform background points

Parameters: theta = [radius_mean, radius_std]
"""
import torch
from topofisher import (
    NoisyRingSimulator,
    AlphaComplexLayer,
    TopKLayer,
    CombinedVectorization,
    MOPEDCompression,
    FisherAnalyzer,
    FisherPipeline,
    FisherConfig
)


def main():
    print("=" * 80)
    print("TopoFisher: Noisy Circle with Alpha Complex")
    print("=" * 80)

    # 1. Set up components
    print("\n1. Initializing components...")

    # Simulator: noisy circle with background
    simulator = NoisyRingSimulator(
        ncirc=200,      # Points on the circle
        nback=20,       # Background points
        bgm_avg=1.0,    # Mean background radius
        device="cpu"
    )
    print("   ✓ Noisy Ring Simulator (200 circle + 20 background points)")

    # Filtration: alpha complex
    filtration = AlphaComplexLayer(
        homology_dimensions=[0, 1],
        min_persistence=[0.0, 0.0],
        show_progress=True
    )
    print("   ✓ Alpha Complex Filtration (H0, H1)")

    # Vectorization: Top-K features
    vectorization = CombinedVectorization([
        TopKLayer(k=100),  # Top-20 for H0
        TopKLayer(k=60),  # Top-30 for H1
    ])
    print("   ✓ Top-K Vectorization (k=20 for H0, k=30 for H1)")

    # Compression: MOPED
    compression = MOPEDCompression()
    print("   ✓ MOPED Compression (analytical optimal)")

    # Fisher analyzer
    fisher = FisherAnalyzer(clean_data=True)
    print("   ✓ Fisher Analyzer")

    # 2. Create pipeline
    print("\n2. Building pipeline...")
    pipeline = FisherPipeline(
        simulator=simulator,
        filtration=filtration,
        vectorization=vectorization,
        compression=compression,
        fisher_analyzer=fisher
    )
    print("   ✓ Pipeline assembled")

    # 3. Configure analysis
    print("\n3. Configuring Fisher analysis...")
    config = FisherConfig(
        theta_fid=torch.tensor([1.0, 0.2]),      # Fiducial: radius_mean=1.0, radius_std=0.2
        delta_theta=torch.tensor([0.05, 0.01]),  # Step sizes (will be divided by 2)
        n_s=5000,                                 # Simulations for covariance
        n_d=5000,                                 # Simulations for derivatives
        find_derivative=[True, True],             # Compute derivatives for both parameters
        seed_cov=42,                              # Reproducibility
        seed_ders=[43, 44]
    )
    print(f"   Fiducial parameters: radius_mean={config.theta_fid[0]}, radius_std={config.theta_fid[1]}")
    print(f"   Step sizes: δμ={config.delta_theta[0]:.4f}, δσ={config.delta_theta[1]:.4f}")
    print(f"   Simulations: {config.n_s} (covariance), {config.n_d} (derivatives)")

    # 4. Generate test data and print PD statistics
    print("\n4. Persistence Diagram Statistics...")
    test_data = simulator.generate(config.theta_fid, n_samples=100, seed=999)
    test_diagrams = filtration(test_data)

    # Count points in each diagram
    n_points_h0 = torch.tensor([len(pd) for pd in test_diagrams[0]], dtype=torch.float32)
    n_points_h1 = torch.tensor([len(pd) for pd in test_diagrams[1]], dtype=torch.float32)

    print(f"   H0 points per diagram: min={n_points_h0.min():.0f}, "
          f"median={n_points_h0.median():.0f}, max={n_points_h0.max():.0f}")
    print(f"   H1 points per diagram: min={n_points_h1.min():.0f}, "
          f"median={n_points_h1.median():.0f}, max={n_points_h1.max():.0f}")

    # 5. Run Fisher analysis
    print("\n5. Running Fisher analysis...")
    result = pipeline.run(config)

    # 6. Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Compare with theoretical Fisher if available
    if hasattr(simulator, 'theoretical_fisher_matrix'):
        F_theory = simulator.theoretical_fisher_matrix(config.theta_fid)
        log_det_theory = torch.logdet(F_theory)
        constraints_theory = torch.sqrt(torch.diag(torch.linalg.inv(F_theory)))

        print(f"\n{'Method':<30} {'log|F|':>10} {'σ(μ)':>10} {'σ(σ)':>10} {'F[0,0]':>10} {'F[0,1]':>10} {'F[1,1]':>10}")
        print("-" * 93)
        print(f"{'Theoretical':<30} {log_det_theory.item():>10.2f} {constraints_theory[0].item():>10.4f} {constraints_theory[1].item():>10.4f} {F_theory[0,0].item():>10.2f} {F_theory[0,1].item():>10.2f} {F_theory[1,1].item():>10.2f}")
        print(f"{'Alpha Complex + MOPED':<30} {result.log_det_fisher.item():>10.2f} {result.constraints[0].item():>10.4f} {result.constraints[1].item():>10.4f} {result.fisher_matrix[0,0].item():>10.2f} {result.fisher_matrix[0,1].item():>10.2f} {result.fisher_matrix[1,1].item():>10.2f}")
    else:
        print(f"\nFisher Matrix:\n{result.fisher_matrix}")
        print(f"\nlog|F|: {result.log_det_fisher.item():.2f}")
        print(f"Marginal constraints (1σ):")
        print(f"  σ(radius_mean) = {result.constraints[0].item():.4f}")
        print(f"  σ(radius_std)  = {result.constraints[1].item():.4f}")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

    return result


if __name__ == "__main__":
    result = main()
