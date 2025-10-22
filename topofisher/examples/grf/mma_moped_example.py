"""
Fisher information analysis with MMA + MOPED.

This script demonstrates the recommended approach for GRF analysis:
- MMALayer (multiparameter persistence using field + gradient)
- MMATopKLayer (select top-k corners by lexicographic order)
- MOPEDCompression (analytical optimal compression)

Achieves ~40% efficiency with optimal k values.
"""
import torch

from topofisher import (
    GRFSimulator,
    MMALayer,
    MMATopKLayer,
    CombinedVectorization,
    MOPEDCompression,
    FisherAnalyzer,
    FisherPipeline,
    FisherConfig
)


def main():
    print("=" * 80)
    print("Fisher Information Analysis: MMA + MOPED")
    print("=" * 80)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Components
    simulator = GRFSimulator(N=16, dim=2, device=str(device))
    filtration = MMALayer(homology_dimensions=[0, 1])

    # Optimal k values based on corner count analysis:
    # H0: ~358 corners (use k=100), H1: ~91 corners (use k=50)
    vectorization = CombinedVectorization([
        MMATopKLayer(k=100),  # H0
        MMATopKLayer(k=50)    # H1
    ])

    compression = MOPEDCompression()
    fisher_analyzer = FisherAnalyzer(clean_data=True)

    # Assemble pipeline
    pipeline = FisherPipeline(
        simulator=simulator,
        filtration=filtration,
        vectorization=vectorization,
        compression=compression,
        fisher_analyzer=fisher_analyzer
    )

    # Fisher config
    config = FisherConfig(
        theta_fid=torch.tensor([1.0, 2.0]),
        delta_theta=torch.tensor([0.1, 0.2]),
        n_s=1000,
        n_d=1000,
        find_derivative=[True, True],
        seed_cov=42,
        seed_ders=[43, 44]
    )

    print("\nRunning pipeline...")
    print(f"   MMA features: k_H0=100, k_H1=50 → 300 features total")
    print(f"   MOPED compression: 300 → 2 (analytical optimal)")

    # Run pipeline
    result = pipeline(config)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Theoretical Fisher
    F_theory = simulator.theoretical_fisher_matrix(config.theta_fid)
    inv_F_theory = torch.linalg.inv(F_theory)
    constraints_theory = torch.sqrt(torch.diag(inv_F_theory))
    sign, logdet = torch.linalg.slogdet(F_theory)
    log_det_theory = sign * logdet

    print(f"\n{'Method':<30} {'log|F|':>10} {'σ(A)':>10} {'σ(B)':>10} {'F[0,0]':>10} {'F[0,1]':>10} {'F[1,1]':>10}")
    print("-"*93)
    print(f"{'Theoretical':<30} {log_det_theory.item():>10.2f} {constraints_theory[0].item():>10.4f} {constraints_theory[1].item():>10.4f} {F_theory[0,0].item():>10.2f} {F_theory[0,1].item():>10.2f} {F_theory[1,1].item():>10.2f}")
    print(f"{'MMA + MOPED':<30} {result.log_det_fisher.item():>10.2f} {result.constraints[0].item():>10.4f} {result.constraints[1].item():>10.4f} {result.fisher_matrix[0,0].item():>10.2f} {result.fisher_matrix[0,1].item():>10.2f} {result.fisher_matrix[1,1].item():>10.2f}")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
