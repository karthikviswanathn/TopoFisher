"""
Train linear compression for Fourier magnitudes to maximize Fisher information.

This script uses the refactored FisherPipeline with:
- RawGRFSimulator (returns Fourier magnitudes directly)
- IdentityFiltration (pass-through, no persistence)
- IdentityVectorization (pass-through)
- MLPCompression with no hidden layers (linear compression)
- Built-in training via compression.train_compression()
"""
import torch

from topofisher import (
    FisherAnalyzer, FisherConfig, TrainingConfig, FisherPipeline
)
from topofisher.compressions import MLPCompression
from topofisher.filtrations import IdentityFiltration
from topofisher.vectorizations import IdentityVectorization
from topofisher.examples.grf.raw.raw_grf_simulator import RawGRFSimulator


def main():
    print("=" * 80)
    print("Training Linear Compression for Fourier Magnitudes")
    print("=" * 80)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Components
    simulator = RawGRFSimulator(
        N=16,
        dim=2,
        return_type='fourier_recovered',
        device=str(device)
    )

    filtration = IdentityFiltration()  # Pass-through (no persistence)
    vectorization = IdentityVectorization()  # Pass-through

    compression = MLPCompression(
        hidden_dims=[],  # Linear only (no hidden layers)
        dropout=0.0
    )

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
        n_s=20000,
        n_d=20000,
        find_derivative=[True, True],
        seed_cov=42,
        seed_ders=[43, 45]
    )

    # Training config
    training_config = TrainingConfig(
        n_epochs=2000,
        lr=1e-3,
        batch_size=500,
        weight_decay=0.0,
        train_frac=0.5,
        val_frac=0.25,
        validate_every=50,  # Validate every 50 epochs
        verbose=True,
        check_gaussianity=True
    )

    print("\nRunning pipeline with training...")
    print(f"   Input will be Fourier magnitudes (dim = {16*16})")
    print(f"   Output will be linear compression to {len(config.theta_fid)} features")

    # Run pipeline (generates data, trains compression, computes Fisher)
    result = pipeline.run(config, training_config=training_config)

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION ON TEST SET")
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
    print(f"{'Fourier + Linear':<30} {result.log_det_fisher.item():>10.2f} {result.constraints[0].item():>10.4f} {result.constraints[1].item():>10.4f} {result.fisher_matrix[0,0].item():>10.2f} {result.fisher_matrix[0,1].item():>10.2f} {result.fisher_matrix[1,1].item():>10.2f}")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
