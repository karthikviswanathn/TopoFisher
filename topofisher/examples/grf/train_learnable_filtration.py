#!/usr/bin/env python
"""
Train learnable filtration for GRF Fisher information maximization.

This script trains a CNN-based learnable filtration that transforms input fields
to maximize Fisher information. The CNN learns to enhance topological features
relevant to parameter inference.

Example usage:
    # Quick test (20 epochs)
    python topofisher/examples/grf/train_learnable_filtration.py --n_epochs 20 --n_s 500 --n_d 500

    # Full training (500 epochs)
    python topofisher/examples/grf/train_learnable_filtration.py --n_epochs 500 --n_s 2000 --n_d 2000

    # Custom architecture
    python topofisher/examples/grf/train_learnable_filtration.py \
        --input_size 16 --hidden_channels 64 128 64 --activation leaky_relu
"""
import argparse
import torch
from datetime import datetime

from topofisher import (
    GRFSimulator,
    LearnableFiltration,
    CombinedVectorization,
    TopKLayer,
    TopKBirthsDeathsLayer,
    LearnableFiltrationPipeline,
    FisherConfig
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train learnable filtration for GRF Fisher information maximization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model architecture
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--input_size', type=int, default=8,
                             help='Input grid size (N×N)')
    model_group.add_argument('--hidden_channels', type=int, nargs='+', default=[32, 64, 32],
                             help='CNN hidden channel dimensions')
    model_group.add_argument('--kernel_size', type=int, default=3,
                             help='Convolution kernel size (must be odd)')
    model_group.add_argument('--activation', type=str, default='relu',
                             choices=['relu', 'leaky_relu', 'tanh'],
                             help='Activation function')
    model_group.add_argument('--upscale_factor', type=int, default=2,
                             choices=[1, 2],
                             help='CNN upscaling factor (1=no upscale N→N, 2=double size N→2N)')
    model_group.add_argument('--topk', type=int, default=10,
                             help='Number of top persistence pairs to keep')
    model_group.add_argument('--vectorization', type=str, default='topk',
                             choices=['topk', 'birthdeath'],
                             help='Vectorization method: topk (persistence-based) or birthdeath (independent births/deaths)')

    # Training configuration
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--n_epochs', type=int, default=100,
                             help='Number of training epochs')
    train_group.add_argument('--lr', type=float, default=1e-4,
                             help='Learning rate (weight_decay = lr/100)')
    train_group.add_argument('--validate_every', type=int, default=5,
                             help='Validation interval (epochs)')

    # Data configuration
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--n_s', type=int, default=2000,
                            help='Number of samples for covariance estimation')
    data_group.add_argument('--n_d', type=int, default=2000,
                            help='Number of samples for derivative estimation')
    data_group.add_argument('--theta_fid', type=float, nargs=2, default=[1.0, 2.0],
                            help='Fiducial parameter values [A, B]')
    data_group.add_argument('--delta_theta', type=float, nargs=2, default=[0.1, 0.2],
                            help='Finite difference step sizes [dA, dB]')
    data_group.add_argument('--seed_cov', type=int, default=42,
                            help='Random seed for covariance samples')
    data_group.add_argument('--seed_ders', type=int, nargs=2, default=[43, 44],
                            help='Random seeds for derivative samples')

    # Output configuration
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument('--save_dir', type=str, default='ai-code/models',
                              help='Directory to save trained models')
    output_group.add_argument('--save_name', type=str, default=None,
                              help='Model filename (default: auto-generated with timestamp)')
    output_group.add_argument('--no_save', action='store_true',
                              help='Do not save the trained model')

    # Device configuration
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use for training')

    return parser.parse_args()


def main():
    """Main training script."""
    args = parse_args()

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("=" * 80)
    print("Learnable Filtration Training for GRF Fisher Information")
    print("=" * 80)
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create components
    print("\n" + "=" * 80)
    print("1. Creating Pipeline Components")
    print("=" * 80)

    simulator = GRFSimulator(N=args.input_size, dim=2, device=device)
    print(f"Simulator: N={simulator.N}, dim={simulator.dim}")

    filtration = LearnableFiltration(
        input_size=args.input_size,
        homology_dimensions=[0, 1],
        hidden_channels=args.hidden_channels,
        kernel_size=args.kernel_size,
        activation=args.activation,
        upscale_factor=args.upscale_factor
    )
    print(f"Filtration: {filtration.get_num_parameters():,} parameters")
    print(f"  Input: {args.input_size}×{args.input_size}")
    print(f"  Output: {filtration.get_output_size()}×{filtration.get_output_size()}")
    print(f"  Upscale factor: {args.upscale_factor}× ({'no upscaling' if args.upscale_factor == 1 else 'upsampling'})")
    print(f"  Hidden channels: {args.hidden_channels}")
    print(f"  Activation: {args.activation}")

    # Create vectorization layer based on argument
    if args.vectorization == 'topk':
        layer_h0 = TopKLayer(k=args.topk)
        layer_h1 = TopKLayer(k=args.topk)
        vec_description = f"TopK (persistence-based) k={args.topk}"
    elif args.vectorization == 'birthdeath':
        layer_h0 = TopKBirthsDeathsLayer(k=args.topk)
        layer_h1 = TopKBirthsDeathsLayer(k=args.topk)
        vec_description = f"TopKBirthsDeaths (independent) k={args.topk}"
    else:
        raise ValueError(f"Unknown vectorization: {args.vectorization}")

    vectorization = CombinedVectorization([layer_h0, layer_h1])
    print(f"Vectorization: {vec_description} for H0 and H1")

    pipeline = LearnableFiltrationPipeline(
        simulator=simulator,
        filtration=filtration,
        vectorization=vectorization,
        use_moped=True
    )
    print("Pipeline: LearnableFiltrationPipeline with MOPED compression")

    # Training configuration
    print("\n" + "=" * 80)
    print("2. Training Configuration")
    print("=" * 80)

    config = FisherConfig(
        theta_fid=torch.tensor(args.theta_fid),
        delta_theta=torch.tensor(args.delta_theta),
        n_s=args.n_s,
        n_d=args.n_d,
        find_derivative=[True, True],
        seed_cov=args.seed_cov,
        seed_ders=args.seed_ders
    )

    print(f"Fiducial parameters: θ = {args.theta_fid}")
    print(f"Step sizes: Δθ = {args.delta_theta}")
    print(f"Samples: n_s={args.n_s}, n_d={args.n_d}")
    print(f"Seeds: cov={args.seed_cov}, ders={args.seed_ders}")
    print(f"\nTraining: {args.n_epochs} epochs, lr={args.lr} (weight_decay = lr/100)")
    print(f"Validation: every {args.validate_every} epochs")

    # Generate save path
    if not args.no_save:
        if args.save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"learnable_filtration_N{args.input_size}_{timestamp}.pt"
        else:
            save_name = args.save_name
        save_path = f"{args.save_dir}/{save_name}"
        print(f"Model will be saved to: {save_path}")
    else:
        save_path = None
        print("Model will NOT be saved")

    # Train
    print("\n" + "=" * 80)
    print("3. Starting Training")
    print("=" * 80)

    history = pipeline.train(
        config=config,
        n_epochs=args.n_epochs,
        lr=args.lr,
        validate_every=args.validate_every,
        verbose=True,
        save_path=save_path
    )

    # Final results
    print("\n" + "=" * 80)
    print("4. Final Results")
    print("=" * 80)

    final_result = history['final_result']

    # Compute theoretical Fisher for comparison
    if hasattr(simulator, 'theoretical_fisher_matrix'):
        F_theory = simulator.theoretical_fisher_matrix(config.theta_fid)
        log_det_theory = torch.logdet(F_theory)
        constraints_theory = torch.sqrt(torch.diag(torch.linalg.inv(F_theory)))
        efficiency = torch.exp(final_result.log_det_fisher - log_det_theory).item()

        print(f"\n{'Method':<30} {'log|F|':>10} {'σ(A)':>10} {'σ(B)':>10} {'F[0,0]':>10} {'F[0,1]':>10} {'F[1,1]':>10}")
        print("-" * 93)
        print(f"{'Theoretical':<30} {log_det_theory.item():>10.2f} {constraints_theory[0].item():>10.4f} {constraints_theory[1].item():>10.4f} {F_theory[0,0].item():>10.2f} {F_theory[0,1].item():>10.2f} {F_theory[1,1].item():>10.2f}")
        print(f"{'Learnable Filtration':<30} {final_result.log_det_fisher.item():>10.2f} {final_result.constraints[0].item():>10.4f} {final_result.constraints[1].item():>10.4f} {final_result.fisher_matrix[0,0].item():>10.2f} {final_result.fisher_matrix[0,1].item():>10.2f} {final_result.fisher_matrix[1,1].item():>10.2f}")
        print(f"\nEfficiency: {efficiency*100:.1f}%")
    else:
        print(f"Final log|F|: {final_result.log_det_fisher.item():.4f}")
        print(f"Constraints (1σ): {final_result.constraints}")

    print("\n" + "=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)

    if not args.no_save:
        print(f"\nModel saved to: {save_path}")
        print("\nTo load the model:")
        print(f"  checkpoint = torch.load('{save_path}')")
        print(f"  filtration.load_state_dict(checkpoint['filtration_state_dict'])")


if __name__ == '__main__':
    main()
