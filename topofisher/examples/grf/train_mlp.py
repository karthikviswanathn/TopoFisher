"""
Train learnable compression for GRF Fisher information maximization.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
from topofisher import (
    GRFSimulator,
    CubicalLayer,
    TopKLayer,
    CombinedVectorization,
    FisherAnalyzer,
    FisherConfig,
    generate_and_save_diagrams,
    load_diagrams
)


class CompressionMLP(nn.Module):
    """MLP for compressing Top-K features to maximize Fisher information."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list = None, dropout: float = 0.2):
        """
        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (n_params)
            hidden_dims: List of hidden layer dimensions. None for linear, [] for same as None,
                        [h1] for 1 hidden layer
            dropout: Dropout probability (default 0.2)
        """
        super().__init__()

        layers = []
        if hidden_dims is None or len(hidden_dims) == 0:
            # Linear compression
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # Add hidden layers
            prev_dim = input_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(nn.LeakyReLU(negative_slope=0.01))
                layers.append(nn.Dropout(dropout))
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def compute_fisher_loss(summaries, delta_theta, fisher_analyzer):
    """
    Compute negative log determinant of Fisher matrix as loss.

    Args:
        summaries: List of summary tensors [fiducial, theta_minus_0, theta_plus_0, ...]
        delta_theta: Step sizes for derivatives
        fisher_analyzer: FisherAnalyzer instance

    Returns:
        Negative log det Fisher (to minimize)
    """
    result = fisher_analyzer(summaries, delta_theta)
    return -result.log_det_fisher


def generate_or_load_data(config, simulator, filtration, vectorization, cache_path):
    """
    Generate or load precomputed diagrams and vectorize them.

    Args:
        config: FisherConfig
        simulator: GRFSimulator
        filtration: CubicalLayer
        vectorization: TopKLayer or CombinedVectorization
        cache_path: Path to diagram cache

    Returns:
        List of Top-K tensors [fiducial, theta_minus_0, theta_plus_0, ...]
    """
    import os

    # Generate and save diagrams if cache doesn't exist
    if not os.path.exists(cache_path):
        print(f"Cache not found, generating diagrams...")
        generate_and_save_diagrams(config, simulator, filtration, cache_path)

    # Load diagrams from cache
    all_diagrams, metadata = load_diagrams(cache_path)

    # Vectorize
    all_summaries = []
    for diagrams in all_diagrams:
        summary = vectorization(diagrams)
        all_summaries.append(summary)

    return all_summaries


def split_data(summaries, train_frac=0.5, val_frac=0.25, seed=42):
    """
    Split summaries into train/val/test sets.

    Important: theta_minus and theta_plus pairs use the same random seed during generation,
    so they must use the same permutation to maintain pairing.

    Args:
        summaries: List of tensors [fiducial, theta_minus_0, theta_plus_0, theta_minus_1, theta_plus_1, ...]
        train_frac: Fraction for training
        val_frac: Fraction for validation
        seed: Random seed for shuffling

    Returns:
        (train_summaries, val_summaries, test_summaries)
    """
    torch.manual_seed(seed)

    def split_with_perm(tensor, perm, train_frac, val_frac):
        """Split a tensor using a given permutation."""
        n = tensor.shape[0]
        shuffled = tensor[perm]
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]

    # Generate permutations
    # Fiducial gets its own permutation
    perm_fid = torch.randperm(summaries[0].shape[0])

    # Each parameter pair (theta_minus, theta_plus) shares a permutation
    n_params = (len(summaries) - 1) // 2
    perms_deriv = [torch.randperm(summaries[1 + 2*i].shape[0]) for i in range(n_params)]

    # Apply permutations and split
    split_summaries_list = []

    # Fiducial
    split_summaries_list.append(split_with_perm(summaries[0], perm_fid, train_frac, val_frac))

    # Derivatives (pairs share permutation to maintain seed pairing)
    for i in range(n_params):
        perm = perms_deriv[i]
        split_summaries_list.append(split_with_perm(summaries[1 + 2*i], perm, train_frac, val_frac))     # theta_minus
        split_summaries_list.append(split_with_perm(summaries[2 + 2*i], perm, train_frac, val_frac))     # theta_plus

    # Reorganize into separate lists
    train_summaries = [s[0] for s in split_summaries_list]
    val_summaries = [s[1] for s in split_summaries_list]
    test_summaries = [s[2] for s in split_summaries_list]

    return train_summaries, val_summaries, test_summaries


def train_compression(model, train_summaries, val_summaries, delta_theta,
                      n_epochs=100, lr=1e-3, batch_size=100, weight_decay=1e-4, test_summaries=None, verbose=False):
    """
    Train compression model to maximize Fisher information.

    Args:
        model: CompressionMLP
        train_summaries: Training data
        val_summaries: Validation data
        delta_theta: Step sizes
        n_epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        weight_decay: L2 regularization weight decay (default 1e-4)
        test_summaries: Optional test data for debugging
        verbose: If True, print training progress (default False)

    Returns:
        (best_model_state, train_losses, val_losses, needs_more_epochs)
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    fisher_analyzer = FisherAnalyzer(clean_data=True)

    train_losses = []
    val_losses = []
    test_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    # Get minimum batch size from all summaries
    min_train_size = min(s.shape[0] for s in train_summaries)
    actual_batch_size = min(batch_size, min_train_size)

    for epoch in range(n_epochs):
        # Training
        model.train()

        # Sample random batch for this epoch
        idx = torch.randperm(min_train_size)[:actual_batch_size]
        # Apply compression to batch
        batch_summaries = [model(s[idx]) for s in train_summaries]
        # import ipdb; ipdb.set_trace()
        # Compute loss
        loss = compute_fisher_loss(batch_summaries, delta_theta, fisher_analyzer)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            # Validation
            model.eval()
            with torch.no_grad():
                val_summaries_compressed = [model(s) for s in val_summaries]
                val_loss = compute_fisher_loss(val_summaries_compressed, delta_theta, fisher_analyzer)
                val_losses.append(val_loss.item())

                # Optional test loss for debugging
                if test_summaries is not None:
                    test_summaries_compressed = [model(s) for s in test_summaries]
                    test_loss = compute_fisher_loss(test_summaries_compressed, delta_theta, fisher_analyzer)
                    test_losses.append(test_loss.item())

                # Save best model
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                    if verbose:
                        print(f"    → New best model at epoch {epoch+1}: {best_val_loss:.3f}")

            if verbose:
                if test_summaries is not None:
                    print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {loss.item():.3f}, Val Loss: {val_losses[-1]:.3f}, Test Loss: {test_losses[-1]:.3f}")
                else:
                    print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {loss.item():.3f}, Val Loss: {val_losses[-1]:.3f}")

    # Check if training might need more epochs (val loss still improving near the end)
    needs_more_epochs = False
    if len(val_losses) >= 3:
        # Check if best val loss was in the last 20% of training
        best_epoch_idx = val_losses.index(min(val_losses))
        final_20_percent = int(len(val_losses) * 0.8)
        if best_epoch_idx >= final_20_percent:
            needs_more_epochs = True

    return best_model_state, train_losses, val_losses, needs_more_epochs


def main():
    print("=" * 80)
    print("Training Learnable Compression for Fisher Information Maximization")
    print("=" * 80)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    N = 32
    simulator = GRFSimulator(N=N, dim=2, device=device)
    filtration = CubicalLayer(homology_dimensions=[0, 1], min_persistence=[0.0, 0.0])
    vectorization = CombinedVectorization([
        TopKLayer(k=50),
        TopKLayer(k=80),
    ])

    config = FisherConfig(
        theta_fid=torch.tensor([1.0, 2.0]),
        delta_theta=torch.tensor([0.1, 0.2]),
        n_s=20000,
        n_d=20000,
        find_derivative=[True, True],
        seed_cov=42,
        seed_ders=[43, 44]
    )

    # Generate or load data
    print("\n1. Generating/loading data...")
    summaries = generate_or_load_data(config, simulator, filtration, vectorization, cache_path='data/diagrams_basic.pkl')
    input_dim = summaries[0].shape[1]
    output_dim = len(config.theta_fid)
    print(f"   Input dim: {input_dim}, Output dim: {output_dim}")

    # Split data
    print("\n2. Splitting data...")
    train_summaries, val_summaries, test_summaries = split_data(summaries)
    print(f"   Train: {train_summaries[0].shape[0]}, Val: {val_summaries[0].shape[0]}, Test: {test_summaries[0].shape[0]}")

    # Move data to device
    print(f"\n3. Moving data to {device}...")
    train_summaries = [s.to(device) for s in train_summaries]
    val_summaries = [s.to(device) for s in val_summaries]
    test_summaries = [s.to(device) for s in test_summaries]
    config.delta_theta = config.delta_theta.to(device)

    # Train models
    architectures = {
        "Linear": {"hidden_dims": None, "lr": 1e-3, "weight_decay": 0.0, "dropout": 0.2, "epochs": 2000},
        "1-Hidden (2d) p=0.2": {"hidden_dims": [2 * input_dim], "lr": 3e-4, "weight_decay": 1e-5, "dropout": 0.2, "epochs": 1000},
        "1-Hidden (2d) p=0.35": {"hidden_dims": [2 * input_dim], "lr": 3e-4, "weight_decay": 1e-5, "dropout": 0.35, "epochs": 1000},
        "1-Hidden (2d) p=0.5": {"hidden_dims": [2 * input_dim], "lr": 3e-4, "weight_decay": 1e-5, "dropout": 0.5, "epochs": 1000},
        "1-Hidden (d) p=0.2": {"hidden_dims": [input_dim], "lr": 3e-4, "weight_decay": 1e-5, "dropout": 0.2, "epochs": 1000},
        "1-Hidden (d) p=0.35": {"hidden_dims": [input_dim], "lr": 3e-4, "weight_decay": 1e-5, "dropout": 0.35, "epochs": 1000},
        "1-Hidden (d) p=0.5": {"hidden_dims": [input_dim], "lr": 3e-4, "weight_decay": 1e-5, "dropout": 0.5, "epochs": 1000},
        "1-Hidden (d/2)": {"hidden_dims": [input_dim // 2], "lr": 3e-4, "weight_decay": 1e-5, "dropout": 0.2, "epochs": 1000},
        "1-Hidden (d/4)": {"hidden_dims": [input_dim // 4], "lr": 3e-4, "weight_decay": 1e-5, "dropout": 0.2, "epochs": 1000},
    }

    trained_models = {}

    for name, config_arch in architectures.items():
        print(f"\n4. Training {name} compression...")
        model = CompressionMLP(input_dim, output_dim, config_arch["hidden_dims"], dropout=config_arch["dropout"]).to(device)
        best_state, train_losses, val_losses, needs_more = train_compression(
            model, train_summaries, val_summaries, config.delta_theta,
            n_epochs=config_arch["epochs"], lr=config_arch["lr"], batch_size=5000,
            weight_decay=config_arch["weight_decay"]
        )
        model.load_state_dict(best_state)
        trained_models[name] = model

        flag = " ⚠️ (may need more epochs)" if needs_more else ""
        print(f"   Best val loss: {min(val_losses):.3f}{flag}")

    # Evaluate on test set
    print("\n" + "=" * 80)
    print("EVALUATION ON TEST SET")
    print("=" * 80)

    # Collect results in a table
    results = []

    # Theoretical Fisher
    F_theory = simulator.theoretical_fisher_matrix(config.theta_fid)
    inv_F_theory = torch.linalg.inv(F_theory)
    constraints_theory = torch.sqrt(torch.diag(inv_F_theory))
    sign, logdet = torch.linalg.slogdet(F_theory)
    log_det_theory = sign * logdet

    results.append({
        'Method': 'Theoretical',
        'F[0,0]': F_theory[0, 0].item(),
        'F[0,1]': F_theory[0, 1].item(),
        'F[1,1]': F_theory[1, 1].item(),
        'log det F': log_det_theory.item(),
        'σ(A)': constraints_theory[0].item(),
        'σ(B)': constraints_theory[1].item()
    })

    # Full Top-K Fisher
    fisher_full = FisherAnalyzer(clean_data=True)
    result_full = fisher_full(test_summaries, config.delta_theta)

    results.append({
        'Method': 'Full Top-K',
        'F[0,0]': result_full.fisher_matrix[0, 0].item(),
        'F[0,1]': result_full.fisher_matrix[0, 1].item(),
        'F[1,1]': result_full.fisher_matrix[1, 1].item(),
        'log det F': result_full.log_det_fisher.item(),
        'σ(A)': result_full.constraints[0].item(),
        'σ(B)': result_full.constraints[1].item()
    })

    # MOPED Fisher
    fisher_moped = FisherAnalyzer(clean_data=True, use_moped=True, moped_compress_frac=0.5)
    result_moped = fisher_moped(test_summaries, config.delta_theta)

    results.append({
        'Method': 'MOPED',
        'F[0,0]': result_moped.fisher_matrix_moped[0, 0].item(),
        'F[0,1]': result_moped.fisher_matrix_moped[0, 1].item(),
        'F[1,1]': result_moped.fisher_matrix_moped[1, 1].item(),
        'log det F': result_moped.log_det_fisher_moped.item(),
        'σ(A)': result_moped.constraints_moped[0].item(),
        'σ(B)': result_moped.constraints_moped[1].item()
    })

    # Learned compressions
    for name, model in trained_models.items():
        model.eval()
        with torch.no_grad():
            test_compressed = [model(s) for s in test_summaries]

        fisher_learned = FisherAnalyzer(clean_data=True)
        result_learned = fisher_learned(test_compressed, config.delta_theta)

        results.append({
            'Method': name,
            'F[0,0]': result_learned.fisher_matrix[0, 0].item(),
            'F[0,1]': result_learned.fisher_matrix[0, 1].item(),
            'F[1,1]': result_learned.fisher_matrix[1, 1].item(),
            'log det F': result_learned.log_det_fisher.item(),
            'σ(A)': result_learned.constraints[0].item(),
            'σ(B)': result_learned.constraints[1].item()
        })

    # Sort by log det F (highest to lowest)
    results.sort(key=lambda x: x['log det F'], reverse=True)

    # Print table
    print("\n")
    header = f"{'Method':<25} {'log det F':>12} {'σ(A)':>10} {'σ(B)':>10} {'F[0,0]':>10} {'F[1,1]':>10} {'F[0,1]':>10}"
    print(header)
    print("-" * len(header))

    for r in results:
        print(f"{r['Method']:<25} {r['log det F']:>12.2f} {r['σ(A)']:>10.2f} {r['σ(B)']:>10.2f} {r['F[0,0]']:>10.2f} {r['F[1,1]']:>10.2f} {r['F[0,1]']:>10.2f}")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)

    # Save results to JSON
    save_results_to_json(
        results=results,
        architectures=architectures,
        simulator=simulator,
        filtration=filtration,
        vectorization=vectorization,
        config=config,
        device=device,
        train_summaries=train_summaries,
        val_summaries=val_summaries,
        test_summaries=test_summaries,
        input_dim=input_dim,
        output_dim=output_dim
    )


def save_results_to_json(results, architectures, simulator, filtration, vectorization,
                          config, device, train_summaries, val_summaries, test_summaries,
                          input_dim, output_dim):
    """Save results and metadata to JSON file."""

    # Create output directory
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get git commit hash if available
    try:
        import subprocess
        git_commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except:
        git_commit = "unknown"

    # Build metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_commit,
        "device": str(device),
        "simulator": {
            "type": "GRFSimulator",
            "N": simulator.N,
            "dim": simulator.dim,
            "boxlength": simulator.boxlength
        },
        "filtration": {
            "type": "CubicalLayer",
            "homology_dimensions": filtration.dimensions,
            "min_persistence": filtration.min_persistence
        },
        "vectorization": {
            "type": "CombinedVectorization",
            "layers": [f"TopKLayer(k={layer.k})" for layer in vectorization.layers],
            "total_features": input_dim
        },
        "fisher_config": {
            "theta_fid": config.theta_fid.cpu().tolist(),
            "delta_theta": config.delta_theta.cpu().tolist(),
            "n_s": config.n_s,
            "n_d": config.n_d,
            "seed_cov": config.seed_cov,
            "seed_ders": config.seed_ders
        },
        "data_split": {
            "train": train_summaries[0].shape[0],
            "val": val_summaries[0].shape[0],
            "test": test_summaries[0].shape[0]
        }
    }

    # Build results array with additional metadata
    json_results = []

    for r in results:
        method_name = r['Method']
        result_entry = {
            "method": method_name,
            "log_det_F": round(r['log det F'], 2),
            "sigma_A": round(r['σ(A)'], 2),
            "sigma_B": round(r['σ(B)'], 2),
            "F_00": round(r['F[0,0]'], 2),
            "F_11": round(r['F[1,1]'], 2),
            "F_01": round(r['F[0,1]'], 2)
        }

        # Add filtration, vectorization, compression info
        if method_name == "Theoretical":
            result_entry.update({
                "filtration": "N/A",
                "vectorization": "N/A",
                "compression": "N/A"
            })
        elif method_name == "Full Top-K":
            result_entry.update({
                "filtration": "Cubical",
                "vectorization": f"Top-K ({'+'.join([str(layer.k) for layer in vectorization.layers])})",
                "compression": "None"
            })
        elif method_name == "MOPED":
            result_entry.update({
                "filtration": "Cubical",
                "vectorization": f"Top-K ({'+'.join([str(layer.k) for layer in vectorization.layers])})",
                "compression": "MOPED"
            })
        else:
            # Learned compression
            arch_config = architectures.get(method_name, {})
            hidden_dims = arch_config.get("hidden_dims")
            dropout = arch_config.get("dropout", 0.0)

            if hidden_dims is None:
                comp_desc = f"Linear MLP (dropout={dropout})"
            else:
                comp_desc = f"{len(hidden_dims)}-Hidden MLP (dropout={dropout})"

            result_entry.update({
                "filtration": "Cubical",
                "vectorization": f"Top-K ({'+'.join([str(layer.k) for layer in vectorization.layers])})",
                "compression": comp_desc,
                "architecture": {
                    "hidden_dims": hidden_dims,
                    "dropout": dropout,
                    "lr": arch_config.get("lr"),
                    "weight_decay": arch_config.get("weight_decay"),
                    "epochs": arch_config.get("epochs")
                }
            })

        json_results.append(result_entry)

    # Combine metadata and results
    output_data = {
        "metadata": metadata,
        "results": json_results
    }

    # Save to file
    output_path = output_dir / "latest.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
