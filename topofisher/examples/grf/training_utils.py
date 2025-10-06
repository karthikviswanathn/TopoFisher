"""
Shared utilities for GRF compression training (MLP and CNN).
"""
import torch
from topofisher import FisherAnalyzer, generate_and_save_diagrams, load_diagrams
import os


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


def generate_or_load_diagrams(config, simulator, filtration, cache_path):
    """
    Generate or load precomputed diagrams.

    Args:
        config: FisherConfig
        simulator: GRFSimulator
        filtration: CubicalLayer
        cache_path: Path to diagram cache

    Returns:
        (all_diagrams, metadata) where all_diagrams is a list of diagram sets
    """
    # Generate and save diagrams if cache doesn't exist
    if not os.path.exists(cache_path):
        print(f"Cache not found, generating diagrams...")
        generate_and_save_diagrams(config, simulator, filtration, cache_path)

    # Load diagrams from cache
    return load_diagrams(cache_path)


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
        split_summaries_list.append(split_with_perm(summaries[1 + 2*i], perm, train_frac, val_frac))
        split_summaries_list.append(split_with_perm(summaries[2 + 2*i], perm, train_frac, val_frac))

    # Reorganize into separate lists
    train_summaries = [s[0] for s in split_summaries_list]
    val_summaries = [s[1] for s in split_summaries_list]
    test_summaries = [s[2] for s in split_summaries_list]

    return train_summaries, val_summaries, test_summaries


def train_compression(model, train_summaries, val_summaries, delta_theta,
                      n_epochs=100, lr=1e-3, batch_size=100, weight_decay=1e-4,
                      test_summaries=None, verbose=False, gaussianity_check_fn=None):
    """
    Train compression model to maximize Fisher information.

    Args:
        model: Compression model (MLP or CNN)
        train_summaries: Training data
        val_summaries: Validation data
        delta_theta: Step sizes
        n_epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        weight_decay: L2 regularization weight decay (default 1e-4)
        test_summaries: Optional test data for debugging
        verbose: If True, print training progress (default False)
        gaussianity_check_fn: Optional function(tensor) -> (dict, bool) to check Gaussianity during training

    Returns:
        Dictionary with results: {
            'best_model_state': state dict of best model,
            'train_losses': list of training losses,
            'val_losses': list of validation losses,
            'test_losses': list of test losses (if test_summaries provided),
            'needs_more_epochs': bool indicating if training might benefit from more epochs
        }
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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

                # Check Gaussianity constraint across all sets if function provided
                passes_gaussianity = True
                if gaussianity_check_fn is not None:
                    # Check Gaussianity for all sets (fiducial and derivatives)
                    for val_set in val_summaries_compressed:
                        _, is_gaussian = gaussianity_check_fn(val_set, verbose=False)
                        if not is_gaussian:
                            passes_gaussianity = False
                            break

                # Save best model (with Gaussianity constraint if enabled)
                if val_loss.item() < best_val_loss and passes_gaussianity:
                    best_val_loss = val_loss.item()
                    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                    if verbose:
                        print(f"    → New best model at epoch {epoch+1}: {best_val_loss:.3f}")

            if verbose:
                # Format Gaussianity status for display
                gaussianity_status = ""
                if gaussianity_check_fn is not None:
                    gaussianity_status = f", Gaussian: {'✓' if passes_gaussianity else '✗'}"

                if test_summaries is not None:
                    print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {loss.item():.3f}, Val Loss: {val_losses[-1]:.3f}, Test Loss: {test_losses[-1]:.3f}{gaussianity_status}")
                else:
                    print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {loss.item():.3f}, Val Loss: {val_losses[-1]:.3f}{gaussianity_status}")

    # Check if training might need more epochs (val loss still improving near the end)
    needs_more_epochs = False
    if len(val_losses) >= 3:
        # Check if best val loss was in the last 20% of training
        best_epoch_idx = val_losses.index(min(val_losses))
        final_20_percent = int(len(val_losses) * 0.8)
        if best_epoch_idx >= final_20_percent:
            needs_more_epochs = True

    return {
        'best_model_state': best_model_state,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_losses': test_losses,
        'needs_more_epochs': needs_more_epochs
    }
