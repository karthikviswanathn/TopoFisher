"""
Train learnable CNN compression for GRF Fisher information maximization using persistence images.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from scipy import stats
from topofisher import (
    GRFSimulator,
    CubicalLayer,
    PersistenceImageLayer,
    FisherAnalyzer,
    FisherConfig
)
from training_utils import (
    generate_or_load_diagrams,
    split_data,
    train_compression
)
from persistence_image_cache import (
    save_persistence_images,
    load_persistence_images,
    get_cache_path,
    cache_exists,
    validate_cache_metadata
)


class CompressionCNN(nn.Module):
    """
    CNN for compressing persistence images to maximize Fisher information.

    Based on Yip et al. 2025 (https://doi.org/10.1088/2632-2153/ade114)
    with parallel CNN + Dense architecture.
    """

    def __init__(
        self,
        n_channels: int,
        n_pixels: int,
        output_dim: int,
        dropout: float = 0.2
    ):
        """
        Args:
            n_channels: Number of input channels (homology dimensions, e.g., 2 for H0+H1)
            n_pixels: Resolution of persistence images (n_pixels x n_pixels)
            output_dim: Output dimension (n_params)
            dropout: Dropout probability (default 0.2)
        """
        super().__init__()

        self.n_channels = n_channels
        self.n_pixels = n_pixels
        self.output_dim = output_dim

        # CNN side (operates on 2D images)
        # Architecture: expand channels first, then conv+pool blocks
        # Input: 2 channels -> expand to 32 channels
        # Then: conv (3x3, padding=1) -> maxpool (2x2) -> ... -> flatten -> dense
        # With padding=1, conv preserves spatial size, maxpool halves it
        # 16x16 -> expand -> 32ch -> conv -> 32ch -> pool -> 8x8 -> conv -> 64ch -> pool -> 4x4 -> conv -> 128ch -> pool -> 2x2
        # 32x32 -> 32ch -> 16ch -> 16 -> 32ch -> 8 -> 64ch -> 4 -> 128ch -> 2
        # 64x64 -> 32ch -> 32 -> 64ch -> 16 -> 128ch -> 8 -> 256ch -> 4 -> 256ch -> 2

        cnn_layers = []

        # Initial expansion: 2 channels -> 32 channels
        cnn_layers.extend([
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.LeakyReLU()
        ])

        # Number of blocks: keep going until we reach 2x2
        n_blocks = int(np.log2(n_pixels)) - 1  # 16->3, 32->4, 64->5

        in_ch = 32
        for i in range(n_blocks):
            out_ch = 32 * (2 ** min(i, 3))  # 32, 64, 128, 256, 256, ...
            cnn_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),  # padding=1 preserves size
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)  # halves size
            ])
            in_ch = out_ch

        self.cnn = nn.Sequential(*cnn_layers)

        # Calculate CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, n_pixels, n_pixels)
            cnn_out = self.cnn(dummy)
            cnn_flat_size = cnn_out.view(1, -1).shape[1]

        # CNN dense layers
        self.cnn_dense = nn.Sequential(
            nn.Linear(cnn_flat_size, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

        # Dense side (operates on summed 1D features)
        # Sum along birth and persistence axes: n_channels x 2 x n_pixels -> n_channels * 2 * n_pixels
        dense_input_size = n_channels * 2 * n_pixels
        self.dense_stack = nn.Sequential(
            nn.Linear(dense_input_size, 512),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        """
        Args:
            x: Persistence images of shape (batch, n_channels, n_pixels, n_pixels)

        Returns:
            Compressed features of shape (batch, output_dim)
        """
        # CNN side
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)
        cnn_out = self.cnn_dense(cnn_features)

        # Dense side - sum along spatial dimensions
        # Sum over birth axis (dim=2) and persistence axis (dim=3)
        sum_birth = x.sum(dim=2)  # (batch, n_channels, n_pixels)
        sum_pers = x.sum(dim=3)   # (batch, n_channels, n_pixels)
        dense_features = torch.cat([sum_birth, sum_pers], dim=2)  # (batch, n_channels, 2*n_pixels)
        dense_features = dense_features.view(x.size(0), -1)  # (batch, n_channels * 2 * n_pixels)
        dense_out = self.dense_stack(dense_features)

        # Average outputs from both sides
        return (cnn_out + dense_out) / 2

    def print_layer_shapes(self, input_shape):
        """
        Print how tensor shapes change through the network.

        Args:
            input_shape: Tuple (n_channels, height, width)
        """
        print("\nCNN Layer Shape Transformations:")
        print("=" * 60)

        # Get device from model parameters
        device = next(self.parameters()).device
        x = torch.zeros(1, *input_shape, device=device)
        print(f"Input: {tuple(x.shape)}")

        # Track through CNN layers
        for i, layer in enumerate(self.cnn):
            x = layer(x)
            layer_name = layer.__class__.__name__
            print(f"  Layer {i} ({layer_name}): {tuple(x.shape)}")

        # Flatten
        x_flat = x.view(x.size(0), -1)
        print(f"Flatten: {tuple(x_flat.shape)}")

        # CNN dense layers
        for i, layer in enumerate(self.cnn_dense):
            x_flat = layer(x_flat)
            layer_name = layer.__class__.__name__
            print(f"  CNN Dense {i} ({layer_name}): {tuple(x_flat.shape)}")

        print(f"Final output: {tuple(x_flat.shape)}")
        print("=" * 60)


def test_gaussianity(compressed_summaries, alpha=0.05, verbose=True):
    """
    Test Gaussianity of compressed summaries using Kolmogorov-Smirnov test.

    Note: The compressed summary statistics are learned features that do not
    directly correspond to the original parameters. We test each learned
    dimension for Gaussianity.

    Args:
        compressed_summaries: Tensor of shape (n_samples, n_features)
        alpha: Significance level for hypothesis test (default 0.05)
        verbose: If True, print detailed results (default True)

    Returns:
        Tuple of (results_dict, all_gaussian_flag) where:
            - results_dict: Dictionary with test results per learned dimension
            - all_gaussian_flag: True if all learned dimensions pass Gaussianity test
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Gaussianity Test (Kolmogorov-Smirnov)")
        print("Note: Testing learned compressed features, not original parameters")
        print("=" * 60)

    n_features = compressed_summaries.shape[1]
    compressed_np = compressed_summaries.cpu().numpy()

    results = {}
    all_gaussian = True

    for i in range(n_features):
        feature_data = compressed_np[:, i]

        # Standardize to mean=0, std=1
        mean = feature_data.mean()
        std = feature_data.std()
        standardized = (feature_data - mean) / std

        # KS test against standard normal
        ks_stat, ks_pvalue = stats.kstest(standardized, 'norm')

        is_gaussian = ks_pvalue > alpha

        label = f"Feature {i+1}"
        results[label] = {
            'mean': mean,
            'std': std,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'is_gaussian': is_gaussian
        }

        if not is_gaussian:
            all_gaussian = False

        if verbose:
            print(f"\n{label}:")
            print(f"  Mean: {mean:.4f}")
            print(f"  Std:  {std:.4f}")
            print(f"  KS statistic: {ks_stat:.4f}")
            print(f"  KS p-value:   {ks_pvalue:.4f} {'✓ Gaussian' if is_gaussian else '✗ Non-Gaussian'}")

    if verbose:
        print("=" * 60)
        print(f"Note: p-value > {alpha} suggests data is consistent with Gaussian")
        print(f"Overall: {'✓ All learned features Gaussian' if all_gaussian else '✗ Not all learned features Gaussian'}")
        print("=" * 60)

    return results, all_gaussian


def main():
    # Configuration
    N = 32
    dim = 2
    boxlength = 256.0

    # Fisher configuration
    theta_fid = torch.tensor([1.0, 2.0])
    delta_theta = torch.tensor([0.1, 0.2])
    n_s = 20000
    n_d = 20000

    config = FisherConfig(
        theta_fid=theta_fid,
        delta_theta=delta_theta,
        n_s=n_s,
        n_d=n_d,
        find_derivative=[True, True],
        seed_cov=42,
        seed_ders=[43, 44]
    )

    # Setup
    simulator = GRFSimulator(N=N, dim=dim, boxlength=boxlength)
    filtration = CubicalLayer(homology_dimensions=[0, 1], min_persistence=[0.0, 0.0])
    # Persistence image hyperparameters
    n_pixels = 16  # Start with 16x16, can increase to 32 or 64
    bandwidth = 0.25  # Start with 1.0, tune as hyperparameter
    weighting = "persistence"  # "persistence" or "uniform"

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load diagrams
    cache_path = "data/diagrams_basic.pkl"
    print("Loading diagrams...")
    all_diagrams, metadata = generate_or_load_diagrams(config, simulator, filtration, cache_path)

    # Validate and transpose loaded diagrams structure
    # load_diagrams returns: [set][homology_dim][simulation]
    # We need: [set][simulation][homology_dim]
    assert isinstance(all_diagrams, list), f"all_diagrams should be list, got {type(all_diagrams)}"
    assert len(all_diagrams) == 5, f"Expected 5 diagram sets (1 fiducial + 2*2 derivatives), got {len(all_diagrams)}"

    print(f"Loaded {len(all_diagrams)} diagram sets, transposing structure...")
    transposed_diagrams = []
    for i, diagram_set in enumerate(all_diagrams):
        # diagram_set is [homology_dim_0_list, homology_dim_1_list, ...]
        # where each list contains n_simulations diagrams
        n_hdims = len(diagram_set)
        n_sims = len(diagram_set[0]) if n_hdims > 0 else 0

        # Transpose to [simulation][homology_dim]
        transposed_set = []
        for sim_idx in range(n_sims):
            sim_diagrams = [diagram_set[h_dim][sim_idx] for h_dim in range(n_hdims)]
            transposed_set.append(sim_diagrams)

        transposed_diagrams.append(transposed_set)
        print(f"  Set {i}: {n_sims} simulations, {n_hdims} homology dimensions")

    all_diagrams = transposed_diagrams

    # Create and fit persistence image vectorization
    print(f"\nFitting PersistenceImageLayer...")
    print(f"  n_pixels: {n_pixels}x{n_pixels}")
    print(f"  bandwidth: {bandwidth}")
    print(f"  weighting: {weighting}")

    vectorization = PersistenceImageLayer(
        n_pixels=n_pixels,
        bandwidth=bandwidth,
        homology_dimensions=[0, 1],  # H0 and H1 only
        weighting=weighting
    )

    # Check if persistence images are cached
    cache_base_path = "data/persistence_images"
    # Extract diagram cache name from path (e.g., "diagrams_basic" from "data/diagrams_basic.pkl")
    diagram_cache_name = Path(cache_path).stem
    pi_cache_path = get_cache_path(
        cache_base_path,
        diagram_cache_name=diagram_cache_name,
        n_pixels=n_pixels,
        bandwidth=bandwidth,
        weighting=weighting,
        homology_dimensions=[0, 1]
    )

    if cache_exists(pi_cache_path):
        print(f"\nFound cached persistence images at {pi_cache_path}")
        all_summaries, cached_metadata = load_persistence_images(pi_cache_path)

        # Validate cache matches current parameters
        expected_params = {
            'n_pixels': n_pixels,
            'bandwidth': bandwidth,
            'weighting': weighting,
            'homology_dimensions': [0, 1]
        }

        if validate_cache_metadata(cached_metadata, expected_params):
            print("✓ Cache validated successfully!")
            # Restore bounds to vectorization layer for display
            if 'bounds' in cached_metadata:
                for h_dim in [0, 1]:
                    print(f"  H{h_dim}: birth [{cached_metadata['bounds'][h_dim][0]:.3f}, {cached_metadata['bounds'][h_dim][1]:.3f}], "
                          f"pers [{cached_metadata['bounds'][h_dim][2]:.3f}, {cached_metadata['bounds'][h_dim][3]:.3f}]")
        else:
            print("✗ Cache validation failed, regenerating persistence images...")
            all_summaries = None
    else:
        print(f"\nNo cache found at {pi_cache_path}")
        all_summaries = None

    # Generate persistence images if not cached or cache invalid
    if all_summaries is None:
        # Fit on ALL diagrams to get fixed bounds
        vectorization.fit(all_diagrams)

        # Vectorize all diagrams to persistence images
        print("\nVectorizing diagrams to persistence images...")
        all_summaries = []
        for i, diagram_set in enumerate(all_diagrams):
            # diagram_set is a list of simulations for one type (fiducial, theta_minus_0, etc.)
            # Each element is [dgm_H0, dgm_H1, dgm_H2, ...] for one simulation
            images = vectorization(diagram_set)  # Shape: (n_sims, n_channels, n_pixels, n_pixels)
            assert isinstance(images, torch.Tensor), f"Vectorization output should be tensor, got {type(images)}"
            assert images.ndim == 4, f"Expected 4D tensor (batch, channels, height, width), got {images.ndim}D"
            assert images.shape[1] == len(vectorization.homology_dimensions), \
                f"Expected {len(vectorization.homology_dimensions)} channels, got {images.shape[1]}"
            assert images.shape[2] == n_pixels and images.shape[3] == n_pixels, \
                f"Expected {n_pixels}x{n_pixels} images, got {images.shape[2]}x{images.shape[3]}"
            all_summaries.append(images)
            print(f"  Set {i}: {images.shape}")

        # Save to cache
        print(f"\n💾 Saving persistence images to cache...")
        metadata = {
            'n_pixels': n_pixels,
            'bandwidth': bandwidth,
            'weighting': weighting,
            'homology_dimensions': [0, 1],
            'bounds': {h_dim: vectorization.pi_layers[h_dim].im_range
                      for h_dim in vectorization.homology_dimensions}
        }
        save_persistence_images(all_summaries, metadata, pi_cache_path)

    # Split data
    train_summaries, val_summaries, test_summaries = split_data(all_summaries)

    # Validate split data
    assert len(train_summaries) == 5, f"Expected 5 train sets, got {len(train_summaries)}"
    assert len(val_summaries) == 5, f"Expected 5 val sets, got {len(val_summaries)}"
    assert len(test_summaries) == 5, f"Expected 5 test sets, got {len(test_summaries)}"
    print(f"\nData split:")
    for i in range(5):
        assert train_summaries[i].ndim == 4, f"Train set {i} should be 4D, got {train_summaries[i].ndim}D"
        assert val_summaries[i].ndim == 4, f"Val set {i} should be 4D, got {val_summaries[i].ndim}D"
        assert test_summaries[i].ndim == 4, f"Test set {i} should be 4D, got {test_summaries[i].ndim}D"
    print(f"  Train: {train_summaries[0].shape[0]} samples per set")
    print(f"  Val: {val_summaries[0].shape[0]} samples per set")
    print(f"  Test: {test_summaries[0].shape[0]} samples per set")

    # Move to device
    train_summaries = [s.to(device) for s in train_summaries]
    val_summaries = [s.to(device) for s in val_summaries]
    test_summaries = [s.to(device) for s in test_summaries]
    config.delta_theta = config.delta_theta.to(device)

    # Get dimensions
    n_channels = len(vectorization.homology_dimensions)
    output_dim = len(theta_fid)

    print(f"\nPersistence Image Configuration:")
    print(f"  Resolution: {n_pixels}x{n_pixels}")
    print(f"  Channels: {n_channels} (H0, H1)")
    print(f"  Bandwidth: {bandwidth}")
    print(f"  Weighting: {weighting}")
    print(f"  Input shape: ({n_channels}, {n_pixels}, {n_pixels})")
    print(f"  Output dim: {output_dim}")

    # Train CNN
    print(f"\nTraining CNN compression...")
    model = CompressionCNN(
        n_channels=n_channels,
        n_pixels=n_pixels,
        output_dim=output_dim,
        dropout=0.2
    ).to(device)

    # Print layer shape transformations
    model.print_layer_shapes(input_shape=(n_channels, n_pixels, n_pixels))

    result = train_compression(
        model=model,
        train_summaries=train_summaries,
        val_summaries=val_summaries,
        delta_theta=config.delta_theta,
        n_epochs=2500,
        lr=2e-4,
        weight_decay=2e-5,
        test_summaries=test_summaries,
        verbose=True,
        gaussianity_check_fn=test_gaussianity
    )

    # Load best model and evaluate
    model.load_state_dict(result['best_model_state'])
    model.eval()

    with torch.no_grad():
        # Evaluate on test set
        compressed_test = [model(s) for s in test_summaries]
        fisher_analyzer = FisherAnalyzer()
        final_result = fisher_analyzer(compressed_test, config.delta_theta)

    F = final_result.fisher_matrix.cpu().numpy()
    print(f"\n{'='*80}")
    print(f"CNN Compression Results (PI {n_pixels}x{n_pixels}, BW={bandwidth}, {weighting})")
    print(f"{'='*80}")
    print(f"log det F: {final_result.log_det_fisher.item():.2f}")
    print(f"σ(A): {1/np.sqrt(F[0,0]):.2f}")
    print(f"σ(B): {1/np.sqrt(F[1,1]):.2f}")
    print(f"F[0,0]: {F[0,0]:.2f}")
    print(f"F[1,1]: {F[1,1]:.2f}")
    print(f"F[0,1]: {F[0,1]:.2f}")
    print(f"{'='*80}")

    # Test Gaussianity on test set (fiducial and all derivatives)
    print(f"\n{'='*80}")
    print("Gaussianity Check on Test Set")
    print(f"{'='*80}")

    set_names = ['Fiducial', 'A-', 'A+', 'B-', 'B+']
    all_pass = True

    for i, (name, compressed) in enumerate(zip(set_names, compressed_test)):
        _, is_gaussian = test_gaussianity(compressed, verbose=False)
        if not is_gaussian:
            all_pass = False
        print(f"{name}: {'✓ Gaussian' if is_gaussian else '✗ Non-Gaussian'}")

    print(f"\n{'='*80}")
    print(f"Overall Gaussianity: {'✓ ALL PASSED' if all_pass else '✗ SOME FAILED'}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
