"""
Utilities for caching GRF data.
"""
import torch
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm


def save_grf_data(
    grfs: List[torch.Tensor],
    metadata: Dict[str, Any],
    filepath: str
):
    """
    Save GRF data and metadata to disk.

    Args:
        grfs: List of GRF tensors [fiducial, theta_minus_0, theta_plus_0, ...]
        metadata: Dictionary containing simulator and config parameters
        filepath: Path to save the cache file
    """
    # Convert tensors to numpy for efficient storage
    grfs_np = [grf.cpu().numpy() for grf in grfs]

    data = {
        'grfs': grfs_np,
        'metadata': metadata
    }

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Use pickle with compression
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Calculate file size
    file_size_mb = filepath.stat().st_size / (1024**2)
    print(f"Saved GRF data to {filepath} ({file_size_mb:.2f} MB)")


def load_grf_data(filepath: str) -> tuple:
    """
    Load GRF data and metadata from disk.

    Args:
        filepath: Path to the cache file

    Returns:
        (grfs, metadata) where grfs is a list of torch tensors
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    # Convert numpy arrays back to torch tensors
    grfs = [torch.from_numpy(grf).float() for grf in data['grfs']]

    file_size_mb = Path(filepath).stat().st_size / (1024**2)
    print(f"Loaded GRF data from {filepath} ({file_size_mb:.2f} MB)")
    return grfs, data['metadata']


def get_cache_path(
    base_path: str,
    N: int,
    theta_fid: List[float],
    delta_theta: List[float],
    n_s: int,
    n_d: int,
    seed_cov: int,
    seed_ders: List[int]
) -> str:
    """
    Generate cache file path based on GRF generation parameters.

    Args:
        base_path: Base directory for cache files
        N: Grid size
        theta_fid: Fiducial parameter values
        delta_theta: Step sizes for derivatives
        n_s: Number of fiducial samples
        n_d: Number of derivative samples
        seed_cov: Random seed for covariance
        seed_ders: Random seeds for derivatives

    Returns:
        Path to cache file
    """
    theta_str = "_".join([f"{t:.1f}" for t in theta_fid])
    delta_str = "_".join([f"{d:.2f}" for d in delta_theta])
    seed_der_str = "_".join(map(str, seed_ders))

    filename = f"grf_N{N}_theta{theta_str}_delta{delta_str}_ns{n_s}_nd{n_d}_seedcov{seed_cov}_seedder{seed_der_str}.pkl"
    return str(Path(base_path) / filename)


def cache_exists(cache_path: str) -> bool:
    """Check if cache file exists."""
    return Path(cache_path).exists()


def validate_cache_metadata(
    cached_metadata: Dict[str, Any],
    expected_params: Dict[str, Any]
) -> bool:
    """
    Validate that cached metadata matches expected parameters.

    Args:
        cached_metadata: Metadata from cache file
        expected_params: Expected parameters to match

    Returns:
        True if metadata matches, False otherwise
    """
    for key, expected_val in expected_params.items():
        if key not in cached_metadata:
            return False

        cached_val = cached_metadata[key]

        # Handle lists/arrays
        if isinstance(expected_val, (list, np.ndarray, torch.Tensor)):
            if isinstance(cached_val, (list, np.ndarray)):
                if not np.allclose(cached_val, expected_val):
                    return False
            else:
                return False
        # Handle scalars
        elif cached_val != expected_val:
            return False

    return True


def generate_and_cache_grfs(config, simulator, cache_path: str, show_progress: bool = True):
    """
    Generate GRF data for Fisher analysis with progress tracking and caching.

    Args:
        config: FisherConfig
        simulator: GRFSimulator
        cache_path: Path to save/load cache
        show_progress: Whether to show progress bars (default True)

    Returns:
        List of GRF tensors [fiducial, theta_minus_0, theta_plus_0, ...]
    """
    # Check if cache exists
    if cache_exists(cache_path):
        print(f"Cache found at {cache_path}")
        grfs, metadata = load_grf_data(cache_path)

        # Validate metadata
        expected_params = {
            'N': simulator.N,
            'theta_fid': config.theta_fid.tolist(),
            'delta_theta': config.delta_theta.tolist(),
            'n_s': config.n_s,
            'n_d': config.n_d,
            'seed_cov': config.seed_cov,
            'seed_ders': config.seed_ders
        }

        if validate_cache_metadata(metadata, expected_params):
            print("Cache validated successfully!")
            return grfs
        else:
            print("Cache parameters don't match, regenerating...")

    # Generate GRF data
    print(f"Generating GRF data...")
    print(f"  Fiducial samples: {config.n_s}")
    print(f"  Derivative samples per parameter: {config.n_d}")

    # Fiducial samples with progress bar
    if show_progress:
        print("Generating fiducial samples...")
    fields_fid = simulator.generate(config.theta_fid, config.n_s, seed=config.seed_cov)

    all_fields = [fields_fid]

    # Generate derivative samples for each parameter
    n_params = len(config.theta_fid)
    iterator = tqdm(range(n_params), desc="Parameters") if show_progress else range(n_params)

    for i in iterator:
        if not config.find_derivative[i]:
            continue

        # Theta minus (centered finite difference)
        theta_minus = config.theta_fid.clone()
        theta_minus[i] -= config.delta_theta[i] / 2.0
        fields_minus = simulator.generate(theta_minus, config.n_d, seed=config.seed_ders[i])
        all_fields.append(fields_minus)

        # Theta plus (centered finite difference)
        theta_plus = config.theta_fid.clone()
        theta_plus[i] += config.delta_theta[i] / 2.0
        fields_plus = simulator.generate(theta_plus, config.n_d, seed=config.seed_ders[i])
        all_fields.append(fields_plus)

    # Save to cache
    metadata = {
        'N': simulator.N,
        'dim': simulator.dim,
        'boxlength': simulator.boxlength,
        'theta_fid': config.theta_fid.tolist(),
        'delta_theta': config.delta_theta.tolist(),
        'n_s': config.n_s,
        'n_d': config.n_d,
        'seed_cov': config.seed_cov,
        'seed_ders': config.seed_ders,
        'find_derivative': config.find_derivative
    }

    save_grf_data(all_fields, metadata, cache_path)

    return all_fields
