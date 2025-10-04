"""
Utilities for caching persistence images.
"""
import torch
import pickle
from pathlib import Path
from typing import List, Dict, Any


def save_persistence_images(
    images: List[torch.Tensor],
    metadata: Dict[str, Any],
    filepath: str
):
    """
    Save persistence images and metadata to disk.

    Args:
        images: List of image tensors [set_0, set_1, ...] where each is (n_sims, n_channels, height, width)
        metadata: Dictionary containing vectorization parameters and bounds
        filepath: Path to save the cache file
    """
    # Convert tensors to numpy for efficient storage
    images_np = [img.cpu().numpy() for img in images]

    data = {
        'images': images_np,
        'metadata': metadata
    }

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved persistence images to {filepath}")


def load_persistence_images(filepath: str) -> tuple:
    """
    Load persistence images and metadata from disk.

    Args:
        filepath: Path to the cache file

    Returns:
        (images, metadata) where images is a list of torch tensors
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    # Convert numpy arrays back to torch tensors
    images = [torch.from_numpy(img).float() for img in data['images']]

    print(f"Loaded persistence images from {filepath}")
    return images, data['metadata']


def get_cache_path(
    base_path: str,
    diagram_cache_name: str,
    n_pixels: int,
    bandwidth: float,
    weighting: str,
    homology_dimensions: List[int]
) -> str:
    """
    Generate cache file path based on diagram cache and vectorization parameters.

    Args:
        base_path: Base directory for cache files
        diagram_cache_name: Name of the source diagram cache file (without extension)
        n_pixels: Image resolution
        bandwidth: Gaussian kernel bandwidth
        weighting: Weighting scheme ("persistence" or "uniform")
        homology_dimensions: List of homology dimensions to include

    Returns:
        Path to cache file
    """
    h_dims_str = "_".join(map(str, homology_dimensions))
    filename = f"pi_{diagram_cache_name}_{n_pixels}x{n_pixels}_bw{bandwidth}_{weighting}_h{h_dims_str}.pkl"
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
        expected_params: Expected vectorization parameters

    Returns:
        True if metadata matches, False otherwise
    """
    required_keys = ['n_pixels', 'bandwidth', 'weighting', 'homology_dimensions']

    for key in required_keys:
        if key not in cached_metadata:
            print(f"Cache validation failed: missing key '{key}'")
            return False
        if cached_metadata[key] != expected_params[key]:
            print(f"Cache validation failed: {key} mismatch")
            print(f"  Cached: {cached_metadata[key]}")
            print(f"  Expected: {expected_params[key]}")
            return False

    # Validate bounds exist
    if 'bounds' not in cached_metadata:
        print("Cache validation failed: missing bounds")
        return False

    return True
