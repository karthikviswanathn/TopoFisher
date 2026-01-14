"""
Component factory with registry pattern for creating pipeline components.

This module provides a clean factory pattern for creating simulators,
filtrations, vectorizations, and compressions from configuration.

MODIFIED FOR MMA BRANCH: Added MMA filtration and vectorization factories.
"""
from typing import Dict, Any, Callable, Optional
import torch

from .data_types import (
    SimulatorConfig, FiltrationConfig, VectorizationConfig,
    CompressionConfig, AnalysisConfig
)


# Component registries
SIMULATORS: Dict[str, Callable] = {}
FILTRATIONS: Dict[str, Callable] = {}
VECTORIZATIONS: Dict[str, Callable] = {}
COMPRESSIONS: Dict[str, Callable] = {}


def register_simulator(name: str):
    """Decorator to register a simulator factory."""
    def decorator(func: Callable):
        SIMULATORS[name] = func
        return func
    return decorator


def register_filtration(name: str):
    """Decorator to register a filtration factory."""
    def decorator(func: Callable):
        FILTRATIONS[name] = func
        return func
    return decorator


def register_vectorization(name: str):
    """Decorator to register a vectorization factory."""
    def decorator(func: Callable):
        VECTORIZATIONS[name] = func
        return func
    return decorator


def register_compression(name: str):
    """Decorator to register a compression factory."""
    def decorator(func: Callable):
        COMPRESSIONS[name] = func
        return func
    return decorator


# ============================================================================
# Simulator Factories
# ============================================================================

@register_simulator('grf')
def create_grf_simulator(params: Dict[str, Any]):
    """Create Gaussian Random Field simulator."""
    from ..simulators import GRFSimulator

    # Auto-detect device if not specified
    if 'device' not in params:
        params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Auto-detected device: {params['device']}")

    return GRFSimulator(**params)


@register_simulator('gaussian_vector')
def create_gaussian_vector_simulator(params: Dict[str, Any]):
    """Create Gaussian Vector simulator."""
    from ..simulators import GaussianVectorSimulator

    # Auto-detect device if not specified
    if 'device' not in params:
        params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Auto-detected device: {params['device']}")

    return GaussianVectorSimulator(**params)


@register_simulator('noisy_ring')
def create_noisy_ring_simulator(params: Dict[str, Any]):
    """Create Noisy Ring (Circle) simulator for point clouds."""
    from ..simulators import NoisyRingSimulator

    # Auto-detect device if not specified
    if 'device' not in params:
        params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Auto-detected device: {params['device']}")

    return NoisyRingSimulator(**params)


# ============================================================================
# Filtration Factories
# ============================================================================

@register_filtration('cubical')
def create_cubical_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create Cubical filtration."""
    from ..filtrations import CubicalLayer
    if trainable:
        raise ValueError("Cubical filtration does not support training. Set trainable=false.")
    return CubicalLayer(**params)


@register_filtration('alpha')
def create_alpha_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create Alpha filtration."""
    from ..filtrations import AlphaComplexLayer
    if trainable:
        raise ValueError("Alpha filtration does not support training. Set trainable=false.")
    return AlphaComplexLayer(**params)


@register_filtration('identity')
def create_identity_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create Identity filtration (pass-through)."""
    from ..filtrations import IdentityFiltration
    if trainable:
        raise ValueError("Identity filtration does not support training. Set trainable=false.")
    return IdentityFiltration(**params)


@register_filtration('learnable')
def create_learnable_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create Learnable filtration."""
    from ..filtrations import LearnableFiltration
    if not trainable:
        print("Warning: 'learnable' filtration type but trainable=false. Setting trainable=true.")
    return LearnableFiltration(**params)


@register_filtration('learnable_point')
def create_learnable_point_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create Learnable Point (Flag Complex) filtration for point clouds."""
    from ..filtrations import LearnablePointFiltration
    if not trainable:
        print("Warning: 'learnable_point' filtration type but trainable=false. Setting trainable=true.")
    return LearnablePointFiltration(**params)


@register_filtration('alpha_dtm')
def create_alpha_dtm_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create Alpha DTM (Distance to Measure) filtration for point clouds.

    This is a non-learnable baseline using the fixed DTM formula:
    vfilt = sqrt(mean(knn_distances^2))
    """
    from ..filtrations import AlphaDTMFiltration
    if trainable:
        raise ValueError("Alpha DTM filtration does not support training. "
                        "Use 'learnable_point' for trainable filtration.")
    return AlphaDTMFiltration(**params)


# ============================================================================
# MMA Filtration Factory (NEW)
# ============================================================================

@register_filtration('mma')
def create_mma_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create MMA (Multiparameter Module Approximation) filtration.
    
    Uses Freudenthal triangulation and multipers for 2-parameter persistence.
    """
    from ..filtrations.mma_simplextree import MMALayer
    if trainable:
        raise ValueError("MMA filtration does not support training. Set trainable=false.")
    return MMALayer(**params)


# ============================================================================
# Vectorization Factories
# ============================================================================

@register_vectorization('topk')
def create_topk_vectorization(params: Dict[str, Any], trainable: bool = False):
    """Create TopK vectorization."""
    from ..vectorizations import TopKLayer
    if trainable:
        raise ValueError("TopK vectorization does not support training. Set trainable=false.")
    return TopKLayer(**params)


@register_vectorization('persistence_image')
def create_persistence_image_vectorization(params: Dict[str, Any], trainable: bool = False):
    """Create Persistence Image vectorization."""
    from ..vectorizations import PersistenceImageLayer
    if trainable:
        # Use learnable version
        from ..vectorizations import LearnablePersistenceImageLayer
        return LearnablePersistenceImageLayer(**params)
    return PersistenceImageLayer(**params)


@register_vectorization('combined')
def create_combined_vectorization(params: Dict[str, Any], trainable: bool = False):
    """Create Combined vectorization."""
    from ..vectorizations import CombinedVectorization

    if trainable:
        raise ValueError("Combined vectorization itself doesn't support training. "
                       "Set trainable=true on individual sub-vectorizations.")

    # Extract sub-vectorization configs (support both 'layers' and 'configs')
    if 'layers' in params:
        sub_configs = params['layers']
    elif 'configs' in params:
        sub_configs = params['configs']
    else:
        raise ValueError("Combined vectorization requires 'layers' (or 'configs') parameter")

    if not isinstance(sub_configs, list) or len(sub_configs) == 0:
        raise ValueError("Combined vectorization 'layers' must be a non-empty list")

    # Create sub-vectorizations
    vectorizations = []
    for config in sub_configs:
        vec_type = config['type']
        vec_params = config.get('params', {})
        vec_trainable = config.get('trainable', False)

        if vec_type not in VECTORIZATIONS:
            raise ValueError(f"Unknown vectorization type: {vec_type}")

        vec = VECTORIZATIONS[vec_type](vec_params, vec_trainable)
        vectorizations.append(vec)

    return CombinedVectorization(vectorizations)


@register_vectorization('identity')
def create_identity_vectorization(params: Dict[str, Any], trainable: bool = False):
    """Create Identity vectorization (pass-through)."""
    from ..vectorizations import IdentityVectorization
    if trainable:
        raise ValueError("Identity vectorization does not support training. Set trainable=false.")
    return IdentityVectorization(**params)


# ============================================================================
# MMA Vectorization Factories (NEW)
# ============================================================================

@register_vectorization('mma_topk')
def create_mma_topk_vectorization(params: Dict[str, Any], trainable: bool = False):
    """Create MMA TopK corner vectorization.
    
    Selects top-k corners ordered lexicographically from MMA modules.
    """
    from ..vectorizations.mma_topk import MMATopKLayer
    if trainable:
        raise ValueError("MMA TopK vectorization does not support training. Set trainable=false.")
    return MMATopKLayer(**params)


@register_vectorization('mma_gaussian')
def create_mma_gaussian_vectorization(params: Dict[str, Any], trainable: bool = False):
    """Create MMA Gaussian kernel vectorization.
    
    Applies Gaussian kernel on a regular grid to MMA intervals.
    """
    from ..vectorizations.mma_kernel import MMAGaussianLayer
    if trainable:
        raise ValueError("MMA Gaussian vectorization does not support training. Set trainable=false.")
    return MMAGaussianLayer(**params)


@register_vectorization('mma_exponential')
def create_mma_exponential_vectorization(params: Dict[str, Any], trainable: bool = False):
    """Create MMA Exponential kernel vectorization.
    
    Applies Exponential kernel on a regular grid to MMA intervals.
    """
    from ..vectorizations.mma_kernel import MMAExponentialLayer
    if trainable:
        raise ValueError("MMA Exponential vectorization does not support training. Set trainable=false.")
    return MMAExponentialLayer(**params)


@register_vectorization('mma_linear')
def create_mma_linear_vectorization(params: Dict[str, Any], trainable: bool = False):
    """Create MMA Linear kernel vectorization.
    
    Applies Linear kernel on a regular grid to MMA intervals.
    """
    from ..vectorizations.mma_kernel import MMALinearLayer
    if trainable:
        raise ValueError("MMA Linear vectorization does not support training. Set trainable=false.")
    return MMALinearLayer(**params)


# ============================================================================
# Compression Factories
# ============================================================================

@register_compression('identity')
def create_identity_compression(params: Dict[str, Any], trainable: bool = False):
    """Create Identity compression (pass-through)."""
    from ..compressions import IdentityCompression
    if trainable:
        raise ValueError("Identity compression does not support training. Set trainable=false.")
    return IdentityCompression(**params)


@register_compression('moped')
def create_moped_compression(params: Dict[str, Any], trainable: bool = False):
    """Create MOPED compression."""
    from ..compressions import MOPEDCompression
    if trainable:
        raise ValueError("MOPED compression does not require training. Set trainable=false.")

    # Ensure reg is a float if provided
    if 'reg' in params:
        params['reg'] = float(params['reg'])

    return MOPEDCompression(**params)


@register_compression('mlp')
def create_mlp_compression(params: Dict[str, Any], trainable: bool = False):
    """Create MLP compression."""
    from ..compressions import MLPCompression
    if not trainable:
        print("Warning: MLP compression typically requires training. Consider setting trainable=true.")
    return MLPCompression(**params)


@register_compression('cnn')
def create_cnn_compression(params: Dict[str, Any], trainable: bool = False):
    """Create CNN compression."""
    from ..compressions import CNNCompression
    if not trainable:
        print("Warning: CNN compression typically requires training. Consider setting trainable=true.")
    return CNNCompression(**params)


@register_compression('inception')
def create_inception_compression(params: Dict[str, Any], trainable: bool = False):
    """Create Inception block compression."""
    from ..compressions import InceptionCompression
    if not trainable:
        print("Warning: Inception compression typically requires training. Consider setting trainable=true.")
    return InceptionCompression(**params)


# ============================================================================
# Main Factory Functions
# ============================================================================

def create_simulator(config: SimulatorConfig):
    """
    Create a simulator from configuration.

    Args:
        config: Simulator configuration

    Returns:
        Simulator instance

    Raises:
        ValueError: If simulator type is unknown
    """
    if config.type not in SIMULATORS:
        available = ', '.join(SIMULATORS.keys())
        raise ValueError(f"Unknown simulator type: {config.type}. Available: {available}")

    return SIMULATORS[config.type](config.params)


def create_filtration(config: FiltrationConfig):
    """
    Create a filtration from configuration.

    Args:
        config: Filtration configuration

    Returns:
        Filtration instance

    Raises:
        ValueError: If filtration type is unknown
    """
    if config.type not in FILTRATIONS:
        available = ', '.join(FILTRATIONS.keys())
        raise ValueError(f"Unknown filtration type: {config.type}. Available: {available}")

    return FILTRATIONS[config.type](config.params, config.trainable)


def create_vectorization(config: VectorizationConfig):
    """
    Create a vectorization from configuration.

    Args:
        config: Vectorization configuration

    Returns:
        Vectorization instance

    Raises:
        ValueError: If vectorization type is unknown
    """
    if config.type not in VECTORIZATIONS:
        available = ', '.join(VECTORIZATIONS.keys())
        raise ValueError(f"Unknown vectorization type: {config.type}. Available: {available}")

    return VECTORIZATIONS[config.type](config.params, config.trainable)


def create_compression(config: CompressionConfig):
    """
    Create a compression from configuration.

    Args:
        config: Compression configuration

    Returns:
        Compression instance

    Raises:
        ValueError: If compression type is unknown
    """
    if config.type not in COMPRESSIONS:
        available = ', '.join(COMPRESSIONS.keys())
        raise ValueError(f"Unknown compression type: {config.type}. Available: {available}")

    return COMPRESSIONS[config.type](config.params, config.trainable)


def create_fisher_analyzer(clean_data: bool = True):
    """
    Create a Fisher analyzer.

    Args:
        clean_data: Whether to clean data before analysis

    Returns:
        FisherAnalyzer instance
    """
    from ..fisher import FisherAnalyzer
    return FisherAnalyzer(clean_data=clean_data)
