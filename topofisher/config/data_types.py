"""
Configuration data types for YAML pipeline loading.

This module defines dataclasses for all configuration sections
used in YAML pipeline definitions.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import torch


@dataclass
class CacheConfig:
    """Configuration for caching diagrams and summaries."""
    mode: str                                # "generate" or "load"
    data_type: str                           # "diagrams" or "summaries"
    save_path: Optional[str] = None          # Path to save (for generate mode)
    load_path: Optional[str] = None          # Path to load from (for load mode)


@dataclass
class ExperimentConfig:
    """Experiment metadata configuration."""
    name: str
    output_dir: str = "experiments"
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class AnalysisConfig:
    """
    Analysis parameters configuration.

    Accepts both lists (from YAML) and tensors (programmatic use).
    Lists are auto-converted to tensors in __post_init__.
    """
    theta_fid: torch.Tensor      # Fiducial parameter values
    delta_theta: torch.Tensor    # Step sizes for finite differences (±Δθ/2)
    n_s: int                     # Number of samples for covariance estimation
    n_d: int                     # Number of samples for derivative estimation
    seed_cov: int = 42           # Seed for fiducial samples
    seed_ders: List[int] = field(default_factory=list)  # Seeds for derivative samples
    cache: Optional[CacheConfig] = None  # Cache configuration for cached pipelines

    def __post_init__(self):
        """Convert lists to tensors and validate."""
        # Auto-convert lists to tensors (for YAML compatibility)
        if not isinstance(self.theta_fid, torch.Tensor):
            self.theta_fid = torch.tensor(self.theta_fid)
        if not isinstance(self.delta_theta, torch.Tensor):
            self.delta_theta = torch.tensor(self.delta_theta)

        # Auto-generate derivative seeds if not provided
        if not self.seed_ders:
            n_params = len(self.theta_fid)
            self.seed_ders = [self.seed_cov + i + 1 for i in range(n_params)]

        # Validate dimensions match
        if len(self.delta_theta) != len(self.theta_fid):
            raise ValueError(
                f"delta_theta length ({len(self.delta_theta)}) must match "
                f"theta_fid length ({len(self.theta_fid)})"
            )
        if len(self.seed_ders) != len(self.theta_fid):
            raise ValueError(
                f"seed_ders length ({len(self.seed_ders)}) must match "
                f"theta_fid length ({len(self.theta_fid)})"
            )


@dataclass
class SimulatorConfig:
    """Simulator configuration."""
    type: str  # e.g., 'grf', 'gaussian_vector'
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FiltrationConfig:
    """Filtration configuration."""
    type: str  # e.g., 'cubical', 'alpha', 'learnable', 'identity'
    trainable: bool = False  # Whether this component requires training
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorizationConfig:
    """Vectorization configuration."""
    type: str  # e.g., 'topk', 'persistence_image', 'combined', 'identity'
    trainable: bool = False  # Whether this component requires training
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionConfig:
    """Compression configuration."""
    type: str  # e.g., 'moped', 'mlp', 'cnn', 'identity'
    trainable: bool = False  # Whether this component requires training
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Training configuration for learnable components."""
    n_epochs: int = 1000
    lr: float = 1e-3
    batch_size: int = 500
    weight_decay: float = 1e-4
    train_frac: float = 0.5
    val_frac: float = 0.25
    validate_every: int = 10
    verbose: bool = True
    patience: Optional[int] = None  # Early stopping patience
    min_delta: float = 1e-6  # Minimum improvement for early stopping
    lambda_k: float = 0.0  # Kurtosis regularization strength (0 = disabled)
    lambda_s: float = 0.0  # Skewness regularization strength (0 = disabled)


@dataclass
class PipelineYAMLConfig:
    """
    Complete pipeline configuration from YAML.

    Simple data container that combines all configuration sections.
    Simulator and filtration are optional for load mode (diagrams already cached).
    """
    experiment: ExperimentConfig
    analysis: AnalysisConfig
    vectorization: VectorizationConfig
    compression: CompressionConfig
    simulator: Optional[SimulatorConfig] = None      # Optional for load mode
    filtration: Optional[FiltrationConfig] = None    # Optional for load mode
    training: Optional[TrainingConfig] = None

    def is_trainable(self) -> bool:
        """
        Check if any component requires training.

        Returns:
            True if any component has trainable=True
        """
        filtration_trainable = self.filtration.trainable if self.filtration else False
        return (filtration_trainable or
                self.vectorization.trainable or
                self.compression.trainable)

    def get_trainable_component(self) -> Optional[str]:
        """
        Get which component is trainable.

        Returns:
            'filtration', 'vectorization', 'compression', or None
        """
        if self.filtration and self.filtration.trainable:
            return 'filtration'
        elif self.vectorization.trainable:
            return 'vectorization'
        elif self.compression.trainable:
            return 'compression'
        return None

    def validate(self) -> None:
        """
        Validate configuration consistency.

        Raises:
            ValueError: If configuration is invalid
        """
        # Require training config for trainable pipelines
        if self.is_trainable() and self.training is None:
            component = self.get_trainable_component()
            raise ValueError(
                f"Training configuration required for trainable {component} component. "
                "Please add a 'training:' section to your YAML config."
            )


# =============================================================================
# Pipeline Output
# =============================================================================

@dataclass
class FisherResult:
    """Results from Fisher information analysis."""
    fisher_matrix: torch.Tensor      # Fisher information matrix (n_params, n_params)
    inverse_fisher: torch.Tensor     # Covariance matrix = F^-1
    derivatives: torch.Tensor        # Parameter derivatives (n_params, n_d, n_features)
    covariance: torch.Tensor         # Summary covariance matrix (n_features, n_features)
    log_det_fisher: torch.Tensor     # log|F| (scalar)
    constraints: torch.Tensor        # 1-sigma constraints = sqrt(diag(F^-1))

    # Optional diagnostic information
    bias_error: Optional[torch.Tensor] = None
    fractional_bias: Optional[torch.Tensor] = None
    is_gaussian: Optional[bool] = None
    gaussianity_details: Optional[Dict[str, Any]] = None

    # Gaussianity regularization penalties (computed when compute_moments=True)
    skewness_penalty: Optional[torch.Tensor] = None   # Mean squared skewness
    kurtosis_penalty: Optional[torch.Tensor] = None   # Mean squared excess kurtosis

    def print_gaussianity(self):
        """Print Gaussianity check result."""
        if self.is_gaussian is None:
            print("\nGaussianity Check: Not performed")
        else:
            gauss_mark = "✓ PASS" if self.is_gaussian else "✗ FAIL"
            print(f"\nGaussianity Check: {gauss_mark}")