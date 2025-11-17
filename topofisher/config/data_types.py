"""
Configuration data types for YAML pipeline loading.

This module defines dataclasses for all configuration sections
used in YAML pipeline definitions.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union


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

    Previously called 'fisher' in old configs, renamed for clarity.
    These parameters control the Fisher information analysis.
    """
    theta_fid: List[float]  # Fiducial parameter values
    delta_theta: List[float]  # Step sizes for finite differences
    n_s: int  # Number of samples for covariance estimation
    n_d: int  # Number of samples for derivative estimation
    seed_cov: int = 42  # Seed for fiducial samples
    seed_ders: List[int] = field(default_factory=list)  # Seeds for derivative samples

    def __post_init__(self):
        """Validate and set defaults."""
        if not self.seed_ders:
            # Auto-generate derivative seeds if not provided
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
    check_gaussianity: bool = True
    patience: Optional[int] = None  # Early stopping patience
    min_delta: float = 1e-6  # Minimum improvement for early stopping


@dataclass
class PipelineYAMLConfig:
    """
    Complete pipeline configuration from YAML.

    Simple data container that combines all configuration sections.
    """
    experiment: ExperimentConfig
    analysis: AnalysisConfig
    simulator: SimulatorConfig
    filtration: FiltrationConfig
    vectorization: VectorizationConfig
    compression: CompressionConfig
    training: Optional[TrainingConfig] = None

    def is_trainable(self) -> bool:
        """
        Check if any component requires training.

        Returns:
            True if any component has trainable=True
        """
        return (self.filtration.trainable or
                self.vectorization.trainable or
                self.compression.trainable)

    def get_trainable_component(self) -> Optional[str]:
        """
        Get which component is trainable.

        Returns:
            'filtration', 'vectorization', 'compression', or None
        """
        if self.filtration.trainable:
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