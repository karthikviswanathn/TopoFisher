"""
Configuration module for YAML pipeline loading.

This module provides a clean way to load and create pipelines from YAML configurations.
"""

# Data types
from .data_types import (
    ExperimentConfig,
    AnalysisConfig,
    SimulatorConfig,
    FiltrationConfig,
    VectorizationConfig,
    CompressionConfig,
    TrainingConfig,
    PipelineYAMLConfig
)

# Component factory
from .component_factory import (
    create_simulator,
    create_filtration,
    create_vectorization,
    create_compression,
    create_fisher_analyzer,
    create_pipeline_config,
    # Registries for extending
    register_simulator,
    register_filtration,
    register_vectorization,
    register_compression
)

# Loader utilities
from .loader import (
    load_pipeline_config,
    create_pipeline_from_config,
    load_and_create_pipeline
)

__all__ = [
    # Data types
    'ExperimentConfig',
    'AnalysisConfig',
    'SimulatorConfig',
    'FiltrationConfig',
    'VectorizationConfig',
    'CompressionConfig',
    'TrainingConfig',
    'PipelineYAMLConfig',
    # Factory functions
    'create_simulator',
    'create_filtration',
    'create_vectorization',
    'create_compression',
    'create_fisher_analyzer',
    'create_pipeline_config',
    # Registry decorators
    'register_simulator',
    'register_filtration',
    'register_vectorization',
    'register_compression',
    # Loader functions
    'load_pipeline_config',
    'create_pipeline_from_config',
    'load_and_create_pipeline',
]