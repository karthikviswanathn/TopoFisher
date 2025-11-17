"""
YAML configuration loading utilities.

This module provides functions to load pipeline configurations from YAML files
and create pipeline instances.
"""
from pathlib import Path
from typing import Union, Optional, Dict, Any
import yaml
import torch

from .data_types import (
    PipelineYAMLConfig, ExperimentConfig, AnalysisConfig,
    SimulatorConfig, FiltrationConfig, VectorizationConfig,
    CompressionConfig, TrainingConfig
)
from .component_factory import (
    create_simulator, create_filtration, create_vectorization,
    create_compression, create_fisher_analyzer, create_pipeline_config
)


def load_pipeline_config(yaml_path: Union[str, Path]) -> PipelineYAMLConfig:
    """
    Load pipeline configuration from YAML file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        PipelineYAMLConfig instance

    Raises:
        FileNotFoundError: If YAML file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        ValueError: If configuration is invalid
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    # Load YAML
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)

    # Parse experiment config
    experiment_data = yaml_data.get('experiment', {})
    experiment_config = ExperimentConfig(
        name=experiment_data.get('name', 'unnamed_experiment'),
        output_dir=experiment_data.get('output_dir', 'experiments'),
        description=experiment_data.get('description'),
        tags=experiment_data.get('tags', [])
    )

    # Parse analysis config (previously 'fisher' or 'simulation')
    analysis_data = yaml_data.get('analysis', yaml_data.get('simulation', yaml_data.get('fisher')))
    if not analysis_data:
        raise ValueError("Missing 'analysis' section in YAML config")

    analysis_config = AnalysisConfig(
        theta_fid=analysis_data['theta_fid'],
        delta_theta=analysis_data['delta_theta'],
        n_s=analysis_data['n_s'],
        n_d=analysis_data['n_d'],
        seed_cov=analysis_data.get('seed_cov', 42),
        seed_ders=analysis_data.get('seed_ders', [])
    )

    # Parse simulator config
    sim_data = yaml_data.get('simulator', {})
    if not sim_data:
        raise ValueError("Missing 'simulator' section in YAML config")

    simulator_config = SimulatorConfig(
        type=sim_data['type'],
        params=sim_data.get('params', {})
    )

    # Parse filtration config
    filt_data = yaml_data.get('filtration', {})
    if not filt_data:
        raise ValueError("Missing 'filtration' section in YAML config")

    filtration_config = FiltrationConfig(
        type=filt_data['type'],
        trainable=filt_data.get('trainable', False),
        params=filt_data.get('params', {})
    )

    # Parse vectorization config
    vec_data = yaml_data.get('vectorization', {})
    if not vec_data:
        raise ValueError("Missing 'vectorization' section in YAML config")

    vectorization_config = VectorizationConfig(
        type=vec_data['type'],
        trainable=vec_data.get('trainable', False),
        params=vec_data.get('params', {})
    )

    # Parse compression config
    comp_data = yaml_data.get('compression', {})
    if not comp_data:
        raise ValueError("Missing 'compression' section in YAML config")

    compression_config = CompressionConfig(
        type=comp_data['type'],
        trainable=comp_data.get('trainable', False),
        params=comp_data.get('params', {})
    )

    # Parse training config (optional)
    training_config = None
    if 'training' in yaml_data:
        train_data = yaml_data['training']
        training_config = TrainingConfig(
            n_epochs=train_data.get('n_epochs', 1000),
            lr=train_data.get('lr', 1e-3),
            batch_size=train_data.get('batch_size', 500),
            weight_decay=train_data.get('weight_decay', 1e-4),
            train_frac=train_data.get('train_frac', 0.5),
            val_frac=train_data.get('val_frac', 0.25),
            validate_every=train_data.get('validate_every', 10),
            verbose=train_data.get('verbose', True),
            check_gaussianity=train_data.get('check_gaussianity', True),
            patience=train_data.get('patience'),
            min_delta=train_data.get('min_delta', 1e-6)
        )

    # Create and validate config
    config = PipelineYAMLConfig(
        experiment=experiment_config,
        analysis=analysis_config,
        simulator=simulator_config,
        filtration=filtration_config,
        vectorization=vectorization_config,
        compression=compression_config,
        training=training_config
    )

    # Validate configuration
    config.validate()

    return config


def create_pipeline_from_config(config: PipelineYAMLConfig):
    """
    Create a pipeline instance from configuration.

    Args:
        config: Pipeline configuration

    Returns:
        Pipeline instance (BasePipeline or learnable variant)

    Raises:
        ValueError: If configuration is invalid
    """
    # Create components
    simulator = create_simulator(config.simulator)
    filtration = create_filtration(config.filtration)
    vectorization = create_vectorization(config.vectorization)
    compression = create_compression(config.compression)
    fisher_analyzer = create_fisher_analyzer(clean_data=True)

    # Determine pipeline type based on trainable component
    if not config.is_trainable():
        # Use base pipeline for non-trainable configurations
        from ..pipelines import BasePipeline

        pipeline = BasePipeline(
            simulator=simulator,
            filtration=filtration,
            vectorization=vectorization,
            compression=compression,
            fisher_analyzer=fisher_analyzer
        )

    else:
        # Use appropriate learnable pipeline
        trainable_component = config.get_trainable_component()

        if trainable_component == 'filtration':
            from ..pipelines.learnable import LearnableFiltrationPipeline

            pipeline = LearnableFiltrationPipeline(
                simulator=simulator,
                filtration=filtration,
                vectorization=vectorization,
                compression=compression,
                fisher_analyzer=fisher_analyzer
            )

        elif trainable_component == 'vectorization':
            from ..pipelines.learnable import LearnableVectorizationPipeline

            pipeline = LearnableVectorizationPipeline(
                simulator=simulator,
                filtration=filtration,
                vectorization=vectorization,
                compression=compression,
                fisher_analyzer=fisher_analyzer
            )

        elif trainable_component == 'compression':
            from ..pipelines.learnable import LearnableCompressionPipeline

            pipeline = LearnableCompressionPipeline(
                simulator=simulator,
                filtration=filtration,
                vectorization=vectorization,
                compression=compression,
                fisher_analyzer=fisher_analyzer
            )

        else:
            raise ValueError(f"Unknown trainable component: {trainable_component}")

    return pipeline, config


def load_and_create_pipeline(yaml_path: Union[str, Path]):
    """
    Convenience function to load config and create pipeline in one step.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        Tuple of (pipeline, config)
    """
    config = load_pipeline_config(yaml_path)
    return create_pipeline_from_config(config)