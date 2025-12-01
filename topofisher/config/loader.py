"""
YAML configuration loading utilities.

This module provides functions to load pipeline configurations from YAML files
and create pipeline instances.
"""
import os
from pathlib import Path
from typing import Union, Optional, Dict, Any
import yaml
import torch

from .data_types import (
    PipelineYAMLConfig, ExperimentConfig, AnalysisConfig,
    SimulatorConfig, FiltrationConfig, VectorizationConfig,
    CompressionConfig, TrainingConfig, CacheConfig
)
from .component_factory import (
    create_simulator, create_filtration, create_vectorization,
    create_compression, create_fisher_analyzer
)


def load_component_config(config_value: Union[str, Dict], component_name: str,
                         base_path: Path) -> Dict:
    """
    Load component configuration from file or return inline config.

    Args:
        config_value: Either a string path to YAML file or inline dict config
        component_name: Name of component (for error messages)
        base_path: Base path for resolving relative paths

    Returns:
        Component configuration dictionary
    """
    if isinstance(config_value, str):
        # It's a file path - load the component config
        component_path = Path(config_value)

        # Try different path resolutions
        if component_path.is_absolute() and component_path.exists():
            resolved_path = component_path
        elif (base_path / component_path).exists():
            # Relative to main config file
            resolved_path = base_path / component_path
        else:
            # Try in components directory
            components_dir = Path(__file__).parent / 'components'
            if (components_dir / component_path).exists():
                resolved_path = components_dir / component_path
            else:
                raise FileNotFoundError(
                    f"{component_name} config file not found: {config_value}\n"
                    f"Tried: {component_path}, {base_path / component_path}, "
                    f"{components_dir / component_path}"
                )

        print(f"Loading {component_name} from: {resolved_path}")
        with open(resolved_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # It's an inline config dictionary
        return config_value


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

    base_path = yaml_path.parent

    # Parse experiment config
    experiment_data = yaml_data.get('experiment', {})
    experiment_config = ExperimentConfig(
        name=experiment_data.get('name', 'unnamed_experiment'),
        output_dir=experiment_data.get('output_dir', 'experiments'),
        description=experiment_data.get('description'),
        tags=experiment_data.get('tags', [])
    )

    # Parse analysis config (can be file path or inline; previously 'fisher' or 'simulation')
    analysis_data = yaml_data.get('analysis', yaml_data.get('simulation', yaml_data.get('fisher')))
    if not analysis_data:
        raise ValueError("Missing 'analysis' section in YAML config")

    analysis_data = load_component_config(analysis_data, 'analysis', base_path)

    # Parse cache config if present
    cache_config = None
    if 'cache' in analysis_data:
        cache_data = analysis_data['cache']
        cache_config = CacheConfig(
            mode=cache_data['mode'],
            data_type=cache_data['data_type'],
            save_path=cache_data.get('save_path'),
            load_path=cache_data.get('load_path')
        )

    # For load mode, get analysis params from cached file's metadata
    if cache_config and cache_config.mode == "load" and cache_config.load_path:
        import pickle
        with open(cache_config.load_path, 'rb') as f:
            cached_data = pickle.load(f)
        analysis_data = cached_data.get('metadata', {})
        print(f"Loaded analysis parameters from cache: {cache_config.load_path}")

    analysis_config = AnalysisConfig(
        theta_fid=analysis_data['theta_fid'],
        delta_theta=analysis_data['delta_theta'],
        n_s=analysis_data['n_s'],
        n_d=analysis_data['n_d'],
        seed_cov=analysis_data.get('seed_cov', 42),
        seed_ders=analysis_data.get('seed_ders', []),
        cache=cache_config
    )

    # Check cache modes to determine which components are optional
    is_diagram_generate = (cache_config is not None and
                           cache_config.mode == "generate" and
                           cache_config.data_type == "diagrams")
    is_diagram_load = (cache_config is not None and
                       cache_config.mode == "load" and
                       cache_config.data_type == "diagrams")

    # Parse simulator config (optional for load mode)
    simulator_config = None
    sim_data = yaml_data.get('simulator')
    if sim_data:
        sim_data = load_component_config(sim_data, 'simulator', base_path)
        simulator_config = SimulatorConfig(
            type=sim_data['type'],
            params=sim_data.get('params', {})
        )
    elif not is_diagram_load:
        raise ValueError("Missing 'simulator' section in YAML config")

    # Parse filtration config (optional for load mode)
    filtration_config = None
    filt_data = yaml_data.get('filtration')
    if filt_data:
        filt_data = load_component_config(filt_data, 'filtration', base_path)
        filtration_config = FiltrationConfig(
            type=filt_data['type'],
            trainable=filt_data.get('trainable', False),
            params=filt_data.get('params', {})
        )
    elif not is_diagram_load:
        raise ValueError("Missing 'filtration' section in YAML config")

    # Parse vectorization config (optional for diagram generation)
    vec_data = yaml_data.get('vectorization')
    if not vec_data and not is_diagram_generate:
        raise ValueError("Missing 'vectorization' section in YAML config")

    if vec_data:
        vec_data = load_component_config(vec_data, 'vectorization', base_path)
        vectorization_config = VectorizationConfig(
            type=vec_data['type'],
            trainable=vec_data.get('trainable', False),
            params=vec_data.get('params', {})
        )
    else:
        # Default identity for diagram generation
        vectorization_config = VectorizationConfig(type='identity', trainable=False, params={})

    # Parse compression config (optional for diagram generation)
    comp_data = yaml_data.get('compression')
    if not comp_data and not is_diagram_generate:
        raise ValueError("Missing 'compression' section in YAML config")

    if comp_data:
        comp_data = load_component_config(comp_data, 'compression', base_path)
        compression_config = CompressionConfig(
            type=comp_data['type'],
            trainable=comp_data.get('trainable', False),
            params=comp_data.get('params', {})
        )
    else:
        # Default identity for diagram generation
        compression_config = CompressionConfig(type='identity', trainable=False, params={})

    # Parse training config (optional, can be file path or inline)
    training_config = None
    if 'training' in yaml_data:
        train_data = yaml_data['training']
        train_data = load_component_config(train_data, 'training', base_path)
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
    # Create components (None configs result in None components)
    simulator = create_simulator(config.simulator) if config.simulator else None
    filtration = create_filtration(config.filtration) if config.filtration else None
    vectorization = create_vectorization(config.vectorization)
    compression = create_compression(config.compression)
    fisher_analyzer = create_fisher_analyzer(clean_data=True)

    # Check for cache mode first - this takes priority
    if config.analysis.cache is not None:
        cache_mode = config.analysis.cache.mode

        if cache_mode == "generate":
            # Use GenerateCachePipeline to generate and save data
            from ..pipelines.cached import GenerateCachePipeline

            pipeline = GenerateCachePipeline(
                simulator=simulator,
                filtration=filtration,
                vectorization=vectorization,
                compression=compression,
                fisher_analyzer=fisher_analyzer
            )
            return pipeline, config

        elif cache_mode == "load":
            # Use CachedCompressionPipeline to load and train
            # simulator and filtration can be None (diagrams already cached)
            from ..pipelines.cached import CachedCompressionPipeline

            pipeline = CachedCompressionPipeline(
                simulator=None,
                filtration=None,
                vectorization=vectorization,
                compression=compression,
                fisher_analyzer=fisher_analyzer
            )
            return pipeline, config

        else:
            raise ValueError(f"Unknown cache mode: {cache_mode}. Use 'generate' or 'load'.")

    # No cache - determine pipeline type based on trainable component
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