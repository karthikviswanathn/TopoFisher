#!/usr/bin/env python3
"""
Run a TopoFisher pipeline from a YAML configuration file.

Usage:
    python run_pipeline.py config.yaml [--train] [--verbose]

Examples:
    # Run inference only
    python run_pipeline.py topofisher/examples/grf/topk_moped.yaml

    # Train learnable components
    python run_pipeline.py topofisher/examples/grf/cubical_mlp.yaml --train
"""
import argparse
import json
import sys
from pathlib import Path
import torch
import numpy as np
import yaml

from topofisher.config import load_pipeline_config, create_pipeline_from_config


def extract_expanded_config(config, pipeline):
    """Extract fully expanded configuration including auto-selected values."""
    from dataclasses import asdict
    from topofisher.vectorizations import CombinedVectorization, TopKLayer

    # Convert config to dict (handle None for optional components)
    expanded = {
        'experiment': asdict(config.experiment),
        'analysis': asdict(config.analysis),
        'vectorization': asdict(config.vectorization),
        'compression': asdict(config.compression),
    }

    if config.simulator:
        expanded['simulator'] = asdict(config.simulator)
    if config.filtration:
        expanded['filtration'] = asdict(config.filtration)
    if config.training:
        expanded['training'] = asdict(config.training)

    # Extract auto-selected k values from TopK layers
    if isinstance(pipeline.vectorization, CombinedVectorization):
        layers = []
        for layer in pipeline.vectorization.layers:
            if isinstance(layer, TopKLayer) and layer.k is not None:
                layers.append({
                    'type': 'topk',
                    'params': {'k': layer.k}  # Include auto-selected k
                })
        if layers:
            expanded['vectorization']['params']['layers'] = layers

    return expanded


def main():
    parser = argparse.ArgumentParser(description='Run TopoFisher pipeline from YAML config')
    parser.add_argument('config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--train', action='store_true', help='Train learnable components')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--save-results', action='store_true', default=True,
                        help='Save results to output directory')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate from config')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory from config')
    parser.add_argument('--lambda-k', type=float, default=None,
                        help='Override kurtosis regularization strength')
    parser.add_argument('--lambda-s', type=float, default=None,
                        help='Override skewness regularization strength')
    parser.add_argument('--n-epochs', type=int, default=None,
                        help='Override number of training epochs')

    args = parser.parse_args()

    # Load configuration
    print(f"\n{'='*60}")
    print(f"Loading configuration from: {args.config}")
    print(f"{'='*60}")

    try:
        config = load_pipeline_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Apply command line overrides
    if args.lr is not None and config.training is not None:
        print(f"Overriding learning rate: {config.training.lr} → {args.lr}")
        config.training.lr = args.lr

    if args.output_dir is not None:
        print(f"Overriding output directory: {config.experiment.output_dir} → {args.output_dir}")
        config.experiment.output_dir = args.output_dir

    if args.lambda_k is not None and config.training is not None:
        print(f"Overriding lambda_k: {config.training.lambda_k} → {args.lambda_k}")
        config.training.lambda_k = args.lambda_k

    if args.lambda_s is not None and config.training is not None:
        print(f"Overriding lambda_s: {config.training.lambda_s} → {args.lambda_s}")
        config.training.lambda_s = args.lambda_s

    if args.n_epochs is not None and config.training is not None:
        print(f"Overriding n_epochs: {config.training.n_epochs} → {args.n_epochs}")
        config.training.n_epochs = args.n_epochs

    # Print configuration summary
    print(f"\nExperiment: {config.experiment.name}")
    if config.experiment.description:
        print(f"Description: {config.experiment.description}")
    print(f"Output directory: {config.experiment.output_dir}")
    print(f"\nAnalysis parameters:")
    print(f"  theta_fid: {config.analysis.theta_fid}")
    print(f"  delta_theta: {config.analysis.delta_theta}")
    print(f"  n_s: {config.analysis.n_s}, n_d: {config.analysis.n_d}")
    print(f"\nPipeline components:")
    print(f"  Simulator: {config.simulator.type if config.simulator else 'N/A (cached)'}")
    print(f"  Filtration: {config.filtration.type if config.filtration else 'N/A (cached)'}")
    print(f"  Vectorization: {config.vectorization.type} (trainable={config.vectorization.trainable})")
    print(f"  Compression: {config.compression.type} (trainable={config.compression.trainable})")

    # Create pipeline
    print(f"\n{'='*60}")
    print("Creating pipeline...")
    print(f"{'='*60}")

    try:
        pipeline, config = create_pipeline_from_config(config)
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        sys.exit(1)

    # Use analysis config directly (no conversion needed)
    analysis_config = config.analysis

    # Check for generate mode (special case - no training, just save data)
    if analysis_config.cache is not None and analysis_config.cache.mode == "generate":
        print(f"\n{'='*60}")
        print("Running in GENERATE mode")
        print(f"{'='*60}")

        result_dict = pipeline.run(analysis_config)

        print(f"\nGeneration complete!")
        print(f"  Data type: {result_dict.get('data_type', 'unknown')}")
        if 'saved_diagrams' in result_dict:
            print(f"  Saved to: {result_dict['saved_diagrams']}")
        if 'saved_summaries' in result_dict:
            print(f"  Saved to: {result_dict['saved_summaries']}")

        print(f"\n{'='*60}")
        print("Generation completed successfully!")
        print(f"{'='*60}\n")
        return

    # Training mode (cache load or regular trainable)
    needs_training = (analysis_config.cache is not None and analysis_config.cache.mode == "load") or \
                     (config.is_trainable() and args.train)

    if needs_training:
        if not config.training:
            print("Error: Training config required")
            sys.exit(1)

        print(f"\nTraining {config.get_trainable_component()} component...")
        print(f"Training config:")
        print(f"  Epochs: {config.training.n_epochs}")
        print(f"  Learning rate: {config.training.lr}")
        print(f"  Batch size: {config.training.batch_size}")

        # Generate/load data
        print("\nPreparing data...")
        data = pipeline.generate_data(analysis_config)

        # Train the pipeline
        training_result = pipeline.run(
            config=analysis_config,
            training_config=config.training,
            data=data
        )
        result = training_result['test_result']

    elif config.is_trainable() and not args.train:
        print("\nWarning: Pipeline has trainable components but --train not specified")
        print("Running inference only (may fail if components aren't pre-trained)")
        result = pipeline(analysis_config)

    else:
        # Run inference
        print("\nRunning pipeline...")
        result = pipeline(analysis_config)

    # Print results
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")

    print(f"\nFisher Information Matrix:")
    print(result.fisher_matrix.detach().cpu().numpy())

    print(f"\nlog|F| = {result.log_det_fisher.cpu().item():.4f}")

    print(f"\nParameter constraints (1σ):")
    for i, (param, sigma) in enumerate(zip(config.analysis.theta_fid, result.constraints.cpu())):
        print(f"  θ_{i} = {param:.2f} ± {sigma.item():.4f}")

    # Gaussianity check
    result.print_gaussianity()

    # Compare with theoretical if available
    if pipeline.simulator and hasattr(pipeline.simulator, 'theoretical_fisher_matrix'):
        print(f"\n{'='*60}")
        print("Comparison with Theoretical")
        print(f"{'='*60}")

        theta_fid_tensor = torch.tensor(config.analysis.theta_fid)
        F_theory = pipeline.simulator.theoretical_fisher_matrix(theta_fid_tensor)
        log_det_theory = torch.logdet(F_theory)
        constraints_theory = torch.sqrt(torch.diag(torch.linalg.inv(F_theory)))

        print(f"\n{'Method':<20} {'log|F|':>10} {'σ(A)':>10} {'σ(B)':>10}")
        print("-"*53)
        print(f"{'Theoretical':<20} {log_det_theory.cpu().item():>10.2f} {constraints_theory[0].cpu().item():>10.4f} {constraints_theory[1].cpu().item():>10.4f}")
        print(f"{'Pipeline':<20} {result.log_det_fisher.cpu().item():>10.2f} {result.constraints[0].cpu().item():>10.4f} {result.constraints[1].cpu().item():>10.4f}")

        ratio = (result.log_det_fisher.cpu() / log_det_theory.cpu()).item()
        print(f"\nEfficiency: {ratio:.1%} of theoretical maximum")

    # Save results if requested
    if args.save_results:
        output_dir = Path(config.experiment.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save expanded config with auto-selected values
        expanded_config = extract_expanded_config(config, pipeline)

        # Custom representer to use flow style for lists
        def represent_list(dumper, data):
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

        yaml.add_representer(list, represent_list)

        with open(output_dir / "config.yaml", 'w') as f:
            yaml.dump(expanded_config, f, default_flow_style=False, sort_keys=False)

        # Collect component representations
        component_metadata = {}
        if hasattr(pipeline, 'simulator'):
            component_metadata['simulator'] = repr(pipeline.simulator)
        if hasattr(pipeline, 'filtration'):
            component_metadata['filtration'] = repr(pipeline.filtration)
        if hasattr(pipeline, 'vectorization'):
            component_metadata['vectorization'] = repr(pipeline.vectorization)
        if hasattr(pipeline, 'compression'):
            component_metadata['compression'] = repr(pipeline.compression)
        if hasattr(pipeline, 'fisher_analyzer'):
            component_metadata['fisher_analyzer'] = repr(pipeline.fisher_analyzer)

        # Save results as JSON (move tensors to CPU first)
        results_dict = {
            'experiment': config.experiment.name,
            'fisher_matrix': result.fisher_matrix.cpu().tolist(),
            'log_det_fisher': result.log_det_fisher.cpu().item(),
            'constraints': result.constraints.cpu().tolist(),
            'is_gaussian': result.is_gaussian,
            'analysis': {
                'theta_fid': config.analysis.theta_fid.tolist(),
                'delta_theta': config.analysis.delta_theta.tolist(),
                'n_s': config.analysis.n_s,
                'n_d': config.analysis.n_d
            },
            'metadata': {
                'components': component_metadata
            }
        }

        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)

        # Save Fisher matrix as numpy (move to CPU first)
        np.save(output_dir / 'fisher_matrix.npy', result.fisher_matrix.detach().cpu().numpy())

        print(f"\nResults saved to: {output_dir}")

    print(f"\n{'='*60}")
    print("Pipeline completed successfully!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()