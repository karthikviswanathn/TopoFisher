# TopoFisher User Guide

## Installation

```bash
pip install -e .
pip install gudhi powerbox
```

## Quick Start

```bash
# Run inference pipeline
python run_pipeline.py topofisher/examples/grf/topk_moped.yaml

# Train learnable components
python run_pipeline.py topofisher/examples/grf/learnable_topk_moped.yaml --train

# Save results
python run_pipeline.py config.yaml --save-results
```

## Pipeline Overview

TopoFisher uses a five-stage pipeline:

```
Simulator → Filtration → Vectorization → Compression → Fisher Analyzer
```

1. **Simulator**: Generates data at parameter values θ
2. **Filtration**: Computes persistence diagrams
3. **Vectorization**: Converts diagrams to fixed-size vectors
4. **Compression**: Reduces dimensionality
5. **Fisher Analyzer**: Computes Fisher information matrix

## Configuration

### YAML Structure

#### Non-Learnable Pipeline (no training required)

```yaml
experiment:
  name: grf_topk_moped
  output_dir: experiments/output

analysis:
  theta_fid: [1.0, 2.0]      # Fiducial parameters
  delta_theta: [0.1, 0.2]    # Step sizes
  n_s: 10000                 # Covariance samples
  n_d: 10000                 # Derivative samples
  seed_cov: 42
  seed_ders: [43, 44]

simulator: simulators/grf.yaml
filtration: filtrations/cubical.yaml
vectorization: vectorizations/topk_auto.yaml

compression:
  type: moped
  trainable: false
  params:
    reg: 1.0e-6
```

Run with: `python run_pipeline.py config.yaml`

#### Learnable Pipeline (requires training)

```yaml
experiment:
  name: grf_learnable_filtration
  output_dir: experiments/output

analysis:
  theta_fid: [1.0, 2.0]
  delta_theta: [0.1, 0.2]
  n_s: 40000                 # More samples recommended for training
  n_d: 40000
  seed_cov: 42
  seed_ders: [43, 44]

simulator: simulators/grf.yaml

filtration:
  type: learnable
  trainable: true            # Mark as trainable
  params:
    homology_dimensions: [0, 1]
    hidden_channels: [32, 64, 32]

vectorization: vectorizations/topk_auto.yaml
compression: compressions/moped.yaml

training:                    # Required when trainable: true
  n_epochs: 1000
  lr: 0.001
  batch_size: 500
  train_frac: 0.875
  val_frac: 0.0625
  validate_every: 10
  lambda_k: 0.0              # Kurtosis regularization (default: 0.0)
  lambda_s: 0.0              # Skewness regularization (default: 0.0)
```

Run with: `python run_pipeline.py config.yaml --train`

### Command-Line Overrides

```bash
python run_pipeline.py config.yaml --train --lr 0.0001 --n-epochs 500
```

## Available Components

### Simulators
- `grf`: Gaussian Random Fields (params: N, dim)
- `noisy_ring`: Point clouds on noisy ring (params: ncirc, nback, bgm_avg)

### Filtrations
- `cubical`: For gridded data/images
- `alpha`: Alpha complex for point clouds
- `alpha_dtm`: Distance-to-measure alpha complex
- `learnable`: CNN-based learnable filtration on cubical complexes
- `learnable_point`: Learnable vertex filtration for point clouds on Alpha complexes

### Vectorizations
- `topk`: Select k most persistent features
- `persistence_image`: Convert to 2D images
- `combined`: Combine multiple vectorizations

### Compressions
- `moped`: Analytical optimal compression
- `identity`: No compression
- `mlp`: Learned MLP compression (works well with topk)
- `cnn`: Learned CNN compression (works well with persistence images)

## Training

For learnable components, set `trainable: true` and add a training section:

```yaml
filtration:
  type: learnable
  trainable: true
  params:
    hidden_channels: [32, 64, 32]

training:
  n_epochs: 1000
  lr: 0.001
  batch_size: 500
  lambda_k: 0.1    # Kurtosis regularization
  lambda_s: 0.1    # Skewness regularization
```

Run with: `python run_pipeline.py config.yaml --train`

## Output

```python
from topofisher.config import load_pipeline_config, create_pipeline_from_config

# Load and create pipeline
config = load_pipeline_config("config.yaml")
pipeline, config = create_pipeline_from_config(config)

# Run (just a forward pass, i.e., after training)
result = pipeline(config.analysis)

# Access results
print(f"Fisher Matrix: {result.fisher_matrix}")
print(f"log|F|: {result.log_det_fisher}")
print(f"Constraints: {result.constraints}")
```

Results include:
- **Fisher Matrix**: Information about parameter constraints
- **log|F|**: Log determinant (higher = more information)
- **Constraints**: 1-sigma uncertainties on parameters
- **Gaussianity Check**: Must pass for valid Fisher estimates

## Examples

Example configs are in `topofisher/examples/`:
- `grf/topk_moped.yaml`: Basic GRF analysis
- `grf/learnable_topk_moped.yaml`: Learnable filtration for GRF
- `circle/learnable_flag.yaml`: Learnable filtration for point clouds

## Extending TopoFisher

To add custom simulators, filtrations, vectorizations, or compressions, see the [Extending TopoFisher](README.md#extending-topofisher) section in README.md.