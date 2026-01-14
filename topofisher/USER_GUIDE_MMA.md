# TopoFisher MMA User Guide

This guide covers the MMA (Multiparameter Module Approximation) branch of TopoFisher, which uses 2-parameter persistent homology instead of standard 1-parameter persistence.

## Installation

```bash
pip install -e .
pip install gudhi powerbox multipers
```

Note: `multipers` is required for MMA computation.

## Quick Start

```bash
# Run MMA pipeline with TopK vectorization
python run_pipeline.py topofisher/examples/grf/mma_topk_moped.yaml

# Run MMA pipeline with Exponential kernel vectorization
python run_pipeline.py topofisher/examples/grf/mma_exponential_moped.yaml

# Save results
python run_pipeline.py topofisher/examples/grf/mma_topk_moped.yaml --save-results
```

## Pipeline Overview

TopoFisher MMA uses a five-stage pipeline:

```
Simulator → MMA Filtration → MMA Vectorization → Compression → Fisher Analyzer
```

### Comparison with 1PH (main branch)

| Stage | 1PH (main) | MMA (this branch) |
|-------|------------|-------------------|
| Input | field | field + gradient |
| Filtration | Cubical/Alpha → PD | Freudenthal → SimplexTreeMulti → Module Approximation |
| Output | Persistence Diagrams | MMA intervals (births/deaths in 2D) |
| Vectorization | TopK on (birth, death) | TopK or Kernel on 2D corners |

### How MMA Works

1. **Bifiltration**: The field and its gradient magnitude form a 2-parameter filtration
2. **Freudenthal triangulation**: Converts the 2D grid to a simplicial complex
3. **Module approximation**: Computes the MMA using `multipers.module_approximation()`
4. **Vectorization**: Extracts features from the MMA intervals

## Configuration

### YAML Structure for MMA

```yaml
experiment:
  name: grf_mma_experiment
  output_dir: experiments/grf/results/mma_output

analysis:
  theta_fid: [1.0, 2.0]      # Fiducial parameters
  delta_theta: [0.1, 0.2]    # Step sizes for derivatives
  n_s: 10000                 # Covariance samples
  n_d: 10000                 # Derivative samples
  seed_cov: 42
  seed_ders: [43, 44]

simulator:
  type: grf
  params:
    N: 16
    dim: 2

filtration:
  type: mma                  # Use MMA filtration
  trainable: false
  params:
    nlines: 500              # Number of lines for MMA approximation

vectorization:
  type: combined
  trainable: false
  params:
    layers:
      - type: mma_topk       # or mma_exponential, mma_gaussian, mma_linear
        params:
          k: 100
          homology_dimension: 0
      - type: mma_topk
        params:
          k: 50
          homology_dimension: 1

compression:
  type: moped
  trainable: false
  params:
    reg: 1.0e-6
```

## Available Components

### Simulators
- `grf`: Gaussian Random Fields (params: N, dim)

### Filtrations
- `mma`: Multiparameter Module Approximation
  - params:
    - `nlines`: Number of lines for approximation (default: 500)
    - `max_error`: Alternative to nlines, specify max error threshold

### MMA Vectorizations

#### TopK Corners (`mma_topk`)
Selects the k corners with largest L1 norm from the MMA intervals.

```yaml
- type: mma_topk
  params:
    k: 100                    # Number of corners to keep
    homology_dimension: 0     # H0 or H1
    pad_value: 0.0           # Padding if fewer than k corners
```

Output: `k * 2` features (k corners × 2 coordinates)

#### Exponential Kernel (`mma_exponential`)
Applies exponential kernel on a regular grid. **Recommended for best performance.**

```yaml
- type: mma_exponential
  params:
    resolution: 15            # Grid resolution (15x15)
    bandwidth: 0.5            # Kernel bandwidth
    homology_dimension: 0     # H0 or H1
```

Output: `resolution * resolution` features per homology dimension

#### Linear Kernel (`mma_linear`)
Applies linear kernel on a regular grid.

```yaml
- type: mma_linear
  params:
    resolution: 15
    bandwidth: 0.5
    homology_dimension: 0
```

#### Gaussian Kernel (`mma_gaussian`)
Applies Gaussian kernel on a regular grid. **Note: May perform poorly in some cases.**

```yaml
- type: mma_gaussian
  params:
    resolution: 15
    bandwidth: 0.5
    homology_dimension: 0
```

### Compressions
- `moped`: Analytical optimal compression (recommended)
- `identity`: No compression
- `mlp`: Learned MLP compression
- `cnn`: Learned CNN compression

## Examples

### Example 1: MMA + TopK + MOPED

```yaml
# mma_topk_moped.yaml
experiment:
  name: grf_mma_topk_moped
  output_dir: experiments/grf/results/mma_topk_moped

analysis:
  theta_fid: [1.0, 2.0]
  delta_theta: [0.1, 0.2]
  n_s: 10000
  n_d: 10000
  seed_cov: 42
  seed_ders: [43, 44]

simulator:
  type: grf
  params:
    N: 16
    dim: 2

filtration:
  type: mma
  params:
    nlines: 500

vectorization:
  type: combined
  params:
    layers:
      - type: mma_topk
        params:
          k: 100
          homology_dimension: 0
      - type: mma_topk
        params:
          k: 50
          homology_dimension: 1

compression:
  type: moped
  params:
    reg: 1.0e-6
```

### Example 2: MMA + Exponential Kernel + MOPED

```yaml
# mma_exponential_moped.yaml
experiment:
  name: grf_mma_exponential_moped
  output_dir: experiments/grf/results/mma_exponential_moped

analysis:
  theta_fid: [1.0, 2.0]
  delta_theta: [0.1, 0.2]
  n_s: 10000
  n_d: 10000
  seed_cov: 42
  seed_ders: [43, 44]

simulator:
  type: grf
  params:
    N: 16
    dim: 2

filtration:
  type: mma
  params:
    nlines: 500

vectorization:
  type: combined
  params:
    layers:
      - type: mma_exponential
        params:
          resolution: 15
          bandwidth: 0.5
          homology_dimension: 0
      - type: mma_exponential
        params:
          resolution: 15
          bandwidth: 0.5
          homology_dimension: 1

compression:
  type: moped
  params:
    reg: 1.0e-6
```

## Output

```python
from topofisher.config import load_pipeline_config, create_pipeline_from_config

# Load and create pipeline
config = load_pipeline_config("mma_topk_moped.yaml")
pipeline, config = create_pipeline_from_config(config)

# Run
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

## Performance Comparison

Based on GRF experiments with `theta_fid = [1.0, 2.0]`:

| Method | log|F| | Efficiency vs Theoretical |
|--------|--------|---------------------------|
| Theoretical | 6.73 | 100% |
| MMA + Exponential + MOPED | 4.80 | ~71% |
| MMA + TopK + MOPED | 4.58 | ~68% |
| MMA + Gaussian + MOPED | 0.06 | ~1% (not recommended) |

## Programmatic Usage (without YAML)

```python
import torch
from topofisher.filtrations.mma_simplextree import MMALayer
from topofisher.vectorizations.mma_topk import MMATopKLayer
from topofisher.vectorizations.mma_kernel import MMAExponentialLayer
from topofisher import (
    GRFSimulator,
    CombinedVectorization,
    MOPEDCompression,
    FisherAnalyzer,
    FisherPipeline,
    FisherConfig
)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Components
simulator = GRFSimulator(N=16, dim=2, device=str(device))
filtration = MMALayer(nlines=500)
vectorization = CombinedVectorization([
    MMATopKLayer(k=100, homology_dimension=0),
    MMATopKLayer(k=50, homology_dimension=1)
])
compression = MOPEDCompression()
fisher_analyzer = FisherAnalyzer(clean_data=True)

# Assemble pipeline
pipeline = FisherPipeline(
    simulator=simulator,
    filtration=filtration,
    vectorization=vectorization,
    compression=compression,
    fisher_analyzer=fisher_analyzer
)

# Fisher config
config = FisherConfig(
    theta_fid=torch.tensor([1.0, 2.0]),
    delta_theta=torch.tensor([0.1, 0.2]),
    n_s=10000,
    n_d=10000,
    seed_cov=42,
    seed_ders=[43, 44]
)

# Run
result = pipeline(config)
print(f"log|F| = {result.log_det_fisher.item():.2f}")
```

## Troubleshooting

### "No module named 'multipers'"
Install multipers: `pip install multipers`

### KeOps CUDA Warning
This is normal on CPU-only machines. The warning can be ignored:
```
[KeOps] Warning : CUDA libraries not found or could not be loaded; Switching to CPU only.
```

### Gaussian kernel poor performance
The Gaussian kernel may perform poorly on some datasets. Use `mma_exponential` or `mma_topk` instead.

## Extending TopoFisher MMA

To add custom MMA vectorizations, create a new class inheriting from `nn.Module` with:
- `__init__`: Set `self.n_features` for output size
- `forward(self, mma_objects, field, gradient)`: Return tensor of shape `(n_samples, n_features)`

Register it in `topofisher/config/component_factory.py`:
```python
@register_vectorization('my_mma_vectorization')
def create_my_vectorization(params, trainable):
    from ..vectorizations.my_module import MyMMALayer
    return MyMMALayer(**params)
```