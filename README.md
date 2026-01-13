# TopoFisher

**Topological Fisher Information Analysis in PyTorch**

TopoFisher is a modular PyTorch implementation for computing Fisher information matrices using persistent homology. It enables parameter inference from data by combining topological data analysis with Fisher information theory.

## Overview

The pipeline consists of five customizable components:

1. **Simulator**: Generate data at different parameter values (e.g., Gaussian Random Fields)
2. **Filtration**: Compute persistence diagrams from data (e.g., Cubical Complex)
3. **Vectorization**: Convert persistence diagrams to feature vectors (e.g., Top-K)
4. **Compression**: Reduce dimensionality while preserving Fisher information (e.g., MOPED, MLP)
5. **Fisher Analyzer**: Compute Fisher information matrix from summary statistics


## Quick Start

### Method 1: YAML Configuration (Recommended)

```bash
# Run with YAML config
python run_pipeline.py topofisher/examples/grf/topk_moped.yaml --save-results

# Train learnable components
python run_pipeline.py topofisher/examples/grf/learnable_mlp.yaml --train --save-results
```

### Method 2: Python API

```python
import torch
from topofisher.simulators.grf import GRFSimulator
from topofisher.filtrations.cubical import CubicalLayer
from topofisher.vectorizations.topk import TopKLayer
from topofisher.vectorizations.combined import CombinedVectorization
from topofisher.compressions.moped import MOPEDCompression
from topofisher.fisher.analyzer import FisherAnalyzer
from topofisher.pipelines.base import BasePipeline
from topofisher.config.data_types import AnalysisConfig

# 1. Set up components
simulator = GRFSimulator(N=32, dim=2)

filtration = CubicalLayer(homology_dimensions=[0, 1])

# Auto-k selection (NEW!)
vectorization = CombinedVectorization([
    TopKLayer(),  # k automatically determined from data
    TopKLayer(),  # k automatically determined from data
])

compression = MOPEDCompression(reg=1e-8)

fisher_analyzer = FisherAnalyzer(clean_data=True)

# 2. Create pipeline
pipeline = BasePipeline(
    simulator=simulator,
    filtration=filtration,
    vectorization=vectorization,
    compression=compression,
    fisher_analyzer=fisher_analyzer
)

# 3. Configure analysis
config = AnalysisConfig(
    theta_fid=[1.0, 2.0],      # Fiducial parameters [A, B]
    delta_theta=[0.1, 0.2],    # Step sizes (divided by 2 internally)
    n_s=1000,                  # Samples for covariance
    n_d=1000,                  # Samples for derivatives
    seed_cov=42,
    seed_ders=[43, 44]
)

# 4. Run pipeline
result = pipeline(config)

# 5. Inspect results
print(f"Fisher Matrix:\n{result.fisher_matrix}")
print(f"Constraints (1σ): {result.constraints}")
print(f"log|F|: {result.log_det_fisher}")
```

## Key Features

### 🚀 Training Learned Compressions

```python
from topofisher.compressions.mlp import MLPCompression
from topofisher.config.data_types import TrainingConfig

# Create uninitialized compression (dimensions inferred from data)
compression = MLPCompression(hidden_dims=[64, 32])

# Train with pipeline
training_config = TrainingConfig(
    n_epochs=1000,
    lr=1e-3,
    batch_size=500,
    check_gaussianity=True  # Ensure compressed features are Gaussian
)

# Pipeline automatically trains before running
result = pipeline.run(config=analysis_config, training_config=training_config)
```

## Structure

```
topofisher/
├── config/              # Configuration system
│   ├── data_types.py    # Config dataclasses
│   ├── loader.py        # YAML loading
│   └── component_factory.py  # Component creation
├── pipelines/           # Pipeline orchestrators
│   ├── base.py          # Standard pipeline
│   └── learnable.py     # Pipelines with training
├── simulators/
│   ├── __init__.py      # Base Simulator class
│   ├── grf.py           # Gaussian Random Field
│   └── gaussian_vector.py
├── filtrations/
│   ├── cubical.py       # Cubical complex (GUDHI)
│   ├── mma.py           # Multi-parameter persistence
│   └── identity.py      # Pass-through
├── vectorizations/
│   ├── topk.py          # Top-K with auto-selection
│   ├── persistence_image.py
│   ├── combined.py      # Combine multiple vectorizations
│   └── mma_topk.py
├── compressions/
│   ├── __init__.py      # Base Compression class
│   ├── identity.py      # No compression
│   ├── moped.py         # MOPED (analytical)
│   ├── mlp.py           # Multi-layer perceptron
│   ├── cnn.py           # Convolutional network
│   └── inception.py     # Inception blocks
├── fisher/
│   └── analyzer.py      # Fisher matrix computation
└── examples/
    └── grf/
        ├── topk_moped.yaml     # Standard config
        ├── learnable_mlp.yaml  # Trainable MLP
        └── train_*.py          # Training scripts
```

## YAML Configuration

Create minimal, readable configs:

```yaml
experiment:
  name: my_experiment
  output_dir: experiments/my_experiment

analysis:
  theta_fid: [1.0, 2.0]
  delta_theta: [0.1, 0.2]
  n_s: 1000
  n_d: 1000
  seed_cov: 42
  seed_ders: [43, 44]

simulator:
  type: grf
  params:
    N: 32
    dim: 2

filtration:
  type: cubical
  params:
    homology_dimensions: [0, 1]

vectorization:
  type: combined
  params:
    layers:
      - type: topk
        params: {}  # Auto-k selection
      - type: topk
        params:
          k: 10     # Manual k

compression:
  type: moped
  params:
    reg: 1.0e-8

# Optional: for trainable components
training:
  n_epochs: 1000
  lr: 0.001
  batch_size: 500
```

## Compression Methods

### Identity (No Compression)
```python
from topofisher.compressions.identity import IdentityCompression
compression = IdentityCompression()
```

### MOPED (Analytical)
```python
from topofisher.compressions.moped import MOPEDCompression
compression = MOPEDCompression(
    train_frac=0.5,   # Train/test split ratio
    clean_data=True,  # Remove zero-variance features
    reg=1e-8         # Regularization
)
```

### MLP (Learned)
```python
from topofisher.compressions.mlp import MLPCompression

# Dimensions can be inferred from data
compression = MLPCompression(
    hidden_dims=[64, 32],  # Hidden layers
    activation='gelu',
    dropout=0.1
)

# Train using pipeline.run(config, training_config)
```

### CNN (for Persistence Images)
```python
from topofisher.compressions.cnn import CNNCompression

compression = CNNCompression(
    channels=[32, 64, 128],  # CNN channels
    kernel_size=3,
    pool_size=2
)
```

## Important Implementation Details

### Critical Rules 

1. **Finite Differences**: Always use ±Δθ/2, never ±Δθ
2. **Seed Management**: Same seeds for θ- and θ+ pairs
3. **Data Flow**: Maintain order [fid, minus_0, plus_0, minus_1, plus_1, ...]


## Examples

```bash
# Basic GRF with TopK and MOPED
python run_pipeline.py topofisher/examples/grf/topk_moped.yaml

```

## Extending TopoFisher

### Add a New Simulator

```python
from topofisher.simulators import Simulator

class MySimulator(Simulator):
    def generate_single(self, theta, seed):
        # Generate one sample at theta with given seed
        np.random.seed(seed)
        return data  # numpy array
```

### Add a New Filtration

Create `topofisher/filtrations/my_filtration.py`:

```python
import torch
import torch.nn as nn
from typing import List

class MyFiltration(nn.Module):
    def __init__(self, homology_dimensions: List[int], my_param: float = 1.0):
        super().__init__()
        self.dimensions = homology_dimensions
        self.my_param = my_param

    def forward(self, data: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Args:
            data: Input tensor of shape (n_samples, ...)

        Returns:
            List[List[Tensor]]: Outer list per homology dimension,
                inner list per sample. Each tensor shape (n_points, 2)
                with (birth, death) pairs.
        """
        n_samples = data.shape[0]
        all_diagrams = [[] for _ in self.dimensions]

        for i in range(n_samples):
            for dim_idx, dim in enumerate(self.dimensions):
                diagram = self._compute_diagram(data[i], dim)
                all_diagrams[dim_idx].append(diagram)

        return all_diagrams
```

### Add a New Vectorization

Create `topofisher/vectorizations/my_vectorization.py`:

```python
import torch
import torch.nn as nn
from typing import List

class MyVectorization(nn.Module):
    def __init__(self, output_dim: int = 50):
        super().__init__()
        self.output_dim = output_dim
        self.n_features = output_dim  # Required attribute

    def fit(self, diagrams: List[torch.Tensor]):
        """Optional: compute data-dependent parameters."""
        pass

    def forward(self, diagrams: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            diagrams: List of diagrams, each shape (n_points, 2)

        Returns:
            Tensor of shape (len(diagrams), n_features)
        """
        features = []
        for dgm in diagrams:
            if dgm.shape[0] == 0:
                vec = torch.zeros(self.output_dim, device=dgm.device)
            else:
                vec = self._vectorize(dgm)
            features.append(vec)
        return torch.stack(features)
```

### Add a New Compression

```python
from topofisher.compressions import Compression

class MyCompression(Compression):
    def forward(self, summaries):
        # summaries: [fid, minus_0, plus_0, minus_1, plus_1, ...]
        return [self.compress_fn(s) for s in summaries]
```

### Registering Components for YAML

Add to `topofisher/config/component_factory.py`:

```python
@register_filtration('my_filtration')
def create_my_filtration(params: Dict[str, Any], trainable: bool = False):
    from ..filtrations.my_filtration import MyFiltration
    return MyFiltration(**params)

@register_vectorization('my_vectorization')
def create_my_vectorization(params: Dict[str, Any], trainable: bool = False):
    from ..vectorizations.my_vectorization import MyVectorization
    return MyVectorization(**params)
```

Then use in YAML:

```yaml
filtration:
  type: my_filtration
  params:
    homology_dimensions: [0, 1]
    my_param: 2.0
```


## TODOs

- [ ] **Wrap filtration, vectorization and compression into a single model**: Currently these components are separate, but they could be wrapped together into a single `nn.Module` (perhaps as a PyTorch model) for cleaner learnable pipeline implementations. This would enable end-to-end optimization and better gradient flow through all components.

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- NumPy
- GUDHI
- powerbox (patched version in external/)
- multipers (for MMA filtration)

