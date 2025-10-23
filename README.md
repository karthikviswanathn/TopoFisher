# TopoFisher

**Topological Fisher Information Analysis in PyTorch**

TopoFisher is a clean, modular PyTorch implementation for computing Fisher information matrices using persistent homology. It enables parameter inference from data by combining topological data analysis with Fisher information theory.

**Interactive Dashboard**: See [GRF compression results](https://raw.githack.com/karthikviswanathn/TopoFisher/dev/topofisher/examples/grf/dashboard.html)

## Overview

The pipeline consists of five customizable components:

1. **Simulator**: Generate data at different parameter values (e.g., Gaussian Random Fields)
2. **Filtration**: Compute persistence diagrams from data (e.g., Cubical Complex)
3. **Vectorization**: Convert persistence diagrams to feature vectors (e.g., Top-K)
4. **Compression**: Compress features to maximize Fisher information (e.g., MOPED, MLP, CNN)
5. **Fisher Analyzer**: Compute Fisher information matrix from summary statistics

## Installation

```bash
# Install dependencies
pip install torch numpy scipy gudhi multipers powerbox

# Install TopoFisher
cd TopoFisher
pip install -e .
```

## Quick Start

```python
import torch
from topofisher import (
    GRFSimulator,
    CubicalLayer,
    TopKLayer,
    CombinedVectorization,
    IdentityCompression,
    MOPEDCompression,
    FisherAnalyzer,
    FisherPipeline,
    FisherConfig
)

# 1. Set up components
simulator = GRFSimulator(N=32, dim=2, boxlength=1.0)

filtration = CubicalLayer(
    homology_dimensions=[0, 1],
    min_persistence=[0.0, 0.0]
)

vectorization = CombinedVectorization([
    TopKLayer(k=10),  # H0
    TopKLayer(k=10),  # H1
])

# Choose compression method
compression = IdentityCompression()  # No compression
# compression = MOPEDCompression(compress_frac=0.5)  # Or use MOPED

fisher = FisherAnalyzer(clean_data=True)

# 2. Create pipeline
pipeline = FisherPipeline(
    simulator=simulator,
    filtration=filtration,
    vectorization=vectorization,
    compression=compression,
    fisher_analyzer=fisher
)

# 3. Configure analysis
config = FisherConfig(
    theta_fid=torch.tensor([1.0, 2.0]),      # Fiducial parameters [A, B]
    delta_theta=torch.tensor([0.1, 0.2]),    # Step sizes for derivatives
    n_s=100,                                  # Simulations for covariance
    n_d=100,                                  # Simulations for derivatives
    find_derivative=[True, True]              # Compute derivatives for both params
)

# 4. Run pipeline
result = pipeline(config)

# 5. Inspect results
print("Fisher Matrix:")
print(result.fisher_matrix)

print("\nParameter Constraints (1-sigma):")
print(result.constraints)

print("\nLog Fisher Information:")
print(result.log_det_fisher)
```

## Architecture

```
topofisher/
├── core/
│   ├── data_types.py     # Data structures (FisherConfig, FisherResult)
│   ├── interfaces.py     # Abstract base classes
│   └── pipeline.py       # Pipeline orchestrator
├── simulators/
│   ├── grf.py           # Gaussian Random Field simulator
│   └── gaussian_vector.py  # Gaussian vector simulator (for testing)
├── filtrations/
│   ├── cubical.py       # Cubical complex filtration
│   └── mma.py           # MMA filtration
├── vectorizations/
│   ├── topk.py          # Top-K vectorization
│   ├── persistence_image.py  # Persistence images
│   └── combined.py      # Combined vectorizations
├── compressions/
│   ├── moped.py         # MOPED compression
│   ├── mlp.py           # MLP learned compression
│   ├── cnn.py           # CNN learned compression
│   └── inception.py     # Inception block compression
└── fisher/
    └── analyzer.py      # Fisher information computation
```

## Compression Methods

TopoFisher supports multiple compression methods to reduce dimensionality while maximizing Fisher information:

### IdentityCompression (No Compression)
```python
from topofisher import IdentityCompression
compression = IdentityCompression()  # Pass-through, no compression
```

### MOPED (Maximum a Posteriori with Exponential Distribution)
```python
from topofisher import MOPEDCompression
compression = MOPEDCompression(
    compress_frac=0.5,  # Use 50% of data to compute compression matrix
    clean_data=True     # Remove zero-variance features
)
```

### MLP (Multi-Layer Perceptron)
```python
from topofisher import MLPCompression

# Create and train MLP (see examples/grf/train_mlp.py for training)
compression = MLPCompression(
    input_dim=260,      # Input feature dimension
    output_dim=2,       # Number of parameters
    hidden_dims=[512],  # Hidden layer dimensions
    dropout=0.2
)
# Or load pre-trained
compression = MLPCompression.from_pretrained("model.pth")
```

### CNN (Convolutional Neural Network for Persistence Images)
```python
from topofisher import CNNCompression

# For persistence images
compression = CNNCompression(
    n_channels=2,        # Number of homology dimensions
    n_pixels=16,         # Resolution (16x16)
    output_dim=2,        # Number of parameters
    dropout=0.2,
    use_dense_path=False  # Use only CNN path (or True for CNN+Dense)
)
```

### InceptBlock (IMNN-style Inception Network)
```python
from topofisher import InceptBlockCompression

# IMNN-style inception blocks
compression = InceptBlockCompression(
    n_channels=2,
    n_pixels=16,
    output_dim=2,
    n_filters=16
)
```

## Extending TopoFisher

### Add a New Simulator

```python
from topofisher.core.interfaces import Simulator

class MySimulator(Simulator):
    def generate(self, theta, n_samples, seed=None):
        # Your simulation logic
        return data  # shape: (n_samples, ...)
```

### Add a New Vectorization

```python
import torch.nn as nn

class MyVectorization(nn.Module):
    def forward(self, diagrams):
        # diagrams: List of (n_points, 2) tensors
        features = []
        for dgm in diagrams:
            # Your vectorization logic
            features.append(my_features)
        return torch.stack(features)
```

### Add a New Compression

```python
from topofisher.core.interfaces import Compression

class MyCompression(Compression):
    def forward(self, summaries, delta_theta=None):
        # Compress summaries
        # summaries: List of tensors [fiducial, theta_minus_0, theta_plus_0, ...]
        compressed = [self.compress(s) for s in summaries]
        return compressed
```

## TODO

- [ ] Add support for selective derivative computation (find_derivative flag)
- [ ] Add Fisher bias error computation
- [x] Add compression stage to pipeline (MOPED, MLP, CNN, InceptBlock)
- [x] Add Persistence Images vectorization
- [ ] Add more vectorization methods (Landscapes, Silhouettes, etc.)
- [ ] Add distributed computing support
- [ ] Add GPU acceleration for filtration
- [ ] Add more simulators (LSS, N-body, etc.)

## Requirements

- Python >= 3.8
- PyTorch >= 1.10
- NumPy
- GUDHI
- powerbox (for GRF simulation)

## License

MIT

## Citation

If you use TopoFisher in your research, please cite:

```bibtex
@software{topofisher2024,
  title={TopoFisher: Topological Fisher Information Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/TopoFisher}
}
```
