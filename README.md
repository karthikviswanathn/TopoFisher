# TopoFisher

**Topological Fisher Information Analysis in PyTorch**

TopoFisher is a clean, modular PyTorch implementation for computing Fisher information matrices using persistent homology. It enables parameter inference from data by combining topological data analysis with Fisher information theory.

**Interactive Dashboard**: See [GRF compression results](https://htmlpreview.github.io/?https://github.com/karthikviswanathn/TopoFisher/blob/dev/topofisher/examples/grf/dashboard.html)

## Overview

The pipeline consists of four customizable components:

1. **Simulator**: Generate data at different parameter values (e.g., Gaussian Random Fields)
2. **Filtration**: Compute persistence diagrams from data (e.g., Cubical Complex)
3. **Vectorization**: Convert persistence diagrams to feature vectors (e.g., Top-K)
4. **Fisher Analyzer**: Compute Fisher information matrix from summary statistics

## Installation

```bash
# Install dependencies
pip install torch numpy gudhi powerbox

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

fisher = FisherAnalyzer(clean_data=True)

# 2. Create pipeline
pipeline = FisherPipeline(
    simulator=simulator,
    filtration=filtration,
    vectorization=vectorization,
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
│   ├── types.py          # Data structures (FisherConfig, FisherResult)
│   ├── interfaces.py     # Abstract base classes
│   └── pipeline.py       # Pipeline orchestrator
├── simulators/
│   └── grf.py           # Gaussian Random Field simulator
├── filtrations/
│   └── cubical.py       # Cubical complex filtration
├── vectorizations/
│   └── topk.py          # Top-K vectorization
└── fisher/
    └── analyzer.py      # Fisher information computation
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

## TODO

- [ ] Add support for selective derivative computation (find_derivative flag)
- [ ] Add Fisher bias error computation
- [ ] Add more vectorization methods (Persistence Images, Landscapes, etc.)
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
