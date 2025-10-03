# TopoFisher (TensorFlow Legacy)

> **Note:** This is the legacy TensorFlow implementation. The current development is on the `main` branch using PyTorch.

TopoFisher is a Python package for computing Fisher information matrices from topological summaries of simulated data, with a focus on cosmological applications.

## Branch Information

This `tensorflow` branch contains the original TensorFlow-based implementation. For the latest PyTorch implementation with improved performance and features, please switch to the `main` branch:

```bash
git checkout main
```

## Overview

This legacy implementation uses TensorFlow for computing Fisher information from persistent homology features extracted from simulation data.

## Installation

```bash
pip install -e .
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Scipy
- gudhi (for persistent homology computation)

## Migration to PyTorch

The `main` branch contains the refactored PyTorch implementation with:
- GPU acceleration support
- Improved modularity and extensibility
- Learnable compression networks
- MOPED compression
- Better performance and maintainability

For new projects, we recommend using the PyTorch version on the `main` branch.

## License

[Add your license information here]

## Citation

If you use this code, please cite:
[Add citation information here]
