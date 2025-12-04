# GRF Experiment Results

## Fisher Information Summary

| Method | log\|F\| | σ(A) | σ(B) |
|--------|---------|------|------|
| Theoretical | 6.73 | 0.2658 | 0.2768 |
| Cubical (raw field) | 4.77 | 0.3939 | 0.4238 |
| Cubical (learned transform) | 5.94 | 0.3092 | 0.3376 |

## Experiment Details

- **Simulator**: GRFSimulator with N=16, dim=2, boxlength=16
- **Power spectrum**: P(k) = A × k^(-B)
- **Fiducial parameters**: θ = (A=1.0, B=2.0)
- **Delta θ**: (0.1, 0.2)

### Methods

1. **Theoretical**: Analytical Fisher matrix from Gaussian field theory
2. **Cubical (raw field)**: CubicalLayer persistence on raw GRF field
   - Vectorization: TopK (k=10 H0, k=9 H1)
   - Compression: MOPED
   - Samples: n_s=10,000, n_d=10,000

3. **Cubical (learned transform)**: LearnableFiltration CNN followed by CubicalLayer
   - CNN architecture: [1→32→64→32→1] with 3×3 convolutions
   - Vectorization: TopK (k=7 H0, k=5 H1)
   - Compression: MOPED
   - Samples: n_s=40,000, n_d=40,000
   - Training: lr=3e-2, best step 540

## Correlation Structure

| Method | ρ(A,B) |
|--------|--------|
| Theoretical | 0.883 |
| Cubical (raw field) | 0.835 |
| Cubical (learned transform) | 0.870 |

## Files

- `results.json` - Raw results data
- `fisher_contours.png` - Fisher confidence ellipses comparison (68%)
- `validation_curve.png` - Training convergence for learnable filtration
