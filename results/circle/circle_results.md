# Circle Experiment Results

## Fisher Information Summary

| Method | log\|F\| | σ(r) | σ(σ) |
|--------|---------|------|------|
| Theoretical | 15.82 | 0.0216 | 0.0170 |
| k=10 Alpha-DTM | 14.17 | 0.0376 | 0.0243 |
| k=10 Learnable | 13.59 | 0.0412 | 0.0306 |
| k=20 Alpha-DTM | 14.36 | 0.0331 | 0.0255 |
| k=20 Learnable | 13.89 | 0.0366 | 0.0300 |
| k=40 Alpha-DTM | 14.40 | 0.0277 | 0.0282 |
| k=40 Learnable | 13.97 | 0.0325 | 0.0307 |

## Experiment Details

- **Simulator**: NoisyRingSimulator with ncirc=100, nback=10, bgm_avg=1.0
- **Fiducial parameters**: θ = (r_mean=1.0, r_std=0.2)
- **Delta θ**: (0.05, 0.01)
- **Sample sizes**: n_s=10,000 (DTM), n_s=40,000 (Learnable)

## Best Learnable Models Selected

| k | Experiment | Architecture | Learning Rate |
|---|------------|--------------|---------------|
| 10 | k10_lr1e-2_h4k-2k | [40, 20, 1] | 1e-2 |
| 20 | k20_lr1e-2_h4k-2k | [80, 40, 1] | 1e-2 |
| 40 | k40_lr5e-3_hk | [40, 1] | 5e-3 |

## Files

- `results.json` - Raw results data
- `filtration_comparison.png` - Visualization comparing DTM, Learned, and Random filtrations
- `validation_curve_k40.png` - Training convergence for k=40 learnable filtration
