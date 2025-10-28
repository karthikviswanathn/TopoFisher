# Fisher Information Experiment Results

## N=16 GRF Results

| Method | log\|F\| | σ(A) | σ(B) | F[0,0] | F[0,1] | F[1,1] | Efficiency | Comments |
|--------|----------|------|------|--------|--------|--------|------------|----------|
| **Theoretical** | 6.73 | 0.2658 | 0.2768 | 64.00 | -54.23 | 59.00 | 100% | GRF N=16, θ=[1.0, 2.0] |
| **Fourier + Linear** | 6.59 | 0.2574 | 0.2773 | 55.82 | -44.25 | 48.09 | 78.7% | Fourier space analysis with linear compression |
| **Field+Grad 1PH + MOPED** | 5.84 | 0.3176 | 0.3763 | 48.59 | -36.59 | 34.62 | 41.1% | ✓ k_field(25,30), k_grad(30,35), n_s=10000, n_d=10000 |
| **1PH + MOPED** | 5.26 | 0.3403 | 0.3602 | 24.95 | -19.07 | 22.28 | 23.0% | ✓ k_H0=20, k_H1=25, n_s=10000, n_d=10000 |
| **Field+Grad+MMA + MOPED** | 5.20 | 0.4090 | 0.4753 | 41.06 | -32.65 | 30.40 | 21.8% | ✓ k_field(25,30), k_grad(30,35), k_mma(100,50), n_s=10000, n_d=10000 |
| **MMA + MOPED** | 3.85 | 0.4401 | 0.6685 | 20.92 | -11.96 | 9.07 | 5.6% | ✗ k_H0=100, k_H1=50, n_s=10000, n_d=10000 |

**Notes:**
- ✓/✗ indicates whether compressed features pass Gaussianity test (Kolmogorov-Smirnov, α=0.05)
- Efficiency = exp(log|F_empirical| - log|F_theory|) shows fraction of theoretical Fisher information recovered
- MOPED compression uses 50% train / 50% test split
- All methods use proper train/test splitting for Fisher estimation
