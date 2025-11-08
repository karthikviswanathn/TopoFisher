"""
Test MMA Layer - Validation against 1PH

This test validates that MMA layer produces correct results by comparing with 1PH:
1. Test with constant gradient + horizontal slice should match pure 1PH
2. Test diagonal slice should match 1PH(field+gradient) within error bound 2δ

The gradient is computed numerically from the field using finite differences
(magnitude of spatial gradient).

This test should be placed in: topofisher/tests/test_mma_validation.py
"""
import numpy as np
import torch
import gudhi as gd
from gudhi.wasserstein import wasserstein_distance

# Import from topofisher
from topofisher import GRFSimulator
from topofisher.filtrations.mma import MMALayer


# ============================================================================
# TEST CONFIGURATION
# ============================================================================
DELTA = 0.01  # Error threshold for MMA approximation
TEST_TOLERANCE = 2 * DELTA  # Maximum allowed error: 2δ
GRID_SIZE = 32
BOX_LENGTH = 1.0
THETA_FIDUCIAL = [1.0, 2.0]  # GRF parameters [A, B]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def compute_gradient_magnitude(field):
    """
    Compute gradient magnitude from field using finite differences.
    
    Uses central differences for interior points, forward/backward 
    differences at boundaries.
    
    Args:
        field: Tensor of shape (1, H, W) or (H, W)
        
    Returns:
        Gradient magnitude tensor of same shape as input
    """
    if field.ndim == 3:
        # Remove batch dimension for computation
        field_2d = field[0]
        add_batch = True
    else:
        field_2d = field
        add_batch = False
    
    H, W = field_2d.shape
    
    # Initialize gradient components
    grad_x = torch.zeros_like(field_2d)
    grad_y = torch.zeros_like(field_2d)
    
    # Central differences for interior points
    grad_x[:, 1:-1] = (field_2d[:, 2:] - field_2d[:, :-2]) / 2.0
    grad_y[1:-1, :] = (field_2d[2:, :] - field_2d[:-2, :]) / 2.0
    
    # Forward difference at left/top boundaries
    grad_x[:, 0] = field_2d[:, 1] - field_2d[:, 0]
    grad_y[0, :] = field_2d[1, :] - field_2d[0, :]
    
    # Backward difference at right/bottom boundaries
    grad_x[:, -1] = field_2d[:, -1] - field_2d[:, -2]
    grad_y[-1, :] = field_2d[-1, :] - field_2d[-2, :]
    
    # Compute magnitude
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    
    if add_batch:
        gradient_magnitude = gradient_magnitude.unsqueeze(0)
    
    return gradient_magnitude


def freudenthal_triangulation(field_2d):
    """
    Freudenthal triangulation for computing 1PH.
    Same implementation as in MMALayer._freudenthal_triangulation.
    """
    height, width = field_2d.shape
    simplices = []
    idx = lambda i, j: i * width + j
    
    # Vertices
    for i in range(height):
        for j in range(width):
            simplices.append(([idx(i, j)], field_2d[i, j].item()))
    
    # Horizontal edges
    for i in range(height):
        for j in range(width - 1):
            v1, v2 = idx(i, j), idx(i, j + 1)
            filt = max(field_2d[i, j].item(), field_2d[i, j + 1].item())
            simplices.append(([v1, v2], filt))
    
    # Vertical edges
    for i in range(height - 1):
        for j in range(width):
            v1, v2 = idx(i, j), idx(i + 1, j)
            filt = max(field_2d[i, j].item(), field_2d[i + 1, j].item())
            simplices.append(([v1, v2], filt))
    
    # Diagonal edges and triangles
    for i in range(height - 1):
        for j in range(width - 1):
            v_bl = idx(i, j)
            v_br = idx(i, j + 1)
            v_tl = idx(i + 1, j)
            v_tr = idx(i + 1, j + 1)
            
            filt_diag = max(field_2d[i, j].item(), field_2d[i + 1, j + 1].item())
            simplices.append(([v_bl, v_tr], filt_diag))
            
            filt_tri1 = max(
                field_2d[i, j].item(), 
                field_2d[i, j + 1].item(), 
                field_2d[i + 1, j + 1].item()
            )
            simplices.append(([v_bl, v_br, v_tr], filt_tri1))
            
            filt_tri2 = max(
                field_2d[i, j].item(), 
                field_2d[i + 1, j].item(), 
                field_2d[i + 1, j + 1].item()
            )
            simplices.append(([v_bl, v_tl, v_tr], filt_tri2))
    
    return simplices


def compute_1ph(field_2d):
    """Compute standard 1-parameter persistent homology."""
    simplices = freudenthal_triangulation(field_2d)
    
    st = gd.SimplexTree()
    for simplex, filt in simplices:
        st.insert(simplex, filtration=filt)
    
    st.persistence()
    diagram = list(st.persistence())
    
    pd_H0 = np.array([p[1] for p in diagram if p[0] == 0])
    pd_H1 = np.array([p[1] for p in diagram if p[0] == 1])
    
    return pd_H0, pd_H1


# ============================================================================
# TEST FUNCTIONS
# ============================================================================
def test_constant_gradient_horizontal_slice():
    """
    TEST 1: Constant gradient + horizontal slice should match pure 1PH.
    
    When gradient is constant, a horizontal slice of MMA should exactly
    reproduce the standard 1PH of the field.
    """
    print("\n" + "=" * 80)
    print("TEST 1: Constant Gradient + Horizontal Slice vs Pure 1PH")
    print("=" * 80)
    
    # Generate GRF data
    print("\n[1.1] Generating GRF data...")
    simulator = GRFSimulator(N=GRID_SIZE, dim=2, boxlength=BOX_LENGTH)
    data = simulator.generate(torch.tensor(THETA_FIDUCIAL), n_samples=1, seed=42)
    field = data[0]  # Shape: (1, N, N)
    
    print(f"  Field shape: {field.shape}")
    print(f"  Field range: [{field.min():.3f}, {field.max():.3f}]")
    
    # Compute pure 1PH on field
    print("\n[1.2] Computing pure 1PH on field...")
    pd_H0_1ph, pd_H1_1ph = compute_1ph(field)
    print(f"  1PH: H0={len(pd_H0_1ph)} bars, H1={len(pd_H1_1ph)} bars")
    
    # Compute MMA with constant gradient
    print("\n[1.3] Computing MMA with constant gradient=1...")
    gradient_constant = torch.ones_like(field)
    
    mma_layer = MMALayer(nlines=500)
    mma_obj = mma_layer(field, gradient_constant)[0]
    
    # Extract horizontal slice
    print("\n[1.4] Extracting horizontal slice from MMA...")
    basepoint = [field.min().item() - 1.0, 1.5]
    direction = [1, 0]  # Horizontal
    
    barcode_horizontal = mma_obj.barcode2(
        basepoint, 
        direction=direction, 
        degree=-1, 
        threshold=False, 
        keep_inf=False
    )
    
    bars_h0_param = barcode_horizontal[0]
    bars_h1_param = barcode_horizontal[1] if len(barcode_horizontal) > 1 else np.empty((0, 2))
    
    # Reparametrize
    bars_h0 = bars_h0_param.copy()
    bars_h0[:, 0] = basepoint[0] + bars_h0_param[:, 0]
    bars_h0[:, 1][np.isfinite(bars_h0_param[:, 1])] = \
        basepoint[0] + bars_h0_param[:, 1][np.isfinite(bars_h0_param[:, 1])]
    
    bars_h1 = bars_h1_param.copy()
    if len(bars_h1) > 0:
        bars_h1[:, 0] = basepoint[0] + bars_h1_param[:, 0]
        bars_h1[:, 1][np.isfinite(bars_h1_param[:, 1])] = \
            basepoint[0] + bars_h1_param[:, 1][np.isfinite(bars_h1_param[:, 1])]
    
    print(f"  MMA slice: H0={len(bars_h0)} bars, H1={len(bars_h1)} bars")
    
    # Compare
    print("\n[1.5] Computing distances (should be ≈ 0)...")
    test_pass = True
    
    if len(bars_h0) > 0 and len(pd_H0_1ph) > 0:
        bottleneck_H0 = gd.bottleneck_distance(pd_H0_1ph, bars_h0)
        wasserstein_H0 = wasserstein_distance(pd_H0_1ph, bars_h0, order=1)
        print(f"  H0: Bottleneck={bottleneck_H0:.6f}, 1-Wasserstein={wasserstein_H0:.6f}")
        if bottleneck_H0 > 1e-6:
            print(f"  ⚠ H0 distance larger than expected")
            test_pass = False
    
    if len(bars_h1) > 0 and len(pd_H1_1ph) > 0:
        bottleneck_H1 = gd.bottleneck_distance(pd_H1_1ph, bars_h1)
        wasserstein_H1 = wasserstein_distance(pd_H1_1ph, bars_h1, order=1)
        print(f"  H1: Bottleneck={bottleneck_H1:.6f}, 1-Wasserstein={wasserstein_H1:.6f}")
        if bottleneck_H1 > 1e-6:
            print(f"  ⚠ H1 distance larger than expected")
            test_pass = False
    
    print(f"\n[TEST 1 RESULT] {'✓ PASSED' if test_pass else '✗ FAILED'}")
    return test_pass


def test_diagonal_slice_with_gradient():
    """
    TEST 2: Diagonal slice with gradient vs 1PH(field+gradient).
    
    A diagonal slice of MMA computed with gradient (computed numerically from field)
    should approximate 1PH computed on the combined filtration max(field, gradient).
    Error should be bounded by 2δ.
    
    The gradient is computed as the magnitude of the spatial gradient using finite
    differences.
    """
    print("\n" + "=" * 80)
    print("TEST 2: Diagonal Slice vs 1PH(field+gradient)")
    print("=" * 80)
    print(f"Expected: Distance ≤ 2δ = {TEST_TOLERANCE}")
    
    # Generate GRF data with gradient
    print("\n[2.1] Generating GRF data and computing gradient...")
    simulator = GRFSimulator(N=GRID_SIZE, dim=2, boxlength=BOX_LENGTH)
    data = simulator.generate(torch.tensor(THETA_FIDUCIAL), n_samples=1, seed=42)
    field = data[0]
    
    # Compute gradient magnitude numerically from field
    gradient = compute_gradient_magnitude(field)
    
    print(f"  Field shape: {field.shape}")
    print(f"  Field range: [{field.min():.3f}, {field.max():.3f}]")
    print(f"  Gradient shape: {gradient.shape}")
    print(f"  Gradient range: [{gradient.min():.3f}, {gradient.max():.3f}]")
    
    # Compute MMA with max_error=delta
    print(f"\n[2.2] Computing MMA with gradient (max_error={DELTA})...")
    mma_layer = MMALayer(max_error=DELTA)
    mma_obj = mma_layer(field, gradient)[0]
    
    # Extract diagonal slice
    print("\n[2.3] Extracting diagonal slice from MMA...")
    basepoint_mma = [-3, -3]
    direction_mma = [1, 1]  # Diagonal
    
    barcode_diag = mma_obj.barcode2(
        basepoint_mma, 
        direction=direction_mma, 
        degree=-1, 
        threshold=False, 
        keep_inf=False
    )
    
    bars_h0_diag_param = barcode_diag[0]
    bars_h1_diag_param = barcode_diag[1] if len(barcode_diag) > 1 else np.empty((0, 2))
    
    # Reparametrize MMA slice
    bars_h0_diag = bars_h0_diag_param.copy()
    bars_h0_diag[:, 0] = basepoint_mma[0] + bars_h0_diag_param[:, 0]
    bars_h0_diag[:, 1][np.isfinite(bars_h0_diag_param[:, 1])] = \
        basepoint_mma[0] + bars_h0_diag_param[:, 1][np.isfinite(bars_h0_diag_param[:, 1])]
    
    bars_h1_diag = bars_h1_diag_param.copy()
    if len(bars_h1_diag) > 0:
        bars_h1_diag[:, 0] = basepoint_mma[0] + bars_h1_diag_param[:, 0]
        bars_h1_diag[:, 1][np.isfinite(bars_h1_diag_param[:, 1])] = \
            basepoint_mma[0] + bars_h1_diag_param[:, 1][np.isfinite(bars_h1_diag_param[:, 1])]
    
    print(f"  MMA diagonal slice: H0={len(bars_h0_diag)} bars, H1={len(bars_h1_diag)} bars")
    
    # Compute 1PH on combined filtration
    print("\n[2.4] Computing 1PH on combined filtration (field+gradient)...")
    basepoint_1ph = [0, basepoint_mma[1] - basepoint_mma[0]]
    new_filt_1d = torch.maximum(field - basepoint_1ph[0], gradient - basepoint_1ph[1])
    
    pd_H0_1d, pd_H1_1d = compute_1ph(new_filt_1d)
    print(f"  1PH combined: H0={len(pd_H0_1d)} bars, H1={len(pd_H1_1d)} bars")
    
    # Compare
    print("\n[2.5] Computing distances...")
    print(f"  Threshold: ≤ 2δ = {TEST_TOLERANCE}")
    
    test_pass = True
    
    if len(bars_h0_diag) > 0 and len(pd_H0_1d) > 0:
        bottleneck_H0 = gd.bottleneck_distance(pd_H0_1d, bars_h0_diag)
        wasserstein_H0 = wasserstein_distance(pd_H0_1d, bars_h0_diag, order=1)
        print(f"  H0: Bottleneck={bottleneck_H0:.6f}, 1-Wasserstein={wasserstein_H0:.6f}")
        if bottleneck_H0 > TEST_TOLERANCE:
            print(f"  ✗ H0 exceeds threshold")
            test_pass = False
        else:
            print(f"  ✓ H0 within threshold")
    
    if len(bars_h1_diag) > 0 and len(pd_H1_1d) > 0:
        bottleneck_H1 = gd.bottleneck_distance(pd_H1_1d, bars_h1_diag)
        wasserstein_H1 = wasserstein_distance(pd_H1_1d, bars_h1_diag, order=1)
        print(f"  H1: Bottleneck={bottleneck_H1:.6f}, 1-Wasserstein={wasserstein_H1:.6f}")
        if bottleneck_H1 > TEST_TOLERANCE:
            print(f"  ✗ H1 exceeds threshold")
            test_pass = False
        else:
            print(f"  ✓ H1 within threshold")
    
    print(f"\n[TEST 2 RESULT] {'✓ PASSED' if test_pass else '✗ FAILED'}")
    return test_pass


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Run all MMA validation tests."""
    print("=" * 80)
    print("MMA LAYER VALIDATION TESTS")
    print("=" * 80)
    
    test1_pass = test_constant_gradient_horizontal_slice()
    test2_pass = test_diagonal_slice_with_gradient()
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Test 1 (Constant gradient): {'✓ PASSED' if test1_pass else '✗ FAILED'}")
    print(f"Test 2 (Diagonal slice):    {'✓ PASSED' if test2_pass else '✗ FAILED'}")
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if (test1_pass and test2_pass) else '✗ SOME TESTS FAILED'}")
    print("=" * 80)
    
    return test1_pass and test2_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
