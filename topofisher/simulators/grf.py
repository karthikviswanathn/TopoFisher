"""
Gaussian Random Field (GRF) simulator.
"""
from typing import Optional
import torch
import numpy as np

try:
    import powerbox as pbox
except ImportError:
    raise ImportError("powerbox is required for GRF simulation. Install with: pip install powerbox")

from ..core.interfaces import Simulator


class GRFSimulator(Simulator):
    """
    Gaussian Random Field simulator using power-law power spectrum.

    The power spectrum is: P(k) = A * k^(-B)
    where A is amplitude and B is slope.
    """

    def __init__(
        self,
        N: int,
        dim: int = 2,
        boxlength: float = None,
        ensure_physical: bool = False,
        vol_normalised_power: bool = True,
        device: str = "cpu"
    ):
        """
        Initialize GRF simulator.

        Args:
            N: Number of grid points along each dimension
            dim: Number of dimensions (2 or 3)
            boxlength: Physical size of the simulation box (default: N)
            ensure_physical: Whether to ensure physicality of the field
            vol_normalised_power: Whether power spectrum is volume normalized
            device: Device to place generated tensors on ('cpu' or 'cuda')
        """
        self.N = N
        self.dim = dim
        # TODO: Accommodate other boxlength values. Currently fixed to N for consistency
        # between real-space and Fourier-space sampling (dx = dk = 1).
        self.boxlength = float(N) if boxlength is None else boxlength
        self.ensure_physical = ensure_physical
        self.vol_normalised_power = vol_normalised_power
        self.device = device

    def generate(
        self,
        theta: torch.Tensor,
        n_samples: int,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate GRF samples.

        Calls parent class generate() and moves result to device.

        Args:
            theta: Parameter tensor [A, B] where A is amplitude, B is slope
            n_samples: Number of GRF realizations to generate
            seed: Random seed for reproducibility

        Returns:
            Tensor of shape (n_samples, N, N) for 2D or (n_samples, N, N, N) for 3D
        """
        # Validate parameters
        if theta.numel() != 2:
            raise ValueError(f"Expected 2 parameters [A, B], got {theta.numel()}")

        # Call parent class generate (handles seeding, looping, stacking)
        result = super().generate(theta, n_samples, seed)

        # Move to device
        return result.to(self.device)

    def generate_single(self, theta: torch.Tensor, seed: int):
        """
        Generate a single GRF realization.

        Args:
            theta: Parameter tensor [A, B] where A is amplitude, B is slope
            seed: Random seed for this sample

        Returns:
            Numpy array of shape (N, N) for 2D or (N, N, N) for 3D
        """
        # Extract parameters
        A = float(theta[0])
        B = float(theta[1])

        # Generate single GRF
        pb = pbox.PowerBox(
            N=self.N,
            dim=self.dim,
            pk=lambda k: A * k**(-B),
            boxlength=self.boxlength,
            vol_normalised_power=self.vol_normalised_power,
            ensure_physical=self.ensure_physical,
            seed=seed
        )
        grf = pb.delta_x()  # Real space field (numpy array)

        return grf

    def theoretical_fisher_matrix(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Calculate the theoretical Fisher information matrix for GRF.

        This is the Fisher matrix for the power spectrum parameters [A, B]
        assuming Gaussian fields.

        Args:
            theta: Parameter tensor [A, B] where A is amplitude, B is slope

        Returns:
            Theoretical Fisher matrix of shape (2, 2)

        Raises:
            ValueError: If dim is not 2 (only implemented for 2D)
        """
        if self.dim != 2:
            raise ValueError("Theoretical Fisher matrix only implemented for 2D GRFs")

        if theta.numel() != 2:
            raise ValueError(f"Expected 2 parameters [A, B], got {theta.numel()}")

        A = float(theta[0])
        B = float(theta[1])

        # Create powerbox with same settings as simulation
        pb = pbox.PowerBox(
            N=self.N,
            dim=self.dim,
            pk=lambda k: A * k**(-B),
            boxlength=self.boxlength,
            vol_normalised_power=self.vol_normalised_power,
            ensure_physical=self.ensure_physical,
        )

        # Get k values (Fourier modes)
        # Use simplified k-mode extraction to match old behavior
        k = pb.k()
        k_modes = k[0:self.N//2, 0:self.N].flatten()

        # Power spectrum at these modes
        Pk = A * k_modes**(-B)

        # Inverse covariance (diagonal in Fourier space)
        Cinv = np.diag(1.0 / Pk)

        # Derivatives of covariance w.r.t. parameters
        C_A = np.diag(k_modes ** (-B))  # dC/dA
        C_B = np.diag(-Pk * np.log(k_modes))  # dC/dB

        # Fisher matrix: F_ij = 0.5 * Tr(C^-1 dC/di C^-1 dC/dj)
        F_AA = 0.5 * np.trace(C_A @ Cinv @ C_A @ Cinv)
        F_AB = 0.5 * np.trace(C_A @ Cinv @ C_B @ Cinv)
        F_BA = 0.5 * np.trace(C_B @ Cinv @ C_A @ Cinv)
        F_BB = 0.5 * np.trace(C_B @ Cinv @ C_B @ Cinv)

        fisher_matrix = np.array([[F_AA, F_AB], [F_BA, F_BB]])

        return torch.tensor(fisher_matrix, dtype=torch.float32)
