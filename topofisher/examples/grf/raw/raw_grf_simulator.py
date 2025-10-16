"""
Specialized GRF simulator for raw/analytical summaries.

This simulator extends GRFSimulator to optionally return Fourier modes
or power spectra directly, instead of real-space grids.
"""
import numpy as np
import torch

try:
    import powerbox as pbox
    from powerbox import dft
except ImportError:
    raise ImportError("powerbox is required for GRF simulation. Install with: pip install powerbox")

from topofisher import GRFSimulator


class RawGRFSimulator(GRFSimulator):
    """
    GRF simulator that can return different representations.

    Extends base GRFSimulator to optionally return:
    - Fourier coefficient magnitudes |delta_k|
    - Radially-binned power spectrum P(k)
    - Original real-space grids (default)
    """

    def __init__(
        self,
        N: int,
        dim: int = 2,
        boxlength: float = None,
        ensure_physical: bool = False,
        vol_normalised_power: bool = True,
        device: str = "cpu",
        return_type: str = "realspace",
        n_power_bins: int = 20
    ):
        """
        Initialize raw GRF simulator.

        Args:
            N: Number of grid points along each dimension
            dim: Number of dimensions (2 or 3)
            boxlength: Physical size of the simulation box (default: N)
            ensure_physical: Whether to ensure physicality of the field
            vol_normalised_power: Whether power spectrum is volume normalized
            device: Device to place generated tensors on
            return_type: What format to return from generate_single():
                - 'realspace': delta_x grid (default, shape (N, N))
                - 'fourier': |delta_k| magnitudes from pb.delta_k() (flattened, shape (N*N,))
                - 'fourier_recovered': |delta_k| magnitudes from FFT(delta_x) (flattened, shape (N*N,))
                - 'power_spectrum': P(k) binned (shape (n_bins,))
            n_power_bins: Number of radial bins for power spectrum
        """
        # TODO: Accommodate other boxlength values. Currently defaults to N for consistency.
        # Initialize parent (which will set boxlength = N if None)
        super().__init__(
            N=N,
            dim=dim,
            boxlength=boxlength,
            ensure_physical=ensure_physical,
            vol_normalised_power=vol_normalised_power,
            device=device
        )

        self.return_type = return_type
        self.n_power_bins = n_power_bins

        # Validate return_type
        valid_types = ['realspace', 'fourier', 'fourier_recovered', 'power_spectrum']
        if return_type not in valid_types:
            raise ValueError(f"return_type must be one of {valid_types}, got '{return_type}'")

    def generate_single(self, theta: torch.Tensor, seed: int):
        """
        Generate a single GRF realization in specified format.

        Args:
            theta: Parameter tensor [A, B] where A is amplitude, B is slope
            seed: Random seed for this sample

        Returns:
            Numpy array in format specified by return_type:
            - 'realspace': shape (N, N) or (N, N, N)
            - 'fourier': shape (N*N,) or (N*N*N,)
            - 'fourier_recovered': shape (N*N,) or (N*N*N,)
            - 'power_spectrum': shape (n_power_bins,)
        """
        # Extract parameters
        A = float(theta[0])
        B = float(theta[1])

        # Create PowerBox
        pb = pbox.PowerBox(
            N=self.N,
            dim=self.dim,
            pk=lambda k: A * k**(-B),
            boxlength=self.boxlength,
            vol_normalised_power=self.vol_normalised_power,
            ensure_physical=self.ensure_physical,
            seed=seed
        )

        # Return based on type
        if self.return_type == 'realspace':
            return pb.delta_x()

        elif self.return_type == 'fourier':
            # Get Fourier coefficients (complex)
            delta_k = pb.delta_k()
            # Return magnitudes, flattened
            magnitudes = np.absolute(delta_k).flatten()
            return magnitudes

        elif self.return_type == 'fourier_recovered':
            # Get real-space field
            delta_x = pb.delta_x()
            # Compute Fourier coefficients via FFT
            # Following PowerBox convention: delta_x = V * ifft(delta_k)
            # So: delta_k = fft(delta_x / V)
            delta_k, _ = dft.fft(
                delta_x / pb.V,
                L=pb.boxlength,
                a=pb.fourier_a,
                b=pb.fourier_b
            )
            # Return magnitudes, flattened
            magnitudes = np.absolute(delta_k).flatten()
            return magnitudes

        elif self.return_type == 'power_spectrum':
            # Get real-space field first
            delta_x = pb.delta_x()

            # Compute radially-binned power spectrum
            p_k_field, _ = pbox.get_power(
                deltax=delta_x,
                boxlength=self.boxlength
            )

            # Return only P(k) values (bins not needed)
            return p_k_field

        else:
            raise ValueError(f"Unknown return_type: {self.return_type}")

    def __repr__(self):
        return (f"RawGRFSimulator(N={self.N}, dim={self.dim}, device={self.device}, "
                f"return_type='{self.return_type}')")
