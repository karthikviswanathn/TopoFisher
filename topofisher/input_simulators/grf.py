"""
Created on Fri Mar 10 16:04:44 2023

@author: karthikviswanathan
"""
import powerbox as pbox
import numpy as np
from . import input_simulator

class GRFSimulator(input_simulator.FisherSimulator):
    def __init__(self, N, dim, boxlength, ensure_physical=False,
                 vol_normalised_power=True, xspace=True, name='grf'):
        """
        Initialize a Gaussian Random Field (GRF) simulator.

        Parameters:
        ----------
        N : int
            Number of grid points along each dimension.
        dim : int
            Number of dimensions (e.g., 2 for 2D, 3 for 3D).
        boxlength : float
            Size of the box (physical size) for the GRF.
        ensure_physical : bool, optional
            Whether to ensure physicality (default is False).
        vol_normalised_power : bool, optional
            Whether the power spectrum of the GRF is volume normalized (True) 
            or not (False).
        xspace : bool, optional
            Whether to simulate in real space (True) or Fourier space (False)
            (default is True).
        name : str, optional
            Name of the GRF simulator (default is 'grf').
        """
        self.N = N
        self.dim = dim
        self.boxlength = boxlength
        self.ensure_physical = ensure_physical
        self.vol_normalised_power = vol_normalised_power
        self.xspace = xspace
        super().__init__(name)
       
    def generateInstances(self, θ, num_sims, seed=None):
        """
        Simulate 'num_sims' Gaussian Random Fields (GRFs) for a given 
        fiducial parameter.

        Parameters:
        ----------
        θ : tf.Tensor
            The fiducial value for which the GRFs are simulated. It corresponds
            to (A, B), the amplitude and the slope of the power spectrum of the
            GRFs.
        num_sims : int
            The number of GRF instances to generate.
        seed : int, optional
            Seed for random number generation used for the generating the 
            GRFs.

        Returns:
        ----------
        grfs : np.ndarray
            An array of shape (num_sims, N, N) if dim = 2, containing the 
            simulated GRFs.
        """
        a, b = θ
        if(seed is not None):
            np.random.seed(seed)
        seeds = np.random.randint(1e7, size=(num_sims))
        grfs_list = []
        for idx in range(num_sims):
            pb = pbox.PowerBox(
                N=self.N,
                dim=self.dim,
                pk=lambda k: a * k**(-b),
                boxlength=self.boxlength,
                vol_normalised_power=self.vol_normalised_power,
                ensure_physical=self.ensure_physical,
                seed=seeds[idx]
            )
            grfs_list.append(pb.delta_x()) if(self.xspace) \
                else grfs_list.append(pb.delta_k())
        return np.array(grfs_list)
    
    def TFM(self, θ):
        """
        Calculate the theoretical Fisher information matrix for a given 
        fiducial parameter.

        Parameters:
        ----------
        θ : tf.Tensor
            The fiducial value for which the theoretical Fisher matrix is computed.

        Returns:
        ----------
        F : np.ndarray
            The Fisher information matrix.
        """
        if (self.dim != 2) : 
            raise ValueError("This method is implemented only for 2D GRFs.")
        A, B = θ
        pb = pbox.PowerBox(
            N=self.N,
            dim=self.dim,
            pk=lambda k: A * k**(-B),
            boxlength=self.boxlength,
            vol_normalised_power=self.vol_normalised_power,
            ensure_physical=self.ensure_physical,
        )
        N = self.N
        k = pb.k()
        k_center = k[1:N//2, 0:N].flatten()
        k_b1 = k[0, 0:N//2].flatten()
        k_b2 = k[N//2, 0:N//2].flatten()
        k = np.concatenate([k_center, k_b1, k_b2])
        Pk = A * k**(-B)
    
        Cinv = np.diag(1. / Pk)
        C_A =  np.diag(k ** -B)
        C_B =  np.diag(- Pk * np.log(k))
    
        F_AA = 0.5 * np.trace((C_A @ Cinv @ C_A @ Cinv))
        F_AB = 0.5 * np.trace((C_A @ Cinv @ C_B @ Cinv))
        F_BA = 0.5 * np.trace((C_B @ Cinv @ C_A @ Cinv))
        F_BB = 0.5 * np.trace((C_B @ Cinv @ C_B @ Cinv))
    
        return np.array([[F_AA, F_AB], [F_BA, F_BB]])
