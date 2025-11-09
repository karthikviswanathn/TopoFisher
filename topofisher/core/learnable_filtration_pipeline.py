"""
Learnable filtration pipeline for Fisher information maximization.

This module implements a specialized pipeline for training learnable filtrations
(CNN-based field transformations) to maximize Fisher information. Key differences
from standard pipeline:

1. Data regeneration: Fresh samples each epoch (filtration changes)
2. Adaptive TopK: Automatically adjusts k based on minimum diagram size
3. End-to-end training: Gradients flow through CNN → persistence → Fisher loss
4. MOPED compression: Analytical optimal compression

Default setup: N=8 → N=16 upscale (can scale to N=16 → N=32 if successful)
"""
from typing import List, Optional, Dict
import torch
import torch.nn as nn
import numpy as np
import time

from .data_types import FisherConfig, FisherResult
from .pipeline import FisherPipeline
from topofisher.filtrations.learnable import LearnableFiltration
from topofisher.compressions.moped import MOPEDCompression
from topofisher.fisher.analyzer import FisherAnalyzer


class LearnableFiltrationPipeline(FisherPipeline):
    """    
    Pipeline for training learnable filtrations to maximize Fisher information.

    Pipeline:
        Simulator → LearnableFiltration (CNN → Persistence) → Vectorization → MOPED → Fisher

    Training loop:
        - Regenerate data each epoch (since CNN filtration changes)
        - Compute Fisher information
        - Minimize -log|F| to maximize Fisher determinant
        - Gradients flow back to CNN parameters

    Example:
        >>> from topofisher import (
        ...     GRFSimulator, LearnableFiltration, CombinedVectorization, TopKLayer
        ... )
        >>>
        >>> # Create components
        >>> simulator = GRFSimulator(N=16, dim=2)
        >>> filtration = LearnableFiltration(input_size=16, homology_dimensions=[0, 1])
        >>> vectorization = CombinedVectorization([TopKLayer(k=10), TopKLayer(k=10)])
        >>>
        >>> # Create pipeline
        >>> pipeline = LearnableFiltrationPipeline(
        ...     simulator=simulator,
        ...     filtration=filtration,
        ...     vectorization=vectorization
        ... )
        >>>
        >>> # Train
        >>> config = FisherConfig(...)
        >>> result = pipeline.train(config, n_epochs=1000, lr=1e-3)
    """

    def __init__(
        self,
        simulator: nn.Module,
        filtration: LearnableFiltration,
        vectorization: nn.Module,
        use_moped: bool = True
    ):
        """
        Initialize learnable filtration pipeline.

        Args:
            simulator: Data simulator (e.g., GRFSimulator)
            filtration: Learnable filtration with CNN
            vectorization: Vectorization module (e.g., TopKLayer)
            use_moped: Whether to use MOPED compression (recommended)
        """
        # Initialize parent with placeholder compression (will be created during training)
        compression = MOPEDCompression() if use_moped else None
        fisher_analyzer = FisherAnalyzer(clean_data=True)

        super().__init__(
            simulator=simulator,
            filtration=filtration,
            vectorization=vectorization,
            compression=compression,
            fisher_analyzer=fisher_analyzer
        )

        self.use_moped = use_moped

    def train(
        self,
        config: FisherConfig,
        n_epochs: int = 1000,
        lr: float = 1e-3,
        validate_every: int = 10,
        verbose: bool = True,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Train learnable filtration to maximize Fisher information.

        Args:
            config: Fisher configuration (theta_fid, delta_theta, n_s, n_d, seeds)
            n_epochs: Number of training epochs
            lr: Learning rate (weight_decay automatically set to lr/100)
            validate_every: Validation interval
            verbose: Print training progress
            save_path: Path to save best model (e.g., 'ai-code/models/model.pt')

        Returns:
            Dictionary with training history and final result
        """
        device = getattr(self.simulator, 'device', 'cpu')
        if isinstance(device, str):
            device = torch.device(device)

        # Move filtration to device
        self.filtration.to(device)

        # Compute weight decay as lr / 100 (standard practice for Adam)
        weight_decay = lr / 100

        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.filtration.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Training history
        history = {
            'train_losses': [],
            'val_losses': [],
            'fisher_dets': [],
            'epoch_times': [],
            'best_epoch': 0,
            'best_val_loss': float('inf'),
            'best_model_state': None
        }

        if verbose:
            print("=" * 80)
            print("Training Learnable Filtration")
            print("=" * 80)
            print(f"Filtration: {self.filtration.__class__.__name__}")
            print(f"Parameters: {self.filtration.get_num_parameters():,}")
            print(f"Epochs: {n_epochs}")
            print(f"Learning rate: {lr}")
            print(f"Weight decay: {weight_decay} (= lr/100)")

            # Print theoretical Fisher information if available
            if hasattr(self.simulator, 'theoretical_fisher_matrix'):
                try:
                    F_theory = self.simulator.theoretical_fisher_matrix(config.theta_fid)
                    log_det_theory = torch.logdet(F_theory).item()
                    print(f"Theoretical log|F|: {log_det_theory:.4f} (100% baseline)")
                except Exception as e:
                    print(f"Could not compute theoretical Fisher: {e}")

            print("=" * 80)
            print()

        # Training loop
        total_start_time = time.time()

        for epoch in range(n_epochs):
            epoch_start_time = time.time()

            # Ensure filtration is in training mode (enables dropout if used, gradient tracking)
            self.filtration.train()

            # Create training config with different seeds each epoch
            # Use base seed + epoch to ensure different data each time
            train_config = FisherConfig(
                theta_fid=config.theta_fid,
                delta_theta=config.delta_theta,
                n_s=config.n_s,
                n_d=config.n_d,
                find_derivative=config.find_derivative,
                seed_cov=config.seed_cov + epoch * 1000,  # Different seed each epoch
                seed_ders=[config.seed_ders[0] + epoch * 1000, config.seed_ders[1] + epoch * 1000]
            )

            # Forward pass: generate fresh data, apply pipeline
            # MOPED internally splits 2000→1000 for B computation, 1000 for Fisher
            result_train = super().run(train_config, training_config=None, verbose_gaussianity=False)
            loss = -result_train.log_det_fisher  # Minimize negative log det

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_time = time.time() - epoch_start_time

            history['train_losses'].append(loss.item())
            history['fisher_dets'].append(result_train.log_det_fisher.item())
            history['epoch_times'].append(epoch_time)

            # Print training progress every epoch
            if verbose:
                print(f"Epoch {epoch+1}/{n_epochs} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Time: {epoch_time:.1f}s", end="")

            # Validation (periodic, with fixed seeds)
            if (epoch + 1) % validate_every == 0:
                self.filtration.eval()
                with torch.no_grad():
                    result_val = self.validate(config, verbose=False)
                    val_loss = -result_val.log_det_fisher

                history['val_losses'].append(val_loss.item())

                # Save best model
                if val_loss.item() < history['best_val_loss']:
                    history['best_val_loss'] = val_loss.item()
                    history['best_epoch'] = epoch + 1
                    history['best_model_state'] = {
                        k: v.clone().cpu() for k, v in self.filtration.state_dict().items()
                    }

                if verbose:
                    print(f" | Val Loss: {val_loss.item():.4f}")
            else:
                if verbose:
                    print()  # Newline for non-validation epochs

        # Training complete
        total_time = time.time() - total_start_time
        history['total_time'] = total_time
        history['avg_epoch_time'] = np.mean(history['epoch_times'])

        if verbose:
            print(f"\nTraining completed in {total_time/60:.1f} minutes")
            print(f"Average time per epoch: {history['avg_epoch_time']:.1f}s")

        # Load best model
        if history['best_model_state'] is not None:
            self.filtration.load_state_dict(history['best_model_state'])
            if verbose:
                print(f"Loaded best model from epoch {history['best_epoch']}")

            # Save model to disk if path provided
            if save_path is not None:
                # Ensure parent directory exists
                import os
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                # Save model state and metadata
                save_dict = {
                    'filtration_state_dict': history['best_model_state'],
                    'best_epoch': history['best_epoch'],
                    'best_val_loss': history['best_val_loss'],
                    'config': {
                        'input_size': self.filtration.input_size,
                        'output_size': self.filtration.output_size,
                        'homology_dimensions': self.filtration.homology_dimensions,
                    },
                    'training_config': {
                        'n_epochs': n_epochs,
                        'lr': lr,
                        'weight_decay': weight_decay,
                    }
                }
                torch.save(save_dict, save_path)
                if verbose:
                    print(f"Saved best model to {save_path}")

        # Final evaluation
        if verbose:
            print("\n" + "=" * 80)
            print("Final Evaluation")
            print("=" * 80)

        final_result = self.evaluate(config, verbose=verbose)
        history['final_result'] = final_result

        return history

    def evaluate(
        self,
        config: FisherConfig,
        verbose: bool = True
    ) -> FisherResult:
        """
        Evaluate learnable filtration on fresh data.

        Uses parent's run() which handles everything:
        - Generate data (2000 samples)
        - Apply filtration
        - Vectorize
        - MOPED splits internally (1000 for B, 1000 for Fisher)
        - Compute Fisher

        Args:
            config: Fisher configuration
            verbose: Print evaluation results

        Returns:
            FisherResult with Fisher matrix and analysis
        """
        self.filtration.eval()

        with torch.no_grad():
            result = super().run(config, training_config=None, verbose_gaussianity=verbose)

            if verbose:
                print(f"log|F|: {result.log_det_fisher.item():.4f}")
                print(f"Constraints (1σ): {result.constraints}")

        return result

    def validate(
        self,
        config: FisherConfig,
        verbose: bool = True
    ) -> FisherResult:
        """
        Validate learnable filtration with fixed validation seeds.

        Uses n_s = n_d = 2000 and seeds [100, 101, 102] for reproducibility.

        Args:
            config: Fisher configuration (theta_fid and delta_theta will be used)
            verbose: Print validation results

        Returns:
            FisherResult with Fisher matrix and analysis
        """
        # Create validation config with fixed seeds
        val_config = FisherConfig(
            theta_fid=config.theta_fid,
            delta_theta=config.delta_theta,
            n_s=2000,
            n_d=2000,
            find_derivative=config.find_derivative,
            seed_cov=100,
            seed_ders=[101, 102]
        )

        if verbose:
            print("\n" + "=" * 80)
            print("Validation (seeds: 100, 101, 102 | n_s=n_d=2000)")
            print("=" * 80)

        return self.evaluate(val_config, verbose=verbose)

    def test(
        self,
        config: FisherConfig,
        verbose: bool = True
    ) -> FisherResult:
        """
        Test learnable filtration with fixed test seeds.

        Uses n_s = n_d = 2000 and seeds [200, 201, 202] for reproducibility.

        Args:
            config: Fisher configuration (theta_fid and delta_theta will be used)
            verbose: Print test results

        Returns:
            FisherResult with Fisher matrix and analysis
        """
        # Create test config with fixed seeds
        test_config = FisherConfig(
            theta_fid=config.theta_fid,
            delta_theta=config.delta_theta,
            n_s=2000,
            n_d=2000,
            find_derivative=config.find_derivative,
            seed_cov=200,
            seed_ders=[201, 202]
        )

        if verbose:
            print("\n" + "=" * 80)
            print("Test (seeds: 200, 201, 202 | n_s=n_d=2000)")
            print("=" * 80)

        return self.evaluate(test_config, verbose=verbose)
