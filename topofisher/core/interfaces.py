"""
Abstract base classes (interfaces) for TopoFisher components.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Callable
import torch
import torch.nn as nn
import numpy as np

from topofisher.core.data_types import FisherResult, TrainingConfig


class Simulator(ABC):
    """Base class for data simulators."""

    def generate(self, theta: torch.Tensor, n_samples: int, seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate n_samples at parameter value theta.

        This method handles seeding, looping, and tensor conversion.
        Child classes should implement generate_single() instead.

        Args:
            theta: Parameter values
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility (master seed)

        Returns:
            Tensor of shape (n_samples, *data_shape)
        """
        # Set master seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Generate individual seeds for each sample
        seeds = np.random.randint(1e7, size=n_samples)

        # Generate samples
        samples = []
        for i in range(n_samples):
            sample = self.generate_single(theta, seed=int(seeds[i]))
            samples.append(sample)

        # Stack into tensor
        if isinstance(samples[0], torch.Tensor):
            return torch.stack(samples)
        else:
            # Convert numpy arrays to torch
            samples_array = np.array(samples)
            return torch.from_numpy(samples_array).float()

    @abstractmethod
    def generate_single(self, theta: torch.Tensor, seed: int):
        """
        Generate a single sample at parameter value theta.

        Child classes must implement this method.

        Args:
            theta: Parameter values
            seed: Random seed for this specific sample

        Returns:
            Single sample tensor or array of shape (*data_shape)
        """
        pass


class Filtration(ABC):
    """Base class for computing persistence diagrams."""

    @abstractmethod
    def compute_diagrams(self, data: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Compute persistence diagrams from input data.

        Args:
            data: Input data of shape (n_samples, *data_shape)

        Returns:
            List of lists of persistence diagrams.
            Outer list: homology dimensions
            Inner list: diagrams for each sample
            Each diagram: tensor of shape (n_points, 2) with (birth, death) pairs
        """
        pass

# TODO: Change this to forward instead of fit and transform
class Vectorization(ABC):
    """Base class for vectorizing persistence diagrams."""

    @abstractmethod
    def fit(self, diagrams: List[torch.Tensor]) -> None:
        """
        Fit the vectorization to a set of diagrams (e.g., determine ranges).

        Args:
            diagrams: List of persistence diagrams
        """
        pass

    @abstractmethod
    def transform(self, diagrams: List[torch.Tensor]) -> torch.Tensor:
        """
        Transform persistence diagrams to feature vectors.

        Args:
            diagrams: List of persistence diagrams

        Returns:
            Tensor of shape (n_diagrams, n_features)
        """
        pass

    def fit_transform(self, diagrams: List[torch.Tensor]) -> torch.Tensor:
        """Convenience method to fit and transform in one call."""
        self.fit(diagrams)
        return self.transform(diagrams)


class Compression(nn.Module, ABC):
    """Base class for compression methods in the TopoFisher pipeline.

    Compression sits between vectorization and Fisher analysis:
        vectorization → compression → fisher_analyzer

    TODO (long-term): Consider extracting training logic (train_compression, _train_model,
    _validate_model) into a separate CompressionTrainer class or TrainingMixin to reduce
    base class complexity and improve separation of concerns. Current implementation works
    but violates Single Responsibility Principle (235 lines of training code in base class).
    """

    @abstractmethod
    def forward(
        self,
        summaries: List[torch.Tensor],
        delta_theta: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Apply compression to summaries.

        Args:
            summaries: List of summary tensors.
                summaries[0]: shape (n_s, n_features) at theta_fid (for covariance)
                summaries[1:]: shape (n_d, n_features) at perturbed values
                    Ordered as [theta_minus_0, theta_plus_0, theta_minus_1, theta_plus_1, ...]
            delta_theta: Optional step sizes for derivatives (needed for MOPED)

        Returns:
            List of compressed summary tensors with same structure as input
        """
        pass

    def is_trainable(self) -> bool:
        """
        Return True if this compression requires training.

        Returns:
            True if compression is trainable (e.g., MLP, CNN), False otherwise (e.g., MOPED, Identity)
        """
        return False

    def returns_test_only(self) -> bool:
        """
        Return True if this compression splits data and returns only the test set.

        When True, the compression:
        - Splits summaries internally into train/test
        - Learns compression on train set
        - Applies compression to test set
        - Returns only the test set (smaller than input)

        When False (default), the compression:
        - Applies compression to all input summaries
        - Returns all summaries (same size as input)

        Returns:
            True if compression returns test set only, False if returns all data
        """
        return False

    def is_initialized(self) -> bool:
        """
        Return True if compression network is initialized.

        For compressions with lazy initialization (e.g., MLP with input_dim=None),
        this returns False until initialize() is called.

        Returns:
            True if initialized and ready to use, False if needs initialization
        """
        return True

    def initialize(self, input_dim: int, output_dim: int) -> None:
        """
        Initialize compression network with inferred dimensions.

        For compressions that support lazy initialization, this builds the network
        once dimensions are known from data.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (typically n_params)
        """
        pass

    def train_compression(
        self,
        summaries: List[torch.Tensor],
        delta_theta: torch.Tensor,
        training_config: Optional[TrainingConfig] = None
    ) -> Optional[Dict]:
        """
        Train this compression to maximize Fisher information.

        Only trainable compressions can be trained. Non-trainable compressions
        (MOPED, Identity) return None immediately.

        Args:
            summaries: Pre-computed summaries [fid, minus_0, plus_0, minus_1, plus_1, ...]
            delta_theta: Step sizes for derivatives
            training_config: Training hyperparameters (uses defaults if None)

        Returns:
            Training history dict or None if compression is not trainable
        """
        if not self.is_trainable():
            print(f"Compression {type(self).__name__} is not trainable, skipping training")
            return None

        # Import here to avoid circular dependency
        from ..core.training_utils import split_data
        from ..fisher.gaussianity import test_gaussianity
        from ..fisher.analyzer import FisherAnalyzer

        # Use default training config if not provided
        if training_config is None:
            print("No TrainingConfig provided, using defaults:")
            training_config = TrainingConfig()

        print("Using TrainingConfig:")
        print(f"  n_epochs: {training_config.n_epochs}")
        print(f"  lr: {training_config.lr}")
        print(f"  batch_size: {training_config.batch_size}")
        print(f"  weight_decay: {training_config.weight_decay}")
        print(f"  train_frac: {training_config.train_frac}")
        print(f"  val_frac: {training_config.val_frac}")
        print(f"  validate_every: {training_config.validate_every}")
        print(f"  check_gaussianity: {training_config.check_gaussianity}")
        print()

        print("=" * 80)
        print("Training Compression")
        print("=" * 80)

        # Step 1: Initialize compression if needed
        if not self.is_initialized():
            input_dim = summaries[0].shape[1]
            output_dim = (len(summaries) - 1) // 2  # Infer n_params from summaries structure
            print(f"\n1. Initializing compression: input_dim={input_dim}, output_dim={output_dim}")
            self.initialize(input_dim, output_dim)
        else:
            print(f"\n1. Compression already initialized: {self}")

        # Step 2: Split data
        print(f"\n2. Splitting data (train={training_config.train_frac:.0%}, val={training_config.val_frac:.0%})...")

        train_summaries, val_summaries, test_summaries = split_data(
            summaries,
            train_frac=training_config.train_frac,
            val_frac=training_config.val_frac
        )

        print(f"   Train: {train_summaries[0].shape[0]}, Val: {val_summaries[0].shape[0]}, Test: {test_summaries[0].shape[0]}")

        # Step 3: Train
        print(f"\n3. Training for {training_config.n_epochs} epochs...")

        history = self._train_model(
            train_summaries=train_summaries,
            val_summaries=val_summaries,
            test_summaries=None,
            delta_theta=delta_theta,
            training_config=training_config
        )

        # Step 4: Load best model
        if history['best_model_state'] is not None:
            self.load_state_dict(history['best_model_state'])
            print(f"\n   Loaded best model (val loss: {min(history['val_losses']):.3f})")
        else:
            print("\n   Warning: No valid model found (Gaussianity constraints may have been too strict)")

        if history.get('needs_more_epochs', False):
            print("   Note: Training may benefit from more epochs (best model found late in training)")

        print("\n" + "=" * 80)
        print("Training Complete")
        print("=" * 80)

        # Add test summaries to history for post-training evaluation
        history['test_summaries'] = test_summaries

        return history

    def _validate_model(
        self,
        val_summaries: List[torch.Tensor],
        test_summaries: Optional[List[torch.Tensor]],
        compute_fisher_loss: Callable,
        training_config: TrainingConfig,
        best_val_loss: float,
        best_model_state: Optional[Dict],
        epoch: int
    ) -> Dict:
        """Run validation and optionally test evaluation."""
        from ..fisher.gaussianity import test_gaussianity

        self.eval()
        with torch.no_grad():
            val_summaries_compressed = self(val_summaries)
            val_loss = compute_fisher_loss(val_summaries_compressed)

            # Optional test loss for debugging
            test_loss = None
            if test_summaries is not None:
                test_summaries_compressed = self(test_summaries)
                test_loss = compute_fisher_loss(test_summaries_compressed).item()

            # Check Gaussianity constraint if enabled
            passes_gaussianity = True
            if training_config.check_gaussianity:
                _, passes_gaussianity = test_gaussianity(val_summaries_compressed, alpha=0.05, mode="summary", verbose=False)

            # Save best model (with Gaussianity constraint if enabled)
            if val_loss.item() < best_val_loss and passes_gaussianity:
                best_val_loss = val_loss.item()
                best_model_state = {k: v.clone() for k, v in self.state_dict().items()}
                if training_config.verbose:
                    print(f"    → New best model at epoch {epoch+1}: {best_val_loss:.3f}")

        return {
            'val_loss': val_loss.item(),
            'passes_gaussianity': passes_gaussianity,
            'test_loss': test_loss,
            'best_val_loss': best_val_loss,
            'best_model_state': best_model_state
        }

    def _train_model(
        self,
        train_summaries: List[torch.Tensor],
        val_summaries: List[torch.Tensor],
        test_summaries: Optional[List[torch.Tensor]],
        delta_theta: torch.Tensor,
        training_config: TrainingConfig
    ) -> Dict:
        """Train compression model to maximize Fisher information."""
        from ..fisher.analyzer import FisherAnalyzer

        optimizer = torch.optim.Adam(self.parameters(), lr=training_config.lr, weight_decay=training_config.weight_decay)
        fisher_analyzer = FisherAnalyzer(clean_data=True)

        train_losses = []
        val_losses = []
        test_losses = []
        best_val_loss = float('inf')
        best_model_state = None

        # Get minimum batch size from all summaries
        min_train_size = min(s.shape[0] for s in train_summaries)
        actual_batch_size = min(training_config.batch_size, min_train_size)

        def compute_fisher_loss(summaries):
            """Compute negative log determinant of Fisher matrix as loss."""
            result = fisher_analyzer(summaries, delta_theta)
            return -result.log_det_fisher

        for epoch in range(training_config.n_epochs):
            # Training
            self.train()

            # Sample random batch for this epoch
            idx = torch.randperm(min_train_size)[:actual_batch_size]
            # Apply compression to batch (pass list of batched tensors)
            batch_summaries = self([s[idx] for s in train_summaries])
            # Compute loss
            loss = compute_fisher_loss(batch_summaries)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            if (epoch + 1) % training_config.validate_every == 0:
                # Validation
                val_result = self._validate_model(
                    val_summaries=val_summaries,
                    test_summaries=test_summaries,
                    compute_fisher_loss=compute_fisher_loss,
                    training_config=training_config,
                    best_val_loss=best_val_loss,
                    best_model_state=best_model_state,
                    epoch=epoch
                )

                val_losses.append(val_result['val_loss'])
                if val_result['test_loss'] is not None:
                    test_losses.append(val_result['test_loss'])

                # Update best model state
                best_val_loss = val_result['best_val_loss']
                best_model_state = val_result['best_model_state']

                if training_config.verbose:
                    # Format Gaussianity status for display
                    gaussianity_status = ""
                    if training_config.check_gaussianity:
                        gaussianity_status = f", Gaussian: {'✓' if val_result['passes_gaussianity'] else '✗'}"

                    if val_result['test_loss'] is not None:
                        print(f"Epoch {epoch+1}/{training_config.n_epochs} - Train Loss: {loss.item():.3f}, Val Loss: {val_result['val_loss']:.3f}, Test Loss: {val_result['test_loss']:.3f}{gaussianity_status}")
                    else:
                        print(f"Epoch {epoch+1}/{training_config.n_epochs} - Train Loss: {loss.item():.3f}, Val Loss: {val_result['val_loss']:.3f}{gaussianity_status}")

        # Check if training might need more epochs
        needs_more_epochs = False
        if len(val_losses) >= 3:
            best_epoch_idx = val_losses.index(min(val_losses))
            final_20_percent = int(len(val_losses) * 0.8)
            if best_epoch_idx >= final_20_percent:
                needs_more_epochs = True

        return {
            'best_model_state': best_model_state,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_losses': test_losses,
            'needs_more_epochs': needs_more_epochs
        }


class FisherAnalyzer(ABC):
    """Base class for Fisher information analysis."""

    @abstractmethod
    def compute_fisher(
        self,
        summaries: List[torch.Tensor],
        delta_theta: torch.Tensor
    ) -> 'FisherResult':
        """
        Compute Fisher information from summaries.

        Args:
            summaries: List of summary statistics.
                summaries[0]: at theta_fid (for covariance)
                summaries[1:]: at perturbed values (for derivatives)
            delta_theta: Step sizes for derivative estimation

        Returns:
            FisherResult object containing Fisher matrix and related quantities
        """
        pass
