"""Base class for learnable pipelines.

TODO: Wrap filtration, vectorization and compression into a single model.
Currently these components are separate, but they could be wrapped together
into a single nn.Module (as a PyTorch model) for cleaner learnable pipeline implementations.
This would enable end-to-end optimization and better gradient flow through
all components.
"""

import torch
import torch.nn as nn
from abc import abstractmethod
from typing import List, Tuple, Dict, Any
from ..base import BasePipeline
from ..configs.data_types import PipelineConfig, TrainingConfig, FisherResult
from ...fisher.gaussianity import test_gaussianity

class LearnablePipeline(BasePipeline):
    """
    Base class for pipelines with learnable components.

    Provides common infrastructure for training compression, vectorization,
    or filtration components to maximize Fisher information.
    """

    def split_data(
        self,
        data_list: List[torch.Tensor],
        train_frac: float = 0.5,
        val_frac: float = 0.25,
        seed: int = 42
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Split data into train/val/test sets.

        IMPORTANT: For derivative data, theta_minus and theta_plus pairs
        must use the same permutation to maintain pairing from same seed.

        Args:
            data_list: [fid, minus_0, plus_0, minus_1, plus_1, ...]
            train_frac: Fraction for training
            val_frac: Fraction for validation
            seed: Random seed for reproducibility

        Returns:
            train_data, val_data, test_data (each maintaining list structure)
        """
        generator = torch.Generator()
        generator.manual_seed(seed)

        train_data = []
        val_data = []
        test_data = []

        # Split fiducial data
        n_fid = data_list[0].shape[0]
        n_train = int(n_fid * train_frac)
        n_val = int(n_fid * val_frac)

        perm_fid = torch.randperm(n_fid, generator=generator)
        train_idx = perm_fid[:n_train]
        val_idx = perm_fid[n_train:n_train + n_val]
        test_idx = perm_fid[n_train + n_val:]

        train_data.append(data_list[0][train_idx])
        val_data.append(data_list[0][val_idx])
        test_data.append(data_list[0][test_idx])

        # Split derivative data (pairs must share permutation)
        n_params = (len(data_list) - 1) // 2
        for param_idx in range(n_params):
            minus_data = data_list[1 + 2 * param_idx]
            plus_data = data_list[2 + 2 * param_idx]

            n_deriv = minus_data.shape[0]
            n_train_d = int(n_deriv * train_frac)
            n_val_d = int(n_deriv * val_frac)

            # Same permutation for both minus and plus
            perm_deriv = torch.randperm(n_deriv, generator=generator)
            train_idx = perm_deriv[:n_train_d]
            val_idx = perm_deriv[n_train_d:n_train_d + n_val_d]
            test_idx = perm_deriv[n_train_d + n_val_d:]

            train_data.append(minus_data[train_idx])
            train_data.append(plus_data[train_idx])
            val_data.append(minus_data[val_idx])
            val_data.append(plus_data[val_idx])
            test_data.append(minus_data[test_idx])
            test_data.append(plus_data[test_idx])

        return train_data, val_data, test_data

    @abstractmethod
    def _compute_summaries(self, data: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Compute compressed summaries from input data.

        Each pipeline type implements this differently:
        - Filtration: data → filtration → vectorization → compression
        - Compression: summaries → compression
        - Vectorization: diagrams → vectorization → compression

        Args:
            data: Input data in pipeline-specific format

        Returns:
            List of compressed summary tensors [fid, minus_0, plus_0, ...]
        """
        pass

    def compute_fisher_result(
        self,
        data: List[torch.Tensor],
        delta_theta: torch.Tensor,
        check_gaussianity: bool = False
    ):
        """
        Compute Fisher result with optional Gaussianity check.

        This method eliminates redundancy by centralizing the Fisher computation
        and Gaussianity checking logic. Subclasses only need to implement
        _compute_summaries().

        Args:
            data: Input data
            delta_theta: Parameter step sizes
            check_gaussianity: If True, also check Gaussianity (no extra computation)

        Returns:
            FisherResult with fisher_matrix, constraints, log_det_fisher
            If check_gaussianity=True, also includes 'passes_gaussianity' attribute
        """
        # Get compressed summaries (subclass-specific implementation)
        compressed = self._compute_summaries(data)

        # Compute Fisher result
        fisher_result = self.fisher_analyzer(compressed, delta_theta)

        # Gaussianity check and updating fisher_result
        if check_gaussianity:
            _, passes_gaussianity = test_gaussianity(compressed, verbose=False)
            fisher_result.passes_gaussianity = passes_gaussianity

        return fisher_result

    def train_model(
        self,
        train_data: List[torch.Tensor],
        val_data: List[torch.Tensor],
        delta_theta: torch.Tensor,
        training_config: TrainingConfig
    ) -> Dict[str, Any]:
        """
        Train all learnable components automatically.

        PyTorch automatically tracks all parameters in the pipeline,
        so we just use self.parameters() to train everything that's learnable.

        Args:
            train_data: Training data [fid, minus_0, plus_0, ...]
            val_data: Validation data with same structure
            delta_theta: Parameter step sizes for Fisher computation
            training_config: Training hyperparameters

        Returns:
            Training history with loss curves, best epoch, etc.
        """
        # PyTorch automatically tracks all parameters - just use self.parameters()!
        params = list(self.parameters())

        if len(params) == 0:
            raise ValueError(
                "No trainable parameters found in pipeline. "
                "Ensure at least one component (filtration, vectorization, or compression) "
                "has trainable parameters."
            )

        if training_config.verbose:
            # Skip parameter counting if there are uninitialized parameters (e.g., LazyLinear)
            try:
                n_params = sum(p.numel() for p in params)
                print(f"  Training {n_params} parameters")
            except (ValueError, RuntimeError):
                print(f"  Training parameters (lazy initialization - will be determined on first forward pass)")

        # Setup optimizer with ALL parameters automatically
        optimizer = torch.optim.Adam(
            self.parameters(),  # This gets ALL parameters automatically!
            lr=training_config.lr,
            weight_decay=training_config.weight_decay
        )

        # Training state
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_state = None
        best_epoch = 0

        # Batch size handling
        min_train_size = min(
            s.shape[0] if isinstance(s, torch.Tensor) else len(s)
            for s in train_data
        )
        batch_size = min(training_config.batch_size, min_train_size)

        for epoch in range(training_config.n_epochs):
            # Training step
            self.train()  # Set entire pipeline to train mode

            # Sample random batch
            idx = torch.randperm(min_train_size)[:batch_size]
            batch_data = self._extract_batch(train_data, idx)

            # Forward pass: compute Fisher result and extract loss
            fisher_result = self.compute_fisher_result(batch_data, delta_theta, check_gaussianity=False)
            loss = -fisher_result.log_det_fisher

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            # Validation
            if (epoch + 1) % training_config.validate_every == 0:
                self.eval()  # Set entire pipeline to eval mode
                with torch.no_grad():
                    val_result = self.compute_fisher_result(val_data, delta_theta, check_gaussianity=training_config.check_gaussianity)
                    val_loss = -val_result.log_det_fisher

                val_losses.append(val_loss.item())

                # Check validation conditions
                is_best_loss = val_loss.item() < best_val_loss
                passes_gaussianity = val_result.passes_gaussianity  

                # Save best model (both conditions must pass)
                if is_best_loss and passes_gaussianity:
                    best_val_loss = val_loss.item()
                    best_epoch = epoch + 1
                    best_model_state = {
                        k: v.clone() for k, v in self.state_dict().items()
                    }

                    if training_config.verbose:
                        print(f"  Epoch {epoch+1}/{training_config.n_epochs}: "
                              f"Train Loss={loss.item():.3f}, "
                              f"Val Loss={val_loss.item():.3f}, "
                              f"Best Loss: ✓, Gaussianity: ✓ → Model updated")
                elif training_config.verbose:
                    best_loss_mark = "✓" if is_best_loss else "✗"
                    gaussian_mark = "✓" if passes_gaussianity else "✗"
                    print(f"  Epoch {epoch+1}/{training_config.n_epochs}: "
                          f"Train Loss={loss.item():.3f}, "
                          f"Val Loss={val_loss.item():.3f}, "
                          f"Best Loss: {best_loss_mark}, Gaussianity: {gaussian_mark}")

        # Load best model (entire pipeline)
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
            if training_config.verbose:
                print(f"\n  Loaded best model from epoch {best_epoch} "
                      f"(val loss: {best_val_loss:.3f})")

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch
        }

    def _extract_batch(self, data_list: List, indices: torch.Tensor) -> List:
        """
        Extract batch from data list.

        Handles both tensor and list types.

        Args:
            data_list: List of data elements
            indices: Batch indices

        Returns:
            Batched data maintaining structure
        """
        batch = []
        for data in data_list:
            if isinstance(data, torch.Tensor):
                batch.append(data[indices])
            elif isinstance(data, list):
                # Handle nested lists (e.g., persistence diagrams)
                batch.append([data[i] for i in indices])
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
        return batch

    def evaluate(
        self,
        data: List[torch.Tensor],
        delta_theta: torch.Tensor
    ) -> FisherResult:
        """
        Evaluate the pipeline on a dataset.

        Always checks Gaussianity on test data for final verification.

        Args:
            data: Data to evaluate [fid, minus_0, plus_0, ...]
            delta_theta: Parameter step sizes

        Returns:
            Fisher result with matrix, constraints, log determinant, and passes_gaussianity
        """
        # Always check Gaussianity on test data
        return self.compute_fisher_result(data, delta_theta, check_gaussianity=True)

    def run(
        self,
        config: PipelineConfig,
        training_config: TrainingConfig,
        data: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Run the complete learnable pipeline.

        Steps:
        1. Split data into train/val/test
        2. Train component (uses evaluate() on val during training)
        3. Evaluate on test set

        Args:
            config: Pipeline configuration
            training_config: Training configuration
            data: Pre-generated data to use [fid, minus_0, plus_0, ...]

        Returns:
            Dictionary with test result and training history
        """
        # 1. Split data
        if training_config.verbose:
            print("\n" + "="*80)
            print("Splitting Data")
            print("="*80)

        train_data, val_data, test_data = self.split_data(
            data,
            train_frac=training_config.train_frac,
            val_frac=training_config.val_frac,
            seed=42  # Fixed seed for reproducibility
        )

        if training_config.verbose:
            n_train = train_data[0].shape[0]
            n_val = val_data[0].shape[0]
            n_test = test_data[0].shape[0]
            print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")

        # 2. Train (internally uses evaluate() on validation)
        if training_config.verbose:
            print("\n" + "="*80)
            print("Training Component")
            print("="*80)

        history = self.train_model(
            train_data=train_data,
            val_data=val_data,
            delta_theta=config.delta_theta,
            training_config=training_config
        )

        # 3. Evaluate on test set
        if training_config.verbose:
            print("\n" + "="*80)
            print("Evaluating on Test Set")
            print("="*80)

        test_result = self.evaluate(test_data, config.delta_theta)

        if training_config.verbose:
            print(f"  Test log|F|: {test_result.log_det_fisher:.3f}")
            print(f"  Test constraints: {test_result.constraints.detach().cpu().numpy()}")

        return {
            'test_result': test_result,
            'training_history': history,
            'n_train': n_train,
            'n_val': n_val,
            'n_test': n_test
        }