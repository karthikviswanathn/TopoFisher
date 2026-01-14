#!/usr/bin/env python3
"""
Training script for LearnableMMAFiltration.

Trains the CNN to produce a second filtration parameter that maximizes
Fisher information.

Usage:
    python train_learnable_mma.py [--config CONFIG_PATH]
    python train_learnable_mma.py --n_epochs 50 --lr 0.001 --batch_size 32
"""

import argparse
import copy
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Optional

# TopoFisher imports
from topofisher.simulators import GRFSimulator
from topofisher.filtrations.learnable_mma import LearnableMMAFiltration
from topofisher.vectorizations.mma_topk import MMATopKLayer
from topofisher.vectorizations.mma_kernel import MMAExponentialLayer, MMALinearLayer, MMAGaussianLayer
from topofisher.vectorizations.combined import CombinedVectorization
from topofisher.compressions.moped import MOPEDCompression
from topofisher.fisher.analyzer import FisherAnalyzer


class LearnableMMATrainer:
    """
    Trainer for LearnableMMAFiltration.
    
    Trains the CNN to maximize Fisher information by:
    1. Forward: field → CNN → (field, cnn_output) → MMA → vectorization → compression → Fisher
    2. Loss: -log|F|
    3. Backward: gradients flow through CNN via evaluate_mod_in_grid
    """
    
    def __init__(
        self,
        simulator: nn.Module,
        filtration: LearnableMMAFiltration,
        vectorization: nn.Module,
        compression: nn.Module,
        fisher_analyzer: nn.Module,
        device: str = 'cpu'
    ):
        self.simulator = simulator
        self.filtration = filtration
        self.vectorization = vectorization
        self.compression = compression
        self.fisher_analyzer = fisher_analyzer
        self.device = device
        
    def generate_data(
        self,
        theta_fid: torch.Tensor,
        delta_theta: torch.Tensor,
        n_samples: int,
        seed: int = None
    ) -> List[torch.Tensor]:
        """
        Generate simulation data for Fisher analysis.
        
        Returns:
            [fiducial, theta0_minus, theta0_plus, theta1_minus, theta1_plus, ...]
        """
        n_params = len(theta_fid)
        
        # Fiducial
        data_fid = self.simulator.generate(theta_fid, n_samples, seed=seed)
        all_data = [data_fid]
        
        # Derivatives
        for i in range(n_params):
            seed_i = seed + i + 1 if seed else None
            
            theta_minus = theta_fid.clone()
            theta_minus[i] -= delta_theta[i] / 2.0
            data_minus = self.simulator.generate(theta_minus, n_samples, seed=seed_i)
            
            theta_plus = theta_fid.clone()
            theta_plus[i] += delta_theta[i] / 2.0
            data_plus = self.simulator.generate(theta_plus, n_samples, seed=seed_i)
            
            all_data.extend([data_minus, data_plus])
        
        return all_data
    
    def compute_summaries(self, data: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Compute summaries through full pipeline.
        
        field → CNN → MMA → vectorization → summaries
        """
        all_summaries = []
        
        for field_batch in data:
            # Filtration: CNN produces second param
            mma_objects = self.filtration(field_batch)
            field_for_vec, second_param = self.filtration.get_stored_tensors()
            
            # Vectorization
            all_features = []
            for layer in self.vectorization.layers:
                features = layer(mma_objects, field_for_vec, second_param)
                all_features.append(features)
            summary = torch.cat(all_features, dim=-1)
            
            all_summaries.append(summary)
        
        return all_summaries
    
    def compute_fisher_loss(
        self,
        data: List[torch.Tensor],
        delta_theta: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Fisher loss = -log|F|.
        """
        # Get summaries
        summaries = self.compute_summaries(data)
        
        # Apply compression
        compressed = self.compression(summaries, delta_theta)
        
        # Compute Fisher
        result = self.fisher_analyzer(compressed, delta_theta)
        
        # Loss = -log|F| (we want to maximize log|F|)
        loss = -result.log_det_fisher
        
        return loss, result
    
    def split_data(
        self,
        data: List[torch.Tensor],
        train_frac: float = 0.5,
        val_frac: float = 0.25,
        seed: int = 42
    ):
        """Split data into train/val/test."""
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        train_data, val_data, test_data = [], [], []
        
        # Fiducial
        n = data[0].shape[0]
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        
        perm = torch.randperm(n, generator=generator)
        train_data.append(data[0][perm[:n_train]])
        val_data.append(data[0][perm[n_train:n_train+n_val]])
        test_data.append(data[0][perm[n_train+n_val:]])
        
        # Derivatives (pairs share permutation)
        n_params = (len(data) - 1) // 2
        for i in range(n_params):
            minus = data[1 + 2*i]
            plus = data[2 + 2*i]
            
            n = minus.shape[0]
            n_train = int(n * train_frac)
            n_val = int(n * val_frac)
            
            perm = torch.randperm(n, generator=generator)
            
            train_data.append(minus[perm[:n_train]])
            train_data.append(plus[perm[:n_train]])
            val_data.append(minus[perm[n_train:n_train+n_val]])
            val_data.append(plus[perm[n_train:n_train+n_val]])
            test_data.append(minus[perm[n_train+n_val:]])
            test_data.append(plus[perm[n_train+n_val:]])
        
        return train_data, val_data, test_data
    
    def train(
        self,
        theta_fid: torch.Tensor,
        delta_theta: torch.Tensor,
        n_samples: int = 1000,
        n_epochs: int = 50,
        batch_size: int = 64,
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        validate_every: int = 5,
        seed: int = 42,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the CNN to maximize Fisher information.
        """
        if verbose:
            print("=" * 60)
            print("Training LearnableMMAFiltration")
            print("=" * 60)
            print(f"  theta_fid: {theta_fid.tolist()}")
            print(f"  delta_theta: {delta_theta.tolist()}")
            print(f"  n_samples: {n_samples}")
            print(f"  n_epochs: {n_epochs}")
            print(f"  batch_size: {batch_size}")
            print(f"  lr: {lr}")
            n_params = sum(p.numel() for p in self.filtration.cnn.parameters())
            print(f"  CNN parameters: {n_params}")
            print()
        
        # Generate data
        if verbose:
            print("Generating training data...")
        all_data = self.generate_data(theta_fid, delta_theta, n_samples, seed=seed)
        train_data, val_data, test_data = self.split_data(all_data, seed=seed)
        
        if verbose:
            print(f"  Train: {train_data[0].shape[0]} samples")
            print(f"  Val: {val_data[0].shape[0]} samples")
            print(f"  Test: {test_data[0].shape[0]} samples")
            print()
        
        # Optimizer (only CNN parameters)
        optimizer = torch.optim.Adam(
            self.filtration.cnn.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Training state
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_state = None
        best_epoch = 0
        
        # Training loop
        if verbose:
            print("Training...")
        
        for epoch in tqdm(range(n_epochs), desc="Epochs", disable=not verbose):
            self.filtration.train()
            
            # Sample batch
            min_size = min(d.shape[0] for d in train_data)
            batch_size_actual = min(batch_size, min_size)
            idx = torch.randperm(min_size)[:batch_size_actual]
            batch_data = [d[idx] for d in train_data]
            
            # Forward + loss
            loss, _ = self.compute_fisher_loss(batch_data, delta_theta)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Validation
            if (epoch + 1) % validate_every == 0:
                self.filtration.eval()
                with torch.no_grad():
                    val_loss, val_result = self.compute_fisher_loss(val_data, delta_theta)
                
                val_losses.append(val_loss.item())
                
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_epoch = epoch + 1
                    best_model_state = copy.deepcopy(self.filtration.state_dict())
                    
                    if verbose:
                        tqdm.write(f"  Epoch {epoch+1}: train={loss.item():.3f}, "
                                   f"val={val_loss.item():.3f}, log|F|={-val_loss.item():.3f} ✓")
        
        # Load best model
        if best_model_state is not None:
            self.filtration.load_state_dict(best_model_state)
            if verbose:
                print(f"\nLoaded best model from epoch {best_epoch}")
        
        # Final evaluation on test set
        if verbose:
            print("\nEvaluating on test set...")
        
        self.filtration.eval()
        with torch.no_grad():
            test_loss, test_result = self.compute_fisher_loss(test_data, delta_theta)
        
        if verbose:
            print(f"\n{'=' * 60}")
            print("Results")
            print(f"{'=' * 60}")
            print(f"log|F| = {-test_loss.item():.4f}")
            print(f"σ(A) = {test_result.constraints[0].item():.4f}")
            print(f"σ(B) = {test_result.constraints[1].item():.4f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'test_loss': test_loss.item(),
            'test_result': test_result
        }


def main():
    parser = argparse.ArgumentParser(description='Train LearnableMMAFiltration')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--nlines', type=int, default=500, help='MMA nlines')
    parser.add_argument('--k0', type=int, default=50, help='TopK for H0')
    parser.add_argument('--k1', type=int, default=25, help='TopK for H1')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save', type=str, default=None, help='Save model to path')
    parser.add_argument('--vectorization', type=str, default='topk', choices=['topk', 'exponential', 'linear', 'gaussian'], help='Vectorization type')
    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Parameters
    theta_fid = torch.tensor([1.0, 2.0])
    delta_theta = torch.tensor([0.1, 0.2])
    
    # Components
    simulator = GRFSimulator(N=16, dim=2, device=device)
    
    filtration = LearnableMMAFiltration(
        nlines=args.nlines,
        hidden_channels=[32, 64, 32],
        kernel_size=3,
        activation='relu'
    )
    
    if args.vectorization == 'topk':
        vectorization = CombinedVectorization([
            MMATopKLayer(k=args.k0, homology_dimension=0),
            MMATopKLayer(k=args.k1, homology_dimension=1)
        ])
    else:
        vectorization = CombinedVectorization([
            MMAExponentialLayer(resolution=15, bandwidth=0.5, homology_dimension=0),
            MMAExponentialLayer(resolution=15, bandwidth=0.5, homology_dimension=1)
        ])
    
    compression = MOPEDCompression(train_frac=0.5, reg=1e-6)
    fisher_analyzer = FisherAnalyzer(clean_data=True)
    
    # Trainer
    trainer = LearnableMMATrainer(
        simulator=simulator,
        filtration=filtration,
        vectorization=vectorization,
        compression=compression,
        fisher_analyzer=fisher_analyzer,
        device=device
    )
    
    # Train
    history = trainer.train(
        theta_fid=theta_fid,
        delta_theta=delta_theta,
        n_samples=args.n_samples,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        verbose=True
    )
    
    # Save model
    if args.save:
        torch.save({
            'filtration_state': filtration.state_dict(),
            'history': history,
            'args': vars(args)
        }, args.save)
        print(f"\nModel saved to {args.save}")


if __name__ == '__main__':
    main()
