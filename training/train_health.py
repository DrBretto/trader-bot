"""
Health model training script.

Trains an Autoencoder or VAE for asset health scoring.
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from typing import Dict, Optional, Tuple
from tqdm import tqdm

from training.models import create_health_model, vae_loss, ASSET_FEATURES, VOL_BUCKETS, BEHAVIORS
from training.utils.data_loader import create_health_dataloaders, HealthDataset
from training.utils.metrics import HealthMetrics, EarlyStopping, compute_health_labels_from_baseline


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    metrics: HealthMetrics,
    is_vae: bool = False,
    kl_weight: float = 0.001
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    metrics.reset()

    for batch in tqdm(train_loader, desc='Training', leave=False):
        features = batch['features'].to(device)

        optimizer.zero_grad()

        output = model(features)

        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(output['reconstruction'], features)

        # Total loss
        if is_vae:
            total_loss, loss_dict = vae_loss(
                output['reconstruction'],
                features,
                output['mu'],
                output['log_var'],
                kl_weight=kl_weight
            )
        else:
            total_loss = recon_loss
            loss_dict = {'reconstruction': recon_loss.item(), 'total': total_loss.item()}

        total_loss.backward()
        optimizer.step()

        # Update metrics
        health_target = batch.get('health')
        if health_target is not None:
            health_target = health_target.to(device)

        metrics.update(
            reconstruction=output['reconstruction'],
            original=features,
            health_pred=output['health_score'],
            health_target=health_target,
            vol_pred=output['vol_logits'],
            behavior_pred=output['behavior_logits'],
            losses=loss_dict
        )

    return metrics.compute()


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    metrics: HealthMetrics,
    is_vae: bool = False,
    kl_weight: float = 0.001
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    metrics.reset()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating', leave=False):
            features = batch['features'].to(device)

            output = model(features)

            # Loss
            recon_loss = nn.functional.mse_loss(output['reconstruction'], features)

            if is_vae:
                total_loss, loss_dict = vae_loss(
                    output['reconstruction'],
                    features,
                    output['mu'],
                    output['log_var'],
                    kl_weight=kl_weight
                )
            else:
                total_loss = recon_loss
                loss_dict = {'reconstruction': recon_loss.item(), 'total': total_loss.item()}

            health_target = batch.get('health')
            if health_target is not None:
                health_target = health_target.to(device)

            metrics.update(
                reconstruction=output['reconstruction'],
                original=features,
                health_pred=output['health_score'],
                health_target=health_target,
                vol_pred=output['vol_logits'],
                behavior_pred=output['behavior_logits'],
                losses=loss_dict
            )

    return metrics.compute()


def train_health_model(
    features_df,
    context_df=None,
    model_type: str = 'autoencoder',
    latent_dim: int = 16,
    hidden_dims: list = None,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 100,
    patience: int = 15,
    kl_weight: float = 0.001,
    device: str = 'auto',
    save_dir: Optional[str] = None
) -> Tuple[nn.Module, Dict]:
    """
    Train health scoring autoencoder model.

    Args:
        features_df: DataFrame with asset features
        context_df: Optional context features (not used currently)
        model_type: 'autoencoder' or 'vae'
        latent_dim: Dimension of latent space
        hidden_dims: List of hidden layer dimensions
        batch_size: Training batch size
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        epochs: Maximum training epochs
        patience: Early stopping patience
        kl_weight: Weight for KL divergence (VAE only)
        device: 'auto', 'cuda', 'mps', or 'cpu'
        save_dir: Directory to save model checkpoints

    Returns:
        (trained_model, training_history)
    """
    if hidden_dims is None:
        hidden_dims = [64, 32]

    print(f"Training health model ({model_type})...")
    print(f"Data: {len(features_df)} rows")

    # Set device
    if device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    # Generate labels using baseline model
    print("Generating health labels from baseline model...")
    features_df = features_df.copy()
    health_scores, vol_buckets, behaviors = compute_health_labels_from_baseline(features_df)
    features_df['health_score_label'] = health_scores
    features_df['vol_bucket_label'] = vol_buckets
    features_df['behavior_label'] = behaviors

    # Filter feature columns that exist in the data
    available_features = [f for f in ASSET_FEATURES if f in features_df.columns]
    print(f"Using features: {available_features}")

    if len(available_features) < 3:
        raise ValueError(f"Not enough features available. Found: {available_features}")

    # Create dataloaders
    label_cols = {
        'health': 'health_score_label',
        'vol': 'vol_bucket_label',
        'behavior': 'behavior_label'
    }

    train_loader, val_loader, test_loader = create_health_dataloaders(
        features_df=features_df,
        feature_cols=available_features,
        batch_size=batch_size,
        train_ratio=0.7,
        val_ratio=0.15,
        label_cols=label_cols
    )

    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Create model
    is_vae = model_type == 'vae'
    model = create_health_model(
        model_type=model_type,
        input_dim=len(available_features),
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        num_vol_buckets=len(VOL_BUCKETS),
        num_behaviors=len(BEHAVIORS),
        dropout=0.2
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Metrics and early stopping
    train_metrics = HealthMetrics(VOL_BUCKETS, BEHAVIORS)
    val_metrics = HealthMetrics(VOL_BUCKETS, BEHAVIORS)
    early_stopping = EarlyStopping(patience=patience, mode='min')

    # Training history
    history = {
        'train_loss': [],
        'train_recon_mse': [],
        'val_loss': [],
        'val_recon_mse': [],
        'learning_rate': []
    }

    best_model_state = None
    best_val_loss = float('inf')

    # Check if validation set is available
    has_validation = len(val_loader.dataset) > 0

    if not has_validation:
        print("Warning: No validation data available, using training loss for model selection")

    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        train_results = train_epoch(
            model, train_loader, optimizer, device,
            train_metrics, is_vae=is_vae, kl_weight=kl_weight
        )
        print(f"  Train - Loss: {train_results.get('loss_total', 0):.4f}, "
              f"Recon MSE: {train_results['reconstruction_mse']:.4f}")

        # Validate (if data available)
        if has_validation:
            val_results = validate(
                model, val_loader, device,
                val_metrics, is_vae=is_vae, kl_weight=kl_weight
            )
            print(f"  Val   - Loss: {val_results.get('loss_total', 0):.4f}, "
                  f"Recon MSE: {val_results['reconstruction_mse']:.4f}")
            loss_for_scheduler = val_results.get('loss_total', val_results['reconstruction_mse'])
        else:
            val_results = train_results  # Use train results when no validation data
            loss_for_scheduler = train_results.get('loss_total', train_results['reconstruction_mse'])

        # Update scheduler
        scheduler.step(loss_for_scheduler)

        # Record history
        history['train_loss'].append(train_results.get('loss_total', train_results['reconstruction_mse']))
        history['train_recon_mse'].append(train_results['reconstruction_mse'])
        history['val_loss'].append(val_results.get('loss_total', val_results['reconstruction_mse']))
        history['val_recon_mse'].append(val_results['reconstruction_mse'])
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        # Save best model
        if loss_for_scheduler < best_val_loss:
            best_val_loss = loss_for_scheduler
            best_model_state = model.state_dict().copy()

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))

        # Early stopping
        if early_stopping(loss_for_scheduler, epoch):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Final evaluation on test set (if available)
    has_test = len(test_loader.dataset) > 0
    if has_test:
        print("\nEvaluating on test set...")
        test_metrics = HealthMetrics(VOL_BUCKETS, BEHAVIORS)
        test_results = validate(model, test_loader, device, test_metrics, is_vae=is_vae, kl_weight=kl_weight)
        print(f"Test - Recon MSE: {test_results['reconstruction_mse']:.4f}")
        if 'health_mse' in test_results:
            print(f"Health MSE: {test_results['health_mse']:.4f}, "
                  f"Health Correlation: {test_results.get('health_correlation', 0):.4f}")
    else:
        print("\nNo test set available, using training results for evaluation")
        test_results = {'reconstruction_mse': history['train_recon_mse'][-1]}

    # Add test results to history
    history['test_recon_mse'] = test_results['reconstruction_mse']
    history['test_health_mse'] = test_results.get('health_mse')
    history['test_health_correlation'] = test_results.get('health_correlation')
    history['best_epoch'] = early_stopping.best_epoch
    history['feature_cols'] = available_features
    history['model_type'] = model_type
    history['latent_dim'] = latent_dim
    history['hidden_dims'] = hidden_dims

    # Store normalization params
    history['normalization'] = {
        'mean': train_loader.dataset.feature_mean.tolist(),
        'std': train_loader.dataset.feature_std.tolist()
    }

    return model, history


def save_health_model(
    model: nn.Module,
    history: Dict,
    save_path: str,
    version: Optional[str] = None
):
    """Save trained health model."""
    if version is None:
        version = datetime.now().strftime('%Y%m%d')

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    # Move model to CPU before saving for Lambda compatibility.
    model_cpu = model.cpu()

    # Save model state and config
    model_data = {
        'version': version,
        'model_state': model_cpu.state_dict(),
        'model_config': {
            'model_type': history.get('model_type', 'autoencoder'),
            'input_dim': len(history.get('feature_cols', ASSET_FEATURES)),
            'latent_dim': history.get('latent_dim', 16),
            'hidden_dims': history.get('hidden_dims', [64, 32]),
            'num_vol_buckets': len(VOL_BUCKETS),
            'num_behaviors': len(BEHAVIORS)
        },
        'feature_cols': history.get('feature_cols', ASSET_FEATURES),
        'normalization': history.get('normalization', {}),
        'training_history': {
            'test_recon_mse': history.get('test_recon_mse'),
            'test_health_mse': history.get('test_health_mse'),
            'test_health_correlation': history.get('test_health_correlation'),
            'best_epoch': history.get('best_epoch')
        },
        'timestamp': datetime.now().isoformat()
    }

    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    import pandas as pd
    import argparse

    parser = argparse.ArgumentParser(description='Train health model')
    parser.add_argument('--data', type=str, help='Path to features data parquet')
    parser.add_argument('--output', type=str, default='models/health_model.pkl', help='Output path')
    parser.add_argument('--model-type', type=str, default='autoencoder', choices=['autoencoder', 'vae'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    if args.data:
        features_df = pd.read_parquet(args.data)
    else:
        # Create dummy data for testing
        print("No data provided, using dummy data for testing...")
        n_samples = 5000
        features_df = pd.DataFrame({
            'date': np.repeat(pd.date_range('2020-01-01', periods=100, freq='D'), 50),
            'symbol': np.tile([f'SYM{i}' for i in range(50)], 100),
            'return_1d': np.random.randn(n_samples) * 0.02,
            'return_5d': np.random.randn(n_samples) * 0.04,
            'return_21d': np.random.randn(n_samples) * 0.08,
            'return_63d': np.random.randn(n_samples) * 0.15,
            'vol_21d': np.abs(np.random.randn(n_samples) * 0.05) + 0.1,
            'vol_63d': np.abs(np.random.randn(n_samples) * 0.05) + 0.12,
            'drawdown_21d': -np.abs(np.random.randn(n_samples) * 0.05),
            'drawdown_63d': -np.abs(np.random.randn(n_samples) * 0.08),
            'rel_strength_21d': np.random.randn(n_samples) * 0.03,
            'rel_strength_63d': np.random.randn(n_samples) * 0.05,
        })

    model, history = train_health_model(
        features_df,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    save_health_model(model, history, args.output)
