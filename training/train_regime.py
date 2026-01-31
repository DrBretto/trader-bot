"""
Regime model training script.

Trains a GRU or Transformer model for market regime classification.
"""

import os
import json
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from typing import Dict, Optional, Tuple
from tqdm import tqdm

from training.models import create_regime_model, REGIME_LABELS, CONTEXT_FEATURES
from training.utils.data_loader import create_regime_dataloaders, RegimeDataset
from training.utils.metrics import RegimeMetrics, EarlyStopping, compute_regime_labels_from_baseline


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    metrics: RegimeMetrics
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    metrics.reset()

    for batch in tqdm(train_loader, desc='Training', leave=False):
        features = batch['features'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        output = model(features)
        loss = criterion(output['logits'], labels)

        loss.backward()
        optimizer.step()

        metrics.update(output['probs'], labels, loss.item())

    return metrics.compute()


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    metrics: RegimeMetrics
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    metrics.reset()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating', leave=False):
            features = batch['features'].to(device)
            labels = batch['label'].to(device)

            output = model(features)
            loss = criterion(output['logits'], labels)

            metrics.update(output['probs'], labels, loss.item())

    return metrics.compute()


def train_regime_model(
    context_df,
    features_df=None,
    model_type: str = 'gru',
    seq_length: int = 21,
    hidden_dim: int = 64,
    num_layers: int = 2,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 100,
    patience: int = 15,
    device: str = 'auto',
    save_dir: Optional[str] = None
) -> Tuple[nn.Module, Dict]:
    """
    Train regime classification model.

    Args:
        context_df: DataFrame with context features
        features_df: Optional additional features (not used currently)
        model_type: 'gru' or 'transformer'
        seq_length: Sequence length for model input
        hidden_dim: Hidden dimension of model
        num_layers: Number of GRU/Transformer layers
        batch_size: Training batch size
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        epochs: Maximum training epochs
        patience: Early stopping patience
        device: 'auto', 'cuda', 'mps', or 'cpu'
        save_dir: Directory to save model checkpoints

    Returns:
        (trained_model, training_history)
    """
    print(f"Training regime model ({model_type})...")
    print(f"Data: {len(context_df)} rows")

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
    print("Generating regime labels from baseline model...")
    context_df = context_df.copy()
    context_df['regime_label'] = compute_regime_labels_from_baseline(context_df)

    # Filter feature columns that exist in the data
    available_features = [f for f in CONTEXT_FEATURES if f in context_df.columns]
    print(f"Using features: {available_features}")

    if len(available_features) < 3:
        raise ValueError(f"Not enough features available. Found: {available_features}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_regime_dataloaders(
        context_df=context_df,
        feature_cols=available_features,
        seq_length=seq_length,
        batch_size=batch_size,
        train_ratio=0.7,
        val_ratio=0.15,
        label_col='regime_label',
        regime_labels=REGIME_LABELS
    )

    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Create model
    model = create_regime_model(
        model_type=model_type,
        input_dim=len(available_features),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=len(REGIME_LABELS),
        embedding_dim=8,
        dropout=0.2
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Metrics and early stopping
    train_metrics = RegimeMetrics(REGIME_LABELS)
    val_metrics = RegimeMetrics(REGIME_LABELS)
    early_stopping = EarlyStopping(patience=patience, mode='min')

    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
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
        train_results = train_epoch(model, train_loader, optimizer, criterion, device, train_metrics)
        print(f"  Train - Loss: {train_results['loss']:.4f}, Acc: {train_results['accuracy']:.4f}")

        # Validate (if data available)
        if has_validation:
            val_results = validate(model, val_loader, criterion, device, val_metrics)
            print(f"  Val   - Loss: {val_results['loss']:.4f}, Acc: {val_results['accuracy']:.4f}")
            loss_for_scheduler = val_results['loss']
        else:
            val_results = {'loss': train_results['loss'], 'accuracy': train_results['accuracy']}
            loss_for_scheduler = train_results['loss']

        # Update scheduler
        scheduler.step(loss_for_scheduler)

        # Record history
        history['train_loss'].append(train_results['loss'])
        history['train_accuracy'].append(train_results['accuracy'])
        history['val_loss'].append(val_results['loss'])
        history['val_accuracy'].append(val_results['accuracy'])
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
        test_metrics = RegimeMetrics(REGIME_LABELS)
        test_results = validate(model, test_loader, criterion, device, test_metrics)
        print(f"Test - Loss: {test_results['loss']:.4f}, Acc: {test_results['accuracy']:.4f}")
        print(f"F1 (macro): {test_results['f1_macro']:.4f}")
        print("\nClassification Report:")
        print(test_metrics.get_classification_report())
    else:
        print("\nNo test set available, using training results for evaluation")
        test_results = {'loss': history['train_loss'][-1], 'accuracy': history['train_accuracy'][-1], 'f1_macro': 0.0}

    # Add test results to history
    history['test_loss'] = test_results['loss']
    history['test_accuracy'] = test_results['accuracy']
    history['test_f1_macro'] = test_results.get('f1_macro', 0.0)
    history['best_epoch'] = early_stopping.best_epoch
    history['feature_cols'] = available_features
    history['model_type'] = model_type
    history['seq_length'] = seq_length

    # Store normalization params
    history['normalization'] = {
        'mean': train_loader.dataset.feature_mean.tolist(),
        'std': train_loader.dataset.feature_std.tolist()
    }

    return model, history


def save_regime_model(
    model: nn.Module,
    history: Dict,
    save_path: str,
    version: Optional[str] = None
):
    """Save trained regime model."""
    if version is None:
        version = datetime.now().strftime('%Y%m%d')

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    # Save model state and config
    model_data = {
        'version': version,
        'model_state': model.state_dict(),
        'model_config': {
            'model_type': history.get('model_type', 'gru'),
            'input_dim': len(history.get('feature_cols', CONTEXT_FEATURES)),
            'hidden_dim': model.hidden_dim,
            'num_layers': model.num_layers,
            'num_classes': model.num_classes,
            'embedding_dim': model.embedding_dim,
            'seq_length': history.get('seq_length', 21)
        },
        'feature_cols': history.get('feature_cols', CONTEXT_FEATURES),
        'normalization': history.get('normalization', {}),
        'training_history': {
            'test_accuracy': history.get('test_accuracy'),
            'test_f1_macro': history.get('test_f1_macro'),
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

    parser = argparse.ArgumentParser(description='Train regime model')
    parser.add_argument('--data', type=str, help='Path to context data parquet')
    parser.add_argument('--output', type=str, default='models/regime_model.pkl', help='Output path')
    parser.add_argument('--model-type', type=str, default='gru', choices=['gru', 'transformer'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    if args.data:
        context_df = pd.read_parquet(args.data)
    else:
        # Create dummy data for testing
        print("No data provided, using dummy data for testing...")
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        context_df = pd.DataFrame({
            'date': dates,
            'spy_return_1d': np.random.randn(500) * 0.01,
            'spy_return_21d': np.random.randn(500) * 0.05,
            'spy_vol_21d': np.abs(np.random.randn(500) * 0.05) + 0.1,
            'yield_slope': np.random.randn(500) * 0.5,
            'credit_spread_proxy': np.random.randn(500) * 0.02,
            'vixy_return_21d': np.random.randn(500) * 0.1,
            'risk_off_proxy': np.random.randn(500) * 0.03,
            'gdelt_avg_tone': np.random.randn(500) * 0.5,
        })

    model, history = train_regime_model(
        context_df,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    save_regime_model(model, history, args.output)
