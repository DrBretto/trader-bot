"""
Asset health scoring model using Autoencoder architecture.

This model takes asset features as input and outputs health scores,
volatility buckets, behavior classifications, and latent embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


# Asset features used as input
ASSET_FEATURES = [
    'return_1d',
    'return_5d',
    'return_21d',
    'return_63d',
    'vol_21d',
    'vol_63d',
    'drawdown_21d',
    'drawdown_63d',
    'rel_strength_21d',
    'rel_strength_63d',
]

# Volatility buckets
VOL_BUCKETS = ['low', 'med', 'high']

# Behavior classifications
BEHAVIORS = ['momentum', 'mean_reversion', 'mixed']


class HealthAutoencoder(nn.Module):
    """
    Autoencoder for asset health scoring.

    The encoder compresses asset features into a latent space.
    The decoder reconstructs the features.
    Additional heads predict health score, volatility bucket, and behavior.
    """

    def __init__(
        self,
        input_dim: int = len(ASSET_FEATURES),
        latent_dim: int = 16,
        hidden_dims: List[int] = [64, 32],
        num_vol_buckets: int = len(VOL_BUCKETS),
        num_behaviors: int = len(BEHAVIORS),
        dropout: float = 0.2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_vol_buckets = num_vol_buckets
        self.num_behaviors = num_behaviors

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Health score head (regression, 0-1)
        self.health_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid()
        )

        # Volatility bucket head (classification)
        self.vol_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, num_vol_buckets)
        )

        # Behavior classification head
        self.behavior_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, num_behaviors)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstructed input."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Dict with keys:
                - reconstruction: (batch_size, input_dim)
                - latent: (batch_size, latent_dim)
                - health_score: (batch_size, 1)
                - vol_logits: (batch_size, num_vol_buckets)
                - behavior_logits: (batch_size, num_behaviors)
        """
        # Encode
        latent = self.encode(x)

        # Decode
        reconstruction = self.decode(latent)

        # Prediction heads
        health_score = self.health_head(latent)
        vol_logits = self.vol_head(latent)
        behavior_logits = self.behavior_head(latent)

        return {
            'reconstruction': reconstruction,
            'latent': latent,
            'health_score': health_score,
            'vol_logits': vol_logits,
            'behavior_logits': behavior_logits
        }

    def predict(self, x: torch.Tensor) -> List[Dict]:
        """
        Make predictions for a batch of assets.

        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,)

        Returns:
            List of dicts with health_score, vol_bucket, behavior, latent
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)

            output = self.forward(x)

            vol_probs = F.softmax(output['vol_logits'], dim=-1)
            behavior_probs = F.softmax(output['behavior_logits'], dim=-1)

            results = []
            for i in range(x.shape[0]):
                vol_idx = vol_probs[i].argmax().item()
                behavior_idx = behavior_probs[i].argmax().item()

                results.append({
                    'health_score': output['health_score'][i].item(),
                    'vol_bucket': VOL_BUCKETS[vol_idx],
                    'behavior': BEHAVIORS[behavior_idx],
                    'latent': output['latent'][i].tolist()
                })

            return results


class VariationalHealthAutoencoder(nn.Module):
    """
    Variational Autoencoder for asset health scoring.

    Uses VAE for better latent space regularization and
    potential for generative modeling.
    """

    def __init__(
        self,
        input_dim: int = len(ASSET_FEATURES),
        latent_dim: int = 16,
        hidden_dims: List[int] = [64, 32],
        num_vol_buckets: int = len(VOL_BUCKETS),
        num_behaviors: int = len(BEHAVIORS),
        dropout: float = 0.2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space parameters (mu and log_var)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_var = nn.Linear(prev_dim, latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Prediction heads
        self.health_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid()
        )

        self.vol_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, num_vol_buckets)
        )

        self.behavior_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, num_behaviors)
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns dict with reconstruction, latent, mu, log_var, and predictions.
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)

        health_score = self.health_head(z)
        vol_logits = self.vol_head(z)
        behavior_logits = self.behavior_head(z)

        return {
            'reconstruction': reconstruction,
            'latent': z,
            'mu': mu,
            'log_var': log_var,
            'health_score': health_score,
            'vol_logits': vol_logits,
            'behavior_logits': behavior_logits
        }

    def predict(self, x: torch.Tensor) -> List[Dict]:
        """Make predictions using mean of latent distribution."""
        self.eval()
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)

            mu, _ = self.encode(x)
            reconstruction = self.decode(mu)

            health_score = self.health_head(mu)
            vol_logits = self.vol_head(mu)
            behavior_logits = self.behavior_head(mu)

            vol_probs = F.softmax(vol_logits, dim=-1)
            behavior_probs = F.softmax(behavior_logits, dim=-1)

            results = []
            for i in range(x.shape[0]):
                vol_idx = vol_probs[i].argmax().item()
                behavior_idx = behavior_probs[i].argmax().item()

                results.append({
                    'health_score': health_score[i].item(),
                    'vol_bucket': VOL_BUCKETS[vol_idx],
                    'behavior': BEHAVIORS[behavior_idx],
                    'latent': mu[i].tolist()
                })

            return results


def vae_loss(
    reconstruction: torch.Tensor,
    original: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    kl_weight: float = 0.001
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute VAE loss: reconstruction + KL divergence.

    Args:
        reconstruction: Reconstructed input
        original: Original input
        mu: Latent mean
        log_var: Latent log variance
        kl_weight: Weight for KL term (beta-VAE)

    Returns:
        Total loss and dict of individual losses
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(reconstruction, original, reduction='mean')

    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    total_loss = recon_loss + kl_weight * kl_loss

    return total_loss, {
        'reconstruction': recon_loss.item(),
        'kl': kl_loss.item(),
        'total': total_loss.item()
    }


def create_health_model(
    model_type: str = 'autoencoder',
    **kwargs
) -> nn.Module:
    """
    Factory function to create health model.

    Args:
        model_type: 'autoencoder' or 'vae'
        **kwargs: Model-specific arguments

    Returns:
        Instantiated model
    """
    if model_type == 'autoencoder':
        return HealthAutoencoder(**kwargs)
    elif model_type == 'vae':
        return VariationalHealthAutoencoder(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
