"""Tests for ML model architectures."""

import pytest
import numpy as np

# Skip tests if PyTorch not available
torch = pytest.importorskip("torch")

import sys
sys.path.insert(0, str(__file__).rsplit('/tests', 1)[0])

from training.models.regime_transformer import (
    RegimeGRU,
    RegimeTransformer,
    create_regime_model,
    REGIME_LABELS,
    CONTEXT_FEATURES
)
from training.models.health_autoencoder import (
    HealthAutoencoder,
    VariationalHealthAutoencoder,
    create_health_model,
    vae_loss,
    ASSET_FEATURES,
    VOL_BUCKETS,
    BEHAVIORS
)


class TestRegimeGRU:
    """Tests for RegimeGRU model."""

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shapes."""
        batch_size = 8
        seq_length = 21
        input_dim = len(CONTEXT_FEATURES)

        model = RegimeGRU(input_dim=input_dim)
        x = torch.randn(batch_size, seq_length, input_dim)

        output = model(x)

        assert output['logits'].shape == (batch_size, len(REGIME_LABELS))
        assert output['probs'].shape == (batch_size, len(REGIME_LABELS))
        assert output['embedding'].shape == (batch_size, 8)  # default embedding_dim
        assert output['attention_weights'].shape == (batch_size, seq_length)

    def test_probs_sum_to_one(self):
        """Test that output probabilities sum to 1."""
        model = RegimeGRU()
        x = torch.randn(4, 21, len(CONTEXT_FEATURES))

        output = model(x)
        prob_sums = output['probs'].sum(dim=-1)

        assert torch.allclose(prob_sums, torch.ones(4), atol=1e-6)

    def test_predict_returns_correct_structure(self):
        """Test that predict method returns expected structure."""
        model = RegimeGRU()
        x = torch.randn(1, 21, len(CONTEXT_FEATURES))

        result = model.predict(x)

        assert 'regime_label' in result
        assert 'regime_probs' in result
        assert 'regime_embedding' in result
        assert result['regime_label'] in REGIME_LABELS
        assert len(result['regime_probs']) == len(REGIME_LABELS)
        assert len(result['regime_embedding']) == 8

    def test_batch_predict(self):
        """Test prediction on batch."""
        model = RegimeGRU()
        x = torch.randn(4, 21, len(CONTEXT_FEATURES))

        results = model.predict(x)

        assert isinstance(results, list)
        assert len(results) == 4


class TestRegimeTransformer:
    """Tests for RegimeTransformer model."""

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shapes."""
        batch_size = 8
        seq_length = 21
        input_dim = len(CONTEXT_FEATURES)

        model = RegimeTransformer(input_dim=input_dim)
        x = torch.randn(batch_size, seq_length, input_dim)

        output = model(x)

        assert output['logits'].shape == (batch_size, len(REGIME_LABELS))
        assert output['probs'].shape == (batch_size, len(REGIME_LABELS))
        assert output['embedding'].shape == (batch_size, 8)

    def test_predict_returns_correct_structure(self):
        """Test that predict method returns expected structure."""
        model = RegimeTransformer()
        x = torch.randn(1, 21, len(CONTEXT_FEATURES))

        result = model.predict(x)

        assert 'regime_label' in result
        assert 'regime_probs' in result
        assert 'regime_embedding' in result


class TestCreateRegimeModel:
    """Tests for create_regime_model factory."""

    def test_create_gru(self):
        """Test creating GRU model."""
        model = create_regime_model('gru')
        assert isinstance(model, RegimeGRU)

    def test_create_transformer(self):
        """Test creating Transformer model."""
        model = create_regime_model('transformer')
        assert isinstance(model, RegimeTransformer)

    def test_invalid_type_raises(self):
        """Test that invalid type raises error."""
        with pytest.raises(ValueError):
            create_regime_model('invalid_type')


class TestHealthAutoencoder:
    """Tests for HealthAutoencoder model."""

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shapes."""
        batch_size = 16
        input_dim = len(ASSET_FEATURES)
        latent_dim = 16

        model = HealthAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
        x = torch.randn(batch_size, input_dim)

        output = model(x)

        assert output['reconstruction'].shape == (batch_size, input_dim)
        assert output['latent'].shape == (batch_size, latent_dim)
        assert output['health_score'].shape == (batch_size, 1)
        assert output['vol_logits'].shape == (batch_size, len(VOL_BUCKETS))
        assert output['behavior_logits'].shape == (batch_size, len(BEHAVIORS))

    def test_health_score_in_range(self):
        """Test that health scores are in [0, 1]."""
        model = HealthAutoencoder()
        x = torch.randn(16, len(ASSET_FEATURES))

        output = model(x)

        assert (output['health_score'] >= 0).all()
        assert (output['health_score'] <= 1).all()

    def test_encode_decode_shape(self):
        """Test that encode/decode preserve shapes."""
        model = HealthAutoencoder()
        x = torch.randn(8, len(ASSET_FEATURES))

        latent = model.encode(x)
        reconstructed = model.decode(latent)

        assert latent.shape == (8, 16)
        assert reconstructed.shape == x.shape

    def test_predict_returns_correct_structure(self):
        """Test that predict method returns expected structure."""
        model = HealthAutoencoder()
        x = torch.randn(1, len(ASSET_FEATURES))

        results = model.predict(x)

        assert len(results) == 1
        result = results[0]
        assert 'health_score' in result
        assert 'vol_bucket' in result
        assert 'behavior' in result
        assert 'latent' in result
        assert result['vol_bucket'] in VOL_BUCKETS
        assert result['behavior'] in BEHAVIORS


class TestVariationalHealthAutoencoder:
    """Tests for VariationalHealthAutoencoder model."""

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shapes."""
        batch_size = 16
        input_dim = len(ASSET_FEATURES)
        latent_dim = 16

        model = VariationalHealthAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
        x = torch.randn(batch_size, input_dim)

        output = model(x)

        assert output['reconstruction'].shape == (batch_size, input_dim)
        assert output['latent'].shape == (batch_size, latent_dim)
        assert output['mu'].shape == (batch_size, latent_dim)
        assert output['log_var'].shape == (batch_size, latent_dim)

    def test_vae_loss_computation(self):
        """Test VAE loss function."""
        model = VariationalHealthAutoencoder()
        x = torch.randn(8, len(ASSET_FEATURES))

        output = model(x)
        loss, loss_dict = vae_loss(
            output['reconstruction'],
            x,
            output['mu'],
            output['log_var']
        )

        assert 'reconstruction' in loss_dict
        assert 'kl' in loss_dict
        assert 'total' in loss_dict
        assert loss.item() > 0


class TestCreateHealthModel:
    """Tests for create_health_model factory."""

    def test_create_autoencoder(self):
        """Test creating autoencoder model."""
        model = create_health_model('autoencoder')
        assert isinstance(model, HealthAutoencoder)

    def test_create_vae(self):
        """Test creating VAE model."""
        model = create_health_model('vae')
        assert isinstance(model, VariationalHealthAutoencoder)

    def test_invalid_type_raises(self):
        """Test that invalid type raises error."""
        with pytest.raises(ValueError):
            create_health_model('invalid_type')


class TestModelGradients:
    """Tests for model gradient flow."""

    def test_regime_model_gradients(self):
        """Test that gradients flow through regime model."""
        model = RegimeGRU()
        x = torch.randn(4, 21, len(CONTEXT_FEATURES), requires_grad=True)
        labels = torch.randint(0, len(REGIME_LABELS), (4,))

        output = model(x)
        loss = torch.nn.functional.cross_entropy(output['logits'], labels)
        loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_health_model_gradients(self):
        """Test that gradients flow through health model."""
        model = HealthAutoencoder()
        x = torch.randn(8, len(ASSET_FEATURES), requires_grad=True)

        output = model(x)
        loss = torch.nn.functional.mse_loss(output['reconstruction'], x)
        loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
