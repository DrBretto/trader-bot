"""Model loader for loading trained models from S3."""

import json
import pickle
import io
import numpy as np
from typing import Dict, Any, Optional

from src.utils.s3_client import S3Client
from src.models.baseline_regime import baseline_regime_model
from src.models.baseline_health import baseline_health_model
from src.models.ensemble_regime import EnsembleRegimeModel, baseline_ensemble_regime

# Try to import PyTorch (optional for Lambda)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TrainedRegimeModel:
    """Wrapper for trained regime model."""

    def __init__(self, model_data: dict):
        """
        Initialize from pickled model data.

        Args:
            model_data: Dict containing model_state, model_config, normalization
        """
        self.config = model_data.get('model_config', {})
        self.feature_cols = model_data.get('feature_cols', [])
        self.normalization = model_data.get('normalization', {})
        self.version = model_data.get('version', 'unknown')

        # Lazy load model on first prediction
        self._model = None
        self._model_state = model_data.get('model_state')

    def _load_model(self):
        """Lazy load PyTorch model."""
        if self._model is not None:
            return

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available. Cannot load trained model.")

        # Import model class
        from training.models.regime_transformer import create_regime_model

        self._model = create_regime_model(
            model_type=self.config.get('model_type', 'gru'),
            input_dim=self.config.get('input_dim', 10),
            hidden_dim=self.config.get('hidden_dim', 64),
            num_layers=self.config.get('num_layers', 2),
            num_classes=self.config.get('num_classes', 5),
            embedding_dim=self.config.get('embedding_dim', 8)
        )
        self._model.load_state_dict(self._model_state)
        self._model.eval()

    def predict(self, context) -> Dict[str, Any]:
        """
        Predict regime from context features.

        Args:
            context: Dict or Series with context features

        Returns:
            Dict with regime_label, regime_probs, regime_embedding
        """
        self._load_model()

        import pandas as pd

        # Extract features in correct order
        if isinstance(context, dict):
            context = pd.Series(context)

        features = []
        for col in self.feature_cols:
            val = context.get(col, 0.0)
            if pd.isna(val):
                val = 0.0
            features.append(float(val))

        # Normalize
        features = np.array(features, dtype=np.float32)
        mean = np.array(self.normalization.get('mean', [0] * len(features)))
        std = np.array(self.normalization.get('std', [1] * len(features)))
        features = (features - mean) / std

        # Create sequence (repeat for seq_length)
        seq_length = self.config.get('seq_length', 21)
        seq = np.tile(features, (seq_length, 1))

        # Convert to tensor
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

        # Predict
        return self._model.predict(x)


class TrainedHealthModel:
    """Wrapper for trained health model."""

    def __init__(self, model_data: dict):
        """
        Initialize from pickled model data.

        Args:
            model_data: Dict containing model_state, model_config, normalization
        """
        self.config = model_data.get('model_config', {})
        self.feature_cols = model_data.get('feature_cols', [])
        self.normalization = model_data.get('normalization', {})
        self.version = model_data.get('version', 'unknown')

        # Lazy load model
        self._model = None
        self._model_state = model_data.get('model_state')

    def _load_model(self):
        """Lazy load PyTorch model."""
        if self._model is not None:
            return

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available. Cannot load trained model.")

        from training.models.health_autoencoder import create_health_model

        self._model = create_health_model(
            model_type=self.config.get('model_type', 'autoencoder'),
            input_dim=self.config.get('input_dim', 10),
            latent_dim=self.config.get('latent_dim', 16),
            hidden_dims=self.config.get('hidden_dims', [64, 32]),
            num_vol_buckets=self.config.get('num_vol_buckets', 3),
            num_behaviors=self.config.get('num_behaviors', 3)
        )
        self._model.load_state_dict(self._model_state)
        self._model.eval()

    def predict(self, features_df, latest_date):
        """
        Predict health scores for assets.

        Args:
            features_df: DataFrame with asset features
            latest_date: Date to score

        Returns:
            DataFrame with health scores
        """
        import pandas as pd

        self._load_model()

        # Filter to latest date
        latest = features_df[features_df['date'] == latest_date].copy()

        if len(latest) == 0:
            return pd.DataFrame()

        # Extract features
        feature_matrix = []
        for _, row in latest.iterrows():
            features = []
            for col in self.feature_cols:
                val = row.get(col, 0.0)
                if pd.isna(val):
                    val = 0.0
                features.append(float(val))
            feature_matrix.append(features)

        # Normalize
        features = np.array(feature_matrix, dtype=np.float32)
        mean = np.array(self.normalization.get('mean', [0] * features.shape[1]))
        std = np.array(self.normalization.get('std', [1] * features.shape[1]))
        features = (features - mean) / std

        # Convert to tensor
        x = torch.tensor(features, dtype=torch.float32)

        # Predict
        predictions = self._model.predict(x)

        # Build result DataFrame
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                'symbol': latest.iloc[i].get('symbol', f'UNKNOWN_{i}'),
                'health_score': pred['health_score'],
                'vol_bucket': pred['vol_bucket'],
                'behavior': pred['behavior'],
                'latent': pred['latent']
            })

        return pd.DataFrame(results)


class ModelLoader:
    """Loads models from S3 or falls back to baselines."""

    def __init__(self, s3_client: S3Client):
        self.s3 = s3_client
        self.regime_model = None
        self.ensemble_model = None
        self.health_model = None
        self.model_version = 'baseline_v1'
        self.using_trained_models = False
        self.is_ensemble = False

    def load_models(self) -> Dict[str, str]:
        """
        Attempt to load trained models from S3.

        Returns:
            Dict with model versions
        """
        # Try to load latest.json to find current model versions
        latest_config = self.s3.read_json('models/latest.json')

        if latest_config is None:
            print("No trained models found, using baselines")
            return {
                'regime': 'baseline_v1',
                'health': 'baseline_v1',
                'ensemble': False
            }

        self.model_version = latest_config.get('version', 'baseline_v1')
        self.is_ensemble = latest_config.get('ensemble', False)

        # Try to load trained models
        if TORCH_AVAILABLE:
            try:
                # Check if ensemble mode
                if self.is_ensemble:
                    gru_key = latest_config.get('regime_gru_model')
                    transformer_key = latest_config.get('regime_transformer_model')

                    if gru_key and transformer_key:
                        print(f"Loading ensemble regime models...")
                        print(f"  GRU: {gru_key}")
                        print(f"  Transformer: {transformer_key}")

                        gru_data = self._load_pickle(gru_key)
                        transformer_data = self._load_pickle(transformer_key)

                        if gru_data and transformer_data:
                            # Create ensemble wrapper
                            self.ensemble_model = EnsembleRegimeModel(
                                gru_model=gru_data.get('model'),
                                transformer_model=transformer_data.get('model'),
                                gru_weight=0.5,
                                transformer_weight=0.5,
                                disagreement_threshold=0.3
                            )
                            self.regime_model = self.ensemble_model
                            print(f"  Loaded ensemble regime model v{self.model_version}")
                else:
                    # Load single regime model
                    regime_key = latest_config.get('regime_model')
                    if regime_key:
                        print(f"Loading trained regime model: {regime_key}")
                        regime_data = self._load_pickle(regime_key)
                        if regime_data:
                            self.regime_model = TrainedRegimeModel(regime_data)
                            print(f"  Loaded regime model v{self.regime_model.version}")

                # Load health model
                health_key = latest_config.get('health_model')
                if health_key:
                    print(f"Loading trained health model: {health_key}")
                    health_data = self._load_pickle(health_key)
                    if health_data:
                        self.health_model = TrainedHealthModel(health_data)
                        print(f"  Loaded health model v{self.health_model.version}")

                if self.regime_model and self.health_model:
                    self.using_trained_models = True

            except Exception as e:
                print(f"Error loading trained models: {e}")
                print("Falling back to baseline models")
                self.regime_model = None
                self.ensemble_model = None
                self.health_model = None
        else:
            print("PyTorch not available, using baseline models")

        return {
            'regime': f'ensemble_v{self.model_version}' if self.is_ensemble and self.ensemble_model else (
                f'trained_v{self.model_version}' if self.regime_model else 'baseline_v1'
            ),
            'health': f'trained_v{self.model_version}' if self.health_model else 'baseline_v1',
            'ensemble': self.is_ensemble and self.ensemble_model is not None
        }

    def _load_pickle(self, key: str) -> Optional[dict]:
        """Load pickled model from S3."""
        try:
            data = self.s3.read_bytes(key)
            if data:
                return pickle.loads(data)
        except Exception as e:
            print(f"Error loading pickle {key}: {e}")
        return None

    def predict_regime(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict market regime.

        Uses trained model if available, otherwise baseline.
        For ensemble models, includes disagreement and confidence info.

        Returns:
            Dict with:
                - regime_label: Predicted regime
                - regime_probs: Probability distribution
                - regime_embedding: Latent embedding
                - confidence: Model confidence (ensemble: agreement-weighted)
                - disagreement: Model disagreement (0-1, ensemble only)
                - position_size_multiplier: Sizing adjustment (ensemble only)
        """
        import pandas as pd

        # Convert dict to Series if needed
        if isinstance(context, dict):
            context = pd.Series(context)

        # Try ensemble model first
        if self.is_ensemble and self.ensemble_model:
            try:
                result = self.ensemble_model.predict(context)
                return result
            except Exception as e:
                print(f"Ensemble regime model error: {e}, falling back to baseline")
                return baseline_ensemble_regime(context)

        # Try single trained model
        if self.regime_model:
            try:
                result = self.regime_model.predict(context)
                # Add default ensemble fields for compatibility
                result['confidence'] = max(result.get('regime_probs', {}).values()) if result.get('regime_probs') else 1.0
                result['disagreement'] = 0.0
                result['agreement'] = 1.0
                result['position_size_multiplier'] = 1.0
                return result
            except Exception as e:
                print(f"Trained regime model error: {e}, falling back to baseline")

        # Baseline fallback
        result = baseline_regime_model(context)
        result['confidence'] = 1.0
        result['disagreement'] = 0.0
        result['agreement'] = 1.0
        result['position_size_multiplier'] = 1.0
        return result

    def predict_health(self, features_df, latest_date) -> 'pd.DataFrame':
        """
        Predict asset health scores.

        Uses trained model if available, otherwise baseline.
        """
        if self.health_model:
            try:
                return self.health_model.predict(features_df, latest_date)
            except Exception as e:
                print(f"Trained health model error: {e}, falling back to baseline")

        return baseline_health_model(features_df, latest_date)
