"""Model loader for loading trained models from S3."""

import json
from typing import Dict, Any, Optional

from src.utils.s3_client import S3Client
from src.models.baseline_regime import baseline_regime_model
from src.models.baseline_health import baseline_health_model


class ModelLoader:
    """Loads models from S3 or falls back to baselines."""

    def __init__(self, s3_client: S3Client):
        self.s3 = s3_client
        self.regime_model = None
        self.health_model = None
        self.model_version = 'baseline_v1'

    def load_models(self) -> Dict[str, str]:
        """
        Attempt to load trained models from S3.

        Returns:
            Dict with model versions
        """
        # Try to load latest.json to find current model versions
        latest_config = self.s3.read_json('models/latest.json')

        if latest_config is None:
            # No trained models, use baselines
            print("No trained models found, using baselines")
            return {
                'regime': 'baseline_v1',
                'health': 'baseline_v1'
            }

        # In Phase 2+, we would load the actual model files here
        # For Phase 1, we always use baselines
        self.model_version = latest_config.get('version', 'baseline_v1')

        return {
            'regime': self.model_version,
            'health': self.model_version
        }

    def predict_regime(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict market regime.

        For Phase 1, always uses baseline model.
        """
        import pandas as pd

        # Convert dict to Series if needed
        if isinstance(context, dict):
            context = pd.Series(context)

        return baseline_regime_model(context)

    def predict_health(self, features_df, latest_date) -> 'pd.DataFrame':
        """
        Predict asset health scores.

        For Phase 1, always uses baseline model.
        """
        return baseline_health_model(features_df, latest_date)
