"""Model package exports.

Avoid importing optional `torch`-backed modules at package import time.
"""

from .baseline_regime import baseline_regime_model
from .baseline_health import baseline_health_model

__all__ = [
    "baseline_regime_model",
    "baseline_health_model",
    "ModelLoader",
    "EnsembleRegimeModel",
    "baseline_ensemble_regime",
]


def __getattr__(name):
    if name == "ModelLoader":
        from .loader import ModelLoader
        return ModelLoader
    if name in {"EnsembleRegimeModel", "baseline_ensemble_regime"}:
        from .ensemble_regime import EnsembleRegimeModel, baseline_ensemble_regime
        if name == "EnsembleRegimeModel":
            return EnsembleRegimeModel
        return baseline_ensemble_regime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
