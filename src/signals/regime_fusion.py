"""Regime Fusion v3 — multi-expert regime decision layer.

Combines ensemble regime model outputs with four expert signals to produce
a final regime label, confidence, position size modifier, and risk throttle.

Priority-ordered logic:
1. Hard override — Panic (vol or ensemble panic)
2. Hard override — Unstable calm (VVIX stress before VIX spike)
3. Caution gate — Fragility (cap exposure)
4. Caution gate — Entropy shift (reduce trust)
5. Macro modulation (shift regime based on macro backdrop)
6. Ensemble disagreement (existing logic preserved)
"""

import numpy as np
from typing import Dict, Any, Optional


def decide_regime_v3(
    ensemble_regime_label: str,
    trend_risk_on_prob: float,
    panic_prob: float,
    ensemble_disagreement: float,
    ensemble_multiplier: float,
    macro_credit_score: float,
    vol_uncertainty_score: float,
    vol_regime_label: str,
    fragility_score: float,
    entropy_score: float,
    entropy_shift_flag: bool,
) -> Dict[str, Any]:
    """
    Compute final regime decision from all expert inputs.

    Args:
        ensemble_regime_label: Raw label from ensemble regime model
        trend_risk_on_prob: Probability of risk-on trend from ensemble
        panic_prob: Probability of panic from ensemble
        ensemble_disagreement: Disagreement between GRU and Transformer (0-1)
        ensemble_multiplier: Position size multiplier from ensemble (0.5-1.0)
        macro_credit_score: Macro/credit expert score (-1 to +1)
        vol_uncertainty_score: Vol uncertainty score (0 to 1)
        vol_regime_label: Vol regime label (calm/unstable_calm/panic)
        fragility_score: Cross-asset fragility score (0 to 1)
        entropy_score: Entropy score (0 to 1)
        entropy_shift_flag: Whether entropy shift is detected

    Returns:
        Dict with final_regime_label, regime_confidence,
        position_size_modifier, risk_throttle_factor.
    """
    # Start with defaults from ensemble
    final_regime = ensemble_regime_label
    confidence = 1.0 - ensemble_disagreement
    position_size_mod = 1.0
    risk_throttle = 0.0

    override_reason = None

    # --- 1. Hard Override: Panic ---
    if panic_prob > 0.7 or vol_regime_label == 'panic':
        final_regime = 'high_vol_panic'
        position_size_mod = 0.25
        risk_throttle = 1.0
        confidence = max(panic_prob, vol_uncertainty_score)
        override_reason = 'panic_override'

    # --- 2. Hard Override: Unstable Calm ---
    elif vol_regime_label == 'unstable_calm' and panic_prob > 0.3:
        final_regime = 'risk_off_trend'
        position_size_mod = 0.50
        risk_throttle = 0.7
        confidence = vol_uncertainty_score
        override_reason = 'unstable_calm_override'

    # --- 3. Macro Modulation (only if no hard override) ---
    else:
        if macro_credit_score < -0.5 and final_regime in ('risk_on_trend', 'calm_uptrend'):
            final_regime = 'choppy'
            override_reason = 'macro_downgrade'
        elif macro_credit_score > 0.5 and final_regime == 'choppy':
            final_regime = 'risk_on_trend'
            override_reason = 'macro_upgrade'

    # --- 4. Caution Gate: Fragility ---
    if fragility_score > 0.75:
        position_size_mod = min(position_size_mod, 0.60)
        risk_throttle = min(risk_throttle + 0.2, 1.0)

    # --- 5. Caution Gate: Entropy Shift ---
    if entropy_shift_flag:
        position_size_mod *= 0.7
        risk_throttle = min(risk_throttle + 0.15, 1.0)

    # --- 6. Ensemble Disagreement (preserves existing behavior) ---
    position_size_mod *= ensemble_multiplier

    # --- Final Clamps ---
    position_size_mod = float(np.clip(position_size_mod, 0.25, 1.0))
    risk_throttle = float(np.clip(risk_throttle, 0.0, 1.0))
    confidence = float(np.clip(confidence, 0.0, 1.0))

    return {
        'final_regime_label': final_regime,
        'regime_confidence': confidence,
        'position_size_modifier': position_size_mod,
        'risk_throttle_factor': risk_throttle,
        'override_reason': override_reason,
    }
