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
from typing import Dict, Any


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
    params: Dict[str, Any] | None = None,
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
    params = params or {}
    panic_prob_threshold = float(params.get('panic_prob_threshold', 0.70))
    unstable_calm_panic_threshold = float(params.get('unstable_calm_panic_threshold', 0.30))
    macro_downgrade_threshold = float(params.get('macro_downgrade_threshold', -0.50))
    macro_upgrade_threshold = float(params.get('macro_upgrade_threshold', 0.50))
    fragility_threshold = float(params.get('fragility_threshold', 0.75))
    entropy_size_multiplier = float(params.get('entropy_size_multiplier', 0.70))
    panic_position_size = float(params.get('panic_position_size', 0.25))
    panic_risk_throttle = float(params.get('panic_risk_throttle', 1.0))
    unstable_position_size = float(params.get('unstable_position_size', 0.50))
    unstable_risk_throttle = float(params.get('unstable_risk_throttle', 0.7))
    fragility_position_cap = float(params.get('fragility_position_cap', 0.60))
    fragility_throttle_increment = float(params.get('fragility_throttle_increment', 0.2))
    entropy_throttle_increment = float(params.get('entropy_throttle_increment', 0.15))
    throttle_to_exposure_scale = float(params.get('throttle_to_exposure_scale', 0.5))
    final_position_size_clip_min = float(params.get('final_position_size_clip_min', 0.25))
    final_position_size_clip_max = float(params.get('final_position_size_clip_max', 1.0))
    final_risk_throttle_clip_min = float(params.get('final_risk_throttle_clip_min', 0.0))
    final_risk_throttle_clip_max = float(params.get('final_risk_throttle_clip_max', 1.0))
    throttle_mapping = (
        "effective_exposure_multiplier = position_size_modifier * "
        f"(1 - {throttle_to_exposure_scale:.2f} * risk_throttle_factor)"
    )

    # Start with defaults from ensemble
    final_regime = ensemble_regime_label
    confidence = 1.0 - ensemble_disagreement
    position_size_mod = 1.0
    risk_throttle = 0.0
    override_reason = None
    hard_override = False
    fusion_rules = []

    def _add_rule(
        order: int,
        code: str,
        label: str,
        fired: bool,
        inputs: str,
        threshold: str,
        effect: str,
    ) -> None:
        fusion_rules.append(
            {
                'order': order,
                'code': code,
                'label': label,
                'fired': fired,
                'inputs': inputs,
                'threshold': threshold,
                'effect': effect,
            }
        )

    # --- 1. Hard Override: Panic ---
    panic_fired = panic_prob > panic_prob_threshold or vol_regime_label == 'panic'
    if panic_fired:
        final_regime = 'high_vol_panic'
        position_size_mod = panic_position_size
        risk_throttle = panic_risk_throttle
        confidence = max(panic_prob, vol_uncertainty_score)
        override_reason = 'panic_override'
        hard_override = True
    _add_rule(
        order=1,
        code='panic_override',
        label='Panic Override',
        fired=panic_fired,
        inputs=f"panic_prob={panic_prob:.2f}, vol_regime={vol_regime_label}",
        threshold=f"panic_prob>{panic_prob_threshold:.2f} OR vol_regime='panic'",
        effect=(
            f"regime={final_regime}, size={position_size_mod:.2f}, "
            f"throttle={risk_throttle:.2f}"
            if panic_fired
            else "no change"
        ),
    )

    # --- 2. Hard Override: Unstable Calm ---
    unstable_fired = (not hard_override) and (
        vol_regime_label == 'unstable_calm' and panic_prob > unstable_calm_panic_threshold
    )
    if unstable_fired:
        final_regime = 'risk_off_trend'
        position_size_mod = unstable_position_size
        risk_throttle = unstable_risk_throttle
        confidence = vol_uncertainty_score
        override_reason = 'unstable_calm_override'
        hard_override = True
    _add_rule(
        order=2,
        code='unstable_calm_override',
        label='Unstable Calm Override',
        fired=unstable_fired,
        inputs=f"vol_regime={vol_regime_label}, panic_prob={panic_prob:.2f}",
        threshold=(
            "vol_regime='unstable_calm' AND "
            f"panic_prob>{unstable_calm_panic_threshold:.2f}"
        ),
        effect=(
            f"regime={final_regime}, size={position_size_mod:.2f}, "
            f"throttle={risk_throttle:.2f}"
            if unstable_fired
            else ("skipped (higher-priority hard override)" if panic_fired else "no change")
        ),
    )

    # --- 3. Macro Modulation (only if no hard override) ---
    macro_fired = False
    if not hard_override:
        if macro_credit_score < macro_downgrade_threshold and final_regime in (
            'risk_on_trend',
            'calm_uptrend',
        ):
            final_regime = 'choppy'
            override_reason = 'macro_downgrade'
            macro_fired = True
        elif macro_credit_score > macro_upgrade_threshold and final_regime == 'choppy':
            final_regime = 'risk_on_trend'
            override_reason = 'macro_upgrade'
            macro_fired = True
    _add_rule(
        order=3,
        code='macro_modulation',
        label='Macro Modulation',
        fired=macro_fired,
        inputs=f"macro_credit_score={macro_credit_score:.2f}, regime={ensemble_regime_label}",
        threshold=(
            f"score<{macro_downgrade_threshold:.2f} (downgrade) OR "
            f"score>{macro_upgrade_threshold:.2f} (upgrade)"
        ),
        effect=(
            f"regime={final_regime}"
            if macro_fired
            else ("skipped (hard override active)" if hard_override else "no change")
        ),
    )

    # --- 4. Caution Gate: Fragility ---
    fragility_fired = fragility_score > fragility_threshold
    if fragility_fired:
        position_size_mod = min(position_size_mod, fragility_position_cap)
        risk_throttle = min(risk_throttle + fragility_throttle_increment, 1.0)
    _add_rule(
        order=4,
        code='fragility_gate',
        label='Fragility Gate',
        fired=fragility_fired,
        inputs=f"fragility_score={fragility_score:.2f}",
        threshold=f"fragility_score>{fragility_threshold:.2f}",
        effect=(
            f"size={position_size_mod:.2f}, throttle={risk_throttle:.2f}"
            if fragility_fired
            else "no change"
        ),
    )

    # --- 5. Caution Gate: Entropy Shift ---
    entropy_fired = bool(entropy_shift_flag)
    if entropy_fired:
        position_size_mod *= entropy_size_multiplier
        risk_throttle = min(risk_throttle + entropy_throttle_increment, 1.0)
    _add_rule(
        order=5,
        code='entropy_shift',
        label='Entropy Shift Gate',
        fired=entropy_fired,
        inputs=f"entropy_shift_flag={entropy_shift_flag}, entropy_score={entropy_score:.2f}",
        threshold="entropy_shift_flag=True",
        effect=(
            f"size×{entropy_size_multiplier:.2f}, throttle={risk_throttle:.2f}"
            if entropy_fired
            else "no change"
        ),
    )

    # --- 6. Ensemble Disagreement (preserves existing behavior) ---
    disagreement_fired = abs(ensemble_multiplier - 1.0) > 1e-9
    position_size_mod *= ensemble_multiplier
    _add_rule(
        order=6,
        code='ensemble_disagreement',
        label='Ensemble Disagreement Sizing',
        fired=disagreement_fired,
        inputs=(
            f"disagreement={ensemble_disagreement:.2f}, "
            f"ensemble_multiplier={ensemble_multiplier:.2f}"
        ),
        threshold="multiplier<1.00 when disagreement is elevated",
        effect=f"size={position_size_mod:.2f}",
    )

    # --- Final Clamps ---
    position_size_mod = float(np.clip(
        position_size_mod,
        final_position_size_clip_min,
        final_position_size_clip_max,
    ))
    risk_throttle = float(np.clip(
        risk_throttle,
        final_risk_throttle_clip_min,
        final_risk_throttle_clip_max,
    ))
    confidence = float(np.clip(confidence, 0.0, 1.0))
    effective_exposure_multiplier = float(
        np.clip(position_size_mod * (1.0 - throttle_to_exposure_scale * risk_throttle), 0.0, 1.0)
    )

    return {
        'final_regime_label': final_regime,
        'regime_confidence': confidence,
        'position_size_modifier': position_size_mod,
        'risk_throttle_factor': risk_throttle,
        'override_reason': override_reason,
        'effective_exposure_multiplier': effective_exposure_multiplier,
        'target_gross_exposure': effective_exposure_multiplier,
        'throttle_mapping': throttle_mapping,
        'fusion_rules': fusion_rules,
    }
