"""PKT-TB-004 ablation battery — cell definitions.

Single source of truth for the cell list. PREREG_BATTERY.json is generated
from this module (and hashed) BEFORE any holdout replay executes; the runner
refuses cells whose built bundle hash does not match the prereg entry.

Baseline C00 = champion-as-deployed (active bundle) with the stored nightly
LLM risks wired in (use_stored_llm_risks=True). Delta convention everywhere:
delta = cell − C00, so a layer's contribution = −delta of its LOO cell.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Callable, Dict, List

HOLDOUT_START = '2026-03-11'
LLM_ERA_START = '2026-01-29'
SEEDS = [11, 17, 23, 29, 31]

NEUTRAL_VOL_ADJ = {'low': 1.0, 'med': 1.0, 'high': 1.0}
NEUTRAL_REGIME_ADJ = {
    'calm_uptrend': 1.0, 'risk_on_trend': 1.0, 'choppy': 1.0,
    'risk_off_trend': 1.0, 'high_vol_panic': 1.0,
}
NEUTRAL_PSM_SIZE = {
    'panic_position_size': 1.0,
    'unstable_position_size': 1.0,
    'fragility_position_cap': 1.0,
    'entropy_size_multiplier': 1.0,
}
FLAT_CASH_FLOOR = {
    'calm_uptrend': 0.10, 'risk_on_trend': 0.10, 'choppy': 0.10,
    'risk_off_trend': 0.10, 'high_vol_panic': 0.10,
}


def canonical_hash(obj: Any) -> str:
    return 'sha256:' + hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(',', ':'), default=str).encode()
    ).hexdigest()


def _de(b: Dict) -> Dict:
    return b.setdefault('decision_engine', {})


def _abl(b: Dict) -> Dict:
    return _de(b).setdefault('ablation', {})


def _rf(b: Dict) -> Dict:
    return b.setdefault('regime_fusion', {})


def _ps(b: Dict) -> Dict:
    return _de(b).setdefault('position_size', {})


# (cell_id, tier, layers, description, mutator). Mutators receive a deep copy
# of the C00 bundle (LLM already wired ON) and apply ONE layer removal.
def _t1a(b):
    _ps(b)['vol_adj'] = dict(NEUTRAL_VOL_ADJ)


def _t1b(b):
    _ps(b)['regime_adj'] = dict(NEUTRAL_REGIME_ADJ)


def _t1llm(b):
    _de(b)['use_stored_llm_risks'] = False


def _t1d(b):
    _rf(b)['neutralize_ensemble_multiplier'] = True


def _t1e(b):
    _rf(b).update(NEUTRAL_PSM_SIZE)


def _t1f(b):
    _ps(b)['throttle_scale'] = 0.0


def _t1g(b):
    b['regime_compatibility'] = {}


def _t1h(b):
    b['decision_params'].pop('buy_score_threshold_by_regime', None)
    b['decision_params'].pop('min_health_buy_by_regime', None)


def _t1i(b):
    b['decision_params']['min_cash_reserve_by_regime'] = dict(FLAT_CASH_FLOOR)


def _t1j(b):
    _rf(b)['force_raw_ensemble_label'] = True


def _t1k(b):
    b['decision_params']['trailing_stop_base'] = 9.99
    b['decision_params']['trailing_stop_leveraged'] = 9.99


def _t1l(b):
    b['decision_params']['sell_health_threshold'] = -1.0


def _t1m(b):
    _abl(b)['disable_panic_force_sell'] = True


def _t1o(b):
    lc = dict(b['decision_params'].get('leveraged_constraints', {}))
    lc['max_hold_days'] = 100000
    b['decision_params']['leveraged_constraints'] = lc


def _t1p(b):
    b['decision_params']['reduce_health_drop'] = 0


def _t1q(b):
    _abl(b)['disable_reduce_regime_shift'] = True


def _t1s(b):
    _abl(b)['disable_vol_bucket_filter'] = True


def _t1t(b):
    _abl(b)['disable_panic_buy_filter'] = True


def _t1bc(b):
    _abl(b)['disable_llm_size_adj'] = True


def _t1bn(b):
    _abl(b)['disable_llm_sell_veto'] = True


def _t1br(b):
    _abl(b)['disable_llm_buy_veto'] = True


CELL_DEFS: List[tuple] = [
    ('C00', 0, [], 'baseline: champion as deployed, stored LLM risks wired (llm_mode=full)', lambda b: None),
    ('T1-A', 1, ['vol_adj'], 'vol_adj all 1.0', _t1a),
    ('T1-B', 1, ['regime_adj'], 'regime_adj all 1.0', _t1b),
    ('T1-LLM', 1, ['llm_size_adj', 'llm_buy_veto', 'llm_sell_veto'], 'LLM layer off (llm_mode=off)', _t1llm),
    ('T1-D', 1, ['ensemble_disagreement_multiplier'], 'ensemble multiplier neutralized at fusion rule 6', _t1d),
    ('T1-E', 1, ['psm_size_channel'], 'psm size channel neutral: panic/unstable size=1.0, fragility cap=1.0, entropy mult=1.0 (label+throttle kept)', _t1e),
    ('T1-F', 1, ['risk_throttle'], 'throttle consumption off: position_size.throttle_scale=0.0', _t1f),
    ('T1-G', 1, ['regime_compatibility'], 'regime_compatibility = {} (all multipliers 1.0)', _t1g),
    ('T1-H', 1, ['regime_conditional_buy_thresholds'], 'buy_score_threshold_by_regime removed -> flat champion fallbacks (NOTE: min_health_buy_by_regime absent in champion = partially DEGENERATE)', _t1h),
    ('T1-I', 1, ['regime_conditional_cash_floors'], 'min_cash_reserve_by_regime -> flat 0.10', _t1i),
    ('T1-J', 1, ['regime_label_steering'], 'force raw ensemble label (fusion size/throttle effects kept)', _t1j),
    ('T1-K', 1, ['trailing_stop'], 'trailing stops 9.99 (never fire)', _t1k),
    ('T1-L', 1, ['health_collapse_sell'], 'sell_health_threshold=-1.0 (never fires)', _t1l),
    ('T1-M', 1, ['panic_force_sell'], 'ablation.disable_panic_force_sell', _t1m),
    ('T1-O', 1, ['leverage_hold_cap'], 'leveraged max_hold_days=100000', _t1o),
    ('T1-P', 1, ['reduce_health_drop'], 'reduce_health_drop=0 (disabled)', _t1p),
    ('T1-Q', 1, ['reduce_regime_shift'], 'ablation.disable_reduce_regime_shift', _t1q),
    ('T1-S', 1, ['vol_bucket_buy_filter'], 'ablation.disable_vol_bucket_filter', _t1s),
    ('T1-T', 1, ['panic_buy_filter'], 'ablation.disable_panic_buy_filter', _t1t),
    # Tier 1b: conditional on T1-LLM paired |t| >= 2.0 on the LLM-era segment
    ('T1b-C', 1.5, ['llm_size_adj'], 'LLM size adjust off only (conditional)', _t1bc),
    ('T1b-N', 1.5, ['llm_sell_veto'], 'LLM sell veto off only (conditional)', _t1bn),
    ('T1b-R', 1.5, ['llm_buy_veto'], 'LLM buy veto off only (conditional)', _t1br),
]


def _alloff(b):
    for cid, tier, layers, desc, mut in CELL_DEFS:
        if tier == 1:
            mut(b)


CELL_DEFS.append(
    ('T2-ALLOFF', 2, ['ALL'], 'every Tier-1 toggle simultaneously (naked base strategy)', _alloff)
)


def build_cells(base_bundle: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build {cell_id: {'tier','layers','description','bundle','bundle_hash'}}."""
    cells = {}
    for cid, tier, layers, desc, mut in CELL_DEFS:
        b = copy.deepcopy(base_bundle)
        _de(b)['use_stored_llm_risks'] = True
        mut(b)
        cells[cid] = {
            'cell_id': cid,
            'tier': tier,
            'layers': layers,
            'description': desc,
            'bundle': b,
            'bundle_hash': canonical_hash(b),
        }
    return cells
