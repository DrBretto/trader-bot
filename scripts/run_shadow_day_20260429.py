"""Phase-6 shadow-day: replay 2026-04-29 with current_production params and
with each Phase-5 candidate variant. Produces an operator-reviewable diff of
what the new params WOULD HAVE decided today.

This is a simulation, not a live deploy. The actual live-deploy step is
operator-gated per packet rule.
"""

import copy
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from optimizer.config import OptimizerConfig  # noqa: E402
from optimizer.data_access import load_optimizer_dataset  # noqa: E402
from optimizer.fitness import compute_segment_metrics  # noqa: E402
from optimizer.replay import run_replay_for_dates  # noqa: E402
from training.models.ranking_mlp import RankingMLP, RANKING_FEATURES  # noqa: E402

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

VARIANTS = {
    'current_production':       {},
    'fixes_only_no_calibration': {},  # same overrides as current — fixes are code-only
    'cap_080_only':              {'fragility_position_cap': 0.80},
    'thr_095_only':              {'fragility_threshold': 0.95},
    'relax_conf_080':            {
        'fragility_relax_in_risk_on': True,
        'fragility_relax_confidence': 0.80,
    },
}


def _load_model(model_dir='models/ranking_expanded_unconditioned'):
    with open(Path(model_dir) / 'ranking_normalization.json') as f:
        norm = json.load(f)
    model = RankingMLP(input_dim=len(RANKING_FEATURES))
    model.load_state_dict(torch.load(Path(model_dir) / 'ranking_mlp.pt', weights_only=True))
    model.eval()
    return model, norm


def main():
    model, norm = _load_model()
    cfg = OptimizerConfig(bucket='investment-system-data', region='us-east-1',
                          max_days=900, train_days=10, test_days=10, step_days=5, gate_days=10)
    dataset = load_optimizer_dataset(cfg)
    today = '2026-04-28'  # last available date with full snapshot
    if today not in dataset.dates:
        today = dataset.dates[-1]
    print(f"shadow date: {today}")

    with open('config/decision_params.active.json') as f:
        active = json.load(f)
    blend = active.get('decision_engine', {}).get('ranking_blend', 0.35)

    out = {'shadow_date': today, 'blend': blend, 'variants': {}}
    for name, rf in VARIANTS.items():
        bundle = copy.deepcopy(active)
        bundle['regime_fusion'] = rf
        res = run_replay_for_dates(
            dataset=dataset, decision_dates=[today], candidate_bundle=bundle,
            random_seed=42, initial_capital=100000.0,
            ranking_model=model, ranking_normalization=norm, ranking_blend=blend,
        )
        actions = []
        for f in res.fills:
            actions.append({
                'action': f.get('action'), 'symbol': f.get('symbol'),
                'shares': f.get('shares'), 'price': f.get('price'),
                'dollars': f.get('dollars'), 'reason': f.get('reason'),
            })
        step = res.steps[0] if res.steps else None
        out['variants'][name] = {
            'overrides': rf,
            'actions': actions,
            'regime_label': step.regime if step else None,
            'actions_count': step.actions_count if step else 0,
            'start_value': step.start_value if step else None,
            'end_value': step.end_value if step else None,
            'traded_notional': step.traded_notional if step else 0,
        }

    p = Path('runs/shadow_day_20260429.json')
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"shadow day -> {p}")

    print('\n=== shadow-day variants ===')
    for name in VARIANTS:
        v = out['variants'][name]
        regime = v.get('regime_label', '?')
        cnt = v.get('actions_count', 0)
        sv = v.get('start_value', 0) or 0
        ev = v.get('end_value', 0) or 0
        notional = v.get('traded_notional', 0) or 0
        print(f"\n{name}: regime={regime} actions_count={cnt} traded_notional=${notional:,.2f} "
              f"start=${sv:,.2f} end=${ev:,.2f} delta=${ev-sv:+,.2f}")
        if not v['actions']:
            print(f"  (no actions)")
        for a in v['actions']:
            sym = a.get('symbol') or '?'
            sh = a.get('shares')
            pr = a.get('price')
            dol = a.get('dollars')
            print(f"  {a['action']:5s} {sym:6s} sh={sh!s:>10} @ ${pr!s:>8} = ${dol!s:>10} reason={a.get('reason')}")


if __name__ == '__main__':
    main()
