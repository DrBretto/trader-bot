"""Narrower rally-window sweep — test F-7 relax knob on 2026-04-06 → 2026-04-28
where the production lag actually emerged."""

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


def _load_model(model_dir='models/ranking_expanded_unconditioned'):
    with open(Path(model_dir) / 'ranking_normalization.json') as f:
        norm = json.load(f)
    model = RankingMLP(input_dim=len(RANKING_FEATURES))
    model.load_state_dict(torch.load(Path(model_dir) / 'ranking_mlp.pt', weights_only=True))
    model.eval()
    return model, norm


VARIANTS = {
    'current_production':           {},
    'thr_085_only':                 {'fragility_threshold': 0.85},
    'thr_095_only':                 {'fragility_threshold': 0.95},
    'thr_099_only':                 {'fragility_threshold': 0.99},
    'cap_080_only':                 {'fragility_position_cap': 0.80},
    'cap_100_only':                 {'fragility_position_cap': 1.00},
    'relax_conf_080_thr_075':       {
        'fragility_relax_in_risk_on': True,
        'fragility_relax_confidence': 0.80,
    },
    'relax_conf_070_thr_075':       {
        'fragility_relax_in_risk_on': True,
        'fragility_relax_confidence': 0.70,
    },
    'relax_conf_080_cap_100':       {
        'fragility_relax_in_risk_on': True,
        'fragility_relax_confidence': 0.80,
        'fragility_position_cap': 1.00,
    },
    'relax_aggressive':             {
        'fragility_relax_in_risk_on': True,
        'fragility_relax_confidence': 0.50,
        'fragility_position_cap': 1.00,
        'fragility_threshold': 0.99,
    },
}


def _run(dataset, dates, params, model, norm, blend):
    res = run_replay_for_dates(
        dataset=dataset, decision_dates=dates, candidate_bundle=params,
        random_seed=42, initial_capital=100000.0,
        ranking_model=model, ranking_normalization=norm, ranking_blend=blend,
    )
    return compute_segment_metrics(res.steps, res.fills, dates), res


def main():
    model, norm = _load_model()
    cfg = OptimizerConfig(bucket='investment-system-data', region='us-east-1',
                          max_days=900, train_days=10, test_days=10, step_days=5, gate_days=10)
    dataset = load_optimizer_dataset(cfg)

    # Two windows:
    # - rally: 2026-04-06 → 2026-04-28 (the production lag period)
    # - hybrid_live_full: 2026-03-21 → 2026-04-28 (since hybrid went live)
    all_dates = dataset.dates
    rally = [d for d in all_dates if '2026-04-06' <= d <= '2026-04-28']
    hybrid = [d for d in all_dates if '2026-03-21' <= d <= '2026-04-28']
    print(f"rally window:  {rally[0]}..{rally[-1]} ({len(rally)} days)")
    print(f"hybrid window: {hybrid[0]}..{hybrid[-1]} ({len(hybrid)} days)")

    with open('config/decision_params.active.json') as f:
        active = json.load(f)
    blend = active.get('decision_engine', {}).get('ranking_blend', 0.35)

    results = {'windows': {}, 'variants': {}}

    for win_name, dates in [('rally', rally), ('hybrid', hybrid)]:
        print(f"\n=== window: {win_name} ({len(dates)} days) ===")
        results['windows'][win_name] = {'start': dates[0], 'end': dates[-1], 'n': len(dates)}
        results['variants'][win_name] = {}
        for name, rf in VARIANTS.items():
            bundle = copy.deepcopy(active)
            bundle['regime_fusion'] = rf
            m, res = _run(dataset, dates, bundle, model, norm, blend)
            n_buys = sum(1 for f in res.fills if f.get('action') == 'BUY')
            n_sells = sum(1 for f in res.fills if f.get('action') == 'SELL')
            results['variants'][win_name][name] = {
                'overrides': rf, 'metrics': asdict(m),
                'fills': len(res.fills), 'buys': n_buys, 'sells': n_sells,
            }
            print(f"  {name:<28} ann={m.annualized_return*100:+7.2f}% "
                  f"sharpe={m.sharpe:+7.3f} mdd={m.max_drawdown*100:+6.2f}% "
                  f"trips={m.realized_round_trips:>3d} fills={len(res.fills):>3d}")

    out = Path('runs/rally_sweep_20260429.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nresults -> {out}")


if __name__ == '__main__':
    main()
