"""PKT-TB-004 ablation battery runner.

Runs every cell in battery_cells.CELL_DEFS on the optimizer harness:
full-period replay (all available dates) + holdout replay (from cash at
2026-03-11), 5 seeds each. Writes per-cell results.json + manifest.json under
runs/pkt_tb_004_control_attribution/cells/<cell_id>/.

Prereg discipline: refuses to run any cell whose built bundle hash does not
match the committed PREREG_BATTERY.json entry.

Usage:
    .venv/bin/python runs/pkt_tb_004_control_attribution/scripts/battery_runner.py \
        [--cells C00,T1-A,...] [--skip-tier1b]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(__file__))

from battery_cells import (  # noqa: E402
    HOLDOUT_START, LLM_ERA_START, SEEDS, build_cells, canonical_hash,
)

RUN_DIR = os.path.join(REPO, 'runs', 'pkt_tb_004_control_attribution')
PREREG_PATH = os.path.join(RUN_DIR, 'prereg', 'PREREG_BATTERY.json')


def compact_fills(fills):
    out = []
    for f in fills:
        out.append({
            'date': f.get('_valuation_date'),
            'symbol': f.get('symbol'),
            'action': f.get('action'),
            'shares': f.get('shares'),
            'price': f.get('price'),
            'market_price': f.get('market_price'),
            'dollars': f.get('dollars'),
            'cost_bps': f.get('transaction_cost_bps'),
            'pnl': f.get('pnl'),
            'reason': f.get('reason'),
        })
    return out


def run_one(ds, dates, bundle, seed, ranking_model, ranking_norm, blend):
    from optimizer.replay import run_replay_for_dates
    t0 = time.time()
    res = run_replay_for_dates(ds, dates, bundle, seed, 100000.0,
                               ranking_model, ranking_norm, blend)
    wall = time.time() - t0
    steps = res.steps
    return {
        'seed': seed,
        'wall_clock_s': round(wall, 1),
        'valuation_dates': [s.valuation_date for s in steps],
        'values': [round(s.end_value, 4) for s in steps],
        'start_value': steps[0].start_value if steps else None,
        'end_cash': [round(s.end_cash, 4) for s in steps],
        'regimes': [s.regime for s in steps],
        'cumulative_transaction_costs': round(
            float(res.final_portfolio.get('cumulative_transaction_costs', 0.0)), 4),
        'final_value': round(float(res.final_portfolio.get('portfolio_value', 0.0)), 4),
        'fills': compact_fills(res.fills),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', default='')
    ap.add_argument('--skip-tier1b', action='store_true',
                    help='skip the conditional T1b cells (run later if triggered)')
    args = ap.parse_args()

    from optimizer.config import load_optimizer_config
    from optimizer.data_access import load_optimizer_dataset
    from optimizer.promote import load_active_bundle
    from src.utils.transaction_costs import get_cost_config_snapshot

    code_sha = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO).decode().strip()
    git_dirty = bool(subprocess.check_output(
        ['git', 'status', '--porcelain', 'src', 'optimizer'], cwd=REPO).decode().strip())

    with open(PREREG_PATH) as f:
        prereg = json.load(f)
    prereg_cells = {c['cell_id']: c for c in prereg['cells']}
    prereg_hash = canonical_hash(prereg)

    cfg = load_optimizer_config(os.path.join(REPO, 'config/optimizer.committee_20260606.json'))
    cfg.holdout_start = ''  # full dataset; holdout discipline handled here
    print('loading dataset...', flush=True)
    ds = load_optimizer_dataset(cfg)
    dates = sorted(str(s.date) for s in ds.snapshots)
    hold_dates = [d for d in dates if d >= HOLDOUT_START]
    print(f'{len(dates)} dates {dates[0]}..{dates[-1]}; holdout {len(hold_dates)} from {hold_dates[0]}', flush=True)

    base = load_active_bundle(cfg.config_dir_path)

    # Ranking model: canon runs blend 0.35; battery must match. Hard-fail if absent.
    import torch
    from training.models.ranking_mlp import RankingMLP, RANKING_FEATURES
    mdir = os.path.join(REPO, 'models/ranking_expanded_unconditioned')
    ranking_model = RankingMLP(input_dim=len(RANKING_FEATURES))
    ranking_model.load_state_dict(torch.load(os.path.join(mdir, 'ranking_mlp.pt'),
                                             weights_only=True))
    ranking_model.eval()
    ranking_norm = json.load(open(os.path.join(mdir, 'ranking_normalization.json')))
    blend = float(base.get('decision_engine', {}).get('ranking_blend', 0.0) or 0.0)
    print(f'ranking model loaded, blend={blend}', flush=True)

    cells = build_cells(base)
    only = [c.strip() for c in args.cells.split(',') if c.strip()]

    cost_cfg_hash = canonical_hash(get_cost_config_snapshot())

    for cid, cell in cells.items():
        if only and cid not in only:
            continue
        if args.skip_tier1b and cell['tier'] == 1.5:
            continue
        pre = prereg_cells.get(cid)
        if pre is None:
            print(f'!! {cid}: not in prereg — REFUSING', flush=True)
            continue
        if pre['bundle_hash'] != cell['bundle_hash']:
            print(f'!! {cid}: bundle hash mismatch vs prereg — REFUSING', flush=True)
            continue

        cdir = os.path.join(RUN_DIR, 'cells', cid)
        os.makedirs(cdir, exist_ok=True)
        if os.path.exists(os.path.join(cdir, 'results.json')):
            print(f'-- {cid}: results exist, skipping', flush=True)
            continue

        print(f'== {cid}: {cell["description"]}', flush=True)
        with open(os.path.join(cdir, 'bundle.json'), 'w') as f:
            json.dump(cell['bundle'], f, indent=1, sort_keys=True)

        results = {'full': [], 'holdout': []}
        walls = {'full': [], 'holdout': []}
        for seed in SEEDS:
            r_full = run_one(ds, dates, cell['bundle'], seed,
                             ranking_model, ranking_norm, blend)
            r_hold = run_one(ds, hold_dates, cell['bundle'], seed,
                             ranking_model, ranking_norm, blend)
            results['full'].append(r_full)
            results['holdout'].append(r_hold)
            walls['full'].append(r_full['wall_clock_s'])
            walls['holdout'].append(r_hold['wall_clock_s'])
            print(f'   seed {seed}: full={r_full["final_value"]:.0f} '
                  f'hold={r_hold["final_value"]:.0f}', flush=True)

        with open(os.path.join(cdir, 'results.json'), 'w') as f:
            json.dump(results, f)

        is_llm_cell = cid in ('T1-LLM', 'T1b-C', 'T1b-N', 'T1b-R')
        manifest = {
            'cell_id': cid,
            'tier': cell['tier'],
            'layers': cell['layers'],
            'description': cell['description'],
            'harness': 'optimizer',
            'baseline_cell': 'C00',
            'delta_convention': 'cell_minus_baseline',
            'code_sha': code_sha,
            'git_dirty': git_dirty,
            'bundle_path': f'cells/{cid}/bundle.json',
            'bundle_hash': cell['bundle_hash'],
            'seeds': SEEDS,
            'replays': [
                {'kind': 'full', 'date_range': [dates[0], dates[-1]],
                 'initial_capital': 100000.0},
                {'kind': 'holdout', 'date_range': [hold_dates[0], hold_dates[-1]],
                 'initial_capital': 100000.0},
            ],
            'data': {
                'bucket': 'investment-system-data',
                'snapshot_prefix': 'daily/',
                'n_decision_dates': {'full': len(dates), 'holdout': len(hold_dates)},
                'holdout_start': HOLDOUT_START,
                'llm_era_start': LLM_ERA_START,
                'llm_metrics_segment': [LLM_ERA_START, dates[-1]] if is_llm_cell else None,
            },
            'cost_model': {
                'source': 'src/utils/transaction_costs.py',
                'config_hash': cost_cfg_hash,
                'slippage_range_bps': 2.0,
            },
            'prereg': {'path': 'prereg/PREREG_BATTERY.json', 'hash': prereg_hash},
            'command': (f'.venv/bin/python runs/pkt_tb_004_control_attribution/'
                        f'scripts/battery_runner.py --cells {cid}'),
            'wall_clock_s': walls,
            'seed_noise_bound_bps': None,  # filled by battery_assemble.py
            'results_path': f'cells/{cid}/results.json',
            'status': 'RUN',
        }
        with open(os.path.join(cdir, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=1)

    print('battery runner done', flush=True)


if __name__ == '__main__':
    main()
