"""PKT-TB-004 Risk Architect — pre-holdout tuning harness.

Tunes the two Risk Architect candidates on PRE-HOLDOUT dates only
(< 2026-03-11; the holdout is read ONCE by the chair from the prereg):

  1. exposure_trim  — decision_engine_overrides['exposure_trim'] grid
  2. vixy_policy    — decision_params['vol_decay_constraints'] grid

Baseline = C00 bundle (active bundle + use_stored_llm_risks=True), replayed
over the same pre-holdout dates with the same seeds. Every variant tried is
logged to candidates/<name>/variants.json per EVIDENCE_PROTOCOL ("Every
variant logged").

Usage:
    .venv/bin/python runs/pkt_tb_004_control_attribution/scripts/risk_arch_tuner.py \
        [--candidate exposure_trim|vixy_policy|both] [--seeds 11,17,23]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(__file__))

from battery_cells import HOLDOUT_START, canonical_hash  # noqa: E402

RUN_DIR = os.path.join(REPO, 'runs', 'pkt_tb_004_control_attribution')
SCREEN_SEEDS = [11, 17, 23]
FULL_SEEDS = [11, 17, 23, 29, 31]


def metrics_from_run(run: dict) -> dict:
    values = run['values']
    cash = run['end_cash']
    start = run['start_value'] or 100000.0
    rets = []
    prev = start
    for v in values:
        rets.append(v / prev - 1.0 if prev > 0 else 0.0)
        prev = v
    n = len(values)
    total_return = values[-1] / start - 1.0
    years = n / 252.0
    cagr = (values[-1] / start) ** (1 / years) - 1.0 if years > 0 else 0.0
    mean_r = statistics.mean(rets) if rets else 0.0
    sd_r = statistics.stdev(rets) if len(rets) > 1 else 0.0
    sharpe = (mean_r / sd_r) * math.sqrt(252) if sd_r > 0 else 0.0
    peak = start
    max_dd = 0.0
    for v in values:
        peak = max(peak, v)
        max_dd = min(max_dd, v / peak - 1.0)
    invested = [1.0 - c / v for c, v in zip(cash, values) if v > 0]
    fills = run['fills']
    sells = [f for f in fills if f['action'] in ('SELL', 'REDUCE')]
    closed = [f for f in sells if f.get('pnl') is not None]
    wins = sum(1 for f in closed if f['pnl'] > 0)
    return {
        'final_value': values[-1],
        'total_return': total_return,
        'cagr': cagr,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'avg_gross_exposure': statistics.mean(invested) if invested else 0.0,
        'n_fills': len(fills),
        'n_round_trips': len(closed),
        'win_rate': wins / len(closed) if closed else None,
        'traded_notional': sum(f['dollars'] or 0 for f in fills),
        'cumulative_transaction_costs': run['cumulative_transaction_costs'],
        'trim_fills': sum(1 for f in fills if f.get('reason') == 'EXPOSURE_TRIM'),
        'trim_notional': sum(f['dollars'] or 0 for f in fills
                             if f.get('reason') == 'EXPOSURE_TRIM'),
        'decay_cap_fills': sum(1 for f in fills
                               if f.get('reason') == 'VOL_DECAY_HOLD_CAP'),
    }


def paired_daily_t(base_run: dict, cand_run: dict) -> dict:
    """Paired t-stat of daily-return deltas on identical valuation dates."""
    base = dict(zip(base_run['valuation_dates'], base_run['values']))
    cand = dict(zip(cand_run['valuation_dates'], cand_run['values']))
    dates = sorted(set(base) & set(cand))
    if len(dates) < 3:
        return {'t': None, 'mean': None, 'sd': None, 'n': len(dates)}
    deltas = []
    pb = base_run['start_value'] or 100000.0
    pc = cand_run['start_value'] or 100000.0
    for d in dates:
        rb = base[d] / pb - 1.0
        rc = cand[d] / pc - 1.0
        deltas.append(rc - rb)
        pb, pc = base[d], cand[d]
    m = statistics.mean(deltas)
    sd = statistics.stdev(deltas)
    t = m / (sd / math.sqrt(len(deltas))) if sd > 0 else 0.0
    return {'t': t, 'mean': m, 'sd': sd, 'n': len(deltas)}


def run_variant(ds, dates, bundle, seeds, ranking_model, ranking_norm, blend):
    from optimizer.replay import run_replay_for_dates
    out = []
    for seed in seeds:
        t0 = time.time()
        res = run_replay_for_dates(ds, dates, bundle, seed, 100000.0,
                                   ranking_model, ranking_norm, blend)
        steps = res.steps
        out.append({
            'seed': seed,
            'wall_clock_s': round(time.time() - t0, 1),
            'valuation_dates': [s.valuation_date for s in steps],
            'values': [round(s.end_value, 4) for s in steps],
            'start_value': steps[0].start_value if steps else None,
            'end_cash': [round(s.end_cash, 4) for s in steps],
            'cumulative_transaction_costs': round(
                float(res.final_portfolio.get('cumulative_transaction_costs', 0.0)), 4),
            'fills': [{
                'date': f.get('_valuation_date'), 'symbol': f.get('symbol'),
                'action': f.get('action'), 'shares': f.get('shares'),
                'dollars': f.get('dollars'), 'pnl': f.get('pnl'),
                'reason': f.get('reason'),
            } for f in res.fills],
        })
    return out


def summarize(variant_runs, base_runs):
    per_seed = []
    for br, vr in zip(base_runs, variant_runs):
        m = metrics_from_run(vr)
        bm = metrics_from_run(br)
        t = paired_daily_t(br, vr)
        per_seed.append({
            'seed': vr['seed'],
            'metrics': m,
            'delta_total_return': m['total_return'] - bm['total_return'],
            'delta_sharpe': m['sharpe'] - bm['sharpe'],
            'delta_max_dd': m['max_drawdown'] - bm['max_drawdown'],
            'delta_exposure': m['avg_gross_exposure'] - bm['avg_gross_exposure'],
            'delta_fills': m['n_fills'] - bm['n_fills'],
            'delta_costs': (m['cumulative_transaction_costs']
                            - bm['cumulative_transaction_costs']),
            'paired_daily_t': t,
        })
    med = sorted(per_seed, key=lambda r: r['delta_total_return'])[len(per_seed) // 2]
    return {
        'per_seed': per_seed,
        'median_seed': med['seed'],
        'median_delta_total_return': med['delta_total_return'],
        'median_delta_sharpe': med['delta_sharpe'],
        'median_delta_max_dd': med['delta_max_dd'],
        'median_paired_t': med['paired_daily_t']['t'],
        'sign_consistent': (len({r['delta_total_return'] > 0 for r in per_seed}) == 1
                            or all(abs(r['delta_total_return']) < 1e-12
                                   for r in per_seed)),
        'mean_trim_fills': statistics.mean(
            r['metrics']['trim_fills'] for r in per_seed),
        'mean_trim_notional': statistics.mean(
            r['metrics']['trim_notional'] for r in per_seed),
        'mean_delta_costs': statistics.mean(r['delta_costs'] for r in per_seed),
    }


def exposure_trim_grid():
    variants = []
    for trigger in (0.05, 0.10, 0.15, 0.25):
        for hyst in (0.0, 0.05):
            for persist in (1, 2):
                variants.append({
                    'enabled': True, 'trigger_gap': trigger,
                    'hysteresis_gap': hyst, 'persistence_days': persist,
                })
    # targeted extras around the plausible operating point
    variants.append({'enabled': True, 'trigger_gap': 0.10, 'hysteresis_gap': 0.05,
                     'persistence_days': 1, 'block_buys_when_trimming': False})
    variants.append({'enabled': True, 'trigger_gap': 0.10, 'hysteresis_gap': 0.05,
                     'persistence_days': 1, 'skip_regimes': ['high_vol_panic']})
    variants.append({'enabled': True, 'trigger_gap': 0.10, 'hysteresis_gap': 0.05,
                     'persistence_days': 1, 'target_floor': 0.30})
    variants.append({'enabled': True, 'trigger_gap': 0.15, 'hysteresis_gap': 0.05,
                     'persistence_days': 1, 'min_trim_dollars': 500})
    return variants


def vixy_grid():
    return [{'max_hold_days': d, 'sectors': ['volatility']}
            for d in (5, 10, 15, 21)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate', default='both',
                    choices=['exposure_trim', 'vixy_policy', 'both'])
    ap.add_argument('--seeds', default=','.join(map(str, SCREEN_SEEDS)))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]

    from optimizer.config import load_optimizer_config
    from optimizer.data_access import load_optimizer_dataset
    from optimizer.promote import load_active_bundle
    from src.utils.transaction_costs import get_cost_config_snapshot

    code_sha = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO).decode().strip()

    cfg = load_optimizer_config(os.path.join(REPO, 'config/optimizer.committee_20260606.json'))
    cfg.holdout_start = HOLDOUT_START  # tuner NEVER loads holdout dates
    print('loading pre-holdout dataset...', flush=True)
    ds = load_optimizer_dataset(cfg)
    dates = sorted(str(s.date) for s in ds.snapshots)
    assert all(d < HOLDOUT_START for d in dates), 'holdout date leaked into tuner'
    print(f'{len(dates)} pre-holdout dates {dates[0]}..{dates[-1]}', flush=True)

    base = load_active_bundle(cfg.config_dir_path)
    base.setdefault('decision_engine', {})['use_stored_llm_risks'] = True  # = C00

    import torch
    from training.models.ranking_mlp import RankingMLP, RANKING_FEATURES
    mdir = os.path.join(REPO, 'models/ranking_expanded_unconditioned')
    ranking_model = RankingMLP(input_dim=len(RANKING_FEATURES))
    ranking_model.load_state_dict(torch.load(os.path.join(mdir, 'ranking_mlp.pt'),
                                             weights_only=True))
    ranking_model.eval()
    ranking_norm = json.load(open(os.path.join(mdir, 'ranking_normalization.json')))
    blend = float(base.get('decision_engine', {}).get('ranking_blend', 0.0) or 0.0)

    common_manifest = {
        'code_sha': code_sha,
        'harness': 'optimizer (pre-holdout slice only)',
        'date_range': [dates[0], dates[-1]],
        'n_decision_dates': len(dates),
        'holdout_start_excluded': HOLDOUT_START,
        'initial_capital': 100000.0,
        'seeds': seeds,
        'baseline': 'C00 bundle (active + use_stored_llm_risks=True)',
        'baseline_bundle_hash': canonical_hash(base),
        'cost_model': {'source': 'src/utils/transaction_costs.py',
                       'config_hash': canonical_hash(get_cost_config_snapshot()),
                       'slippage_range_bps': 2.0},
        'command': ('.venv/bin/python runs/pkt_tb_004_control_attribution/'
                    f'scripts/risk_arch_tuner.py --candidate {args.candidate} '
                    f'--seeds {args.seeds}'),
    }

    print('running C00 pre-holdout baseline...', flush=True)
    base_runs = run_variant(ds, dates, base, seeds, ranking_model, ranking_norm, blend)
    base_metrics = [metrics_from_run(r) for r in base_runs]
    print('baseline final values:',
          [round(m['final_value']) for m in base_metrics], flush=True)

    jobs = []
    if args.candidate in ('exposure_trim', 'both'):
        jobs.append(('exposure_trim', exposure_trim_grid()))
    if args.candidate in ('vixy_policy', 'both'):
        jobs.append(('vixy_policy', vixy_grid()))

    for name, grid in jobs:
        out_dir = os.path.join(RUN_DIR, 'candidates', name)
        os.makedirs(out_dir, exist_ok=True)
        records = []
        for i, vcfg in enumerate(grid):
            bundle = copy.deepcopy(base)
            if name == 'exposure_trim':
                bundle['decision_engine']['exposure_trim'] = vcfg
            else:
                bundle['decision_params']['vol_decay_constraints'] = vcfg
            runs = run_variant(ds, dates, bundle, seeds,
                               ranking_model, ranking_norm, blend)
            summary = summarize(runs, base_runs)
            rec = {
                'variant_id': f'{name}-V{i:02d}',
                'config': vcfg,
                'bundle_hash': canonical_hash(bundle),
                'summary': summary,
            }
            records.append(rec)
            print(f"{rec['variant_id']} {json.dumps(vcfg)} -> "
                  f"dRet={summary['median_delta_total_return']:+.4f} "
                  f"dSharpe={summary['median_delta_sharpe']:+.3f} "
                  f"dDD={summary['median_delta_max_dd']:+.4f} "
                  f"t={summary['median_paired_t'] and round(summary['median_paired_t'], 2)} "
                  f"trims={summary['mean_trim_fills']:.1f} "
                  f"dCost={summary['mean_delta_costs']:+.1f}", flush=True)
        payload = {
            'candidate': name,
            'manifest': common_manifest,
            'baseline_metrics_per_seed': [
                {'seed': r['seed'], **metrics_from_run(r)} for r in base_runs],
            'n_variants_tried': len(records),
            'variants': records,
        }
        with open(os.path.join(out_dir, 'variants.json'), 'w') as f:
            json.dump(payload, f, indent=1)
        print(f'wrote {out_dir}/variants.json ({len(records)} variants)', flush=True)


if __name__ == '__main__':
    main()
