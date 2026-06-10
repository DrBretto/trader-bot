"""PKT-TB-004 HARNESS-C confirmation runs (three-line canon continuation).

Runs the confirmation set on the canon-line harness: real 2026-03-11 book,
champion overlays on BOTH arms, stored llm_risk.json natively consumed,
deterministic (no cost model -> gross-of-costs, exactly one run per cell).

Confirmation set per the Methodologist: baseline C00c, the SELL group
(T1-K/L/M/O/P/Q), the LLM cells, T2-ALLOFF (+ candidates, run separately).

[C] reads may only confirm/contradict [O] direction; a verdict never rests on
[C] alone. T1-LLM here uses the decision_engine ablation channel flags (the
optimizer-only use_stored_llm_risks flag has no effect on this harness).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(__file__))

import boto3  # noqa: E402

from battery_cells import build_cells, canonical_hash  # noqa: E402
from optimizer.config import load_optimizer_config  # noqa: E402
from optimizer.promote import load_active_bundle  # noqa: E402
from src.utils.three_line_replay.replay_engine import (  # noqa: E402
    S3Cache, VariantConfig, run_variant,
)
from src.utils.three_line_replay.extender import _build_champion_strategy  # noqa: E402

RUN_DIR = os.path.join(REPO, 'runs', 'pkt_tb_004_control_attribution')

CONFIRM_SET = ['C00', 'T1-K', 'T1-L', 'T1-M', 'T1-O', 'T1-P', 'T1-Q',
               'T1-LLM', 'T2-ALLOFF']


def bundle_to_variant(name, bundle):
    return VariantConfig(
        name=name,
        decision_params=bundle['decision_params'],
        regime_compatibility=bundle['regime_compatibility'],
        signal_overrides=bundle.get('signals', {}),
        regime_fusion_overrides=bundle.get('regime_fusion', {}),
        decision_engine_overrides=bundle.get('decision_engine', {}),
        ensemble_overrides=bundle.get('ensemble', {}),
        transaction_cost_overrides=bundle.get('transaction_costs', {}),
    )


def main():
    cfg = load_optimizer_config(os.path.join(REPO, 'config/optimizer.committee_20260606.json'))
    base = load_active_bundle(cfg.config_dir_path)
    cells = build_cells(base)

    # Harness-C equivalent of llm off: channel ablation flags (native llm wiring).
    llm_off = cells['T1-LLM']['bundle']
    llm_off.setdefault('decision_engine', {}).setdefault('ablation', {}).update({
        'disable_llm_size_adj': True,
        'disable_llm_buy_veto': True,
        'disable_llm_sell_veto': True,
    })
    alloff = cells['T2-ALLOFF']['bundle']
    alloff.setdefault('decision_engine', {}).setdefault('ablation', {}).update({
        'disable_llm_size_adj': True,
        'disable_llm_buy_veto': True,
        'disable_llm_sell_veto': True,
    })

    code_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO).decode().strip()
    cache = S3Cache(boto3.client('s3'))
    trading_dates = [d for d in cache.list_daily_dates() if d >= '2026-03-11']
    universe_df = cache.get_csv('config/universe.csv')
    strategy = _build_champion_strategy()

    outdir = os.path.join(RUN_DIR, 'canon_confirm')
    os.makedirs(outdir, exist_ok=True)

    for cid in CONFIRM_SET:
        cell = cells[cid]
        opath = os.path.join(outdir, f'{cid}.json')
        if os.path.exists(opath):
            print(f'-- {cid}: exists, skipping', flush=True)
            continue
        print(f'== [C] {cid}: {cell["description"]}', flush=True)
        t0 = time.time()
        variant = bundle_to_variant(cid, cell['bundle'])
        res = run_variant(cache, variant, strategy, trading_dates, universe_df)
        wall = time.time() - t0
        out = {
            'cell_id': cid,
            'harness': 'three_line',
            'gross_of_costs': True,
            'canon_overlays': {'extend_relax_choppy': 0.50, 'topup_psm': 1.1},
            'book_seed': 'real-2026-03-11',
            'code_sha': code_sha,
            'bundle_hash': canonical_hash(cell['bundle']),
            'wall_clock_s': round(wall, 1),
            'date_value_map': res['date_value_map'],
            'final_value': res['final_value'],
            'final_cash': res['final_cash'],
            'final_date': res['final_date'],
            'actions': res['actions'],
        }
        with open(opath, 'w') as f:
            json.dump(out, f)
        print(f'   final={res["final_value"]} ({wall:.0f}s)', flush=True)

    print('canon confirm done', flush=True)


if __name__ == '__main__':
    main()
