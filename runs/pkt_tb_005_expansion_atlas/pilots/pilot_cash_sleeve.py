"""PKT-TB-005 pilot: S-01 cash sleeve (idle cash parked in SHY; S-16 ladder arm).

Skeptic framing (binding): the carry is established by ARITHMETIC, not by this
window — T-bill yield on structurally idle cash. The replay's job is a HARM
CHECK: does the sleeve liquidate cleanly into buys, does it distort decisions,
does it crater anywhere? The report must NOT claim the carry as a
replay-measured win. Pre-registered holdout arm-reads: 2 (shy, ladder).

Wiring (pilot-only, no src/ changes):
- portfolio_view: engine sees sleeve holdings as cash (reserve gate satisfied
  by sleeve; holdings count excludes sleeve; same dict objects pass through).
- universe_override: sleeve symbols ineligible for scoring (the sleeve is not
  an alpha position).
- intent_transform: SELL sleeve to cover buy shortfall (SELLs execute before
  BUYs by ACTION_PRIORITY); sweep estimated idle cash above a buffer into the
  sleeve.

Run: .venv/bin/python runs/pkt_tb_005_expansion_atlas/pilots/pilot_cash_sleeve.py
Outputs: pilots/results_cash_sleeve.json + manifests/cash_sleeve_manifest.json
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))

import common  # noqa: E402

SEEDS_FULL = [20260217, 20260218]
SEEDS_HOLDOUT = [20260217, 20260218, 20260219]
BUFFER = 500.0
ARMS = {
    'control': None,
    'sleeve_shy': {'SHY': 1.0},
    'sleeve_ladder': {'SHY': 0.5, 'IEF': 0.5},
}


def latest_closes(features_df: pd.DataFrame, symbols) -> dict:
    out = {}
    sub = features_df[features_df['symbol'].isin(list(symbols))]
    if len(sub) == 0:
        return out
    sub = sub.sort_values('date').groupby('symbol').tail(1)
    for _, row in sub.iterrows():
        if pd.notna(row.get('close')):
            out[row['symbol']] = float(row['close'])
    return out


def make_sleeve_hooks(weights: dict, universe_df: pd.DataFrame):
    sleeve_syms = set(weights)

    def view(portfolio):
        holdings = portfolio.get('holdings', [])
        sleeve_value = 0.0
        kept = []
        for h in holdings:
            if h.get('symbol') in sleeve_syms:
                sleeve_value += float(h.get('shares', 0)) * float(
                    h.get('current_price', h.get('entry_price', 0)) or 0)
            else:
                kept.append(h)  # same dict object — mutations persist
        v = dict(portfolio)
        v['holdings'] = kept
        v['cash'] = float(portfolio.get('cash', 0)) + sleeve_value
        return v

    def intents(date, decisions, intents_list, extra):
        portfolio = extra['portfolio']
        snapshot = extra['snapshot']
        marks = latest_closes(snapshot.features_df, sleeve_syms)
        real_cash = float(portfolio.get('cash', 0))
        # ALL sleeve holding fragments, in holdings-list order (each daily sweep
        # appends a new holding entry; paper_trader SELL pops the FIRST symbol
        # match, so SELLs must be emitted per fragment in list order or the
        # share counts mismatch and the pop burns the difference)
        sleeve_frags = [h for h in portfolio.get('holdings', [])
                        if h.get('symbol') in sleeve_syms]

        buy_dollars = sum(float(i.get('dollars', 0) or 0) for i in intents_list
                          if str(i.get('action', '')).upper() == 'BUY')
        out = list(intents_list)

        # 1) free cash for engine buys: liquidate sleeve positions IN FULL.
        # paper_trader.execute_trade SELL pops the entire holding while
        # crediting only the sold shares (partial sells burn the remainder —
        # surfaced as a run-level finding), so the sleeve must always sell
        # whole positions; the same-day sweep below rebuys the excess (SELLs
        # execute before BUYs via ACTION_PRIORITY).
        shortfall = buy_dollars + BUFFER - real_cash
        freed = 0.0
        if shortfall > 0:
            for h in sleeve_frags:
                sym = h['symbol']
                px = marks.get(sym, float(h.get('current_price', 0) or 0))
                shares = int(h.get('shares', 0))
                if px <= 0 or shares <= 0:
                    continue
                out.append({'action': 'SELL', 'symbol': sym, 'shares': shares,
                            'reason': 'SLEEVE_LIQUIDATE_FOR_BUYS'})
                freed += shares * px
                # no early break: liquidate every fragment of every sleeve
                # symbol — partial fragment-set sells leave first-match pops
                # misaligned on the next iteration

        # 2) sweep estimated idle cash above buffer into the sleeve
        est_after = real_cash + freed - buy_dollars
        excess = est_after - BUFFER
        if excess >= 250:
            for sym, w in weights.items():
                px = marks.get(sym)
                dollars = excess * w
                if px and dollars >= 250:
                    out.append({'action': 'BUY', 'symbol': sym, 'dollars': round(dollars, 2),
                                'price': px, 'asset_class': 'bond', 'sector': 'short_treasury',
                                'leverage_flag': 0, 'reason': 'SLEEVE_SWEEP'})
        return out

    uni = universe_df.copy()
    uni.loc[uni['symbol'].isin(sleeve_syms), 'eligible'] = 0
    return view, intents, uni


def replay(ds, dates, bundle, seed, arm_weights, ranking):
    model, norm, blend = ranking
    if arm_weights is None:
        hooks = (None, None, None)
    else:
        hooks = make_sleeve_hooks(arm_weights, ds.universe_df)
    view, intents, uni = hooks
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return common.run_pilot_replay(
            ds, dates, bundle, random_seed=seed,
            intent_transform=intents, portfolio_view=view, universe_override=uni,
            ranking_model=model, ranking_normalization=norm, ranking_blend=blend)


def main() -> None:
    t = common.timer()
    cache = common.load_cache()
    ds = cache['dataset']
    bundle = common.load_active_bundle()
    ranking = common.ranking_setup_from_bundle(bundle)
    segs = common.split_dates([s.date for s in ds.snapshots])

    results = {'preregistered_holdout_arm_reads': ['sleeve_shy', 'sleeve_ladder'],
               'skeptic_framing': 'harm check only; carry is arithmetic, not a replay win',
               'arms': {}}
    daily_store = {}
    for arm, weights in ARMS.items():
        arm_out = {'full': [], 'holdout': []}
        for seed in SEEDS_FULL:
            run = replay(ds, segs['full'], bundle, seed, weights, ranking)
            m = common.segment_metrics(run)
            arm_out['full'].append({'seed': seed, **m})
            if seed == SEEDS_FULL[0]:
                daily_store[(arm, 'full')] = common.daily_returns(run['result'])
                # harm-check diagnostics: sleeve mechanics
                fills = run['result'].fills
                arm_out['sleeve_mechanics_full'] = {
                    'sleeve_buys': sum(1 for f in fills if f.get('reason') == 'SLEEVE_SWEEP'),
                    'sleeve_liquidations': sum(1 for f in fills
                                               if f.get('reason') == 'SLEEVE_LIQUIDATE_FOR_BUYS'),
                    'engine_fills': sum(1 for f in fills
                                        if f.get('reason') not in
                                        ('SLEEVE_SWEEP', 'SLEEVE_LIQUIDATE_FOR_BUYS')),
                }
        for seed in SEEDS_HOLDOUT:
            run = replay(ds, segs['holdout'], bundle, seed, weights, ranking)
            arm_out['holdout'].append({'seed': seed, **common.segment_metrics(run)})
            if seed == SEEDS_HOLDOUT[0]:
                daily_store[(arm, 'holdout')] = common.daily_returns(run['result'])
        results['arms'][arm] = arm_out
        print(f"{arm}: holdout ret={arm_out['holdout'][0]['total_return']} "
              f"expo={arm_out['holdout'][0]['avg_gross_exposure']} [{t():.0f}s]", flush=True)

    for arm in ('sleeve_shy', 'sleeve_ladder'):
        block = {}
        for seg in ('full', 'holdout'):
            d_on, d_off = daily_store[(arm, seg)], daily_store[('control', seg)]
            idx = d_on.index.intersection(d_off.index)
            d = (d_on.loc[idx] - d_off.loc[idx]).astype(float)
            sd = d.std()
            block[f'paired_{seg}'] = {
                'n': len(d), 'mean_daily_delta': round(float(d.mean()), 8),
                'sd_daily_delta': round(float(sd), 8),
                't_stat': round(float(d.mean() / (sd / np.sqrt(len(d)))), 3)
                          if sd > 0 and len(d) > 2 else 0.0}
        results['arms'][arm]['paired_vs_control'] = block

    out = RUN_DIR / 'results_cash_sleeve.json'
    out.write_text(json.dumps(results, indent=2))
    common.write_manifest(
        'cash_sleeve',
        params={'arms': {k: v for k, v in ARMS.items()}, 'buffer': BUFFER,
                'seeds_full': SEEDS_FULL, 'seeds_holdout': SEEDS_HOLDOUT,
                'bundle_version': bundle.get('version_id', 'active'),
                'ranking_blend': ranking[2]},
        command='.venv/bin/python runs/pkt_tb_005_expansion_atlas/pilots/pilot_cash_sleeve.py',
        wall_clock_s=t(),
        variants_logged=[f'{a}/{s}' for a in ARMS for s in ('full', 'holdout')],
        holdout_looks=2,
        notes='Harm check per Skeptic rule 5: carry is arithmetic; replay verifies clean '
              'liquidation into buys and absence of decision distortion. Pilot wiring (view/'
              'intent hooks) approximates ship semantics; known divergence: none material — '
              'sleeve excluded from holdings count and scoring via view+universe override.')
    print(f'wrote {out} in {t():.0f}s', flush=True)


if __name__ == '__main__':
    main()
