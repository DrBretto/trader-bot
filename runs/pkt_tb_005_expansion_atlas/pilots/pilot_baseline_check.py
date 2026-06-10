"""PKT-TB-005 mandatory pilot v2: dumb-baseline regime/health vs trained models.

v1 (results_baseline_check_NORANKING_v1.json) ran without the active bundle's
ranking_blend=0.35 — not production-faithful — and lacked the Skeptic's repairs.
v2 incorporates the Skeptic seat's audit of the design:

  R1. Deep-era conditioning: 128/194 stored inference days are RULE-FALLBACK
      one-hots (conf=1.0); deep-vs-rule paired statistics are reported both
      unconditioned and conditioned on deep-era dates (stored probs not one-hot).
  R2. rules_storedsizing arm = rule labels/probs with stored confidence/
      disagreement/psm — the ONLY admissible label-path verdict (one-hot conf=1
      otherwise hands rules a mechanical sizing boost).
  R3. Teacher-version skew check: label agreement reported overall, deep-era
      pre-holdout, and holdout separately.
  R4. Verdict noise model = paired daily-delta t (threshold t>=3.0 per the
      committee honesty rule, ~12-look Bonferroni); seed-std is fill noise only.
  M-13 twin: health_rules arm (stored regime, rule-based health from
      src/models/baseline_health.py applied per date).

Arms (6): ens_stored (control/canon), ens_nodis, rules_raw, rules_storedsizing,
rules_ema (halflife swept PRE-HOLDOUT only), health_rules.
Pre-registered holdout arm-reads this pilot: 5 (all non-control arms).

Run: .venv/bin/python runs/pkt_tb_005_expansion_atlas/pilots/pilot_baseline_check.py
Outputs: pilots/results_baseline_check.json + manifests/baseline_check_manifest.json
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
from src.models.baseline_health import baseline_health_model  # noqa: E402

REGIMES = ['calm_uptrend', 'risk_on_trend', 'risk_off_trend', 'choppy', 'high_vol_panic']
SEEDS_FULL = [20260217, 20260218]
SEEDS_HOLDOUT = [20260217, 20260218, 20260219]
EMA_HALFLIFE_GRID = [2, 3, 5]

# Rule labeler thresholds — verbatim from training/utils/metrics.py
# compute_regime_labels_from_baseline (the deep pair's teacher).
VOL_P85, VOL_P50, VOL_P40 = 0.14, 0.10, 0.08


def rule_label(row) -> str:
    spy_21d_ret = float(row.get('spy_return_21d', 0) or 0)
    spy_21d_vol = float(row.get('spy_vol_21d', 0.10) or 0.10)
    credit_stress = float(row.get('credit_spread_proxy', 0) or 0)
    vixy_21d_ret = float(row.get('vixy_return_21d', 0) or 0)
    if (vixy_21d_ret > 0.10 or spy_21d_vol > VOL_P85) and spy_21d_ret < -0.02:
        return 'high_vol_panic'
    if spy_21d_ret < -0.01 or credit_stress < -0.01:
        return 'risk_off_trend'
    if spy_21d_ret > 0.03 and spy_21d_vol < VOL_P40:
        return 'calm_uptrend'
    if abs(spy_21d_ret) < 0.01 and spy_21d_vol > VOL_P50:
        return 'choppy'
    return 'risk_on_trend'


def is_fallback(inference: dict) -> bool:
    """Stored inference is rule-fallback when probs are a one-hot (max==1.0)."""
    probs = inference.get('regime', {}).get('probs', {}) or {}
    return (not probs) or max(probs.values()) >= 0.999


def build_prob_series(context_df: pd.DataFrame, dates: list,
                      ema_halflife=None) -> dict:
    ctx = context_df.set_index('date')
    alpha = None if ema_halflife is None else 1 - 0.5 ** (1 / ema_halflife)
    state = np.ones(len(REGIMES)) / len(REGIMES)
    out = {}
    for d in dates:
        if d not in ctx.index:
            out[d] = None
            continue
        onehot = np.zeros(len(REGIMES))
        onehot[REGIMES.index(rule_label(ctx.loc[d]))] = 1.0
        if alpha is None:
            vec = onehot
        else:
            state = alpha * onehot + (1 - alpha) * state
            vec = state / state.sum()
        out[d] = {r: float(v) for r, v in zip(REGIMES, vec)}
    return out


def rule_health_records(features_df: pd.DataFrame) -> list:
    latest_date = features_df['date'].max()
    df = baseline_health_model(features_df, latest_date)
    records = []
    for _, row in df.iterrows():
        records.append({
            'symbol': row['symbol'],
            'health_score': float(row['health_score']),
            'vol_bucket': str(row['vol_bucket']),
            'behavior': str(row.get('behavior', 'mixed')),
            'latent': list(row['latent']) if isinstance(row.get('latent'), (list, np.ndarray)) else [0.0] * 16,
        })
    return records


def make_transform(arm: str, prob_series=None, snap_map=None):
    def transform(date: str, inference: dict) -> dict:
        regime = inference.get('regime', {})
        if arm == 'ens_stored':
            return inference
        if arm == 'ens_nodis':
            probs = dict(regime.get('probs', {}))
            if not probs:
                return inference
            for key in ('gru_prediction', 'transformer_prediction'):
                regime[key] = {'probs': dict(probs),
                               'label': max(probs, key=probs.get),
                               'confidence': max(probs.values())}
            regime['disagreement'] = 0.0
            regime['agreement'] = 1.0
            inference['regime'] = regime
            return inference
        if arm == 'health_rules':
            snapshot = snap_map[date]
            inference['asset_health'] = rule_health_records(snapshot.features_df)
            return inference
        # rules_raw / rules_ema / rules_storedsizing
        probs = prob_series.get(date)
        if probs is None:
            return inference
        label = max(probs, key=probs.get)
        new_regime = dict(regime)
        new_regime.update({'label': label, 'probs': dict(probs)})
        if arm == 'rules_storedsizing':
            # label path swapped; sizing path (conf/disagreement/psm) kept stored
            for k in ('confidence', 'disagreement', 'agreement',
                      'position_size_multiplier'):
                new_regime[k] = regime.get(k, new_regime.get(k))
        else:
            new_regime.update({'confidence': float(probs[label]),
                               'disagreement': 0.0, 'agreement': 1.0,
                               'position_size_multiplier': 1.0})
        for key in ('gru_prediction', 'transformer_prediction'):
            new_regime[key] = {'probs': dict(probs), 'label': label,
                               'confidence': float(probs[label])}
        inference['regime'] = new_regime
        return inference

    return transform


_RANKING = None


def replay(ds, dates, bundle, seed, transform):
    global _RANKING
    if _RANKING is None:
        _RANKING = common.ranking_setup_from_bundle(bundle)
        print(f'ranking: blend={_RANKING[2]} (production-faithful canon line)', flush=True)
    model, norm, blend = _RANKING
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return common.run_pilot_replay(ds, dates, bundle, random_seed=seed,
                                       inference_transform=transform,
                                       ranking_model=model,
                                       ranking_normalization=norm,
                                       ranking_blend=blend)


def paired(d_on: pd.Series, d_off: pd.Series, dates_filter=None) -> dict:
    common_idx = d_on.index.intersection(d_off.index)
    if dates_filter is not None:
        common_idx = [d for d in common_idx if d in dates_filter]
    d = (d_on.loc[common_idx] - d_off.loc[common_idx]).astype(float)
    n = len(d)
    if n < 3:
        return {'n': n, 'note': 'insufficient'}
    sd = d.std()
    t = float(d.mean() / (sd / np.sqrt(n))) if sd > 0 else 0.0
    return {'n': n, 'mean_daily_delta': round(float(d.mean()), 8),
            'sd_daily_delta': round(float(sd), 8), 't_stat': round(t, 3)}


def main() -> None:
    t = common.timer()
    cache = common.load_cache()
    ds, context_df = cache['dataset'], cache['context_df']
    bundle = common.load_active_bundle()
    snap_map = ds.by_date()
    all_dates = [s.date for s in ds.snapshots]
    segs = common.split_dates(all_dates)

    # --- diagnostics: fallback share, agreement by era, flips (R1, R3) ---
    raw_series = build_prob_series(context_df, all_dates, None)
    deep_era = set()
    rows = []
    for d in all_dates:
        inf = snap_map[d].inference
        fb = is_fallback(inf)
        if not fb:
            deep_era.add(d)
        probs = raw_series.get(d)
        rl = max(probs, key=probs.get) if probs else None
        el = inf.get('regime', {}).get('label')
        rows.append({'date': d, 'fallback': fb, 'rule': rl, 'ens': el})
    df = pd.DataFrame(rows)
    deep_pre = df[(~df.fallback) & (df.date < common.HOLDOUT_START)]
    hold = df[df.date >= common.HOLDOUT_START]
    diagnostics = {
        'n_dates': len(df),
        'fallback_days': int(df.fallback.sum()),
        'deep_era_days': int((~df.fallback).sum()),
        'deep_era_preholdout_days': len(deep_pre),
        'agreement_overall_pct': round(100 * (df.rule == df.ens).mean(), 1),
        'agreement_deep_era_preholdout_pct':
            round(100 * (deep_pre.rule == deep_pre.ens).mean(), 1) if len(deep_pre) else None,
        'agreement_holdout_pct': round(100 * (hold.rule == hold.ens).mean(), 1),
        'holdout_fallback_days': int(hold.fallback.sum()),
        'regime_flips_rules': int((df.rule != df.rule.shift()).sum() - 1),
        'regime_flips_ensemble': int((df.ens != df.ens.shift()).sum() - 1),
    }
    print('diagnostics:', json.dumps(diagnostics), flush=True)

    # --- EMA halflife selection: PRE-HOLDOUT ONLY ---
    ema_sweep = {}
    for hl in EMA_HALFLIFE_GRID:
        series = build_prob_series(context_df, all_dates, hl)
        run = replay(ds, segs['pre_holdout'], bundle, SEEDS_FULL[0],
                     make_transform('rules_ema', series))
        m = common.segment_metrics(run)
        ema_sweep[hl] = {k: m[k] for k in ('sharpe', 'total_return', 'max_drawdown')}
        print(f'EMA hl={hl} pre-holdout: {ema_sweep[hl]}', flush=True)
    best_hl = max(EMA_HALFLIFE_GRID, key=lambda h: (ema_sweep[h]['sharpe'] or -9))
    print(f'EMA halflife selected on pre-holdout: {best_hl}', flush=True)

    arms = {
        'ens_stored': make_transform('ens_stored'),
        'ens_nodis': make_transform('ens_nodis'),
        'rules_raw': make_transform('rules_raw', raw_series),
        'rules_storedsizing': make_transform('rules_storedsizing', raw_series),
        'rules_ema': make_transform('rules_ema',
                                    build_prob_series(context_df, all_dates, best_hl)),
        'health_rules': make_transform('health_rules', snap_map=snap_map),
    }

    results = {'diagnostics': diagnostics,
               'ema_halflife_sweep_preholdout': ema_sweep,
               'ema_halflife_selected': best_hl,
               'preregistered_holdout_arm_reads': [a for a in arms if a != 'ens_stored'],
               'arms': {}}
    daily_store = {}
    holdout_looks = 0
    for arm, transform in arms.items():
        arm_out = {'full': [], 'holdout': []}
        for seed in SEEDS_FULL:
            run = replay(ds, segs['full'], bundle, seed, transform)
            arm_out['full'].append({'seed': seed, **common.segment_metrics(run)})
            if seed == SEEDS_FULL[0]:
                daily_store[(arm, 'full')] = common.daily_returns(run['result'])
        for seed in SEEDS_HOLDOUT:
            run = replay(ds, segs['holdout'], bundle, seed, transform)
            arm_out['holdout'].append({'seed': seed, **common.segment_metrics(run)})
            if seed == SEEDS_HOLDOUT[0]:
                daily_store[(arm, 'holdout')] = common.daily_returns(run['result'])
        if arm != 'ens_stored':
            holdout_looks += 1
        results['arms'][arm] = arm_out
        print(f"{arm}: holdout ret={arm_out['holdout'][0]['total_return']} "
              f"sharpe={arm_out['holdout'][0]['sharpe']} [{t():.0f}s]", flush=True)

    # --- paired daily deltas vs control (R4), incl. deep-era conditioning (R1) ---
    # valuation_date = decision_date+1; map deep-era decision dates to valuation dates
    val_of = {s.date: s.next_date for s in ds.snapshots}
    deep_val = {val_of[d] for d in deep_era if d in val_of}
    for arm in arms:
        if arm == 'ens_stored':
            continue
        block = {}
        for seg in ('full', 'holdout'):
            d_on, d_off = daily_store[(arm, seg)], daily_store[('ens_stored', seg)]
            block[f'paired_{seg}'] = paired(d_on, d_off)
            block[f'paired_{seg}_deep_era_only'] = paired(d_on, d_off, deep_val)
        results['arms'][arm]['paired_vs_control'] = block

    out = RUN_DIR / 'results_baseline_check.json'
    out.write_text(json.dumps(results, indent=2))
    wall = t()
    common.write_manifest(
        'baseline_check',
        params={'arms': list(arms), 'ema_grid': EMA_HALFLIFE_GRID,
                'ema_selected': best_hl, 'seeds_full': SEEDS_FULL,
                'seeds_holdout': SEEDS_HOLDOUT,
                'bundle_version': bundle.get('version_id', 'active'),
                'ranking_blend': 0.35,
                'rule_thresholds': {'vol_p85': VOL_P85, 'vol_p50': VOL_P50,
                                    'vol_p40': VOL_P40}},
        command='.venv/bin/python runs/pkt_tb_005_expansion_atlas/pilots/pilot_baseline_check.py',
        wall_clock_s=wall,
        variants_logged=[f'{a}/{s}' for a in arms for s in ('full', 'holdout')]
                        + [f'rules_ema_hl{h}/pre_holdout(sweep)' for h in EMA_HALFLIFE_GRID]
                        + ['v1 no-ranking battery preserved as results_baseline_check_NORANKING_v1.json'],
        holdout_looks=holdout_looks,
        notes='v2 with Skeptic repairs R1-R4. Stored-inference fallback era: '
              f'{diagnostics["fallback_days"]}/{diagnostics["n_dates"]} days are rule one-hots; '
              'deep-vs-rule verdicts conditioned on deep-era dates. rules_storedsizing is the '
              'admissible label-path verdict. Verdict noise model = paired daily t (>=3.0 shows '
              'something; 2-3 suggestive; <2 noise), NOT seed-std. Monthly-retrain process noted: '
              'deep arm reflects production process, not a fixed model.')
    print(f'wrote {out} in {wall:.0f}s', flush=True)


if __name__ == '__main__':
    main()
