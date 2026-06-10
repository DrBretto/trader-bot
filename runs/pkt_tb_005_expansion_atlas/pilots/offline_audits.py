"""PKT-TB-005 zero-holdout-look offline audits (Skeptic-approved riders).

M-16 — disagreement-signal validity: Spearman of stored gru-vs-transformer
disagreement vs (a) forward 5d realized SPY vol, (b) regime-label flip within
5 days. Deep-era dates only (stored probs not one-hot); n≈66 → underpowered,
direction-only (se(rho)≈0.13).

M-05 — member redundancy: correlation between stored gru and transformer prob
vectors per deep-era date; if members are near-duplicates, calibration
reweighting has no headroom and one member can retire.

Consumes 0 holdout arm-reads (no replays; pure stats on stored artifacts).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))

import common  # noqa: E402

REGIMES = ['calm_uptrend', 'risk_on_trend', 'risk_off_trend', 'choppy', 'high_vol_panic']


def main() -> None:
    cache = common.load_cache()
    ds = cache['dataset']
    snaps = ds.snapshots

    spy_close = {}
    for s in snaps:
        spy = s.next_prices_df[s.next_prices_df['symbol'] == 'SPY']
        if len(spy):
            spy_close[s.date] = float(spy.iloc[-1]['close'])
    dates = [s.date for s in snaps if s.date in spy_close]
    closes = pd.Series([spy_close[d] for d in dates], index=dates)
    rets = closes.pct_change()

    rows = []
    for i, s in enumerate(snaps):
        reg = s.inference.get('regime', {})
        probs = reg.get('probs', {}) or {}
        if not probs or max(probs.values()) >= 0.999:
            continue  # fallback era — excluded
        gru = (reg.get('gru_prediction') or {}).get('probs') or {}
        tf = (reg.get('transformer_prediction') or {}).get('probs') or {}
        fwd_vol = None
        if s.date in rets.index:
            loc = rets.index.get_loc(s.date)
            w = rets.iloc[loc + 1: loc + 6]
            if len(w) >= 3:
                fwd_vol = float(w.std())
        flip5 = None
        labels = []
        for j in range(i, min(i + 6, len(snaps))):
            labels.append(snaps[j].inference.get('regime', {}).get('label'))
        if len(labels) >= 2:
            flip5 = int(any(l != labels[0] for l in labels[1:]))
        member_corr = None
        if gru and tf:
            g = np.array([gru.get(r, 0) for r in REGIMES])
            t_ = np.array([tf.get(r, 0) for r in REGIMES])
            if g.std() > 0 and t_.std() > 0:
                member_corr = float(np.corrcoef(g, t_)[0, 1])
        rows.append({'date': s.date,
                     'disagreement': float(reg.get('disagreement', 0) or 0),
                     'fwd_vol_5d': fwd_vol, 'flip_within_5d': flip5,
                     'member_corr': member_corr})

    df = pd.DataFrame(rows)
    out = {'n_deep_era_dates': len(df),
           'caveat': 'underpowered by construction (se(rho)~0.13 at n~66); direction only'}

    sub = df.dropna(subset=['fwd_vol_5d'])
    if len(sub) >= 10:
        rho, p = spearmanr(sub.disagreement, sub.fwd_vol_5d)
        out['m16_disagreement_vs_fwd_vol_5d'] = {
            'n': len(sub), 'spearman_rho': round(float(rho), 3), 'p': round(float(p), 3)}
    sub = df.dropna(subset=['flip_within_5d'])
    if len(sub) >= 10 and sub.flip_within_5d.nunique() > 1:
        rho, p = spearmanr(sub.disagreement, sub.flip_within_5d)
        out['m16_disagreement_vs_label_flip_5d'] = {
            'n': len(sub), 'spearman_rho': round(float(rho), 3), 'p': round(float(p), 3)}
    sub = df.dropna(subset=['member_corr'])
    if len(sub):
        out['m05_member_prob_correlation'] = {
            'n': len(sub), 'mean': round(float(sub.member_corr.mean()), 3),
            'median': round(float(sub.member_corr.median()), 3),
            'p10': round(float(sub.member_corr.quantile(0.1)), 3),
            'share_above_0.9': round(float((sub.member_corr > 0.9).mean()), 3)}
        out['m16_disagreement_summary'] = {
            'mean': round(float(df.disagreement.mean()), 3),
            'p90': round(float(df.disagreement.quantile(0.9)), 3)}

    path = RUN_DIR / 'results_offline_audits.json'
    path.write_text(json.dumps(out, indent=2))
    common.write_manifest(
        'offline_audits', params={'audits': ['M-16', 'M-05']},
        command='.venv/bin/python runs/pkt_tb_005_expansion_atlas/pilots/offline_audits.py',
        wall_clock_s=0.0, variants_logged=['offline_stats_only'], holdout_looks=0,
        notes='Zero-look riders per Skeptic; deep-era dates only.')
    print(json.dumps(out, indent=1), flush=True)


if __name__ == '__main__':
    main()
