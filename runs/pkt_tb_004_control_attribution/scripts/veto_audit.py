#!/usr/bin/env python
"""PKT-TB-004 LLM Veto Auditor — score every historical Haiku veto/downsize.

Inputs (read-only):
  runs/pkt_tb_004_control_attribution/cache/daily/<date>/{llm_risk,decisions,portfolio_state}.json
    (synced from s3://investment-system-data/daily/ — see LLM_VETO_AUDIT.md manifest)
  runs/pkt_tb_004_control_attribution/cache/prices_20260610.parquet
    (= s3://investment-system-data/daily/2026-06-10/prices.parquet, closes through 2026-06-09)

Output:
  runs/pkt_tb_004_control_attribution/veto_audit_data.json

Re-run:  .venv/bin/python runs/pkt_tb_004_control_attribution/scripts/veto_audit.py
Deterministic: no seeds, no network. Price snapshot pinned to the 2026-06-10 daily parquet.
"""
import json
import glob
import os
import collections
import math

import pandas as pd
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
CACHE = os.path.join(ROOT, 'cache', 'daily')
PRICES = os.path.join(ROOT, 'cache', 'prices_20260610.parquet')
OUT = os.path.join(ROOT, 'veto_audit_data.json')

HOLDOUT_START = '2026-03-11'   # config/optimizer.committee_20260606.json holdout_start
HORIZONS = [5, 21, 63]

# ---------------------------------------------------------------- price panel
panel = pd.read_parquet(PRICES)[['date', 'symbol', 'close']].copy()
panel['date'] = pd.to_datetime(panel['date'])
panel = panel.sort_values(['symbol', 'date'])
series = {s: g.set_index('date')['close'] for s, g in panel.groupby('symbol')}
PANEL_END = str(panel['date'].max().date())


def fwd_return(symbol, t0, h):
    """Forward return over h trading days from the last close <= t0.

    Returns (ret, actual_h, capped). None if symbol absent or t0 before series.
    """
    s = series.get(symbol)
    if s is None:
        return None, 0, True
    idx = s.index.searchsorted(pd.Timestamp(t0), side='right') - 1
    if idx < 0:
        return None, 0, True
    end = idx + h
    capped = end >= len(s)
    if capped:
        end = len(s) - 1
    if end <= idx:
        return None, 0, True
    return float(s.iloc[end] / s.iloc[idx] - 1.0), int(end - idx), capped


def fwd_risk(symbol, t0, h=21):
    """Realized daily-return std (annualized) and max adverse excursion over
    the h trading days following t0."""
    s = series.get(symbol)
    if s is None:
        return None, None
    idx = s.index.searchsorted(pd.Timestamp(t0), side='right') - 1
    if idx < 0:
        return None, None
    win = s.iloc[idx:min(idx + h + 1, len(s))]
    if len(win) < 5:
        return None, None
    rets = win.pct_change().dropna()
    vol = float(rets.std() * math.sqrt(252))
    mae = float((win / win.iloc[0] - 1.0).min())
    return vol, mae


# ---------------------------------------------------------------- scan record
days = sorted(
    os.path.basename(os.path.dirname(p))
    for p in glob.glob(os.path.join(CACHE, '*', 'llm_risk.json'))
)

assessments = []          # one row per (run_day, symbol) LLM assessment
veto_events = []          # structural_risk_veto == True
downsize_buy_events = []  # BUY action whose size was cut by confidence_adjustment
buys_no_assessment = 0
buys_total = 0
regime_days = collections.Counter()
rationale_counts = collections.Counter()
calls_total = 0

for day in days:
    base = os.path.join(CACHE, day)
    lr = json.load(open(os.path.join(base, 'llm_risk.json')))
    dec = json.load(open(os.path.join(base, 'decisions.json')))
    risks = lr.get('risks', {})
    calls_total += lr.get('calls_made', 0)
    data_date = dec.get('date', day)          # close the decision saw
    regime_days[dec.get('regime', '?')] += 1
    actions = dec.get('actions', [])
    watch = {c['symbol']: c for c in dec.get('buy_candidates', [])}
    sells_by_sym = {a['symbol']: a for a in actions if a['action'] == 'SELL'}
    buys_by_sym = {a['symbol']: a for a in actions if a['action'] == 'BUY'}

    # post-trade holdings for the day (book the morning execution produced)
    held_post = []
    ps_path = os.path.join(base, 'portfolio_state.json')
    if os.path.exists(ps_path):
        ps = json.load(open(ps_path))
        held_post = [h['symbol'] for h in ps.get('holdings', [])]

    for sym, r in risks.items():
        adj = float(r.get('confidence_adjustment', 0) or 0)
        veto = bool(r.get('structural_risk_veto', False))
        rationale_counts[r.get('one_sentence_rationale', '')] += 1
        assessments.append({
            'run_day': day, 'data_date': data_date, 'symbol': sym,
            'severity': r.get('severity'), 'veto': veto, 'adj': adj,
            'flags': r.get('risk_flags', []),
            'rationale': r.get('one_sentence_rationale', ''),
        })

        if veto:
            # classify what the veto actually did
            if sym in sells_by_sym and sells_by_sym[sym].get('reason') == 'LLM_VETO':
                what = 'forced_sell'
                dollars = sells_by_sym[sym].get('dollars', 0.0)
            elif sym in watch:
                what = 'blocked_watchlist_candidate'
                dollars = watch[sym].get('suggested_size', 0.0)
            elif sym in held_post:
                what = 'held_but_not_sold (veto ignored?)'
                dollars = 0.0
            else:
                what = 'no_op (not held, not a candidate)'
                dollars = 0.0
            ev = {
                'run_day': day, 'data_date': data_date, 'symbol': sym,
                'type': 'veto', 'what_it_blocked': what,
                'dollars_at_stake': round(float(dollars), 2),
                'rationale': r.get('one_sentence_rationale', ''),
            }
            for h in HORIZONS:
                ret, ah, cap = fwd_return(sym, data_date, h)
                ev[f'fwd_{h}d'] = ret
                ev[f'fwd_{h}d_actual'] = ah
            spy21, _, _ = fwd_return('SPY', data_date, 21)
            ev['spy_fwd_21d'] = spy21
            # obey-vs-ignore at +21d (positive = obeying saved money)
            r21 = ev['fwd_21d']
            ev['dollars_saved_by_obeying_21d'] = (
                round(-float(dollars) * r21, 2) if (r21 is not None and dollars) else 0.0
            )
            veto_events.append(ev)

    # downsize channel: only consequential where a BUY was actually sized
    for sym, a in buys_by_sym.items():
        buys_total += 1
        if sym not in risks:
            buys_no_assessment += 1
            continue
        adj = float(risks[sym].get('confidence_adjustment', 0) or 0)
        if adj <= 0:
            continue
        dollars = float(a.get('dollars', 0.0))
        full_size = dollars / (1.0 - adj) if adj < 1 else dollars
        haircut = full_size - dollars
        ev = {
            'run_day': day, 'data_date': data_date, 'symbol': sym,
            'type': 'downsize_on_buy', 'adj': adj,
            'buy_dollars': round(dollars, 2),
            'haircut_dollars': round(haircut, 2),
            'rationale': risks[sym].get('one_sentence_rationale', ''),
        }
        for h in HORIZONS:
            ret, ah, cap = fwd_return(sym, data_date, h)
            ev[f'fwd_{h}d'] = ret
            ev[f'fwd_{h}d_actual'] = ah
        spy21, _, _ = fwd_return('SPY', data_date, 21)
        ev['spy_fwd_21d'] = spy21
        vol21, mae21 = fwd_risk(sym, data_date, 21)
        ev['fwd_vol_21d'] = vol21
        ev['fwd_mae_21d'] = mae21
        r21 = ev['fwd_21d']
        # money the haircut kept out of the asset; positive = haircut saved money
        ev['dollars_saved_by_obeying_21d'] = (
            round(-haircut * r21, 2) if r21 is not None else 0.0
        )
        downsize_buy_events.append(ev)

# ------------------------------------------------------------------ base rates
# (a) everything the engine actually bought (passed all filters)
# (b) SPY on the same decision dates
buy_dates = []
buy_cohort = []
for day in days:
    dec = json.load(open(os.path.join(CACHE, day, 'decisions.json')))
    data_date = dec.get('date', day)
    for a in dec.get('actions', []):
        if a['action'] != 'BUY':
            continue
        row = {'run_day': day, 'data_date': data_date, 'symbol': a['symbol']}
        for h in HORIZONS:
            ret, ah, cap = fwd_return(a['symbol'], data_date, h)
            row[f'fwd_{h}d'] = ret
        buy_cohort.append(row)
        buy_dates.append(data_date)

spy_cohort = []
for d in sorted(set(buy_dates)):
    row = {'data_date': d, 'symbol': 'SPY'}
    for h in HORIZONS:
        ret, ah, cap = fwd_return('SPY', d, h)
        row[f'fwd_{h}d'] = ret
    spy_cohort.append(row)

# watchlist cohort (scored candidates, pre-LLM, top-10 by score)
watch_cohort = []
for day in days:
    dec = json.load(open(os.path.join(CACHE, day, 'decisions.json')))
    data_date = dec.get('date', day)
    for c in dec.get('buy_candidates', []):
        ret21, _, _ = fwd_return(c['symbol'], data_date, 21)
        if ret21 is not None:
            watch_cohort.append(ret21)


def dist(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return {}
    arr = np.array(v)
    return {
        'n': len(v), 'mean': float(arr.mean()), 'median': float(np.median(arr)),
        'sd': float(arr.std(ddof=1)) if len(v) > 1 else 0.0,
        'pct_negative': float((arr < 0).mean()),
        'p10': float(np.percentile(arr, 10)), 'p90': float(np.percentile(arr, 90)),
    }


base_rates = {}
for h in HORIZONS:
    base_rates[f'buys_fwd_{h}d'] = dist([r[f'fwd_{h}d'] for r in buy_cohort])
    base_rates[f'spy_fwd_{h}d'] = dist([r[f'fwd_{h}d'] for r in spy_cohort])
base_rates['watchlist_fwd_21d'] = dist(watch_cohort)

# ----------------------------------------------------- downsize-channel stats
ds = pd.DataFrame(downsize_buy_events)
downsize_stats = {}
if len(ds):
    for full, label in [(ds, 'full'), (ds[ds.run_day >= HOLDOUT_START], 'holdout')]:
        sub = full.dropna(subset=['fwd_21d'])
        st = {
            'n_events': int(len(full)),
            'total_haircut_dollars': round(float(full.haircut_dollars.sum()), 2),
            'dollars_saved_by_obeying_21d':
                round(float(full.dollars_saved_by_obeying_21d.sum()), 2),
            'adj_distribution': full.adj.value_counts().sort_index().to_dict(),
        }
        if len(sub) >= 5:
            st['corr_adj_fwd21_pearson'] = round(float(sub.adj.corr(sub.fwd_21d)), 3)
            st['corr_adj_fwd21_spearman'] = round(
                float(sub.adj.corr(sub.fwd_21d, method='spearman')), 3)
            s2 = sub.dropna(subset=['fwd_vol_21d'])
            st['corr_adj_fwdvol_spearman'] = round(
                float(s2.adj.corr(s2.fwd_vol_21d, method='spearman')), 3)
            s3 = sub.dropna(subset=['fwd_mae_21d'])
            st['corr_adj_fwdmae_spearman'] = round(
                float(s3.adj.corr(s3.fwd_mae_21d, method='spearman')), 3)
            st['hit_rate_fwd21_neg'] = round(float((sub.fwd_21d < 0).mean()), 3)
        downsize_stats[label] = st

# severity → adjustment cross-tab (is the LLM differentiating?)
am = pd.DataFrame(assessments)
sev_adj = (
    am.groupby('severity').adj.agg(['count', 'mean'])
    .rename(columns={'count': 'n', 'mean': 'mean_adj'})
)

# downsize counts by symbol and month (charter item 1)
am['month'] = am.run_day.str[:7]
by_symbol = (
    am[am.adj > 0].groupby('symbol').size().sort_values(ascending=False).to_dict()
)
by_month = am[am.adj > 0].groupby('month').size().to_dict()
veto_by_symbol = am[am.veto].groupby('symbol').size().to_dict()
veto_by_month = am[am.veto].groupby('month').size().to_dict()

# adjustment persistence: same symbol same adj day over day?
adj_changes = 0
adj_pairs = 0
for sym, g in am.sort_values('run_day').groupby('symbol'):
    vals = g.adj.tolist()
    for a, b in zip(vals, vals[1:]):
        adj_pairs += 1
        if abs(a - b) > 1e-9:
            adj_changes += 1

# ------------------------------------------------------------------ cost side
N_DAYS = len(days)
avg_calls = calls_total / N_DAYS if N_DAYS else 0
# token estimate: prompt template ~190 words (~260 tok) + system (~25 tok);
# output JSON ~70-110 tok (observed rationales are one sentence)
IN_TOK, OUT_TOK = 300, 100
P_IN, P_OUT = 0.25 / 1e6, 1.25 / 1e6          # claude-3-haiku Bedrock on-demand
P_IN_HI, P_OUT_HI = 0.80 / 1e6, 4.00 / 1e6    # Haiku 3.5 rate = upper bound
cost = {
    'model_id': 'anthropic.claude-3-haiku-20240307-v1:0 (src/steps/llm_risk_check.py:81)',
    'calls_total_88_days': calls_total,
    'avg_calls_per_night': round(avg_calls, 2),
    'est_tokens_per_call': {'input': IN_TOK, 'output': OUT_TOK},
    'est_cost_per_night_usd': round(avg_calls * (IN_TOK * P_IN + OUT_TOK * P_OUT), 5),
    'est_cost_per_year_usd': round(252 * avg_calls * (IN_TOK * P_IN + OUT_TOK * P_OUT), 2),
    'upper_bound_per_year_usd_at_haiku35_rates':
        round(252 * avg_calls * (IN_TOK * P_IN_HI + OUT_TOK * P_OUT_HI), 2),
    'pricing_note': (
        'claude-3-haiku-20240307 no longer appears on current pricing pages '
        '(legacy model); $0.25/$1.25 per MTok is its long-standing Bedrock '
        'on-demand rate. Upper bound uses Haiku 3.5 rates ($0.80/$4.00). '
        'Calls are sequential in run(); at ~1-2 s/call the step adds roughly '
        f'{int(avg_calls)}-{int(2 * avg_calls)} s latency per night.'
    ),
}

# ------------------------------------------------------------------ veto summary
vw = pd.DataFrame(veto_events) if veto_events else pd.DataFrame()
veto_summary = {
    'n_veto_events': len(veto_events),
    'n_forced_sell': int((vw.what_it_blocked == 'forced_sell').sum()) if len(vw) else 0,
    'n_blocked_candidate': int(
        (vw.what_it_blocked == 'blocked_watchlist_candidate').sum()) if len(vw) else 0,
    'n_no_op': int(vw.what_it_blocked.str.startswith('no_op').sum()) if len(vw) else 0,
    'dollar_impact_obey_vs_ignore_21d_full': round(
        float(vw.dollars_saved_by_obeying_21d.sum()), 2) if len(vw) else 0.0,
    'dollar_impact_obey_vs_ignore_21d_holdout': round(float(
        vw[vw.run_day >= HOLDOUT_START].dollars_saved_by_obeying_21d.sum()
    ), 2) if len(vw) else 0.0,
    'by_symbol': veto_by_symbol, 'by_month': veto_by_month,
}

# the VIXY 2026-06-04 incident (LLM approved the structurally-decaying asset)
vixy_0604 = {'note': 'engine emitted BUY VIXY $5139.93 on 2026-06-04; '
                     'llm gave veto=false adj=0.1; execution failed only on '
                     'broker notional cap ($5139.93 > $5000)'}
for h in HORIZONS:
    ret, ah, cap = fwd_return('VIXY', '2026-06-03', h)
    vixy_0604[f'vixy_fwd_{h}d_from_20260603'] = ret
    vixy_0604[f'actual_h'] = ah

out = {
    'generated': '2026-06-10',
    'window': {'first_run_day': days[0], 'last_run_day': days[-1],
               'n_run_days_with_llm_risk': N_DAYS,
               'n_calendar_folders_in_window': 112,
               'price_panel_end': PANEL_END,
               'holdout_start': HOLDOUT_START},
    'inventory': {
        'n_assessments': len(assessments),
        'n_vetoes': len(veto_events),
        'n_downsize_assessments': int((am.adj > 0).sum()),
        'pct_assessments_with_downsize': round(float((am.adj > 0).mean()), 4),
        'adj_value_counts': am.adj.round(2).value_counts().sort_index().to_dict(),
        'severity_value_counts': am.severity.value_counts().sort_index().to_dict(),
        'severity_to_mean_adj': sev_adj.round(4).to_dict('index'),
        'downsizes_by_symbol': by_symbol,
        'downsizes_by_month': by_month,
        'adj_day_over_day_change_rate': round(adj_changes / adj_pairs, 4) if adj_pairs else None,
        'top_repeated_rationales': rationale_counts.most_common(8),
        'n_buys_total': buys_total,
        'n_buys_without_llm_assessment': buys_no_assessment,
    },
    'veto_summary': veto_summary,
    'veto_events': veto_events,
    'downsize_buy_events': downsize_buy_events,
    'downsize_stats': downsize_stats,
    'base_rates': base_rates,
    'regime_mix_of_run_days': dict(regime_days),
    'cost': cost,
    'vixy_20260604_incident': vixy_0604,
}

with open(OUT, 'w') as f:
    json.dump(out, f, indent=1, default=str)
print(f"wrote {OUT}")
print(json.dumps({k: out[k] for k in ('inventory', 'veto_summary', 'downsize_stats',
                                      'base_rates', 'cost')}, indent=1, default=str)[:6000])
print('regime mix:', dict(regime_days))
print('vixy incident:', vixy_0604)
