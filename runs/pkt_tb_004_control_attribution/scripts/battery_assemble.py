"""PKT-TB-004 battery assembly: metrics, paired stats, triggers, matrix.

Reads cells/<id>/results.json produced by battery_runner.py, computes the
EVIDENCE_PROTOCOL metric set + paired daily-delta statistics vs C00, evaluates
the pre-registered Tier-1b/Tier-3 triggers, and writes:
  - cells/<id>/stats.json (per-cell deltas + decision-rule outcome)
  - trigger_evaluation.json (Tier-1b parent test + S1/S2/S3 results)
  - ATTRIBUTION_MATRIX.md
Updates each manifest's seed_noise_bound_bps.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))
from battery_cells import HOLDOUT_START, LLM_ERA_START, SEEDS  # noqa: E402

RUN_DIR = os.path.join(REPO, 'runs', 'pkt_tb_004_control_attribution')
CELLS_DIR = os.path.join(RUN_DIR, 'cells')

LLM_CELLS = {'T1-LLM', 'T1b-C', 'T1b-N', 'T1b-R'}

# Design-claim table for the S2 wrong-sign trigger (sign of Δreturn or ΔmaxDD
# the designer would predict for the layer's REMOVAL). Encoded as the metric
# checked and the sign that matches the design claim.
#   'dd_worse'  -> removal should make maxDD more negative (guard layer)
#   'ret_up'    -> removal should raise return (constraint layer)
DESIGN_CLAIMS = {
    'T1-A': 'dd_worse', 'T1-B': 'dd_worse', 'T1-LLM': 'dd_worse',
    'T1-D': 'ret_up', 'T1-E': 'dd_worse', 'T1-F': 'ret_up',
    'T1-G': 'dd_worse', 'T1-H': 'ret_up', 'T1-I': 'ret_up',
    'T1-J': 'dd_worse', 'T1-K': 'dd_worse', 'T1-L': 'dd_worse',
    'T1-M': 'dd_worse', 'T1-O': 'dd_worse', 'T1-P': 'dd_worse',
    'T1-Q': 'dd_worse', 'T1-S': 'dd_worse', 'T1-T': 'dd_worse',
}


def daily_returns(run):
    v = [run['start_value']] + run['values']
    dates = run['valuation_dates']
    r = [v[i + 1] / v[i] - 1.0 for i in range(len(dates))]
    return dict(zip(dates, r))


def value_by_date(run):
    return dict(zip(run['valuation_dates'], run['values']))


def seg(dates, lo=None, hi_excl=None):
    return [d for d in dates if (lo is None or d >= lo) and (hi_excl is None or d < hi_excl)]


def run_metrics(run, date_filter=None):
    dates = run['valuation_dates']
    if date_filter:
        keep = set(date_filter)
        idx = [i for i, d in enumerate(dates) if d in keep]
    else:
        idx = list(range(len(dates)))
    if not idx:
        return None
    v0 = run['start_value'] if idx[0] == 0 else run['values'][idx[0] - 1]
    vals = [run['values'][i] for i in idx]
    cash = [run['end_cash'][i] for i in idx]
    n = len(vals)
    series = [v0] + vals
    r = np.diff(series) / series[:-1]
    total = vals[-1] / v0 - 1.0
    cagr = (vals[-1] / v0) ** (252.0 / n) - 1.0 if n >= 5 else None
    sharpe = (float(np.mean(r)) / float(np.std(r, ddof=1)) * math.sqrt(252)
              if n >= 20 and np.std(r, ddof=1) > 0 else None)
    arr = np.array(series)
    maxdd = float((arr / np.maximum.accumulate(arr) - 1.0).min())
    exposure = float(np.mean([1.0 - c / v for c, v in zip(cash, vals) if v > 0]))

    dset = {dates[i] for i in idx}
    fills = [f for f in run['fills'] if f['date'] in dset]
    buys = [f for f in fills if f['action'] == 'BUY']
    sells = [f for f in fills if f['action'] == 'SELL']
    reduces = [f for f in fills if f['action'] == 'REDUCE']
    rts = [f for f in sells if f.get('pnl') is not None]
    win_rate = (sum(1 for f in rts if f['pnl'] > 0) / len(rts)) if rts else None
    cost_sum = sum(abs((f['price'] or 0) - (f['market_price'] or 0)) * (f['shares'] or 0)
                   for f in fills)
    return {
        'n_days': n, 'total_return': round(total, 6),
        'cagr': None if cagr is None else round(cagr, 6),
        'sharpe': None if sharpe is None else round(sharpe, 4),
        'max_drawdown': round(maxdd, 6),
        'exposure': round(exposure, 6),
        'trades': {'BUY': len(buys), 'SELL': len(sells), 'REDUCE': len(reduces)},
        'round_trips': len(rts),
        'win_rate': None if win_rate is None else round(win_rate, 4),
        'txn_costs_dollars': round(cost_sum, 2),
        'txn_costs_pct_v0': round(cost_sum / v0, 6) if v0 else None,
    }


def nw_tstat(d, lag=5):
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 10:
        return None
    mu = d.mean()
    e = d - mu
    s0 = float(e @ e) / n
    s = s0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        s += 2.0 * w * float(e[:-k] @ e[k:]) / n
    if s <= 0:
        return None
    se = math.sqrt(s / n)
    return mu / se if se > 0 else None


def paired_stats(cell_run, base_run, date_filter=None):
    rc = daily_returns(cell_run)
    rb = daily_returns(base_run)
    common = sorted(set(rc) & set(rb))
    if date_filter:
        keep = set(date_filter)
        common = [d for d in common if d in keep]
    d = np.array([rc[x] - rb[x] for x in common])
    n = len(d)
    if n < 10:
        return {'n': n}
    sd = float(np.std(d, ddof=1))
    t = float(d.mean() / (sd / math.sqrt(n))) if sd > 0 else None
    return {
        'n': n,
        'mean_bps_day': round(float(d.mean()) * 1e4, 4),
        'sd_bps_day': round(sd * 1e4, 4),
        't': None if t is None else round(t, 3),
        'nw5_t': None if (x := nw_tstat(d)) is None else round(x, 3),
    }


def seed_noise_bound(cell_run, base_run):
    """B = 3 * 1.15bps * sqrt(sum_ON wf^2 + sum_OFF wf^2), wf = fill $/V."""
    tot = 0.0
    for run in (cell_run, base_run):
        vmap = value_by_date(run)
        for f in run['fills']:
            v = vmap.get(f['date'])
            if v and f['dollars']:
                tot += (f['dollars'] / v) ** 2
    return 3.0 * 1.15 * math.sqrt(tot)  # in bps of total return


def load_cell(cid):
    p = os.path.join(CELLS_DIR, cid, 'results.json')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def median_seed_index(cell_res, base_res, kind):
    deltas = []
    for i, seed in enumerate(SEEDS):
        c = cell_res[kind][i]
        b = base_res[kind][i]
        deltas.append((c['final_value'] / c['start_value']) -
                      (b['final_value'] / b['start_value']))
    order = np.argsort(deltas)
    return int(order[len(order) // 2]), deltas


def assemble():
    base_res = load_cell('C00')
    if base_res is None:
        raise SystemExit('C00 results missing')

    all_dates = base_res['full'][0]['valuation_dates']
    pre_dates = seg(all_dates, hi_excl=HOLDOUT_START)
    llm_dates = seg(all_dates, lo=LLM_ERA_START)

    cell_ids = sorted(os.listdir(CELLS_DIR))
    out = {}
    for cid in cell_ids:
        if cid == 'C00':
            continue
        res = load_cell(cid)
        if res is None:
            continue
        is_llm = cid in LLM_CELLS
        date_filter_full = llm_dates if is_llm else None

        med_i, seed_deltas_full = median_seed_index(res, base_res, 'full')
        med_ih, seed_deltas_hold = median_seed_index(res, base_res, 'holdout')

        cell_stats = {
            'cell_id': cid,
            'llm_era_restricted': is_llm,
            'full': {
                'metrics_cell': run_metrics(res['full'][med_i], date_filter_full),
                'metrics_base': run_metrics(base_res['full'][med_i], date_filter_full),
                'paired': paired_stats(res['full'][med_i], base_res['full'][med_i],
                                       date_filter_full),
                'seed_delta_returns': [round(x, 6) for x in seed_deltas_full],
                'sign_consistent': len({np.sign(x) for x in seed_deltas_full
                                        if x != 0}) <= 1,
            },
            'pre_holdout': {
                'paired': paired_stats(res['full'][med_i], base_res['full'][med_i],
                                       seg(llm_dates if is_llm else all_dates,
                                           hi_excl=HOLDOUT_START)),
                'metrics_cell': run_metrics(res['full'][med_i],
                                            seg(llm_dates if is_llm else all_dates,
                                                hi_excl=HOLDOUT_START)),
                'metrics_base': run_metrics(base_res['full'][med_i],
                                            seg(llm_dates if is_llm else all_dates,
                                                hi_excl=HOLDOUT_START)),
            },
            'holdout': {
                'metrics_cell': run_metrics(res['holdout'][med_ih]),
                'metrics_base': run_metrics(base_res['holdout'][med_ih]),
                'paired': paired_stats(res['holdout'][med_ih],
                                       base_res['holdout'][med_ih]),
                'seed_delta_returns': [round(x, 6) for x in seed_deltas_hold],
                'sign_consistent': len({np.sign(x) for x in seed_deltas_hold
                                        if x != 0}) <= 1,
            },
            'seed_noise_bound_bps': round(
                seed_noise_bound(res['full'][med_i], base_res['full'][med_i]), 3),
        }

        # Decision rule (prereg): gate on full-period read
        f = cell_stats['full']
        p = f['paired']
        b_bps = cell_stats['seed_noise_bound_bps']
        dret = (f['metrics_cell']['total_return'] - f['metrics_base']['total_return'])
        gate = {
            't_ok': p.get('t') is not None and abs(p['t']) >= 2.0,
            'nw_ok': p.get('nw5_t') is not None and abs(p['nw5_t']) >= 1.8,
            'sign_ok': f['sign_consistent'],
            'magnitude_ok': abs(dret) * 1e4 > b_bps,
        }
        gate['meaningful'] = all(gate.values())
        h = cell_stats['holdout']
        dret_h = (h['metrics_cell']['total_return'] - h['metrics_base']['total_return'])
        gate['holdout_sign_agrees'] = bool(np.sign(dret_h) == np.sign(dret)) or dret == 0
        cell_stats['decision_gate'] = gate
        cell_stats['delta_total_return_full'] = round(dret, 6)
        cell_stats['delta_total_return_holdout'] = round(dret_h, 6)

        with open(os.path.join(CELLS_DIR, cid, 'stats.json'), 'w') as fo:
            json.dump(cell_stats, fo, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))

        mpath = os.path.join(CELLS_DIR, cid, 'manifest.json')
        if os.path.exists(mpath):
            with open(mpath) as fo:
                m = json.load(fo)
            m['seed_noise_bound_bps'] = b_bps
            with open(mpath, 'w') as fo:
                json.dump(m, fo, indent=1)
        out[cid] = cell_stats

    # ---- Trigger evaluation (prereg rules) ----
    trig = {'tier1b': None, 'S1': None, 'S2': [], 'S3': []}
    t1llm = out.get('T1-LLM')
    if t1llm:
        t_full = t1llm['full']['paired'].get('t')
        t_hold = t1llm['holdout']['paired'].get('t')
        fired = any(v is not None and abs(v) >= 2.0 for v in (t_full, t_hold))
        trig['tier1b'] = {'parent_t_full_llm_era': t_full,
                          'parent_t_holdout': t_hold, 'fired': bool(fired)}

    t1 = {k: v for k, v in out.items() if k.startswith('T1-')}
    alloff = out.get('T2-ALLOFF')
    if alloff and t1:
        deltas = {k: v['pre_holdout']['paired'].get('mean_bps_day') or 0.0
                  for k, v in t1.items()}
        r_mass = (alloff['pre_holdout']['paired'].get('mean_bps_day') or 0.0) - \
            sum(deltas.values())
        sum_abs = sum(abs(x) for x in deltas.values())
        sds = [v['pre_holdout']['paired'].get('sd_bps_day') or 0.0 for v in t1.values()]
        ns = [v['pre_holdout']['paired'].get('n') or 1 for v in t1.values()]
        se_r = math.sqrt(sum((s / math.sqrt(n)) ** 2 for s, n in zip(sds, ns)))
        s1_fired = abs(r_mass) > max(0.25 * sum_abs, 2 * se_r)
        top4 = sorted(deltas, key=lambda k: -abs(deltas[k]))[:4]
        trig['S1'] = {'R_bps_day': round(r_mass, 4), 'sum_abs': round(sum_abs, 4),
                      'SE_R': round(se_r, 4), 'fired': bool(s1_fired),
                      'top4': top4 if s1_fired else []}

    groups = {
        'SIZE': ['T1-A', 'T1-B', 'T1-LLM', 'T1-D', 'T1-E', 'T1-F'],
        'LABEL': ['T1-G', 'T1-H', 'T1-I', 'T1-J'],
        'SELL': ['T1-K', 'T1-L', 'T1-M', 'T1-O', 'T1-P', 'T1-Q'],
        'BUYFILTER': ['T1-S', 'T1-T'],
    }
    for cid, st in t1.items():
        pp = st['pre_holdout']['paired']
        t_pre = pp.get('t')
        if t_pre is None or abs(t_pre) < 2.0:
            continue
        claim = DESIGN_CLAIMS.get(cid)
        m_c = st['pre_holdout']['metrics_cell']
        m_b = st['pre_holdout']['metrics_base']
        if not (m_c and m_b):
            continue
        ddd = m_c['max_drawdown'] - m_b['max_drawdown']
        dr = m_c['total_return'] - m_b['total_return']
        wrong = ((claim == 'dd_worse' and ddd >= 0 and dr > 0) or
                 (claim == 'ret_up' and dr < 0))
        if wrong:
            grp = next((g for g, mem in groups.items() if cid in mem), None)
            neighbors = [k for k in groups.get(grp, []) if k != cid and k in t1]
            neighbors = sorted(
                neighbors,
                key=lambda k: -abs(t1[k]['pre_holdout']['paired'].get('mean_bps_day') or 0))[:3]
            trig['S2'].append({'cell': cid, 'claim': claim,
                               'd_ret': round(dr, 6), 'd_maxdd': round(ddd, 6),
                               'pair_with': neighbors})
    for cid, st in t1.items():
        tp = st['pre_holdout']['paired'].get('t')
        th = st['holdout']['paired'].get('t')
        mp = st['pre_holdout']['paired'].get('mean_bps_day')
        mh = st['holdout']['paired'].get('mean_bps_day')
        if None in (tp, th, mp, mh):
            continue
        if abs(tp) >= 1.5 and abs(th) >= 1.5 and np.sign(mp) != np.sign(mh):
            trig['S3'].append({'cell': cid, 'pre_t': tp, 'hold_t': th,
                               'pair_with': 'T1-J', 'diagnostic_only': True})

    with open(os.path.join(RUN_DIR, 'trigger_evaluation.json'), 'w') as fo:
        json.dump(trig, fo, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))

    print(json.dumps({k: {'d_full': v['delta_total_return_full'],
                          'd_hold': v['delta_total_return_holdout'],
                          't': v['full']['paired'].get('t'),
                          'nw': v['full']['paired'].get('nw5_t'),
                          'meaningful': v['decision_gate']['meaningful']}
                      for k, v in out.items()}, indent=1))
    print('TRIGGERS:', json.dumps(trig, indent=1))


if __name__ == '__main__':
    assemble()
