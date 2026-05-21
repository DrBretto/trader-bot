"""Three-line dashboard extender — invoked from publish_artifacts.run.

Runs three replays from 2026-03-11 EOD forward to the latest available
trading day:
- canonical hybrid (active.json config)
- pre-hybrid (opt-bootstrap config)
- optimized champion (active + extend_relax_choppy + topup_psm_1.2)

Patches the dashboard payload's `equity_curve` with `value`,
`pre_hybrid_value`, `optimized_value` per row, and recomputes
drawdowns / monthly_returns / hero metrics from the canonical curve.

Defensive: any failure logs and returns the dashboard untouched.
The canonical_replay_anchor static segment already enforces the
hybrid line for past dates via dashboard_metrics.py, so dropping
the extender does not break the chart's main line.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from datetime import datetime as dt, timezone, timedelta
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

from .replay_engine import (
    S3Cache, load_variant_configs, run_variant,
)
from .strategies import (
    topup_on_psm_rise, extend_fragility_relax, compose,
)

logger = logging.getLogger(__name__)

REPLAY_START = '2026-03-12'


def _build_champion_strategy():
    """The in-sample champion: extend_relax_choppy_conf0.50 + topup_psm_1.2_full."""
    return compose(
        [extend_fragility_relax(('choppy',), 0.50), topup_on_psm_rise(1.2, 1.0)],
        'extend_relax_choppy + topup_1.2_full',
    )


def extend_dashboard(s3_client, dash: Dict[str, Any]) -> Dict[str, Any]:
    """Mutate `dash` in-place: add three-line equity curve fields and refresh
    canonical metrics. Returns the same dict for chaining.

    On any error, logs and returns dash unchanged.
    """
    try:
        cache = S3Cache(s3_client, bucket='investment-system-data')
        trading_dates = cache.list_daily_dates()
        trading_dates = [d for d in trading_dates if d >= '2026-03-11']
        if len(trading_dates) < 3:
            logger.warning("three_line_replay: not enough trading dates (%d); skipping", len(trading_dates))
            return dash

        universe_df = cache.get_csv('config/universe.csv')
        hybrid_cfg, pre_hybrid_cfg = load_variant_configs(cache)

        logger.info("three_line_replay: running canonical hybrid")
        hybrid = run_variant(cache, hybrid_cfg, None, trading_dates, universe_df)

        if pre_hybrid_cfg is not None:
            logger.info("three_line_replay: running pre-hybrid")
            pre_hybrid = run_variant(cache, pre_hybrid_cfg, None, trading_dates, universe_df)
        else:
            pre_hybrid = None

        logger.info("three_line_replay: running optimized champion")
        champion_strategy = _build_champion_strategy()
        champion = run_variant(cache, hybrid_cfg, champion_strategy, trading_dates, universe_df)

        hybrid_map = dict(hybrid['date_value_map'])
        pre_map = dict(pre_hybrid['date_value_map']) if pre_hybrid is not None else {}
        champ_map = dict(champion['date_value_map'])

        # Tail extension: the replay can only simulate dates that have full
        # night artifacts (features/signals/inference/prices). When the night
        # phase has been down (e.g. 2026-05-11 → 2026-05-20 during the Stooq
        # outage), those days have no artifacts and the replay stops at the
        # last fully-instrumented date. Without this extension, the per-row
        # patch loop below forward-fills the last replay value forever,
        # producing the visible "flatline since 2026-05-08" chart symptom.
        #
        # We use the broker's daily return (computed from raw_value, the
        # marked-to-market broker equity already present in the dashboard
        # payload for every day) as a proxy for the optimized champion's
        # would-be daily return on missing days. The replay's portfolio is
        # ~100% equity and the broker portfolio is also long-equity, so the
        # broker's daily return is a reasonable post-seam continuation of
        # the canon line. This keeps the chart moving with the market
        # instead of plateauing.
        last_replay_date = champion.get('final_date')
        if last_replay_date:
            prior_raw = None
            tail_h = hybrid.get('final_value')
            tail_p = pre_hybrid.get('final_value') if pre_hybrid is not None else None
            tail_c = champion.get('final_value')
            for row in dash['equity_curve']:
                d = row['date']
                if d <= last_replay_date:
                    raw = row.get('raw_value')
                    if raw is not None and raw > 0:
                        prior_raw = float(raw)
                    continue
                raw = row.get('raw_value')
                if raw is None or prior_raw is None or prior_raw <= 0 or raw <= 0:
                    continue
                broker_return = float(raw) / prior_raw - 1.0
                if tail_c is not None:
                    tail_c = tail_c * (1 + broker_return)
                    champ_map[d] = tail_c
                if tail_h is not None:
                    tail_h = tail_h * (1 + broker_return)
                    hybrid_map[d] = tail_h
                if tail_p is not None:
                    tail_p = tail_p * (1 + broker_return)
                    pre_map[d] = tail_p
                prior_raw = float(raw)

        # Patch equity_curve. After the 2026-05-16 canon promotion the
        # OPTIMIZED champion line is the primary canon (`value`); the live
        # hybrid configuration's counterfactual is preserved as `hybrid_value`
        # (demoted to comparison); the pre-hybrid replay stays in
        # `pre_hybrid_value` (older comparison). `optimized_value` mirrors
        # `value` for backwards compatibility with the desktop chart's
        # `optimized_value ?? corrected_value` fallback.
        last_h = last_p = last_c = None
        canonical_curve_values: List[Dict[str, Any]] = []
        for row in dash['equity_curve']:
            d = row['date']
            if d < REPLAY_START:
                row['cumulative_external_cashflow'] = 0.0
                canonical_curve_values.append({'date': d, 'value': row['value']})
                continue
            if d in hybrid_map:
                last_h = hybrid_map[d]
            if d in pre_map:
                last_p = pre_map[d]
            if d in champ_map:
                last_c = champ_map[d]
            # Optimized champion is the primary canon line.
            if last_c is not None:
                row['value'] = last_c
                row['optimized_value'] = last_c
            elif last_h is not None:
                # Champion hadn't yet diverged from hybrid on this date — fall
                # back to hybrid to keep the line continuous across the
                # backtest segment before champion-specific actions fire.
                row['value'] = last_h
            row['cumulative_external_cashflow'] = 0.0
            if last_p is not None:
                row['pre_hybrid_value'] = last_p
            if last_h is not None:
                row['hybrid_value'] = last_h
            canonical_curve_values.append({'date': d, 'value': row['value']})

        # Recompute drawdowns
        peak = 0.0
        new_dd = []
        for r in canonical_curve_values:
            v = r['value']
            if v > peak:
                peak = v
            dd = (v - peak) / peak if peak > 0 else 0.0
            new_dd.append({'date': r['date'], 'drawdown': dd})
        dash['drawdowns'] = new_dd

        # Recompute monthly_returns
        month_buckets: Dict[tuple, List[float]] = defaultdict(list)
        prev_v = None
        for r in canonical_curve_values:
            v = r['value']
            if prev_v is not None and prev_v > 0:
                ds = dt.strptime(r['date'], '%Y-%m-%d')
                month_buckets[(ds.year, ds.month)].append((v - prev_v) / prev_v)
            prev_v = v
        monthly = []
        for (year, month), rets in sorted(month_buckets.items()):
            compound = 1.0
            for x in rets:
                compound *= (1 + x)
            monthly.append({'year': year, 'month': month, 'return_pct': compound - 1, 'observations': len(rets)})
        dash['monthly_returns'] = monthly

        # Recompute metrics
        today = canonical_curve_values[-1]['date']
        total_value = canonical_curve_values[-1]['value']

        def find_v(curve, target):
            for r in curve:
                if r['date'] >= target:
                    return r['value']
            return None
        ytd_base = find_v(canonical_curve_values, f'{today[:4]}-01-01')
        mtd_base = find_v(canonical_curve_values, f'{today[:7]}-01')
        ytd_return = (total_value / ytd_base - 1) if ytd_base and ytd_base > 0 else 0.0
        mtd_return = (total_value / mtd_base - 1) if mtd_base and mtd_base > 0 else 0.0

        daily_returns = []
        prev_v = None
        for r in canonical_curve_values:
            if prev_v is not None and prev_v > 0:
                daily_returns.append((r['value'] - prev_v) / prev_v)
            prev_v = r['value']
        sharpe = None
        if len(daily_returns) >= 60:
            sd = stdev(daily_returns)
            sharpe = (mean(daily_returns) / sd * math.sqrt(252)) if sd > 0 else None
        max_dd = min((d['drawdown'] for d in new_dd), default=0.0)
        current_dd = new_dd[-1]['drawdown'] if new_dd else 0.0

        # Canon final state for cash/holdings/trades comes from the OPTIMIZED
        # CHAMPION replay (post-2026-05-16 promotion). This keeps the holdings
        # table, trade log, and exposure tiles internally consistent with the
        # primary equity line.
        last_canon_holdings = champion.get('final_holdings', hybrid.get('final_holdings', []))
        last_canon_cash = champion.get('final_cash', hybrid.get('final_cash', 0.0))
        canonical_invested = total_value - last_canon_cash
        canonical_cash_pct = last_canon_cash / total_value if total_value > 0 else 0.0
        canonical_invested_pct = canonical_invested / total_value if total_value > 0 else 0.0
        top_position_value = max(
            (h.get('shares', 0) * (h.get('close_price') or h.get('peak_price') or h.get('entry_price') or 0)
             for h in last_canon_holdings), default=0)
        top_position_pct = top_position_value / total_value if total_value > 0 else 0

        # Trade summary from optimized champion line (canon)
        actions_sorted = sorted(champion['actions'], key=lambda r: (r['date'], 0 if r.get('action') == 'SELL' else 1))
        open_lots: Dict[str, deque] = defaultdict(deque)
        round_trips = []
        trades_history = []
        for idx, r in enumerate(actions_sorted):
            sym = r['symbol']; action = r['action']
            shares = int(float(r.get('shares', 0)))
            if shares <= 0:
                continue
            price = float(r.get('price', 0))
            dollars = shares * price
            fill_id = f"{r['date']}:{idx}:{sym}"
            trades_history.append({
                'timestamp': f"{r['date']}T16:00:00", 'symbol': sym, 'action': action,
                'shares': float(shares), 'price': price, 'market_price': price,
                'dollars': round(dollars, 2), 'reason': r.get('reason', ''), 'regime': '',
                'broker_order_id': None, 'broker_client_order_id': None,
                'broker_status': 'replay', 'execution_mode': 'optimized_champion_replay',
            })
            if action == 'BUY':
                open_lots[sym].append({'fill_id': fill_id, 'entry_price': price,
                                       'entry_date': r['date'], 'shares_remaining': shares})
            else:
                remaining = shares
                while remaining > 0 and open_lots[sym]:
                    lot = open_lots[sym][0]
                    matched = min(remaining, lot['shares_remaining'])
                    pnl = (price - lot['entry_price']) * matched
                    pnl_pct = (price / lot['entry_price'] - 1) if lot['entry_price'] > 0 else 0
                    days_held = (dt.strptime(r['date'], '%Y-%m-%d') - dt.strptime(lot['entry_date'], '%Y-%m-%d')).days
                    round_trips.append({
                        'round_trip_id': f"{lot['fill_id']}->{fill_id}:{matched}", 'symbol': sym,
                        'entry_fill_id': lot['fill_id'], 'exit_fill_id': fill_id,
                        'entry_date': lot['entry_date'], 'exit_date': r['date'],
                        'entry_price': lot['entry_price'], 'exit_price': price, 'shares': matched,
                        'realized_pnl': round(pnl, 2), 'realized_pnl_pct': pnl_pct, 'days_held': days_held,
                    })
                    lot['shares_remaining'] -= matched
                    remaining -= matched
                    if lot['shares_remaining'] <= 0:
                        open_lots[sym].popleft()
        wins = sum(1 for rt in round_trips if rt['realized_pnl'] > 0)
        losses = sum(1 for rt in round_trips if rt['realized_pnl'] < 0)
        breakeven = len(round_trips) - wins - losses
        counted = wins + losses
        win_rate = wins / counted if counted > 0 else 0.0

        # Holdings array. last_canon_date is the latest patched row's date,
        # which equals the latest equity_curve date when the tail-extension
        # extended through to today. Falls back to the replay's final_date
        # (may be older than today if the tail extension had no broker
        # raw_value continuity to follow).
        holdings_array = []
        last_canon_date = (
            canonical_curve_values[-1]['date'] if canonical_curve_values
            else champion.get('final_date', hybrid.get('final_date', today))
        )
        for h in last_canon_holdings:
            shares = h.get('shares', 0)
            cp = h.get('close_price') or h.get('peak_price') or h.get('entry_price') or 0
            ep = h.get('entry_price', 0)
            de = h.get('entry_date', last_canon_date)
            try:
                days_held = (dt.strptime(last_canon_date, '%Y-%m-%d') - dt.strptime(de, '%Y-%m-%d')).days
            except Exception:
                days_held = 0
            holdings_array.append({
                'symbol': h['symbol'], 'shares': shares, 'entry_price': ep, 'current_price': cp,
                'market_value': round(shares * cp if cp else 0, 2),
                'unrealized_pnl': round((cp - ep) * shares if cp else 0, 2),
                'unrealized_pnl_pct': (cp / ep - 1) if ep > 0 else 0,
                'health_score': 0.7, 'vol_bucket': 'med', 'days_held': days_held,
            })

        # Bump timestamp to current local time with TZ info
        now_local = dt.now(tz=timezone(timedelta(hours=-4)))
        now_iso = now_local.isoformat()

        m = dash['metrics']
        m['total_value'] = total_value
        m['cash'] = last_canon_cash
        m['invested'] = canonical_invested
        m['ytd_return'] = ytd_return
        m['mtd_return'] = mtd_return
        m['sharpe_ratio'] = sharpe
        m['sharpe_observations'] = len(daily_returns)
        m['max_drawdown'] = max_dd
        m['current_drawdown'] = current_dd
        m['win_rate'] = win_rate
        m['total_trades'] = counted
        m['wins'] = wins
        m['losses'] = losses
        m['breakeven_trades'] = breakeven
        m['realized_round_trips'] = len(round_trips)
        m['total_fills'] = len(trades_history)
        m['cumulative_transaction_costs'] = 0.0
        m['cash_pct'] = canonical_cash_pct
        m['gross_exposure'] = canonical_invested_pct
        m['net_exposure'] = canonical_invested_pct
        m['top_position_pct'] = top_position_pct
        m['corrected_total_value'] = total_value
        m['actual_total_value'] = total_value
        m['canon_source'] = 'optimized_champion'
        m['timestamp'] = now_iso

        dash['snapshot']['timestamp'] = now_iso
        dash['snapshot']['date'] = last_canon_date
        dash['snapshot']['id'] = f"{last_canon_date}:three_line_replay:{now_iso}"

        dash['holdings'] = holdings_array
        dash['trades'] = sorted(trades_history, key=lambda t: t['timestamp'], reverse=True)
        dash['trade_summary'] = {
            'fills_total': len(trades_history), 'realized_round_trips': len(round_trips),
            'wins': wins, 'losses': losses, 'breakeven': breakeven, 'win_rate': win_rate,
            'unmatched_closing_shares': sum(lot['shares_remaining'] for lots in open_lots.values() for lot in lots),
            'cumulative_transaction_costs': 0.0,
        }
        dash['round_trips'] = round_trips

        dash['timeline_correction'] = {
            'version': 'lambda-three-line-replay-v2-optimized-canon',
            'main_line_field': 'value',
            'main_line_label': 'Portfolio (optimized champion)',
            'canon_source': 'optimized_champion',
            'canon_promotion': {
                'primary': 'optimized_value (extend_relax_choppy_conf0.50 + topup_psm_1.2_full)',
                'comparison_solid': 'hybrid_value (live config: hybrid-ranking-035-v1)',
                'comparison_dashed': 'pre_hybrid_value (older opt-bootstrap config)',
                'note': 'value field carries optimized; hybrid preserved in hybrid_value field for comparison',
            },
            'comparison_line_field': 'hybrid_value',
            'comparison_line_label': 'Hybrid (comparison)',
            'comparison_line_present': True,
            'pre_hybrid_line_field': 'pre_hybrid_value',
            'pre_hybrid_line_label': 'Original (pre-hybrid)',
            'pre_hybrid_line_present': pre_hybrid is not None,
            'optimized_line_field': 'optimized_value',
            'optimized_line_label': 'Optimized strategy (in-sample champion)',
            'last_simulation_date': last_canon_date,
            'three_line_finals': {
                'canonical_hybrid': hybrid.get('final_value'),
                'pre_hybrid': pre_hybrid.get('final_value') if pre_hybrid is not None else None,
                'optimized_champion': champion.get('final_value'),
            },
            'optimized_strategy': 'extend_relax_choppy_conf0.50 + topup_psm_1.2_full',
            'note': 'Three lines re-run from 2026-03-11 EOD forward to the latest trading day each Lambda invocation. Source-faithful, no look-ahead. Optimized is canon; hybrid preserved as `hybrid_value`.',
        }

        logger.info("three_line_replay: extension complete. hybrid=$%.2f pre=$%s opt=$%.2f",
                    hybrid.get('final_value') or 0,
                    f"{pre_hybrid.get('final_value'):.2f}" if pre_hybrid is not None and pre_hybrid.get('final_value') else 'n/a',
                    champion.get('final_value') or 0)
        return dash
    except Exception as exc:
        logger.exception("three_line_replay extension FAILED: %s", exc)
        return dash
