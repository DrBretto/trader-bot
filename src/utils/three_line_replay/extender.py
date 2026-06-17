"""Dashboard primary-line extender — invoked from publish_artifacts.run.

PKT-TB-012 (the New-Brain cutover) rewires this from a three-line CHAMPION
replay into a freeze-and-re-anchor:

  - The in-sample optimized champion line through 2026-06-11 is served from the
    byte-immutable static table ``config/champion_freeze_20260611.json`` and is
    NEVER recomputed (no ``run_variant`` for those dates). This is the displayed
    historical champion artifact (D-AUTO-20260616 calls #1/#2; DESIGN_DOSSIER §2
    STAGE 4 / §4 Attack 1).
  - The New Brain (native two-stage engine) is the PRIMARY displayed line forward
    of 2026-06-12, re-anchored C0-continuous to the frozen 06-11 terminal
    (~$114.9k). The forward line is the realized book's OWN daily returns chained
    onto the frozen terminal — display continuity only, never a computational
    splice of the champion (LIVE_PREREG §1).

The legacy champion-replay helpers below are retained (unit-tested) but are no
longer called: the champion is frozen, not replayed.

Defensive: any failure logs and returns the dashboard untouched. The
canonical_replay_anchor static segment still enforces the historical line for
past dates via dashboard_metrics.py, so degrading to the raw canonical line is
safe.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from datetime import datetime as dt, timezone, timedelta
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

from .replay_engine import (
    S3Cache, load_variant_configs, run_variant, VariantConfig,
)
from .strategies import (
    topup_on_psm_rise, extend_fragility_relax, compose,
)

logger = logging.getLogger(__name__)

REPLAY_START = '2026-03-12'


def _market_return_extend(
    equity_curve: List[Dict[str, Any]],
    champ_map: Dict[str, float],
    hybrid_map: Dict[str, float],
    pre_map: Dict[str, float],
) -> None:
    """Retained, unit-tested gap/tail extender (no longer called: the champion
    is frozen, the New Brain forward line is re-anchored from realized returns).
    """
    if not champ_map:
        return
    anchor_c: Optional[float] = None
    anchor_h: Optional[float] = None
    anchor_p: Optional[float] = None
    prior_mkt: Optional[float] = None
    for row in equity_curve:
        d = row['date']
        if d < REPLAY_START:
            continue
        mkt = row.get('benchmark')
        if d in champ_map:
            anchor_c = champ_map[d]
            if d in hybrid_map:
                anchor_h = hybrid_map[d]
            if d in pre_map:
                anchor_p = pre_map[d]
            if mkt is not None and mkt > 0:
                prior_mkt = float(mkt)
            continue
        if anchor_c is None:
            if mkt is not None and mkt > 0:
                prior_mkt = float(mkt)
            continue
        if mkt is None or mkt <= 0 or prior_mkt is None or prior_mkt <= 0:
            continue
        market_return = float(mkt) / prior_mkt - 1.0
        anchor_c = anchor_c * (1 + market_return)
        champ_map[d] = anchor_c
        if anchor_h is not None:
            anchor_h = anchor_h * (1 + market_return)
            hybrid_map[d] = anchor_h
        if anchor_p is not None:
            anchor_p = anchor_p * (1 + market_return)
            pre_map[d] = anchor_p
        prior_mkt = float(mkt)


def _build_champion_strategy():
    """Retained for the unit tests; no longer called (champion is frozen)."""
    return compose(
        [extend_fragility_relax(('choppy',), 0.50), topup_on_psm_rise(1.1, 1.0)],
        'extend_relax_choppy + topup_1.1_full',
    )


def _build_prev_champion_strategy():
    """Retained for the unit tests; no longer called (champion is frozen)."""
    return compose(
        [extend_fragility_relax(('choppy',), 0.50), topup_on_psm_rise(1.2, 1.0)],
        'PREV extend_relax_choppy + topup_1.2_full',
    )


def _load_prev_champion_config(cache: S3Cache) -> Optional[VariantConfig]:
    """Retained for the unit tests; no longer called (champion is frozen)."""
    try:
        prev = cache.get_json('config/decision_params.prev_champion.json')
    except Exception as exc:
        logger.warning("prev_champion config missing; comparison line omitted: %s", exc)
        return None
    return VariantConfig(
        name='prev_champion',
        decision_params=prev['decision_params'],
        regime_compatibility=prev['regime_compatibility'],
        signal_overrides=prev.get('signals', {}),
        regime_fusion_overrides=prev.get('regime_fusion', {}),
        decision_engine_overrides=prev.get('decision_engine', {}),
        ensemble_overrides=prev.get('ensemble', {}),
        transaction_cost_overrides=prev.get('transaction_costs', {}),
    )


def _detect_engine_dates(s3_client, forward_dates: List[str]) -> set:
    """The set of forward dates the two-stage engine ACTUALLY drove — i.e. dates
    for which the cutover wrote daily/<D>/brain_selected_universe.json. The
    "New Brain" brand attaches ONLY to these dates; everything else forward of
    the boundary is the incumbent (old) algorithm and must NOT wear the name
    (DESIGN_DOSSIER §3 Attack-5: you cannot brand the old book as the rebuild)."""
    out: set = set()
    if s3_client is None:
        return out
    for d in forward_dates:
        try:
            s3_client.head_object(Bucket='investment-system-data',
                                  Key=f'daily/{d}/brain_selected_universe.json')
            out.add(d)
        except Exception:
            pass
    return out


def extend_dashboard(s3_client, dash: Dict[str, Any],
                     engine_driven_dates: Optional[set] = None) -> Dict[str, Any]:
    """Freeze the champion <= 2026-06-11 and re-anchor the forward line.

    Mutates ``dash`` in place:
      - every equity_curve row's ``value`` becomes the PRIMARY line: the frozen
        champion through 2026-06-11, then the re-anchored realized book forward;
      - ``champion_frozen_value`` carries the frozen champion (ends 06-11);
        ``new_brain_value`` carries ONLY the dates the two-stage engine actually
        drove (it wrote brain_selected_universe.json); ``incumbent_value`` carries
        forward dates the OLD algorithm drove (before the engine went live);
      - drawdowns / monthly_returns / hero metrics are recomputed from the
        primary line; holdings / trades are left as the real book's.

    The "New Brain" brand is authorized ONLY when the engine has actually traded
    (Attack-5 ruling). Until then the forward line is the incumbent algorithm and
    is labeled as such — never branded the rebuild.

    On any error, logs and returns dash unchanged.
    """
    try:
        from src.utils.canonical_replay_anchor import (
            champion_freeze_map, NEW_BRAIN_BOUNDARY_DATE,
        )

        frozen_map, term_date, term_value = champion_freeze_map()
        if not frozen_map or term_value is None:
            logger.warning("champion freeze table absent; dashboard left on raw "
                           "canonical line (no re-anchor)")
            return dash

        boundary = NEW_BRAIN_BOUNDARY_DATE
        equity_curve = dash.get('equity_curve', [])
        if not equity_curve:
            return dash

        dates = [r['date'] for r in equity_curve]
        forward_dates = [d for d in dates if d > boundary]
        if engine_driven_dates is None:
            engine_driven_dates = _detect_engine_dates(s3_client, forward_dates)
        new_brain_start = min(engine_driven_dates) if engine_driven_dates else None
        # Realized continuity values BEFORE we overwrite — the source of the
        # New Brain forward line's daily returns (the realized book).
        cont = {r['date']: r.get('value') for r in equity_curve}

        # Realized book value at/just before the boundary (the return base).
        prev_cont: Optional[float] = None
        for d in dates:
            if d <= boundary and cont.get(d) is not None:
                prev_cont = float(cont[d])

        # New Brain forward line: chain realized daily returns from the frozen
        # terminal (C0-continuous, no recompute).
        nb: Dict[str, float] = {}
        nb_prev = float(term_value)
        for d in dates:
            if d <= boundary:
                continue
            c = cont.get(d)
            if c is not None and prev_cont and prev_cont > 0 and c > 0:
                ret = float(c) / prev_cont - 1.0
                nb_prev = nb_prev * (1.0 + ret)
                prev_cont = float(c)
            # else: no priceable move this date -> flat-hold (honest "no data")
            nb[d] = nb_prev

        # Rewrite the primary line.
        primary_curve: List[Dict[str, Any]] = []
        for row in equity_curve:
            d = row['date']
            if d <= boundary:
                fv = frozen_map.get(d)
                v = fv if fv is not None else row.get('value')
                row['value'] = v
                row['champion_frozen_value'] = v
                row['new_brain_value'] = None
                row['optimized_value'] = v          # legacy field compat
            else:
                v = nb.get(d, row.get('value'))
                row['value'] = v
                row['champion_frozen_value'] = None  # champion is frozen, not extended
                row['optimized_value'] = v
                # Brand a forward date "New Brain" ONLY if the engine actually
                # drove it; otherwise it is the incumbent (old) algorithm.
                if new_brain_start is not None and d >= new_brain_start:
                    row['new_brain_value'] = v
                    row['incumbent_value'] = None
                else:
                    row['new_brain_value'] = None
                    row['incumbent_value'] = v
            row['cumulative_external_cashflow'] = 0.0
            if v is not None:
                primary_curve.append({'date': d, 'value': v})

        has_new_brain = new_brain_start is not None

        # ---- recompute drawdowns ------------------------------------------
        peak = 0.0
        new_dd = []
        for r in primary_curve:
            v = r['value']
            if v > peak:
                peak = v
            dd = (v - peak) / peak if peak > 0 else 0.0
            new_dd.append({'date': r['date'], 'drawdown': dd})
        dash['drawdowns'] = new_dd

        # ---- recompute monthly returns ------------------------------------
        month_buckets: Dict[tuple, List[float]] = defaultdict(list)
        prev_v = None
        for r in primary_curve:
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
            monthly.append({'year': year, 'month': month,
                            'return_pct': compound - 1, 'observations': len(rets)})
        dash['monthly_returns'] = monthly

        # ---- recompute hero metrics from the primary line -----------------
        today = primary_curve[-1]['date']
        total_value = primary_curve[-1]['value']

        def find_v(curve, target):
            for r in curve:
                if r['date'] >= target:
                    return r['value']
            return None
        ytd_base = find_v(primary_curve, f'{today[:4]}-01-01')
        mtd_base = find_v(primary_curve, f'{today[:7]}-01')
        ytd_return = (total_value / ytd_base - 1) if ytd_base and ytd_base > 0 else 0.0
        mtd_return = (total_value / mtd_base - 1) if mtd_base and mtd_base > 0 else 0.0

        daily_returns = []
        prev_v = None
        for r in primary_curve:
            if prev_v is not None and prev_v > 0:
                daily_returns.append((r['value'] - prev_v) / prev_v)
            prev_v = r['value']
        sharpe = None
        if len(daily_returns) >= 60:
            sd = stdev(daily_returns)
            sharpe = (mean(daily_returns) / sd * math.sqrt(252)) if sd > 0 else None
        max_dd = min((d['drawdown'] for d in new_dd), default=0.0)
        current_dd = new_dd[-1]['drawdown'] if new_dd else 0.0

        now_local = dt.now(tz=timezone(timedelta(hours=-4)))
        now_iso = now_local.isoformat()

        m = dash['metrics']
        m['total_value'] = total_value
        m['ytd_return'] = ytd_return
        m['mtd_return'] = mtd_return
        m['sharpe_ratio'] = sharpe
        m['sharpe_observations'] = len(daily_returns)
        m['max_drawdown'] = max_dd
        m['current_drawdown'] = current_dd
        m['corrected_total_value'] = total_value
        m['actual_total_value'] = total_value
        # The canon line is now the New Brain (native two-stage engine), NOT the
        # champion. Calling it the champion would be the "lie kept for show" the
        # rebuild retires (DESIGN_DOSSIER §1).
        m['canon_source'] = 'new_brain'
        m['champion_frozen_terminal'] = {'date': term_date, 'value': term_value}
        m['timestamp'] = now_iso

        dash['snapshot']['timestamp'] = now_iso
        dash['snapshot']['date'] = today
        dash['snapshot']['id'] = f"{today}:new_brain_canon:{now_iso}"

        # Brand authorization: "New Brain" attaches ONLY to dates the two-stage
        # engine ACTUALLY drove (it wrote brain_selected_universe.json). Until the
        # engine goes live the forward line is the INCUMBENT (old) algorithm and
        # must NOT wear the rebuild's name (DESIGN_DOSSIER §3 Attack-5: you cannot
        # ship the old book and call it the rebuild).
        brand_authorized = has_new_brain
        has_incumbent_forward = any(d > boundary for d in dates) and (
            new_brain_start is None or new_brain_start > min(
                (d for d in dates if d > boundary), default=boundary))

        if brand_authorized:
            main_line_label = 'Portfolio (New Brain — native two-stage engine, from %s)' % new_brain_start
        elif has_incumbent_forward:
            main_line_label = ('Portfolio (current/incumbent algorithm — New Brain '
                               'pending engine go-live)')
        else:
            main_line_label = 'Portfolio (frozen champion through 2026-06-11)'

        dash['timeline_correction'] = {
            'version': 'lambda-new-brain-canon-v1',
            'main_line_field': 'value',
            'main_line_label': main_line_label,
            'canon_source': 'new_brain',
            'brand_authorized': brand_authorized,
            'brand': 'New Brain' if brand_authorized else None,
            'new_brain_start_date': new_brain_start,
            'incumbent_forward_present': has_incumbent_forward,
            'incumbent_forward_note': (None if not has_incumbent_forward else
                'The line forward of 2026-06-11 is the CURRENT (incumbent) algorithm, '
                'NOT the New Brain. The two-stage engine has not yet written live '
                'intents (it falls back to incumbent on any abort). The New Brain '
                'brand attaches only from its first engine-driven day.'),
            'champion_frozen': {
                'field': 'champion_frozen_value',
                'label': 'Champion (frozen, in-sample, retired 2026-06-11)',
                'boundary_date': boundary,
                'terminal_value': term_value,
                'n_points': len(frozen_map),
                'byte_immutable': True,
                'source': 'config/champion_freeze_20260611.json (never recomputed)',
            },
            'new_brain_line': {
                'field': 'new_brain_value',
                'label': 'New Brain (native two-stage engine)',
                'anchor_date': term_date,
                'anchor_value': term_value,
                'start_date': new_brain_start,
                're_anchor': 'C0-continuous; realized book returns chained onto the '
                             'frozen terminal (display continuity, NOT a splice).',
                'present': has_new_brain,
            },
            'incumbent_line': {
                'field': 'incumbent_value',
                'label': 'Current algorithm (incumbent, pre-engine-go-live)',
                'present': has_incumbent_forward,
            },
            # Kept for the publish-time advance guard: the newest primary-line
            # date the corpus made available this run.
            'champion_frontier': max(dates) if dates else None,
            'forward_confirmed': False,
            'forward_confirmed_note': ('The New Brain go-live universe is '
                                       'forward_confirmed:false (DESIGN_DOSSIER '
                                       'Attack 2). The U-E universe-choice rung is '
                                       'displayed so a backtest-lottery core shows '
                                       'zero/negative forward universe rent live.'),
            'non_assertion': ('The New Brain line measures dollar conversion '
                              'forward; it does NOT assert a dollar edge. The '
                              'exposure-stripped selection residual is measured at '
                              'zero (LIVE_PREREG.md).'),
            'note': ('Champion frozen byte-immutable <= 2026-06-11; New Brain '
                     'primary forward of 2026-06-12, re-anchored C0-continuous. '
                     'No recompute of either line.'),
        }

        logger.info("new-brain canon: champion frozen at $%.2f (%d pts); "
                    "New Brain forward present=%s; total=$%.2f",
                    term_value, len(frozen_map), has_new_brain, total_value)
        return dash
    except Exception as exc:
        logger.exception("new-brain extension FAILED: %s", exc)
        return dash
