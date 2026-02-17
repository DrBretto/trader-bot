"""Deterministic replay harness that uses the canonical live decision path."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import random
from typing import Any, Dict, List, Optional

import pandas as pd

from optimizer.data_access import OptimizerDataset
from src.steps import decision_engine, paper_trader


@dataclass
class ReplayStep:
    decision_date: str
    valuation_date: str
    start_value: float
    end_value: float
    regime: str
    actions_count: int
    traded_notional: float


@dataclass
class ReplayResult:
    steps: List[ReplayStep]
    fills: List[Dict[str, Any]]
    final_portfolio: Dict[str, Any]


ACTION_PRIORITY = {
    'SELL': 0,
    'REDUCE': 1,
    'BUY': 2,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _percentile_score(value: float, thresholds: Dict[str, float]) -> float:
    p20 = _safe_float(thresholds.get('p20'), value)
    p50 = _safe_float(thresholds.get('p50'), value)
    p80 = _safe_float(thresholds.get('p80'), value)
    p95 = _safe_float(thresholds.get('p95'), value)

    if value <= p20:
        return 0.1
    if value <= p50 and p50 != p20:
        return 0.2 + 0.3 * (value - p20) / (p50 - p20)
    if value <= p80 and p80 != p50:
        return 0.5 + 0.3 * (value - p50) / (p80 - p50)
    if value <= p95 and p95 != p80:
        return 0.8 + 0.15 * (value - p80) / (p95 - p80)
    return 0.95


def _build_expert_signals(
    signal_row: Dict[str, Any],
    signal_overrides: Dict[str, Any],
    entropy_state: Dict[str, Any],
) -> Dict[str, Any]:
    macro_cfg = signal_overrides.get('macro_credit', {})
    vol_cfg = signal_overrides.get('vol_uncertainty', {})
    frag_cfg = signal_overrides.get('fragility', {})
    ent_cfg = signal_overrides.get('entropy_shift', {})

    # Macro/Credit
    slope = _safe_float(signal_row.get('yield_slope_10y_3m'))
    hy_spread = _safe_float(signal_row.get('hy_spread_proxy'))
    slope_mean = _safe_float(macro_cfg.get('SLOPE_MEAN'), 1.5)
    slope_std = max(_safe_float(macro_cfg.get('SLOPE_STD'), 1.0), 1e-6)
    hy_mean = _safe_float(macro_cfg.get('HY_SPREAD_MEAN'), 0.0)
    hy_std = max(_safe_float(macro_cfg.get('HY_SPREAD_STD'), 0.02), 1e-6)
    slope_weight = _safe_float(macro_cfg.get('slope_weight'), 0.6)
    hy_weight = _safe_float(macro_cfg.get('hy_spread_weight'), 0.4)
    total_weight = slope_weight + hy_weight
    if total_weight <= 0:
        slope_weight, hy_weight = 0.6, 0.4
    else:
        slope_weight /= total_weight
        hy_weight /= total_weight

    slope_z = (slope - slope_mean) / slope_std
    hy_z = (hy_spread - hy_mean) / hy_std
    macro_score = slope_weight * math.tanh(slope_z) + hy_weight * math.tanh(hy_z)

    # Vol uncertainty
    vix_value = _safe_float(signal_row.get('vix_value'), 0.0)
    vvix_value = _safe_float(signal_row.get('vvix_value'), 0.0)
    skew_value = _safe_float(signal_row.get('skew_value'), 0.0)

    vix_thresholds = vol_cfg.get(
        'vix_thresholds',
        {'p20': 13.0, 'p50': 17.0, 'p80': 25.0, 'p95': 30.0},
    )
    vvix_thresholds = vol_cfg.get(
        'vvix_thresholds',
        {'p20': 75.0, 'p50': 85.0, 'p80': 105.0, 'p95': 120.0},
    )
    skew_thresholds = vol_cfg.get(
        'skew_thresholds',
        {'p20': 115.0, 'p50': 125.0, 'p80': 140.0, 'p95': 150.0},
    )

    vix_pctile = _percentile_score(vix_value, vix_thresholds)
    vvix_pctile = _percentile_score(vvix_value, vvix_thresholds)
    skew_pctile = _percentile_score(skew_value, skew_thresholds)

    vol_score = 0.45 * vix_pctile + 0.35 * vvix_pctile + 0.20 * skew_pctile
    panic = vol_cfg.get('regime_thresholds', {}).get('panic', {})
    unstable = vol_cfg.get('regime_thresholds', {}).get('unstable', {})
    panic_vix = _safe_float(panic.get('vix_pctile'), 0.8)
    panic_vvix = _safe_float(panic.get('vvix_pctile'), 0.8)
    unstable_vvix = _safe_float(unstable.get('vvix_pctile'), 0.8)
    unstable_vix_max = _safe_float(unstable.get('vix_pctile_max'), 0.6)

    if vix_pctile > panic_vix and vvix_pctile > panic_vvix:
        vol_regime = 'panic'
    elif vvix_pctile > unstable_vvix and vix_pctile < unstable_vix_max:
        vol_regime = 'unstable_calm'
    else:
        vol_regime = 'calm'

    # Fragility
    avg_corr = _safe_float(signal_row.get('avg_correlation'))
    pc1 = _safe_float(signal_row.get('pc1_explained'))
    symbols_used = int(_safe_float(signal_row.get('symbols_used'), 8))
    min_symbols = int(_safe_float(frag_cfg.get('MIN_SYMBOLS'), 6))

    avg_corr_mean = _safe_float(frag_cfg.get('AVG_CORR_MEAN'), 0.30)
    avg_corr_std = max(_safe_float(frag_cfg.get('AVG_CORR_STD'), 0.15), 1e-6)
    pc1_mean = _safe_float(frag_cfg.get('PC1_MEAN'), 0.45)
    pc1_std = max(_safe_float(frag_cfg.get('PC1_STD'), 0.12), 1e-6)

    if symbols_used < min_symbols:
        fragility_score = 0.5
    else:
        corr_z = (avg_corr - avg_corr_mean) / avg_corr_std
        pc1_z = (pc1 - pc1_mean) / pc1_std
        fragility_score = 0.5 * ((math.tanh(corr_z) + 1.0) / 2.0) + 0.5 * ((math.tanh(pc1_z) + 1.0) / 2.0)

    # Entropy shift
    entropy_score = _safe_float(signal_row.get('entropy_score'), 0.5)
    entropy_z = _safe_float(signal_row.get('entropy_z_score'), 0.0)
    z_threshold = _safe_float(ent_cfg.get('z_threshold'), 1.5)
    consecutive_days_required = int(_safe_float(ent_cfg.get('consecutive_days_required'), 3))

    above = abs(entropy_z) > z_threshold
    prev_above = bool(entropy_state.get('prev_above', False))
    prev_days = int(entropy_state.get('prev_days', 0))
    if above and prev_above:
        consecutive = prev_days + 1
    elif above:
        consecutive = 1
    else:
        consecutive = 0
    entropy_shift_flag = consecutive >= consecutive_days_required

    entropy_state['prev_above'] = above
    entropy_state['prev_days'] = consecutive

    return {
        'macro_credit': {
            'macro_credit_score': max(-1.0, min(1.0, macro_score)),
            'yield_slope_10y_3m': slope,
            'hy_spread_proxy': hy_spread,
            'slope_z_score': slope_z,
            'hy_spread_z_score': hy_z,
        },
        'vol_uncertainty': {
            'vol_uncertainty_score': max(0.0, min(1.0, vol_score)),
            'vol_regime_label': vol_regime,
            'vix_percentile': vix_pctile,
            'vvix_percentile': vvix_pctile,
            'skew_percentile': skew_pctile,
            'vix_value': vix_value,
            'vvix_value': vvix_value,
            'skew_value': skew_value,
        },
        'fragility': {
            'fragility_score': max(0.0, min(1.0, fragility_score)),
            'avg_correlation': avg_corr,
            'pc1_explained': pc1,
            'pc2_explained': _safe_float(signal_row.get('pc2_explained'), 0.0),
            'symbols_used': symbols_used,
        },
        'entropy_shift': {
            'entropy_score': entropy_score,
            'entropy_z_score': entropy_z,
            'entropy_shift_flag': entropy_shift_flag,
            'entropy_consecutive_days': consecutive,
            'entropy_above_threshold': above,
        },
    }


def _build_transaction_cost_config(bundle: Dict[str, Any]) -> Dict[str, Any]:
    tx = bundle.get('transaction_costs', {}) if isinstance(bundle, dict) else {}
    return {
        'spread_bps': tx.get('spread_bps', {}),
        'asset_class_default_bps': tx.get('asset_class_default_bps', {}),
        'slippage_range_bps': _safe_float(tx.get('slippage_range_bps'), 2.0),
    }


def _execution_price_map(next_prices_df: pd.DataFrame) -> Dict[str, float]:
    price_map: Dict[str, float] = {}
    for _, row in next_prices_df.iterrows():
        symbol = str(row.get('symbol', ''))
        if not symbol:
            continue
        if pd.notna(row.get('open')):
            price = _safe_float(row.get('open'), 0.0)
        else:
            price = _safe_float(row.get('close'), 0.0)
        if price > 0:
            price_map[symbol] = price
    return price_map


def _execute_actions(
    portfolio: Dict[str, Any],
    actions: List[Dict[str, Any]],
    next_prices_df: pd.DataFrame,
    regime_label: str,
    universe_df: pd.DataFrame,
    execution_ts: datetime,
    rng: random.Random,
    transaction_cost_config: Dict[str, Any],
    decision_date: str,
    valuation_date: str,
    decision_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    fills: List[Dict[str, Any]] = []
    price_map = _execution_price_map(next_prices_df)

    sorted_actions = sorted(
        actions,
        key=lambda action: (
            ACTION_PRIORITY.get(str(action.get('action', '')).upper(), 9),
            str(action.get('symbol', '')),
        ),
    )

    for action in sorted_actions:
        symbol = str(action.get('symbol', ''))
        action_type = str(action.get('action', '')).upper()
        if symbol not in price_map:
            continue

        execution_price = price_map[symbol]
        adjusted = dict(action)
        adjusted['price'] = execution_price

        if action_type == 'BUY':
            target_dollars = _safe_float(action.get('dollars'), 0.0)
            if target_dollars <= 0:
                target_dollars = _safe_float(action.get('shares'), 0.0) * _safe_float(action.get('price'), 0.0)
            shares = int(target_dollars / execution_price) if execution_price > 0 else 0
            min_order = _safe_float(decision_params.get('min_order_dollars'), 250.0)
            if shares <= 0 or shares * execution_price < min_order:
                continue
            max_affordable = int(_safe_float(portfolio.get('cash'), 0.0) / execution_price)
            shares = min(shares, max_affordable)
            if shares <= 0:
                continue
            adjusted['shares'] = shares
            adjusted['dollars'] = shares * execution_price
        else:
            shares = int(_safe_float(action.get('shares'), 0.0))
            if shares <= 0:
                continue
            adjusted['shares'] = shares

        trade = paper_trader.execute_trade(
            portfolio=portfolio,
            action=adjusted,
            regime_label=regime_label,
            universe_df=universe_df,
            timestamp=execution_ts,
            rng=rng,
            transaction_cost_config=transaction_cost_config,
        )
        trade['_decision_date'] = decision_date
        trade['_valuation_date'] = valuation_date
        trade['_trade_date'] = valuation_date
        fills.append(trade)

    return fills


def run_replay_for_dates(
    dataset: OptimizerDataset,
    decision_dates: List[str],
    candidate_bundle: Dict[str, Any],
    random_seed: int,
    initial_capital: float,
) -> ReplayResult:
    """Replay decisions across dates and return portfolio path + fills."""
    snapshot_map = dataset.by_date()
    ordered_dates = [date for date in decision_dates if date in snapshot_map]

    portfolio: Dict[str, Any] = {
        'cash': initial_capital,
        'holdings': [],
        'portfolio_value': initial_capital,
        'benchmark_value': initial_capital,
        'benchmark_start_price': None,
        'trades_today': [],
        'last_updated': f"{ordered_dates[0]}T00:00:00" if ordered_dates else datetime.now().isoformat(),
        'cumulative_transaction_costs': 0.0,
    }

    decision_params = candidate_bundle.get('decision_params', {})
    regime_compatibility = candidate_bundle.get('regime_compatibility', {})
    signal_overrides = candidate_bundle.get('signals', {})

    tx_cost_config = _build_transaction_cost_config(candidate_bundle)
    rng = random.Random(random_seed)
    entropy_state = {'prev_days': 0, 'prev_above': False}

    steps: List[ReplayStep] = []
    fills: List[Dict[str, Any]] = []

    for date in ordered_dates:
        snapshot = snapshot_map[date]
        start_value = _safe_float(portfolio.get('portfolio_value'), initial_capital)

        expert_signals = _build_expert_signals(
            snapshot.signal_row,
            signal_overrides,
            entropy_state,
        )

        decision_config = {
            'decision_params': decision_params,
            'regime_compatibility': regime_compatibility,
            'portfolio_state': portfolio,
            'universe': dataset.universe_df,
            'regime_fusion_overrides': candidate_bundle.get('regime_fusion', {}),
            'decision_engine_overrides': candidate_bundle.get('decision_engine', {}),
            'ensemble_overrides': candidate_bundle.get('ensemble', {}),
        }

        decisions = decision_engine.run(
            inference_output=snapshot.inference,
            llm_risks={},
            features_df=snapshot.features_df,
            config=decision_config,
            validation={'price_coverage': 1.0, 'degraded_mode': False},
            expert_signals=expert_signals,
        )

        execution_ts = datetime.fromisoformat(f"{snapshot.next_date}T09:45:00")
        day_fills = _execute_actions(
            portfolio=portfolio,
            actions=decisions.get('actions', []),
            next_prices_df=snapshot.next_prices_df,
            regime_label=str(decisions.get('regime', 'unknown')),
            universe_df=dataset.universe_df,
            execution_ts=execution_ts,
            rng=rng,
            transaction_cost_config=tx_cost_config,
            decision_date=snapshot.date,
            valuation_date=snapshot.next_date,
            decision_params=decision_params,
        )
        fills.extend(day_fills)

        valuation_ts = datetime.fromisoformat(f"{snapshot.next_date}T16:00:00")
        portfolio = paper_trader.update_portfolio_values(
            portfolio=portfolio,
            prices_df=snapshot.next_prices_df,
            current_time=valuation_ts,
        )

        traded_notional = sum(_safe_float(fill.get('dollars')) for fill in day_fills)
        steps.append(
            ReplayStep(
                decision_date=snapshot.date,
                valuation_date=snapshot.next_date,
                start_value=start_value,
                end_value=_safe_float(portfolio.get('portfolio_value'), start_value),
                regime=str(decisions.get('regime', 'unknown')),
                actions_count=len(decisions.get('actions', [])),
                traded_notional=traded_notional,
            )
        )

    return ReplayResult(
        steps=steps,
        fills=fills,
        final_portfolio=portfolio,
    )
