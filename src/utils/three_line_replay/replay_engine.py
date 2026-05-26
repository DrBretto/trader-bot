"""Source-faithful replay engine for the three-line dashboard extension.

Adapted from the v2 replay run that produced the canonical hybrid line
(`book-factory/use_lane_outputs/runs/20260506_trader-bot-mar11-known-bugs-fixed-algorithm-comparison-v2/`).

Constraints:
- No look-ahead. For decision date `D`, only `daily/D/*` artifacts are
  read (which contain data through D-1 close).
- Fill price = next-session OPEN from `daily/D+1/prices.parquet` (the
  file that contains D's OHLC). Mark to D's CLOSE.
- VUG 6:1 split applied on first iteration where `current_date >= 2026-04-20`.
- Per-position `entry_psm` captured at open so strategy hooks can
  trigger top-ups based on PSM rises.

Designed to run inside the Lambda nightly. Reuses the production
`src.steps.decision_engine` and `src.signals.regime_fusion` /
`src.signals.fragility` modules — no patches.
"""
from __future__ import annotations

import io
import json
import logging
import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import boto3
import numpy as np
import pandas as pd

from src.steps import decision_engine
from .strategies import Strategy, StrategyContext

logger = logging.getLogger(__name__)

VUG_SPLIT_EX_DATE = '2026-04-20'
VUG_SPLIT_RATIO = 6
START_PORTFOLIO_DATE = '2026-03-11'
PRICE_GAP_THRESHOLD = 0.05


# ---------- S3 cache ----------

class S3Cache:
    """Lightweight in-memory cache for S3 reads during a single replay run.
    The 3 replays share the same daily artifacts so caching cuts S3 calls 3x."""

    def __init__(self, s3_client, bucket: str = 'investment-system-data'):
        self.s3 = s3_client
        self.bucket = bucket
        self._mem: Dict[str, bytes] = {}

    def get(self, key: str) -> bytes:
        if key not in self._mem:
            self._mem[key] = self.s3.get_object(Bucket=self.bucket, Key=key)['Body'].read()
        return self._mem[key]

    def get_json(self, key: str) -> Any:
        return json.loads(self.get(key).decode())

    def get_parquet(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(io.BytesIO(self.get(key)))

    def get_csv(self, key: str) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(self.get(key)))

    def list_daily_dates(self) -> List[str]:
        paginator = self.s3.get_paginator('list_objects_v2')
        dates = set()
        for p in paginator.paginate(Bucket=self.bucket, Prefix='daily/', Delimiter='/'):
            for cp in p.get('CommonPrefixes', []) or []:
                d = cp['Prefix'].split('/')[-2]
                if d.startswith('2026-'):
                    dates.add(d)
        return sorted(dates)


# ---------- Variant configs ----------

@dataclass
class VariantConfig:
    name: str
    decision_params: Dict[str, Any]
    regime_compatibility: Dict[str, Any]
    signal_overrides: Dict[str, Any]
    regime_fusion_overrides: Dict[str, Any]
    decision_engine_overrides: Dict[str, Any]
    ensemble_overrides: Dict[str, Any]
    transaction_cost_overrides: Dict[str, Any]


def load_variant_configs(cache: S3Cache) -> Tuple[VariantConfig, Optional[VariantConfig]]:
    """Load hybrid (active) + pre-hybrid (opt-bootstrap) configs from S3.

    pre_hybrid is optional: if the bootstrap config is missing in S3, the
    extender still produces the canonical hybrid + champion lines and skips
    the comparison line. Previously a missing bootstrap raised NoSuchKey,
    propagated into the extender's outer try/except, and dropped ALL three
    lines — the chart fell back to dashboard_metrics raw broker values
    and rendered as a zig-zag.
    """
    active = cache.get_json('config/decision_params.active.json')
    hybrid = VariantConfig(
        name='hybrid',
        decision_params=active['decision_params'],
        regime_compatibility=active['regime_compatibility'],
        signal_overrides=active.get('signals', {}),
        regime_fusion_overrides=active.get('regime_fusion', {}),
        decision_engine_overrides=active.get('decision_engine', {}),
        ensemble_overrides=active.get('ensemble', {}),
        transaction_cost_overrides=active.get('transaction_costs', {}),
    )

    pre_hybrid: Optional[VariantConfig]
    try:
        bootstrap = cache.get_json('config/decision_params.history/20260328T111138Z-opt-bootstrap.json')
        pre_hybrid = VariantConfig(
            name='pre_hybrid',
            decision_params=bootstrap['decision_params'],
            regime_compatibility=bootstrap['regime_compatibility'],
            signal_overrides=bootstrap.get('signals', {}),
            regime_fusion_overrides=bootstrap.get('regime_fusion', {}),
            decision_engine_overrides=bootstrap.get('decision_engine', {}),
            ensemble_overrides=bootstrap.get('ensemble', {}),
            transaction_cost_overrides=bootstrap.get('transaction_costs', {}),
        )
    except Exception as exc:
        logger.warning(
            "pre_hybrid bootstrap config missing or unreadable; "
            "comparison line will be omitted: %s", exc
        )
        pre_hybrid = None
    return hybrid, pre_hybrid


# ---------- Ranking model ----------

_ranking_model = None
_ranking_norm = None
_ranking_features = None


def _get_ranking_model(cache: S3Cache, model_dir: str):
    """Load the ranking model from S3 (or local cache). Returns (model, norm, feature_list)."""
    global _ranking_model, _ranking_norm, _ranking_features
    if _ranking_model is not None:
        return _ranking_model, _ranking_norm, _ranking_features
    try:
        import torch
        from training.models.ranking_mlp import RankingMLP, RANKING_FEATURES
    except ImportError as e:
        logger.warning("ranking model unavailable (%s); replays will use health-only", e)
        return None, None, None
    model_blob = cache.get(f'{model_dir}/ranking_mlp.pt')
    norm_blob = cache.get(f'{model_dir}/ranking_normalization.json')
    model = RankingMLP(input_dim=len(RANKING_FEATURES))
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        f.write(model_blob)
        tmp_path = f.name
    try:
        model.load_state_dict(torch.load(tmp_path, weights_only=True))
        model.eval()
    finally:
        os.unlink(tmp_path)
    norm = json.loads(norm_blob.decode())
    _ranking_model, _ranking_norm, _ranking_features = model, norm, RANKING_FEATURES
    return model, norm, RANKING_FEATURES


def _compute_ranking_scores(features_df: pd.DataFrame, cache: S3Cache, model_dir: str) -> Dict[str, float]:
    model, norm, feat_list = _get_ranking_model(cache, model_dir)
    if model is None or len(features_df) == 0:
        return {}
    latest_date = features_df['date'].max()
    latest = features_df[features_df['date'] == latest_date]
    scores: Dict[str, float] = {}
    for _, row in latest.iterrows():
        sym = row.get('symbol')
        if not sym:
            continue
        feat_dict = {f: float(row.get(f, 0) or 0) for f in feat_list}
        scores[sym] = model.predict_scores(feat_dict, norm)
    return scores


# ---------- Fragility recompute ----------

def _recompute_fragility(avg_corr: float, pc1: float,
                          avg_corr_mean: float, avg_corr_std: float,
                          pc1_mean: float, pc1_std: float) -> float:
    corr_z = (avg_corr - avg_corr_mean) / avg_corr_std if avg_corr_std > 0 else 0.0
    pc1_z = (pc1 - pc1_mean) / pc1_std if pc1_std > 0 else 0.0
    norm_corr = (math.tanh(corr_z) + 1) / 2
    norm_pc1 = (math.tanh(pc1_z) + 1) / 2
    return float(np.clip(0.5 * norm_corr + 0.5 * norm_pc1, 0.0, 1.0))


def _build_expert_signals(signals_row: pd.Series, variant: VariantConfig) -> Dict[str, Any]:
    avg_corr = float(signals_row['avg_correlation'])
    pc1 = float(signals_row['pc1_explained'])
    f = variant.signal_overrides.get('fragility', {})
    fragility_score = _recompute_fragility(
        avg_corr, pc1,
        float(f.get('AVG_CORR_MEAN', 0.30)),
        float(f.get('AVG_CORR_STD', 0.15)),
        float(f.get('PC1_MEAN', 0.45)),
        float(f.get('PC1_STD', 0.12)),
    )
    return {
        'macro_credit': {
            'macro_credit_score': float(signals_row['macro_credit_score']),
            'yield_slope_10y_3m': float(signals_row['yield_slope_10y_3m']),
            'hy_spread_proxy': float(signals_row['hy_spread_proxy']),
        },
        'vol_uncertainty': {
            'vol_uncertainty_score': float(signals_row['vol_uncertainty_score']),
            'vol_regime_label': str(signals_row['vol_regime_label']),
            'vix_percentile': float(signals_row['vix_percentile']),
            'vvix_percentile': float(signals_row['vvix_percentile']),
        },
        'fragility': {
            'fragility_score': fragility_score,
            'avg_correlation': avg_corr,
            'pc1_explained': pc1,
        },
        'entropy_shift': {
            'entropy_score': float(signals_row['entropy_score']),
            'entropy_z_score': float(signals_row['entropy_z_score']),
            'entropy_shift_flag': bool(signals_row['entropy_shift_flag']),
        },
    }


# ---------- Simulator portfolio ----------

@dataclass
class Position:
    symbol: str
    shares: float
    entry_price: float
    entry_date: str
    peak_price: float
    asset_class: str
    sector: str
    leverage_flag: int
    consecutive_below_health_days: int = 0
    entry_psm: Optional[float] = None


@dataclass
class Portfolio:
    cash: float
    positions: List[Position] = field(default_factory=list)
    benchmark_shares: float = 0.0
    benchmark_start_price: float = 0.0
    realized_pnl: float = 0.0
    vug_split_applied: bool = False

    def value_at_marks(self, marks: Dict[str, float]) -> Tuple[float, List[str]]:
        v = self.cash
        missing = []
        for p in self.positions:
            if p.symbol in marks:
                v += p.shares * marks[p.symbol]
            else:
                missing.append(p.symbol)
        return v, missing

    def to_state_dict_with_marks(self, marks: Dict[str, float]) -> Tuple[Dict[str, Any], List[str]]:
        holdings = [{'symbol': p.symbol, 'shares': p.shares, 'entry_price': p.entry_price,
                     'entry_date': p.entry_date, 'peak_price': p.peak_price,
                     'asset_class': p.asset_class, 'sector': p.sector,
                     'leverage_flag': p.leverage_flag,
                     'consecutive_below_health_days': p.consecutive_below_health_days}
                    for p in self.positions]
        value, missing = self.value_at_marks(marks)
        return {'cash': self.cash, 'holdings': holdings, 'portfolio_value': value}, missing

    def position_map(self) -> Dict[str, Position]:
        return {p.symbol: p for p in self.positions}


def seed_portfolio(cache: S3Cache) -> Portfolio:
    state = cache.get_json(f'daily/{START_PORTFOLIO_DATE}/portfolio_state.json')
    positions = []
    for h in state.get('holdings', []):
        positions.append(Position(
            symbol=h['symbol'], shares=float(h['shares']),
            entry_price=float(h['entry_price']),
            entry_date=h.get('entry_date', '2026-03-11'),
            peak_price=float(h.get('peak_price', h.get('current_price', h['entry_price']))),
            asset_class=h.get('asset_class', 'equity'),
            sector=h.get('sector', 'broad'),
            leverage_flag=int(h.get('leverage_flag', 0) or 0),
            consecutive_below_health_days=int(h.get('consecutive_below_health_days', 0) or 0),
        ))
    return Portfolio(
        cash=float(state['cash']),
        positions=positions,
        benchmark_shares=float(state.get('benchmark_shares', 0.0) or 0.0),
        benchmark_start_price=float(state.get('benchmark_start_price', 0.0) or 0.0),
    )


def _apply_vug_split(p: Portfolio, current_date: str) -> Optional[Dict[str, Any]]:
    if p.vug_split_applied or current_date < VUG_SPLIT_EX_DATE:
        return None
    p.vug_split_applied = True
    for pos in p.positions:
        if pos.symbol == 'VUG':
            pos.shares *= VUG_SPLIT_RATIO
            pos.entry_price /= VUG_SPLIT_RATIO
            pos.peak_price /= VUG_SPLIT_RATIO
            return {'date': current_date, 'symbol': 'VUG', 'ratio': VUG_SPLIT_RATIO,
                    'new_shares': pos.shares, 'new_entry_price': pos.entry_price}
    return None


def _ohlc_for_date(prices_df: pd.DataFrame, target_date: str) -> Dict[str, Dict[str, float]]:
    df = prices_df.copy()
    df['date_str'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    sub = df[df['date_str'] == target_date]
    return {row['symbol']: {'open': float(row['open']), 'close': float(row['close'])}
            for _, row in sub.iterrows()}


def _latest_close_per_symbol(features_df: pd.DataFrame) -> Dict[str, float]:
    if len(features_df) == 0:
        return {}
    df = features_df[['symbol', 'date', 'close']].copy()
    df = df.sort_values(['symbol', 'date'])
    latest = df.groupby('symbol').tail(1)
    return {row['symbol']: float(row['close']) for _, row in latest.iterrows()
            if pd.notna(row['close'])}


def _execute_intents(portfolio: Portfolio, intents: List[Dict[str, Any]],
                     ohlc: Dict[str, Dict[str, float]], decision_params: Dict[str, Any],
                     date: str) -> List[Dict[str, Any]]:
    executed = []
    pos_map = portfolio.position_map()
    min_order = float(decision_params.get('min_order_dollars', 250))
    for intent in intents:
        sym = intent['symbol']; action = intent['action']
        intent_price = float(intent.get('price', 0) or 0)
        quote = ohlc.get(sym)
        if quote is None:
            continue
        morning_price = quote['open']
        if action == 'BUY':
            if intent_price <= 0:
                continue
            gap = abs(morning_price - intent_price) / intent_price
            if gap > PRICE_GAP_THRESHOLD:
                continue
            target_dollars = float(intent.get('dollars', 0) or (intent.get('shares', 0) * intent_price))
            shares = int(target_dollars / morning_price) if morning_price > 0 else 0
            if shares <= 0 or shares * morning_price < min_order:
                continue
            if shares * morning_price > portfolio.cash:
                shares = int(portfolio.cash / morning_price) if morning_price > 0 else 0
                if shares <= 0:
                    continue
            cost = shares * morning_price
            portfolio.cash -= cost
            new_pos = Position(
                symbol=sym, shares=float(shares),
                entry_price=morning_price, entry_date=date,
                peak_price=morning_price,
                asset_class=intent.get('asset_class', 'equity'),
                sector=intent.get('sector', 'broad'),
                leverage_flag=int(intent.get('leverage_flag', 0) or 0),
            )
            portfolio.positions.append(new_pos)
            pos_map[sym] = new_pos
            executed.append({'date': date, 'symbol': sym, 'action': 'BUY',
                             'shares': float(shares), 'price': morning_price,
                             'dollars': round(cost, 2), 'reason': intent.get('reason', '')})
        elif action in ('SELL', 'REDUCE'):
            holding = pos_map.get(sym)
            if holding is None:
                continue
            shares_to_sell = float(intent.get('shares', holding.shares))
            if action == 'REDUCE':
                shares_to_sell *= 0.5
            shares_to_sell = min(shares_to_sell, holding.shares)
            if shares_to_sell <= 0:
                continue
            reason = intent.get('reason', '')
            if reason == 'STOP_HIT':
                stop_pct = float(decision_params.get(
                    'trailing_stop_leveraged' if holding.leverage_flag == 1 else 'trailing_stop_base',
                    0.10))
                if morning_price > holding.peak_price * (1 - stop_pct):
                    continue
            proceeds = shares_to_sell * morning_price
            portfolio.cash += proceeds
            holding.shares -= shares_to_sell
            executed.append({'date': date, 'symbol': sym, 'action': action,
                             'shares': shares_to_sell, 'price': morning_price,
                             'dollars': round(proceeds, 2), 'reason': reason})
            if holding.shares <= 1e-6:
                portfolio.positions = [p for p in portfolio.positions if p is not holding]
    return executed


def _mark_to_close(portfolio: Portfolio, ohlc: Dict[str, Dict[str, float]]) -> None:
    for p in portfolio.positions:
        q = ohlc.get(p.symbol)
        if q is None:
            continue
        if q['close'] > p.peak_price:
            p.peak_price = q['close']
    spy = ohlc.get('SPY')
    if spy and portfolio.benchmark_shares > 0:
        spy_close = spy['close']
        daily_div_per_share = spy_close * (0.013 / 252)
        div_cash = portfolio.benchmark_shares * daily_div_per_share
        portfolio.benchmark_shares += div_cash / spy_close


# ---------- Top-level run_variant ----------

def run_variant(cache: S3Cache, variant: VariantConfig, strategy: Optional[Strategy],
                trading_dates: List[str], universe_df: pd.DataFrame) -> Dict[str, Any]:
    """Run one variant from Mar 11 EOD forward through trading_dates[-2]
    (last viable inputs_date — needs trading_dates[-1] for next-day prices).

    Returns: {'date_value_map': {date: ending_value}, 'final_holdings': [...],
              'final_cash': ..., 'timeline': [...], 'actions': [...]}
    """
    portfolio = seed_portfolio(cache)
    timeline: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = []
    panic_streak = 0
    last_regime: Optional[str] = None

    for i in range(len(trading_dates) - 2):
        inputs_date = trading_dates[i + 1]
        prices_file_date = trading_dates[i + 2]

        # Load day inputs
        try:
            inference = cache.get_json(f'daily/{inputs_date}/inference.json')
            features = cache.get_parquet(f'daily/{inputs_date}/features.parquet')
            signals = cache.get_parquet(f'daily/{inputs_date}/signals.parquet')
        except Exception as e:
            logger.warning("missing daily artifacts for %s: %s", inputs_date, e)
            continue
        try:
            llm_raw = cache.get_json(f'daily/{inputs_date}/llm_risk.json')
            llm_risks = llm_raw.get('risks', {}) if isinstance(llm_raw, dict) else {}
        except Exception:
            llm_risks = {}

        # Load prices for execution
        try:
            prices_df = cache.get_parquet(f'daily/{prices_file_date}/prices.parquet')
        except Exception as e:
            logger.warning("missing prices file %s: %s", prices_file_date, e)
            continue
        ohlc = _ohlc_for_date(prices_df, inputs_date)
        if not ohlc or 'SPY' not in ohlc:
            continue

        if last_regime == 'high_vol_panic':
            panic_streak += 1
        else:
            panic_streak = 0

        _apply_vug_split(portfolio, inputs_date)

        # Build per-iteration variant config (deep-copy so strategy can mutate)
        variant_config = {
            'decision_params': deepcopy(variant.decision_params),
            'regime_compatibility': variant.regime_compatibility,
            'signal_overrides': deepcopy(variant.signal_overrides),
            'regime_fusion_overrides': deepcopy(variant.regime_fusion_overrides),
            'decision_engine_overrides': deepcopy(variant.decision_engine_overrides),
            'ensemble_overrides': variant.ensemble_overrides,
            'transaction_cost_overrides': variant.transaction_cost_overrides,
        }

        ctx = StrategyContext(
            inputs_date=inputs_date, portfolio=portfolio,
            variant_config=variant_config, expert_signals={}, expert_metrics={},
            decisions={}, panic_streak=panic_streak, last_regime=last_regime,
            features_df=features, inference=inference, llm_risks=llm_risks,
        )
        if strategy and strategy.pre_decision:
            strategy.pre_decision(ctx)

        expert_signals = _build_expert_signals(
            signals.iloc[0],
            type('V', (), {'signal_overrides': ctx.variant_config['signal_overrides']})()
        )
        ctx.expert_signals = expert_signals

        current_marks = _latest_close_per_symbol(features)
        state_dict, _missing = portfolio.to_state_dict_with_marks(current_marks)

        # Ranking model
        ranking_blend = float(ctx.variant_config['decision_engine_overrides'].get('ranking_blend', 0) or 0)
        ranking_model_dir = ctx.variant_config['decision_engine_overrides'].get('ranking_model_dir', '')
        ranking_scores = None
        if ranking_blend > 0 and ranking_model_dir:
            try:
                ranking_scores = _compute_ranking_scores(features, cache, ranking_model_dir)
            except Exception as e:
                logger.warning("ranking model failed: %s", e)
                ranking_blend = 0.0

        config = {**ctx.variant_config, 'universe': universe_df, 'portfolio_state': state_dict}
        decisions = decision_engine.run(
            inference, llm_risks, features, config,
            {'price_coverage': 1.0, 'degraded_mode': False, 'critical_failure': False},
            expert_signals=expert_signals,
            ranking_scores=ranking_scores,
            ranking_blend=ranking_blend,
        )
        decisions['_portfolio_value_passed'] = state_dict['portfolio_value']
        ctx.decisions = decisions
        ctx.expert_metrics = decisions.get('expert_metrics', {})
        last_regime = decisions.get('regime')

        intents = list(decisions.get('actions', []))
        if strategy and strategy.post_decision:
            intents = strategy.post_decision(ctx, intents)

        executed = _execute_intents(portfolio, intents, ohlc,
                                    ctx.variant_config['decision_params'], inputs_date)
        current_psm = ctx.expert_metrics.get('position_size_modifier')
        for tr in executed:
            if tr.get('action') == 'BUY':
                for pos in portfolio.positions:
                    if pos.symbol == tr['symbol'] and pos.entry_psm is None:
                        pos.entry_psm = current_psm
                        break
        # Backfill entry_psm on seed positions
        for pos in portfolio.positions:
            if pos.entry_psm is None:
                pos.entry_psm = current_psm

        for tr in executed:
            tr['variant'] = variant.name
            actions.append(tr)
        _mark_to_close(portfolio, ohlc)

        ending_value = portfolio.cash + sum(
            p.shares * ohlc.get(p.symbol, {}).get('close', p.peak_price)
            for p in portfolio.positions if p.symbol in ohlc
        )
        timeline.append({
            'date': inputs_date,
            'ending_value': round(ending_value, 2),
            'ending_cash': round(portfolio.cash, 2),
            'holdings_count': len(portfolio.positions),
            'regime_used': decisions.get('regime'),
            'holdings_at_close': [
                {'symbol': p.symbol, 'shares': p.shares, 'entry_price': p.entry_price,
                 'peak_price': p.peak_price,
                 'close_price': ohlc.get(p.symbol, {}).get('close')}
                for p in portfolio.positions
            ],
        })

    date_value_map = {row['date']: row['ending_value'] for row in timeline if 'ending_value' in row}
    last = timeline[-1] if timeline else {}
    return {
        'date_value_map': date_value_map,
        'timeline': timeline,
        'actions': actions,
        'final_holdings': last.get('holdings_at_close', []),
        'final_cash': last.get('ending_cash', portfolio.cash),
        'final_value': last.get('ending_value'),
        'final_date': last.get('date'),
    }
