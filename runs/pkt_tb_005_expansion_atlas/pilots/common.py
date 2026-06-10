"""Shared pilot harness for PKT-TB-005 expansion-atlas pilots.

The replay loop is a faithful copy of optimizer/replay.py:run_replay_for_dates
with exactly three extensions, all recording/hook-shaped (decision math,
execution math, and cost model are the production modules, unchanged):

1. Per-step cash + gross exposure recording (the stock loop discards cash).
2. ``inference_transform(date, inference) -> inference`` hook so a pilot can
   swap/modify the stored model output for that date (e.g. baseline regime
   probs instead of the deep ensemble) with no lookahead — the transform only
   ever sees that date's artifacts.
3. ``intent_transform(date, decisions, intents, extra) -> intents`` hook so a
   pilot can gate/scale intents from a pilot signal (sizing-input pilots).

Everything is seeded/pinned per committee/EVIDENCE_PROTOCOL.md: fixed seed,
pinned local data cache (data_cache.pkl built from S3 daily artifacts),
manifest with code SHA + params hash + data range + command + wall-clock.

Holdout boundary: 2026-03-11 (config/optimizer.committee_20260606.json).
E1 = full cached range; E2 = holdout-only read.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import pickle
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
REPO = RUN_DIR.parents[2]
sys.path.insert(0, str(REPO))

from optimizer.replay import (  # noqa: E402
    ReplayResult,
    ReplayStep,
    _build_expert_signals,
    _build_transaction_cost_config,
    _execute_actions,
    _safe_float,
)
from src.steps import decision_engine, paper_trader  # noqa: E402

HOLDOUT_START = '2026-03-11'
SEED = 20260217
INITIAL_CAPITAL = 100000.0

_CACHE = None


def load_cache() -> Dict[str, Any]:
    global _CACHE
    if _CACHE is None:
        with open(RUN_DIR / 'data_cache.pkl', 'rb') as f:
            _CACHE = pickle.load(f)
    return _CACHE


def load_active_bundle() -> Dict[str, Any]:
    """The canon line's params: the active production bundle, read from the
    repo's candidate-of-record copy in config/ (same content the replays in
    runs/sizing_lead_20260606 used via optimizer.promote.load_active_bundle)."""
    from optimizer.promote import load_active_bundle as _load
    from optimizer.config import load_optimizer_config
    cfg = load_optimizer_config(str(REPO / 'config/optimizer.committee_20260606.json'))
    return _load(cfg.config_dir_path)


def split_dates(dates: List[str]) -> Dict[str, List[str]]:
    return {
        'full': list(dates),
        'pre_holdout': [d for d in dates if d < HOLDOUT_START],
        'holdout': [d for d in dates if d >= HOLDOUT_START],
    }


# ---------- replay loop (optimizer/replay.py copy + recording/hooks) ----------

def load_ranking_model(model_dir: str):
    """Load the RankingMLP exactly as src/handler.py does (production path)."""
    import torch
    from training.models.ranking_mlp import RankingMLP, RANKING_FEATURES
    model_path = REPO / model_dir / 'ranking_mlp.pt'
    norm_path = REPO / model_dir / 'ranking_normalization.json'
    model = RankingMLP(input_dim=len(RANKING_FEATURES))
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    norm = json.loads(norm_path.read_text())
    return model, norm, RANKING_FEATURES


def ranking_setup_from_bundle(bundle: Dict[str, Any]):
    """Returns (model, norm, blend) per the bundle's decision_engine overrides —
    the production-faithful configuration (handler passes these explicitly)."""
    de = bundle.get('decision_engine', {}) or {}
    blend = float(de.get('ranking_blend', 0) or 0)
    model_dir = de.get('ranking_model_dir', '')
    if blend > 0 and model_dir:
        model, norm, _ = load_ranking_model(model_dir)
        return model, norm, blend
    return None, None, 0.0


def run_pilot_replay(
    dataset,
    decision_dates: List[str],
    candidate_bundle: Dict[str, Any],
    random_seed: int = SEED,
    initial_capital: float = INITIAL_CAPITAL,
    inference_transform: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
    intent_transform: Optional[Callable[[str, Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]], List[Dict[str, Any]]]] = None,
    ranking_model: Any = None,
    ranking_normalization: Any = None,
    ranking_blend: float = 0.0,
    universe_override: Optional[pd.DataFrame] = None,
    portfolio_view: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    snapshot_map = dataset.by_date()
    ordered_dates = [d for d in decision_dates if d in snapshot_map]
    universe_df = universe_override if universe_override is not None else dataset.universe_df

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
    step_extras: List[Dict[str, Any]] = []

    for date in ordered_dates:
        snapshot = snapshot_map[date]
        start_value = _safe_float(portfolio.get('portfolio_value'), initial_capital)

        expert_signals = _build_expert_signals(
            snapshot.signal_row, signal_overrides, entropy_state)

        inference = snapshot.inference
        if inference_transform is not None:
            inference = inference_transform(date, copy.deepcopy(inference))

        # portfolio_view lets a pilot present a modified state to the engine
        # (e.g. cash-sleeve holdings shown as cash). MUST keep the same holding
        # dict objects for pass-through holdings so the engine's multi-day
        # mutations (health counters, peaks) persist on the real portfolio.
        engine_state = portfolio_view(portfolio) if portfolio_view else portfolio

        decision_config = {
            'decision_params': decision_params,
            'regime_compatibility': regime_compatibility,
            'portfolio_state': engine_state,
            'universe': universe_df,
            'regime_fusion_overrides': candidate_bundle.get('regime_fusion', {}),
            'decision_engine_overrides': candidate_bundle.get('decision_engine', {}),
            'ensemble_overrides': candidate_bundle.get('ensemble', {}),
        }

        # Ranking scores per date — verbatim from optimizer/replay.py:357-366
        date_ranking_scores = None
        if ranking_model is not None and ranking_normalization is not None and ranking_blend > 0:
            from training.models.ranking_mlp import RANKING_FEATURES
            date_ranking_scores = {}
            for _, row in snapshot.features_df.iterrows():
                sym = row.get('symbol')
                if sym:
                    feat_dict = {f: float(row.get(f, 0) or 0) for f in RANKING_FEATURES}
                    date_ranking_scores[sym] = ranking_model.predict_scores(
                        feat_dict, ranking_normalization)

        decisions = decision_engine.run(
            inference_output=inference,
            llm_risks={},
            features_df=snapshot.features_df,
            config=decision_config,
            validation={'price_coverage': 1.0, 'degraded_mode': False},
            expert_signals=expert_signals,
            ranking_scores=date_ranking_scores,
            ranking_blend=ranking_blend,
        )

        intents = list(decisions.get('actions', []))
        if intent_transform is not None:
            intents = intent_transform(date, decisions, intents,
                                       {'expert_signals': expert_signals,
                                        'snapshot': snapshot,
                                        'portfolio': portfolio})

        execution_ts = datetime.fromisoformat(f"{snapshot.next_date}T09:45:00")
        day_fills = _execute_actions(
            portfolio=portfolio,
            actions=intents,
            next_prices_df=snapshot.next_prices_df,
            regime_label=str(decisions.get('regime', 'unknown')),
            universe_df=universe_df,
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

        traded_notional = sum(_safe_float(f.get('dollars')) for f in day_fills)
        end_value = _safe_float(portfolio.get('portfolio_value'), start_value)
        steps.append(ReplayStep(
            decision_date=snapshot.date,
            valuation_date=snapshot.next_date,
            start_value=start_value,
            end_value=end_value,
            regime=str(decisions.get('regime', 'unknown')),
            actions_count=len(intents),
            traded_notional=traded_notional,
        ))
        cash = _safe_float(portfolio.get('cash'), 0.0)
        step_extras.append({
            'decision_date': snapshot.date,
            'valuation_date': snapshot.next_date,
            'cash': cash,
            'gross_exposure': (end_value - cash) / end_value if end_value > 0 else 0.0,
            'regime': str(decisions.get('regime', 'unknown')),
        })

    return {
        'result': ReplayResult(steps=steps, fills=fills, final_portfolio=portfolio),
        'step_extras': step_extras,
    }


# ---------- metrics (EVIDENCE_PROTOCOL required reporting) ----------

def daily_returns(result: ReplayResult) -> pd.Series:
    if not result.steps:
        return pd.Series(dtype=float)
    idx = [s.valuation_date for s in result.steps]
    rets = [(s.end_value / s.start_value - 1.0) if s.start_value > 0 else 0.0
            for s in result.steps]
    return pd.Series(rets, index=idx)


def round_trips(fills: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Average-cost realized round trips: each SELL/REDUCE fill realizes
    (px - avg_entry) * shares; win = realized pnl > 0."""
    pos: Dict[str, Dict[str, float]] = {}
    trips: List[float] = []
    for f in fills:
        sym = str(f.get('symbol', ''))
        action = str(f.get('action', '')).upper()
        shares = _safe_float(f.get('shares'))
        price = _safe_float(f.get('price'))
        if shares <= 0 or price <= 0:
            continue
        if action == 'BUY':
            p = pos.setdefault(sym, {'shares': 0.0, 'cost': 0.0})
            p['shares'] += shares
            p['cost'] += shares * price
        elif action in ('SELL', 'REDUCE'):
            p = pos.get(sym)
            if not p or p['shares'] <= 0:
                continue
            avg = p['cost'] / p['shares']
            sell_n = min(shares, p['shares'])
            trips.append((price - avg) * sell_n)
            p['shares'] -= sell_n
            p['cost'] -= avg * sell_n
    wins = sum(1 for t in trips if t > 0)
    return {'round_trips': len(trips),
            'win_rate': wins / len(trips) if trips else None,
            'realized_pnl_sum': float(sum(trips))}


def segment_metrics(run: Dict[str, Any]) -> Dict[str, Any]:
    result: ReplayResult = run['result']
    steps = result.steps
    if not steps:
        return {'note': 'no steps'}
    vals = [steps[0].start_value] + [s.end_value for s in steps]
    arr = np.array(vals, dtype=float)
    rets = daily_returns(result)
    n_days = len(rets)
    total_return = float(arr[-1] / arr[0] - 1.0)
    years = n_days / 252.0
    cagr = float((arr[-1] / arr[0]) ** (1 / years) - 1.0) if years > 0 and arr[0] > 0 else None
    sharpe = (float(rets.mean() / rets.std() * math.sqrt(252))
              if n_days >= 10 and rets.std() > 0 else None)
    max_dd = float((arr / np.maximum.accumulate(arr) - 1.0).min())
    rt = round_trips(result.fills)
    cum_costs = _safe_float(result.final_portfolio.get('cumulative_transaction_costs'))
    gross = [e['gross_exposure'] for e in run['step_extras']]
    return {
        'n_days': n_days,
        'start_value': float(arr[0]),
        'end_value': float(arr[-1]),
        'total_return': round(total_return, 6),
        'cagr': None if cagr is None else round(cagr, 6),
        'sharpe': None if sharpe is None else round(sharpe, 4),
        'max_drawdown': round(max_dd, 6),
        'win_rate': None if rt['win_rate'] is None else round(rt['win_rate'], 4),
        'round_trips': rt['round_trips'],
        'cumulative_transaction_costs': round(cum_costs, 2),
        'avg_gross_exposure': round(float(np.mean(gross)), 4) if gross else None,
        'n_fills': len(result.fills),
    }


def paired_delta(run_on: Dict[str, Any], run_off: Dict[str, Any]) -> Dict[str, Any]:
    """Distribution honesty: paired daily-return differences on identical dates."""
    r_on, r_off = daily_returns(run_on['result']), daily_returns(run_off['result'])
    common = r_on.index.intersection(r_off.index)
    d = (r_on.loc[common] - r_off.loc[common]).astype(float)
    n = len(d)
    if n < 2:
        return {'n_common_days': n, 'note': 'insufficient overlap'}
    t = float(d.mean() / (d.std() / math.sqrt(n))) if d.std() > 0 else 0.0
    return {
        'n_common_days': n,
        'mean_daily_delta': round(float(d.mean()), 8),
        'sd_daily_delta': round(float(d.std()), 8),
        't_stat_paired': round(t, 3),
        'positive_days': int((d > 0).sum()),
        'negative_days': int((d < 0).sum()),
    }


def spy_segment_return(dataset, dates: List[str]) -> Optional[float]:
    closes = []
    snap_map = dataset.by_date()
    for d in dates:
        s = snap_map.get(d)
        if s is None:
            continue
        spy = s.next_prices_df[s.next_prices_df['symbol'] == 'SPY']
        if len(spy):
            closes.append(float(spy.iloc[-1]['close']))
    return closes[-1] / closes[0] - 1.0 if len(closes) >= 2 else None


# ---------- manifest ----------

def write_manifest(name: str, *, params: Dict[str, Any], command: str,
                   wall_clock_s: float, variants_logged: List[str],
                   holdout_looks: int, notes: str = '') -> Path:
    cache = load_cache()
    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                         capture_output=True, text=True).stdout.strip()
    manifest = {
        'pilot': name,
        'packet': 'PKT-TB-005-EXPANSION-ATLAS-V1-20260609',
        'code_sha': sha,
        'params_hash': hashlib.sha256(
            json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:16],
        'seed': SEED,
        'data_snapshot': {
            'source': cache['source'],
            'range': list(cache['snapshot_range']),
            'cache_built_at': cache['built_at'],
            'n_snapshots': len(cache['dataset'].snapshots),
        },
        'holdout_start': HOLDOUT_START,
        'holdout_looks_this_pilot': holdout_looks,
        'fill_cost_model': 'paper_trader.execute_trade (production, unmodified)',
        'command': command,
        'wall_clock_seconds': round(wall_clock_s, 1),
        'variants_logged': variants_logged,
        'notes': notes,
        'written_at': datetime.now().isoformat(timespec='seconds'),
    }
    out = RUN_DIR.parent / 'manifests' / f'{name}_manifest.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    return out


def timer() -> Callable[[], float]:
    t0 = time.time()
    return lambda: time.time() - t0
