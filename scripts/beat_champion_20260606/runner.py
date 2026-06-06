"""
Real-system runner for PKT-TRADER-BOT-OPTIMIZE-BEAT-CHAMPION-20260606.

Performance object = the REAL displayed system: three_line_replay.run_variant
with the champion overlays (extend_relax_choppy + topup_psm). NEVER the stripped
optimizer.replay base config. NEVER the Alpaca raw_value line.

Provides:
  - seed_at(cache, date): seed Portfolio from any daily/{date}/portfolio_state.json
  - champion_strategy(relax_regimes, relax_conf, topup_trigger, topup_frac): the
    overlays, with the knobs exposed for tuning.
  - run_real(cache, universe, trading_dates, seed_date, ensemble_overrides,
             dp_overrides, rf_overrides, strategy): one real run; returns metrics.
  - spy_return(cache, trading_dates): SPY buy&hold over the window (benchmark).

All numbers come from the real harness.
"""
from __future__ import annotations
import io, json, math, os, sys, contextlib
from typing import Any, Dict, List, Optional, Tuple

import boto3
import pandas as pd

from src.utils.three_line_replay import replay_engine as RE
from src.utils.three_line_replay.replay_engine import (
    S3Cache, load_variant_configs, run_variant, Portfolio, Position, _ohlc_for_date,
)
from src.utils.three_line_replay.strategies import (
    extend_fragility_relax, topup_on_psm_rise, compose,
)


def make_cache():
    s3 = boto3.client("s3", region_name="us-east-1")
    return S3Cache(s3)


class ThrottlePatchCache(S3Cache):
    """Wraps S3Cache; rewrites ONLY regime.position_size_multiplier in each
    daily inference.json via mult_policy(baked_mult, regime_block) -> new_mult.

    Keeps the live-baked regime label/probs/disagreement intact (those are the
    good signal); changes only the buy-size throttle. This isolates the
    "un-nerf the arbitrarily-cut buys" lever from the regime classification.
    """
    def __init__(self, src: S3Cache, mult_policy):
        super().__init__(src.s3, src.bucket)
        self._mem = src._mem  # share the underlying byte cache
        self._policy = mult_policy

    def get_json(self, key):
        obj = super().get_json(key)
        if (self._policy is not None and key.startswith("daily/")
                and key.endswith("/inference.json")
                and isinstance(obj, dict) and "regime" in obj):
            obj = json.loads(json.dumps(obj))  # deep copy so cache stays clean
            r = obj["regime"]
            baked = r.get("position_size_multiplier", 1.0)
            r["position_size_multiplier"] = float(self._policy(baked, r))
        return obj


def seed_at(cache: S3Cache, date: str) -> Portfolio:
    state = cache.get_json(f"daily/{date}/portfolio_state.json")
    positions = []
    for h in state.get("holdings", []):
        positions.append(Position(
            symbol=h["symbol"], shares=float(h["shares"]),
            entry_price=float(h["entry_price"]),
            entry_date=h.get("entry_date", date),
            peak_price=float(h.get("peak_price", h.get("current_price", h["entry_price"]))),
            asset_class=h.get("asset_class", "equity"),
            sector=h.get("sector", "broad"),
            leverage_flag=int(h.get("leverage_flag", 0) or 0),
            consecutive_below_health_days=int(h.get("consecutive_below_health_days", 0) or 0),
            peak_health=h.get("peak_health"),
            consecutive_health_drop_days=int(h.get("consecutive_health_drop_days", 0) or 0),
            entry_regime=h.get("entry_regime"),
        ))
    return Portfolio(
        cash=float(state["cash"]), positions=positions,
        benchmark_shares=float(state.get("benchmark_shares", 0.0) or 0.0),
        benchmark_start_price=float(state.get("benchmark_start_price", 0.0) or 0.0),
    )


def champion_strategy(relax_regimes=("choppy",), relax_conf=0.50,
                      topup_trigger=1.2, topup_frac=1.0):
    """The live champion overlays, knobs exposed. Default == current champion."""
    return compose(
        [extend_fragility_relax(tuple(relax_regimes), relax_conf),
         topup_on_psm_rise(topup_trigger, topup_frac)],
        f"relax_{'_'.join(relax_regimes)}_{relax_conf:.2f}+topup_{topup_trigger:.1f}_{topup_frac:.1f}",
    )


def _start_value(cache: S3Cache, seed: Portfolio, first_input_date: str) -> float:
    prices = cache.get_parquet(f"daily/{first_input_date}/prices.parquet")
    ohlc = _ohlc_for_date(prices, first_input_date)
    return seed.cash + sum(
        p.shares * ohlc.get(p.symbol, {}).get("close", p.entry_price)
        for p in seed.positions
    )


def _maxdd(series: List[float]) -> float:
    peak = -1e18; mdd = 0.0
    for v in series:
        peak = max(peak, v)
        if peak > 0:
            mdd = min(mdd, v / peak - 1.0)
    return mdd


def spy_return(cache: S3Cache, trading_dates: List[str]) -> float:
    def spy_close(d):
        px = cache.get_parquet(f"daily/{d}/prices.parquet")
        o = _ohlc_for_date(px, d)
        return o.get("SPY", {}).get("close")
    # first valid SPY close at/after window start, last valid at/before end
    c0 = c1 = None
    for d in trading_dates[1:]:
        c0 = spy_close(d)
        if c0:
            break
    for d in reversed(trading_dates):
        c1 = spy_close(d)
        if c1:
            break
    return (c1 / c0 - 1.0) if (c0 and c1) else float("nan")


def run_real(cache: S3Cache, universe, trading_dates: List[str], seed_date: str,
             ensemble_overrides: Optional[Dict[str, Any]] = None,
             dp_overrides: Optional[Dict[str, Any]] = None,
             rf_overrides: Optional[Dict[str, Any]] = None,
             strategy=None) -> Dict[str, Any]:
    """One real run of the champion overlay system. Returns metrics dict.

    ensemble_overrides: re-weight GRU/transformer + retune disagreement throttle
        (recomputed from stored per-model probs by decision_engine).
    dp_overrides / rf_overrides: patch decision_params / regime_fusion on the
        active bundle (e.g. max_positions, buy_score_threshold_by_regime, cash floors).
    """
    hybrid, _ = load_variant_configs(cache)
    # deep-ish copy so we can patch without mutating the cached bundle
    dp = dict(hybrid.decision_params); dp.update(dp_overrides or {})
    rf = dict(hybrid.regime_fusion_overrides); rf.update(rf_overrides or {})
    variant = RE.VariantConfig(
        name="real_champion",
        decision_params=dp,
        regime_compatibility=hybrid.regime_compatibility,
        signal_overrides=hybrid.signal_overrides,
        regime_fusion_overrides=rf,
        decision_engine_overrides=hybrid.decision_engine_overrides,
        ensemble_overrides=(ensemble_overrides or {}),
        transaction_cost_overrides=hybrid.transaction_cost_overrides,
    )
    # monkeypatch seed date for this run
    orig = RE.START_PORTFOLIO_DATE
    RE.START_PORTFOLIO_DATE = seed_date
    try:
        seed = seed_at(cache, seed_date)
        start_val = _start_value(cache, seed, trading_dates[1])
        # run_variant calls seed_portfolio(cache) internally -> uses patched date.
        # decision_engine print()s heavily; silence it so summaries are readable.
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            res = run_variant(cache, variant, strategy, trading_dates, universe)
    finally:
        RE.START_PORTFOLIO_DATE = orig
    tl = res["timeline"]
    eq = [start_val] + [r["ending_value"] for r in tl]
    final = eq[-1]
    ret = final / start_val - 1.0
    return {
        "start_value": round(start_val, 2),
        "final_value": round(final, 2),
        "return": round(ret, 4),
        "maxdd": round(_maxdd(eq), 4),
        "n_days": len(tl),
        "equity": eq,
        "dates": [r["date"] for r in tl],
        "regimes": [r.get("regime_used") for r in tl],
    }
