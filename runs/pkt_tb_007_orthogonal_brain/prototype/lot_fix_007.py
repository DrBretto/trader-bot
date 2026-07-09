"""PKT-TB-007 — the C5 lot fixes (BUILD_SPEC_007 §3; PROPOSAL_EXPRESSION §3).

Two sites, one bug class — per-symbol structures built by OVERWRITE instead of
aggregation over multi-lot positions:

  1. Adapter side: aggregate_book() builds held_shares / w_prev / NAV by
     ACCUMULATION (the TB-006 SCHD 2x738-lot defect, strategy_adapter.py:320,
     :335-339). Used by tilt_adapter.py.
  2. Harness side: replay_engine._execute_intents maps position_map() (last
     lot wins) and clamps SELLs to one lot. _execute_intents_lotfix below is a
     behavior-identical copy for single-lot books with exactly two changes:
       - BUY into a held symbol INCREMENTS that Position (shares-weighted
         entry_price, peak_price = max) instead of appending a second lot;
       - SELL iterates ALL lots of the symbol (a full exit sells every lot).
     Applied as a runtime monkeypatch (harness_lot_patch, try/finally) in the
     TB-007 runner, IDENTICALLY to both arms' processes. Production src/ is
     never modified.

test_lot_aggregation.py asserts the four §3 behaviors and the single-lot
byte-identical regression.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

PRICE_GAP_THRESHOLD = 0.05   # mirrors replay_engine.PRICE_GAP_THRESHOLD


# --------------------------------------------------------------- site 1: adapter
def aggregate_book(positions, marks: Dict[str, float], cash: float
                   ) -> Tuple[Dict[str, float], Dict[str, float], float, List[str]]:
    """(held_shares, w_prev, nav, missing_marks) by per-symbol ACCUMULATION.

    Mark preference per lot: marks[sym] -> lot.last_close -> lot.entry_price
    (same fallback order as the TB-006 adapter, aggregation fixed)."""
    held_shares: Dict[str, float] = {}
    mv: Dict[str, float] = {}
    missing: List[str] = []
    nav = float(cash)
    for p in positions:
        mk = marks.get(p.symbol)
        if mk is None:
            mk = getattr(p, "last_close", None) or p.entry_price
            if p.symbol not in missing:
                missing.append(p.symbol)
        held_shares[p.symbol] = held_shares.get(p.symbol, 0.0) + float(p.shares)
        mv[p.symbol] = mv.get(p.symbol, 0.0) + float(p.shares) * float(mk)
        nav += float(p.shares) * float(mk)
    w_prev = {s: (v / nav if nav > 0 else 0.0) for s, v in mv.items()}
    return held_shares, w_prev, nav, missing


# --------------------------------------------------------------- site 2: harness
def _execute_intents_lotfix(portfolio, intents: List[Dict[str, Any]],
                            ohlc: Dict[str, Dict[str, float]],
                            decision_params: Dict[str, Any],
                            date: str, entry_regime: Optional[str] = None
                            ) -> List[Dict[str, Any]]:
    """Copy of replay_engine._execute_intents with the two C5 fixes.
    Single-lot behavior is byte-identical (regression-tested)."""
    executed = []
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
            existing = [p for p in portfolio.positions if p.symbol == sym]
            if existing:
                # C5 fix (a): increment the held Position — never a second lot
                pos = existing[0]
                tot = pos.shares + float(shares)
                pos.entry_price = ((pos.entry_price * pos.shares
                                    + morning_price * float(shares)) / tot)
                pos.peak_price = max(pos.peak_price, morning_price)
                pos.shares = tot
            else:
                from chassis.utils.three_line_replay.replay_engine import Position
                portfolio.positions.append(Position(
                    symbol=sym, shares=float(shares),
                    entry_price=morning_price, entry_date=date,
                    peak_price=morning_price,
                    asset_class=intent.get('asset_class', 'equity'),
                    sector=intent.get('sector', 'broad'),
                    leverage_flag=int(intent.get('leverage_flag', 0) or 0),
                    entry_regime=entry_regime,
                ))
            executed.append({'date': date, 'symbol': sym, 'action': 'BUY',
                             'shares': float(shares), 'price': morning_price,
                             'dollars': round(cost, 2), 'reason': intent.get('reason', '')})
        elif action in ('SELL', 'REDUCE'):
            lots = [p for p in portfolio.positions if p.symbol == sym]
            if not lots:
                continue
            total_held = sum(p.shares for p in lots)
            # C5 fix (b): the default + the clamp see ALL lots, not the last one.
            # Fidelity fix (2026-07-01): mirror the LIVE executor
            # (src/steps/paper_trader.py). The chassis emits exact-share trims in
            # a `reduce_shares` field (PKT-TB-004); the live book sells EXACTLY
            # that. Only a legacy REDUCE with no `reduce_shares` field falls back
            # to the historical half-position trim. Ignoring `reduce_shares` (the
            # prior behavior) under-sold every trim by ~50%, so the shadow book
            # never shed exposure and progressively diverged from the live book.
            reduce_shares = intent.get('reduce_shares')
            if reduce_shares is not None:
                shares_to_sell = float(reduce_shares)
            else:
                shares_to_sell = float(intent.get('shares', total_held))
                if action == 'REDUCE':
                    shares_to_sell *= 0.5
            shares_to_sell = min(shares_to_sell, total_held)
            if shares_to_sell <= 0:
                continue
            reason = intent.get('reason', '')
            if reason == 'STOP_HIT':
                lev = max(int(getattr(p, 'leverage_flag', 0) or 0) for p in lots)
                stop_pct = float(decision_params.get(
                    'trailing_stop_leveraged' if lev == 1 else 'trailing_stop_base',
                    0.10))
                peak = max(p.peak_price for p in lots)
                if morning_price > peak * (1 - stop_pct):
                    continue
            proceeds = shares_to_sell * morning_price
            portfolio.cash += proceeds
            remaining = shares_to_sell
            for p in lots:
                take = min(p.shares, remaining)
                p.shares -= take
                remaining -= take
                if remaining <= 1e-12:
                    break
            executed.append({'date': date, 'symbol': sym, 'action': action,
                             'shares': shares_to_sell, 'price': morning_price,
                             'dollars': round(proceeds, 2), 'reason': reason})
            portfolio.positions = [p for p in portfolio.positions
                                   if not (p.symbol == sym and p.shares <= 1e-6)]
    return executed


@contextmanager
def harness_lot_patch(replay_engine_module):
    """Runtime monkeypatch of replay_engine._execute_intents (the sanctioned
    try/finally pattern). Apply identically around BOTH arms' run_variant."""
    orig = replay_engine_module._execute_intents
    replay_engine_module._execute_intents = _execute_intents_lotfix
    try:
        yield
    finally:
        replay_engine_module._execute_intents = orig
