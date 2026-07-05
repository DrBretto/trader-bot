"""Challenger book primitives — Position + Portfolio (VERBATIM first-class port of
the two dataclasses from ``src/utils/three_line_replay/replay_engine.py``).

The challenger carries its OWN stateful book (cash + per-lot Position objects with
entry metadata) so the incumbent's multi-day sell/trim gates (HEALTH_COLLAPSE,
HEALTH_DROP, REGIME_SHIFT, stops) fire correctly across the reconstruction. The
ported ``lot_fix._execute_intents_lotfix`` mutates this book at the settled OPEN;
``settled_close_value`` marks it at the settled CLOSE — the SAME (holdings x settled
close) valuation the core ``replay.driver.Book.value`` uses for canon and SPY (the
ONE marking machinery; the lines differ only in the holdings each produced).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple


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
    peak_health: Optional[float] = None
    consecutive_health_drop_days: int = 0
    entry_regime: Optional[str] = None
    last_close: Optional[float] = None


@dataclass
class Portfolio:
    cash: float
    positions: List[Position] = field(default_factory=list)
    benchmark_shares: float = 0.0
    benchmark_start_price: float = 0.0
    realized_pnl: float = 0.0
    vug_split_applied: bool = False

    def value_at_marks(self, marks: Mapping[str, float]) -> Tuple[float, List[str]]:
        v = self.cash
        missing = []
        for p in self.positions:
            if p.symbol in marks:
                v += p.shares * marks[p.symbol]
            else:
                missing.append(p.symbol)
        return v, missing

    def to_state_dict_with_marks(self, marks: Mapping[str, float]
                                 ) -> Tuple[Dict[str, Any], List[str]]:
        holdings = [{'symbol': p.symbol, 'shares': p.shares, 'entry_price': p.entry_price,
                     'entry_date': p.entry_date, 'peak_price': p.peak_price,
                     'current_price': marks.get(p.symbol, p.last_close or p.entry_price),
                     'asset_class': p.asset_class, 'sector': p.sector,
                     'leverage_flag': p.leverage_flag,
                     'consecutive_below_health_days': p.consecutive_below_health_days,
                     'peak_health': p.peak_health,
                     'consecutive_health_drop_days': p.consecutive_health_drop_days,
                     'entry_regime': p.entry_regime}
                    for p in self.positions]
        value, missing = self.value_at_marks(marks)
        return {'cash': self.cash, 'holdings': holdings, 'portfolio_value': value}, missing

    def position_map(self) -> Dict[str, Position]:
        return {p.symbol: p for p in self.positions}


def settled_close_value(book: "Portfolio", closes: Mapping[str, float]) -> float:
    """THE ONE settled-close mark for the challenger book: cash + sum(shares x
    settled close). Identical formula to ``replay.driver.Book.value`` (canon/SPY)
    and the legacy ``replay_engine.robust_value`` — a name absent from the close
    set contributes 0 (never-drop-cash). The challenger differs from the other two
    lines only in the holdings it holds, never in this valuation."""
    v = float(book.cash)
    for p in book.positions:
        c = closes.get(p.symbol)
        if p.shares and c is not None:
            v += float(p.shares) * float(c)
    return v
