"""PKT-TB-007 C5 — regression for the 2026-07-01 `reduce_shares` fidelity fix
(``lot_fix_007._execute_intents_lotfix``).

The chassis (PKT-TB-004) emits exact-share trims in a ``reduce_shares`` field and
the LIVE executor (``src/steps/paper_trader.py``) sells EXACTLY that. The shadow
harness previously ignored the field and always trimmed a REDUCE by half — so
every trim under-sold ~50%, the shadow book never shed exposure, and it diverged
from the live book. These tests pin the fix:

  * a REDUCE carrying ``reduce_shares`` sells exactly that many shares;
  * a legacy REDUCE with no ``reduce_shares`` still falls back to the historical
    half-position trim (no behavior change for old intents);
  * ``reduce_shares`` is clamped to the total held (a full exit, never oversell).
"""
import sys
from pathlib import Path

PROTO = Path(__file__).resolve().parents[1]
REPO = PROTO.parents[2]
for p in (str(PROTO), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.utils.three_line_replay.replay_engine import Portfolio, Position  # noqa: E402
from lot_fix_007 import _execute_intents_lotfix                             # noqa: E402


def _lot(sym, shares, entry, peak=None):
    return Position(symbol=sym, shares=float(shares), entry_price=float(entry),
                    entry_date="2026-01-31",
                    peak_price=float(peak if peak is not None else entry),
                    asset_class="equity", sector="broad", leverage_flag=0)


PARAMS = {"min_order_dollars": 250, "trailing_stop_base": 0.10,
          "trailing_stop_leveraged": 0.06}
OHLC = {"XLK": {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}}


def test_reduce_honors_exact_reduce_shares():
    pf = Portfolio(cash=0.0, positions=[_lot("XLK", 100, 90.0)])
    intents = [{"symbol": "XLK", "action": "REDUCE", "reduce_shares": 30}]
    ex = _execute_intents_lotfix(pf, intents, OHLC, PARAMS, "2026-06-30")
    assert len(ex) == 1
    assert ex[0]["shares"] == 30.0                  # EXACT trim, not the half (50)
    held = sum(p.shares for p in pf.positions if p.symbol == "XLK")
    assert held == 70.0
    assert pf.cash == 30.0 * 100.0                  # proceeds credited at the open


def test_legacy_reduce_without_field_falls_back_to_half():
    pf = Portfolio(cash=0.0, positions=[_lot("XLK", 100, 90.0)])
    intents = [{"symbol": "XLK", "action": "REDUCE"}]     # no reduce_shares field
    ex = _execute_intents_lotfix(pf, intents, OHLC, PARAMS, "2026-06-30")
    assert ex[0]["shares"] == 50.0                  # historical half-position trim
    held = sum(p.shares for p in pf.positions if p.symbol == "XLK")
    assert held == 50.0


def test_reduce_shares_clamps_to_total_held():
    pf = Portfolio(cash=0.0, positions=[_lot("XLK", 40, 90.0)])
    intents = [{"symbol": "XLK", "action": "REDUCE", "reduce_shares": 999}]
    ex = _execute_intents_lotfix(pf, intents, OHLC, PARAMS, "2026-06-30")
    assert ex[0]["shares"] == 40.0                  # clamped to holdings -> full exit
    assert not [p for p in pf.positions if p.symbol == "XLK"]   # position removed
