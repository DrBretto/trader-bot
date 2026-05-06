"""Tests for src.utils.alpaca_truth.

Cover:
- Pre-cutoff fills (no broker_order_id) pass through.
- Partially_filled local row gets patched with Alpaca filled_qty/avg_price.
- Filled local row matching Alpaca exactly is unchanged.
- Synthetic split BUYs are NOT injected as orphan fills.
- Bootstrap orders (tb-boot-) are NOT injected.
- Smoke-test orders (smoke-test-) are NOT injected.
- Bot-decided orphan order (e.g. 2026-04-28 SLV SELL) IS injected.
"""

from __future__ import annotations

from src.utils.alpaca_truth import (
    apply_alpaca_truth_to_fills,
    _is_bot_decided,
    _is_synthetic_split,
    _orders_by_symbol,
)


def _alpaca(
    *,
    id: str,
    symbol: str,
    side: str,
    submitted_at: str,
    filled_at: str | None = None,
    filled_qty: float | None = None,
    filled_avg_price: float | None = None,
    qty: float | None = None,
    notional: float | None = None,
    status: str = "filled",
    client_order_id: str = "",
) -> dict:
    return {
        "id": id,
        "symbol": symbol,
        "side": side,
        "submitted_at": submitted_at,
        "filled_at": filled_at or submitted_at,
        "filled_qty": str(filled_qty) if filled_qty is not None else None,
        "filled_avg_price": str(filled_avg_price) if filled_avg_price is not None else None,
        "qty": str(qty) if qty is not None else None,
        "notional": str(notional) if notional is not None else None,
        "status": status,
        "client_order_id": client_order_id,
    }


def test_paper_trader_pre_cutoff_passthrough() -> None:
    fill = {
        "timestamp": "2026-02-15T13:45:00",
        "symbol": "GLD",
        "action": "BUY",
        "shares": 10,
        "price": 470.0,
        "dollars": 4700.0,
        "execution_mode": "simulated",
        # no broker_order_id — paper_trader sim trade
    }
    out = apply_alpaca_truth_to_fills([fill], alpaca_orders=[])
    assert len(out) == 1
    assert out[0]["shares"] == 10
    assert "_alpaca_truth_applied" not in out[0]


def test_partially_filled_gets_patched_to_alpaca_truth() -> None:
    fill = {
        "timestamp": "2026-05-05T13:45:43.652459",
        "symbol": "VUG",
        "action": "SELL",
        "shares": 56.0,
        "price": 84.38,
        "broker_order_id": "ord-vug-sell",
        "broker_status": "partially_filled",
        "execution_mode": "alpaca_paper",
    }
    alpaca = [
        _alpaca(
            id="ord-vug-sell",
            symbol="VUG",
            side="sell",
            submitted_at="2026-05-05T13:45:42.60855Z",
            filled_at="2026-05-05T13:45:45.77209Z",
            filled_qty=102.765769,
            filled_avg_price=84.35,
            qty=102.765769,
        ),
    ]
    out = apply_alpaca_truth_to_fills([fill], alpaca)
    assert len(out) == 1
    assert out[0]["shares"] == 102.765769
    assert out[0]["price"] == 84.35
    assert out[0]["broker_status"] == "filled"
    assert out[0]["_alpaca_truth_applied"] is True
    assert out[0]["_alpaca_truth_local_shares"] == 56.0


def test_filled_match_passes_through_unchanged() -> None:
    fill = {
        "symbol": "FXI",
        "action": "BUY",
        "shares": 212.958462842,
        "price": 35.39,
        "broker_order_id": "ord-fxi",
        "broker_status": "filled",
    }
    alpaca = [
        _alpaca(
            id="ord-fxi",
            symbol="FXI",
            side="buy",
            submitted_at="2026-04-07T13:45:43Z",
            filled_qty=212.958462842,
            filled_avg_price=35.39,
            notional=7536.6,
        ),
    ]
    out = apply_alpaca_truth_to_fills([fill], alpaca)
    assert len(out) == 1
    assert out[0]["shares"] == 212.958462842
    assert "_alpaca_truth_applied" not in out[0]


def test_orphan_bot_decided_fill_injected() -> None:
    # 2026-04-28 SLV SELL: filled at Alpaca, missing from trades.jsonl.
    alpaca = [
        _alpaca(
            id="ord-slv-orphan",
            symbol="SLV",
            side="sell",
            submitted_at="2026-04-28T18:00:21Z",
            filled_at="2026-04-28T18:00:21.727438Z",
            filled_qty=81.107334,
            filled_avg_price=66.34,
            qty=81.107334,
            client_order_id="tb-0bd273a1ed56a848",  # bot-decided
        ),
    ]
    out = apply_alpaca_truth_to_fills([], alpaca)
    assert len(out) == 1
    inj = out[0]
    assert inj["symbol"] == "SLV"
    assert inj["action"] == "SELL"
    assert inj["shares"] == 81.107334
    assert inj["price"] == 66.34
    assert inj["broker_order_id"] == "ord-slv-orphan"
    assert inj["_alpaca_only_injection"] is True


def test_bootstrap_orders_are_not_injected() -> None:
    # tb-boot-* prefix means the bootstrap script placed it; the position
    # is already represented by pre-cutoff sim BUYs in trades.jsonl.
    alpaca = [
        _alpaca(
            id="ord-iyt-boot",
            symbol="IYT",
            side="buy",
            submitted_at="2026-03-12T18:05:11Z",
            filled_qty=67.631543351,
            filled_avg_price=73.93,
            notional=5000.0,
            client_order_id="tb-boot-5ca1ce7ad410dad0",
        ),
    ]
    out = apply_alpaca_truth_to_fills([], alpaca)
    assert out == []


def test_smoke_test_orders_are_not_injected() -> None:
    alpaca = [
        _alpaca(
            id="ord-spy-smoke",
            symbol="SPY",
            side="buy",
            submitted_at="2026-03-12T15:35:02Z",
            filled_qty=0.001497234,
            filled_avg_price=667.898,
            notional=1.0,
            client_order_id="smoke-test-1773329702",
        ),
    ]
    out = apply_alpaca_truth_to_fills([], alpaca)
    assert out == []


def test_synthetic_split_buy_is_not_injected() -> None:
    # Apr 7 BUY 17.127628206 sh @ $440.89 → Apr 21 synthetic BUY 85.638 sh @ $82.35
    # → May 5 SELL 102.765769 sh @ $84.35.
    apr7 = _alpaca(
        id="ord-vug-orig",
        symbol="VUG",
        side="buy",
        submitted_at="2026-04-07T13:45:42Z",
        filled_qty=17.127628206,
        filled_avg_price=440.89,
        notional=7551.4,
        client_order_id="tb-291cd6b7728cfbcc",
    )
    apr21 = _alpaca(
        id="ord-vug-split",
        symbol="VUG",
        side="buy",
        submitted_at="2026-04-21T13:39:21Z",
        filled_qty=85.63814103,
        filled_avg_price=82.35,
        qty=85.63814103,  # qty-based, no notional
        client_order_id="tb-aaaaaaaaaaaaaaaa",  # bot-prefix but synthetic
    )
    may5 = _alpaca(
        id="ord-vug-sell",
        symbol="VUG",
        side="sell",
        submitted_at="2026-05-05T13:45:42Z",
        filled_qty=102.765769,
        filled_avg_price=84.35,
        qty=102.765769,
        client_order_id="tb-b90f8aa4d4624e0b",
    )
    # apr7 is matched by a local trade row; apr21 is the synthetic; may5 is a
    # real bot SELL that the local trade has at the wrong qty.
    local = [
        {
            "symbol": "VUG",
            "action": "BUY",
            "shares": 17.127628206,
            "price": 441.05,
            "broker_order_id": "ord-vug-orig",
            "broker_status": "filled",
        },
        {
            "symbol": "VUG",
            "action": "SELL",
            "shares": 56.0,
            "price": 84.38,
            "broker_order_id": "ord-vug-sell",
            "broker_status": "partially_filled",
        },
    ]
    out = apply_alpaca_truth_to_fills(local, [apr7, apr21, may5])
    # Expect: 2 fills out (the synthetic BUY is NOT injected).
    assert len(out) == 2
    sells = [f for f in out if f["action"] == "SELL"]
    assert len(sells) == 1
    assert sells[0]["shares"] == 102.765769


def test_is_bot_decided_classification() -> None:
    assert _is_bot_decided("tb-291cd6b7728cfbcc") is True
    assert _is_bot_decided("tb-boot-5ca1ce7ad410dad0") is False
    assert _is_bot_decided("smoke-test-1773329702") is False
    assert _is_bot_decided("ca9da032-6c8b-4471-89ca-d57daa346ff9") is False
    assert _is_bot_decided("") is False
    assert _is_bot_decided(None) is False


def test_is_synthetic_split_detection() -> None:
    apr7 = _alpaca(
        id="orig",
        symbol="VUG",
        side="buy",
        submitted_at="2026-04-07T13:45:42Z",
        filled_qty=17.127628206,
        filled_avg_price=440.89,
        notional=7551.4,
    )
    apr21 = _alpaca(
        id="synth",
        symbol="VUG",
        side="buy",
        submitted_at="2026-04-21T13:39:21Z",
        filled_qty=85.63814103,
        filled_avg_price=82.35,
        qty=85.63814103,
    )
    may5 = _alpaca(
        id="sell",
        symbol="VUG",
        side="sell",
        submitted_at="2026-05-05T13:45:42Z",
        filled_qty=102.765769,
        filled_avg_price=84.35,
        qty=102.765769,
    )
    by_sym = _orders_by_symbol([apr7, apr21, may5])
    assert _is_synthetic_split(apr21, by_sym) is True
    # Original Apr 7 BUY is not a synthetic split.
    assert _is_synthetic_split(apr7, by_sym) is False
    # SELLs are not synthetic-split BUYs.
    assert _is_synthetic_split(may5, by_sym) is False
