"""Tests for the PKT-TB-004 Risk Architect candidates.

1. compute_exposure_trims — book-level pro-rata trim toward the fusion
   target_gross_exposure (decision_engine_overrides['exposure_trim']).
2. vol_decay_constraints — structural-decay hold cap for volatility ETPs
   (the VIXY policy candidate).
3. paper_trader exact-share REDUCE (reduce_shares field).

Both candidates default OFF: absent config must produce byte-identical
production behavior.
"""

import pandas as pd

import sys
sys.path.insert(0, str(__file__).rsplit('/tests', 1)[0])

from src.steps.decision_engine import (
    compute_exposure_trims,
    evaluate_holdings,
)
from src.steps import paper_trader


def _prices(rows):
    return pd.DataFrame([
        {'date': pd.Timestamp(d), 'symbol': s, 'close': c}
        for d, s, c in rows
    ])


def _portfolio(holdings, cash=0.0, value=100000.0):
    return {
        'cash': cash,
        'portfolio_value': value,
        'holdings': holdings,
    }


def _holding(symbol, shares, sector='broad', entry_date='2026-01-02',
             entry_price=100.0):
    return {
        'symbol': symbol,
        'shares': shares,
        'entry_price': entry_price,
        'peak_price': entry_price,
        'entry_date': entry_date,
        'sector': sector,
        'asset_class': 'equity',
        'leverage_flag': 0,
    }


TWO_HOLDINGS_PRICES = _prices([
    ('2026-02-02', 'AAA', 100.0),
    ('2026-02-02', 'BBB', 50.0),
])


def _two_holding_book():
    # AAA $40k, BBB $20k -> gross 0.60 of a $100k book, 2:1 pro-rata split
    return _portfolio([
        _holding('AAA', 400),
        _holding('BBB', 400),
    ])


class TestExposureTrimTrigger:

    def test_disabled_returns_nothing(self):
        for cfg in ({}, {'enabled': False}):
            assert compute_exposure_trims(
                _two_holding_book(), [], TWO_HOLDINGS_PRICES, 100000.0,
                target_exposure=0.2, regime_label='choppy', params={},
                trim_cfg=cfg,
            ) == []

    def test_fires_when_gap_exceeds_trigger(self):
        # gross 0.60, target 0.40, trigger 0.10 -> breach; trim to 0.45
        trims = compute_exposure_trims(
            _two_holding_book(), [], TWO_HOLDINGS_PRICES, 100000.0,
            target_exposure=0.40, regime_label='choppy', params={},
            trim_cfg={'enabled': True, 'trigger_gap': 0.10,
                      'hysteresis_gap': 0.05},
        )
        assert {t['symbol'] for t in trims} == {'AAA', 'BBB'}
        assert all(t['action'] == 'REDUCE' for t in trims)
        assert all(t['reason'] == 'EXPOSURE_TRIM' for t in trims)
        total = sum(t['shares'] * t['price'] for t in trims)
        # total trim ~= (0.60 - 0.45) * 100k = 15k (minus int-share rounding)
        assert 14000 <= total <= 15000

    def test_no_fire_inside_trigger_band(self):
        # gross 0.60, target 0.55, trigger 0.10 -> gap 0.05 < trigger
        trims = compute_exposure_trims(
            _two_holding_book(), [], TWO_HOLDINGS_PRICES, 100000.0,
            target_exposure=0.55, regime_label='choppy', params={},
            trim_cfg={'enabled': True, 'trigger_gap': 0.10},
        )
        assert trims == []

    def test_pro_rata_split(self):
        trims = compute_exposure_trims(
            _two_holding_book(), [], TWO_HOLDINGS_PRICES, 100000.0,
            target_exposure=0.30, regime_label='choppy', params={},
            trim_cfg={'enabled': True, 'trigger_gap': 0.10,
                      'hysteresis_gap': 0.05},
        )
        by_sym = {t['symbol']: t['shares'] * t['price'] for t in trims}
        # AAA holds 2x the dollars of BBB -> trimmed ~2x the dollars
        ratio = by_sym['AAA'] / by_sym['BBB']
        assert 1.9 <= ratio <= 2.1

    def test_hysteresis_prevents_churn_after_trim(self):
        # Apply the trims, then re-evaluate: post-trim gross sits at
        # target + hysteresis, inside the trigger band -> no second trim.
        book = _two_holding_book()
        trims = compute_exposure_trims(
            book, [], TWO_HOLDINGS_PRICES, 100000.0,
            target_exposure=0.40, regime_label='choppy', params={},
            trim_cfg={'enabled': True, 'trigger_gap': 0.10,
                      'hysteresis_gap': 0.05},
        )
        assert trims
        price = {'AAA': 100.0, 'BBB': 50.0}
        for t in trims:
            for h in book['holdings']:
                if h['symbol'] == t['symbol']:
                    h['shares'] -= t['reduce_shares']
        gross = sum(h['shares'] * price[h['symbol']] for h in book['holdings'])
        assert gross / 100000.0 <= 0.40 + 0.10  # back inside the band
        assert gross / 100000.0 >= 0.40         # never trimmed below target
        second = compute_exposure_trims(
            book, [], TWO_HOLDINGS_PRICES, 100000.0,
            target_exposure=0.40, regime_label='choppy', params={},
            trim_cfg={'enabled': True, 'trigger_gap': 0.10,
                      'hysteresis_gap': 0.05},
        )
        assert second == []

    def test_min_trim_dollars_suppresses_dust(self):
        # gap just over trigger but total trim below the per-order minimum
        trims = compute_exposure_trims(
            _two_holding_book(), [], TWO_HOLDINGS_PRICES, 100000.0,
            target_exposure=0.49, regime_label='choppy', params={},
            trim_cfg={'enabled': True, 'trigger_gap': 0.10,
                      'hysteresis_gap': 0.05, 'min_trim_dollars': 50000},
        )
        assert trims == []

    def test_pending_sell_excluded(self):
        # AAA pending SELL: not counted in gross (0.20 remains) and never trimmed
        trims = compute_exposure_trims(
            _two_holding_book(),
            [{'action': 'SELL', 'symbol': 'AAA'}],
            TWO_HOLDINGS_PRICES, 100000.0,
            target_exposure=0.05, regime_label='choppy', params={},
            trim_cfg={'enabled': True, 'trigger_gap': 0.05,
                      'hysteresis_gap': 0.02},
        )
        assert {t['symbol'] for t in trims} == {'BBB'}

    def test_pending_reduce_counted_post_reduce(self):
        # BBB pending legacy REDUCE (halved): gross = 0.40 + 0.10 = 0.50;
        # target 0.45, trigger 0.10 -> no breach
        trims = compute_exposure_trims(
            _two_holding_book(),
            [{'action': 'REDUCE', 'symbol': 'BBB'}],
            TWO_HOLDINGS_PRICES, 100000.0,
            target_exposure=0.45, regime_label='choppy', params={},
            trim_cfg={'enabled': True, 'trigger_gap': 0.10},
        )
        assert trims == []

    def test_persistence_days_gate(self):
        book = _two_holding_book()
        cfg = {'enabled': True, 'trigger_gap': 0.10, 'hysteresis_gap': 0.05,
               'persistence_days': 2}
        first = compute_exposure_trims(
            book, [], TWO_HOLDINGS_PRICES, 100000.0,
            target_exposure=0.40, regime_label='choppy', params={},
            trim_cfg=cfg,
        )
        assert first == []
        assert book['exposure_trim_consecutive_over'] == 1
        second = compute_exposure_trims(
            book, [], TWO_HOLDINGS_PRICES, 100000.0,
            target_exposure=0.40, regime_label='choppy', params={},
            trim_cfg=cfg,
        )
        assert second  # breach persisted 2 runs -> fires
        assert book['exposure_trim_consecutive_over'] == 0  # re-armed

    def test_persistence_counter_resets_when_back_under(self):
        book = _two_holding_book()
        cfg = {'enabled': True, 'trigger_gap': 0.10, 'persistence_days': 3}
        compute_exposure_trims(
            book, [], TWO_HOLDINGS_PRICES, 100000.0,
            target_exposure=0.40, regime_label='choppy', params={},
            trim_cfg=cfg,
        )
        assert book['exposure_trim_consecutive_over'] == 1
        compute_exposure_trims(
            book, [], TWO_HOLDINGS_PRICES, 100000.0,
            target_exposure=1.0, regime_label='choppy', params={},
            trim_cfg=cfg,
        )
        assert book['exposure_trim_consecutive_over'] == 0

    def test_skip_regimes(self):
        trims = compute_exposure_trims(
            _two_holding_book(), [], TWO_HOLDINGS_PRICES, 100000.0,
            target_exposure=0.20, regime_label='high_vol_panic', params={},
            trim_cfg={'enabled': True, 'trigger_gap': 0.10,
                      'skip_regimes': ['high_vol_panic']},
        )
        assert trims == []

    def test_target_floor_clips_target(self):
        # target 0.10 floored to 0.50 -> gross 0.60 inside trigger band
        trims = compute_exposure_trims(
            _two_holding_book(), [], TWO_HOLDINGS_PRICES, 100000.0,
            target_exposure=0.10, regime_label='choppy', params={},
            trim_cfg={'enabled': True, 'trigger_gap': 0.15,
                      'target_floor': 0.50},
        )
        assert trims == []


class TestVolDecayHoldCap:

    def _book_and_prices(self, entry_date, asof='2026-02-02'):
        portfolio = _portfolio([
            _holding('VIXY', 200, sector='volatility', entry_date=entry_date,
                     entry_price=30.0),
        ])
        # price above the trailing stop so only the hold cap can fire
        prices = _prices([(asof, 'VIXY', 29.0)])
        return portfolio, prices

    def test_cap_fires_after_max_hold_days(self):
        portfolio, prices = self._book_and_prices('2026-01-02')  # 31 days
        actions = evaluate_holdings(
            portfolio, [], prices,
            {'vol_decay_constraints': {'max_hold_days': 15}},
            'choppy', {},
        )
        assert len(actions) == 1
        assert actions[0]['reason'] == 'VOL_DECAY_HOLD_CAP'
        assert actions[0]['action'] == 'SELL'

    def test_no_fire_inside_window(self):
        portfolio, prices = self._book_and_prices('2026-01-28')  # 5 days
        actions = evaluate_holdings(
            portfolio, [], prices,
            {'vol_decay_constraints': {'max_hold_days': 15}},
            'choppy', {},
        )
        assert actions == []

    def test_days_measured_against_price_asof_not_wall_clock(self):
        # Entry 5 days before the price data's date; wall-clock today is far
        # later. The cap must NOT fire (replay-consistent calendar).
        portfolio, prices = self._book_and_prices('2026-01-28')
        actions = evaluate_holdings(
            portfolio, [], prices,
            {'vol_decay_constraints': {'max_hold_days': 10}},
            'choppy', {},
        )
        assert actions == []

    def test_default_off(self):
        portfolio, prices = self._book_and_prices('2025-06-02')  # 8 months
        actions = evaluate_holdings(portfolio, [], prices, {}, 'choppy', {})
        assert actions == []

    def test_non_decay_sector_untouched(self):
        portfolio = _portfolio([
            _holding('SPY', 100, sector='broad', entry_date='2025-06-02'),
        ])
        prices = _prices([('2026-02-02', 'SPY', 100.0)])
        actions = evaluate_holdings(
            portfolio, [], prices,
            {'vol_decay_constraints': {'max_hold_days': 15}},
            'choppy', {},
        )
        assert actions == []


class TestPaperTraderExactReduce:

    def _trade(self, action):
        portfolio = {
            'cash': 0.0,
            'holdings': [
                {'symbol': 'AAA', 'shares': 100, 'entry_price': 90.0,
                 'entry_date': '2026-01-02T09:45:00', 'peak_price': 100.0},
            ],
            'portfolio_value': 10000.0,
        }
        record = paper_trader.execute_trade(
            portfolio=portfolio,
            action=action,
            regime_label='choppy',
            universe_df=pd.DataFrame(),
            transaction_cost_config={'slippage_range_bps': 0.0},
        )
        return portfolio, record

    def test_exact_reduce_shares_honored(self):
        portfolio, record = self._trade({
            'symbol': 'AAA', 'action': 'REDUCE', 'shares': 30,
            'reduce_shares': 30, 'price': 100.0,
        })
        assert portfolio['holdings'][0]['shares'] == 70
        assert record['shares'] == 30

    def test_legacy_halving_unchanged(self):
        portfolio, record = self._trade({
            'symbol': 'AAA', 'action': 'REDUCE', 'shares': 100,
            'price': 100.0,
        })
        assert portfolio['holdings'][0]['shares'] == 50
        assert record['shares'] == 50

    def test_reduce_clamped_to_held_shares(self):
        portfolio, record = self._trade({
            'symbol': 'AAA', 'action': 'REDUCE', 'shares': 500,
            'reduce_shares': 500, 'price': 100.0,
        })
        assert portfolio['holdings'][0]['shares'] == 0
        assert record['shares'] == 100
