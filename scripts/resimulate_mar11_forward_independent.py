"""Independent resimulation of trader-bot performance from 2026-03-11 EOD forward.

Run from packet:
  PKT-ULP-TRB-MAR11-INDEPENDENT-RESIMULATION-20260506

What this is:
- A fresh, independent simulator that takes:
    * 2026-03-11 EOD starting state (cash + shares from raw daily artifact,
      marks from verified close prices)
    * Each trading day's daily/<D>/decisions.json (raw pipeline artifact —
      strategy decision for trading day D)
    * Verified open + close prices from yfinance (independently fetched) and
      Stooq (cross-check via daily/<D+1>/prices.parquet's open_price column
      for D when available)
- Applies a clean fill-price rule: BUY/SELL fill at D's open; portfolio mark
  at D's close. No bridge cashflows. No historical_corrections. No use of
  dashboard.json or corrected-equity files as inputs.

What this is NOT:
- A re-run of the production decision engine (we trust decisions.json as the
  raw pipeline artifact).
- An audit of the prior failed-correction chain (the operator emergency
  directive forbids using those outputs).
- A patch on top of any corrected-equity helper.

Outputs (under run_root/reports/tables/):
  - DAILY_INDEPENDENT_RESIMULATION.tsv
  - SIMULATED_ACTIONS_AND_FILLS.tsv
  - PRICE_AND_CORPORATE_ACTION_SOURCES.tsv
  - SOURCE_GAPS_AND_BLOCKERS.tsv
  (CURRENT_DASHBOARD_COMPARISON.tsv is produced by a separate compare script.)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import pandas as pd
import yfinance as yf

# ---- Constants -------------------------------------------------------------

S3_BUCKET = "investment-system-data"
START_AS_OF = pd.Timestamp("2026-03-11")
SIM_FIRST_DAY = pd.Timestamp("2026-03-12")
SIM_LAST_DAY = pd.Timestamp("2026-05-06")
SPY_BENCHMARK = "SPY"

CROSS_CHECK_THRESHOLD_PCT = 0.5  # yfinance vs Stooq disagreement threshold
MARK_FALLBACK_MAX_DAYS = 5

# ---- Path setup ------------------------------------------------------------

RUN_ROOT = Path(
    "/Users/drbretto/Desktop/Projects/Infotropy Book/book-factory/use_lane_outputs/runs/"
    "20260506_trader-bot-mar11-independent-resimulation-v1"
)
WORK = RUN_ROOT / "working"
TABLES = RUN_ROOT / "reports" / "tables"
DAILY_CACHE = WORK / "daily_cache"
PRICE_CACHE = WORK / "price_cache.parquet"

WORK.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)
DAILY_CACHE.mkdir(parents=True, exist_ok=True)


# ---- Portfolio state -------------------------------------------------------

@dataclass
class Holding:
    symbol: str
    shares: float
    entry_price: float
    entry_date: str
    asset_class: str = "equity"


@dataclass
class Portfolio:
    cash: float
    holdings: Dict[str, Holding] = field(default_factory=dict)
    benchmark_shares: float = 0.0
    peak: float = 0.0

    def total_value(self, marks: Dict[str, float]) -> float:
        v = self.cash
        for h in self.holdings.values():
            mark = marks.get(h.symbol)
            if mark is None:
                # Hard requirement: don't invent prices. Caller must handle.
                raise ValueError(f"Missing mark for held symbol {h.symbol}")
            v += h.shares * mark
        return v

    def holdings_value(self, marks: Dict[str, float]) -> float:
        v = 0.0
        for h in self.holdings.values():
            v += h.shares * marks[h.symbol]
        return v


# ---- Source loaders --------------------------------------------------------

_S3 = None


def s3():
    global _S3
    if _S3 is None:
        sess = boto3.Session(profile_name="personal")
        _S3 = sess.client("s3", region_name="us-east-1")
    return _S3


def get_daily_decisions(date_str: str) -> Optional[Dict[str, Any]]:
    """Load daily/<date>/decisions.json — raw pipeline artifact."""
    cache_path = DAILY_CACHE / f"{date_str}_decisions.json"
    if not cache_path.exists():
        try:
            obj = s3().get_object(Bucket=S3_BUCKET, Key=f"daily/{date_str}/decisions.json")
            cache_path.write_bytes(obj["Body"].read())
        except s3().exceptions.NoSuchKey:
            return None
        except Exception as e:
            print(f"  WARN: cannot fetch decisions.json for {date_str}: {e}", file=sys.stderr)
            return None
    return json.loads(cache_path.read_text())


def find_decisions_for_trading_day(
    trading_day: pd.Timestamp,
    trading_day_set: set,
    lookback_days: int = 4,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """For trading day D, find the decisions.json that targets D's morning.

    Convention: night phase ran X-1 evening writes to daily/<X>/decisions.json,
    where X is the next CALENDAR date. So trading day D's intents live in
    daily/<X>/ where X is the calendar date right after D-1's night run, and X
    is between [D-lookback, D] inclusive.

    For consecutive trading days, X = D (D-1 was a trading day too). For
    Mondays after Fri-Sun, X = Saturday (Fri-evening's run). For Tuesdays
    after a Mon holiday, X = Tue or Mon depending on whether Sun ran (it
    doesn't; the night phase is Mon-Fri).

    Walk back from D through calendar days; return the first decisions.json
    found in a folder that does NOT correspond to another trading day.
    Returns (decisions_dict, source_label).
    """
    for delta in range(0, lookback_days + 1):
        candidate = trading_day - pd.Timedelta(days=delta)
        cand_str = candidate.strftime("%Y-%m-%d")
        # Skip candidate folders owned by ANOTHER trading day
        if delta != 0 and candidate in trading_day_set:
            # That folder belongs to that other trading day; don't poach it.
            return None, "no_decisions_intermediate_trading_day"
        dec = get_daily_decisions(cand_str)
        if dec is not None:
            return dec, f"daily/{cand_str}/decisions.json"
    return None, "no_decisions_found_in_lookback"


def get_daily_prices_parquet(date_str: str) -> Optional[pd.DataFrame]:
    """Load daily/<date>/prices.parquet — Stooq snapshot from night phase."""
    cache_path = DAILY_CACHE / f"{date_str}_prices.parquet"
    if not cache_path.exists():
        try:
            obj = s3().get_object(Bucket=S3_BUCKET, Key=f"daily/{date_str}/prices.parquet")
            cache_path.write_bytes(obj["Body"].read())
        except Exception:
            return None
    try:
        return pd.read_parquet(cache_path)
    except Exception:
        return None


def get_alpaca_split_data(symbol: str) -> pd.Series:
    """Return verified split events for symbol from yfinance (split-ratio per date)."""
    try:
        return yf.Ticker(symbol).splits
    except Exception:
        return pd.Series(dtype=float)


# ---- Price fetching --------------------------------------------------------

class PriceBook:
    """Caches verified open/close prices per (date, symbol)."""

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        if cache_path.exists():
            self._df = pd.read_parquet(cache_path)
            self._df["date"] = pd.to_datetime(self._df["date"]).dt.normalize()
        else:
            self._df = pd.DataFrame(
                columns=[
                    "date", "symbol", "yf_open", "yf_close",
                    "stooq_open", "stooq_close", "yf_split",
                ]
            )

    def _persist(self):
        self._df.to_parquet(self.cache_path, index=False)

    def fetch(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> None:
        """Fetch yfinance bars for [start, end] inclusive and merge into book."""
        # Pull fresh always for the requested window — yfinance is the
        # primary verified-price source. Stooq cross-check is layered on later.
        try:
            df = yf.Ticker(symbol).history(
                start=str(start.date()),
                end=str((end + pd.Timedelta(days=1)).date()),
                auto_adjust=False,
            )
        except Exception as e:
            print(f"  yfinance error for {symbol}: {e}", file=sys.stderr)
            return
        if len(df) == 0:
            return
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        rows = []
        for d, row in df.iterrows():
            rows.append({
                "date": d,
                "symbol": symbol,
                "yf_open": float(row["Open"]),
                "yf_close": float(row["Close"]),
                "stooq_open": None,
                "stooq_close": None,
                "yf_split": float(row.get("Stock Splits", 0.0) or 0.0),
            })
        new_df = pd.DataFrame(rows)
        # Replace any existing rows for this symbol/dates
        mask = (
            self._df["symbol"].eq(symbol)
            & self._df["date"].isin(new_df["date"])
        )
        self._df = self._df[~mask]
        self._df = pd.concat([self._df, new_df], ignore_index=True)

    def attach_stooq(self, date: pd.Timestamp, prices_df: pd.DataFrame) -> None:
        """Attach stooq cross-check from a daily/<date>/prices.parquet snapshot."""
        if prices_df is None or len(prices_df) == 0:
            return
        prices_df = prices_df.copy()
        prices_df["date"] = pd.to_datetime(prices_df["date"]).dt.normalize()
        # We want close for date `date`. Pull just that date's rows.
        target_rows = prices_df[prices_df["date"] == date]
        for _, r in target_rows.iterrows():
            sym = r["symbol"]
            mask = (self._df["symbol"].eq(sym)) & (self._df["date"].eq(date))
            if not mask.any():
                # Add a stooq-only row; will be flagged in source_gaps if yf missing
                self._df = pd.concat(
                    [self._df, pd.DataFrame([{
                        "date": date, "symbol": sym,
                        "yf_open": None, "yf_close": None,
                        "stooq_open": float(r.get("open", float("nan"))) if pd.notna(r.get("open", float("nan"))) else None,
                        "stooq_close": float(r["close"]),
                        "yf_split": 0.0,
                    }])], ignore_index=True)
            else:
                idx = self._df[mask].index[0]
                if pd.notna(r.get("open", float("nan"))):
                    self._df.at[idx, "stooq_open"] = float(r["open"])
                self._df.at[idx, "stooq_close"] = float(r["close"])

    def get_open(self, symbol: str, date: pd.Timestamp) -> Tuple[Optional[float], str]:
        """Return (open_price, source_label)."""
        rows = self._df[(self._df["symbol"] == symbol) & (self._df["date"] == date)]
        if len(rows) == 0:
            return None, "missing"
        r = rows.iloc[0]
        yfo = r["yf_open"]
        sto = r["stooq_open"]
        if yfo is not None and pd.notna(yfo):
            return float(yfo), "yfinance"
        if sto is not None and pd.notna(sto):
            return float(sto), "stooq_only"
        return None, "missing"

    def get_close(self, symbol: str, date: pd.Timestamp) -> Tuple[Optional[float], str]:
        rows = self._df[(self._df["symbol"] == symbol) & (self._df["date"] == date)]
        if len(rows) == 0:
            return None, "missing"
        r = rows.iloc[0]
        yfc = r["yf_close"]
        sto = r["stooq_close"]
        if yfc is not None and pd.notna(yfc):
            return float(yfc), "yfinance"
        if sto is not None and pd.notna(sto):
            return float(sto), "stooq_only"
        return None, "missing"

    def get_yf_split(self, symbol: str, date: pd.Timestamp) -> float:
        rows = self._df[(self._df["symbol"] == symbol) & (self._df["date"] == date)]
        if len(rows) == 0:
            return 0.0
        v = rows.iloc[0]["yf_split"]
        return float(v) if pd.notna(v) else 0.0

    def trading_day_calendar(self, start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
        """Days where SPY traded — taken as the verified trading-day calendar."""
        rows = self._df[(self._df["symbol"] == SPY_BENCHMARK)
                        & (self._df["date"] >= start)
                        & (self._df["date"] <= end)]
        if "yf_close" in rows.columns:
            rows = rows[rows["yf_close"].notna()]
        return sorted(set(pd.to_datetime(rows["date"]).dt.normalize().tolist()))

    def disagreement_pct(self, date: pd.Timestamp, symbol: str, kind: str) -> Optional[float]:
        rows = self._df[(self._df["symbol"] == symbol) & (self._df["date"] == date)]
        if len(rows) == 0:
            return None
        r = rows.iloc[0]
        if kind == "open":
            yf_v, st_v = r["yf_open"], r["stooq_open"]
        else:
            yf_v, st_v = r["yf_close"], r["stooq_close"]
        if yf_v is None or st_v is None or pd.isna(yf_v) or pd.isna(st_v) or st_v == 0:
            return None
        return float((yf_v - st_v) / st_v * 100.0)

    def persist(self):
        self._persist()


# ---- Simulator core --------------------------------------------------------

def build_starting_portfolio(state_json: Dict[str, Any], price_book: PriceBook) -> Tuple[Portfolio, float]:
    """Build EOD-2026-03-11 starting portfolio with verified close marks."""
    cash = float(state_json["cash"])
    benchmark_shares = float(state_json["benchmark_shares"])
    holdings: Dict[str, Holding] = {}
    for h in state_json["holdings"]:
        holdings[h["symbol"]] = Holding(
            symbol=h["symbol"],
            shares=float(h["shares"]),
            entry_price=float(h["entry_price"]),
            entry_date=h.get("entry_date", "2026-03-11"),
            asset_class=h.get("asset_class", "equity"),
        )
    # Mark to verified close 2026-03-11
    marks_0311 = {}
    for sym in list(holdings.keys()) + [SPY_BENCHMARK]:
        c, _src = price_book.get_close(sym, START_AS_OF)
        if c is None:
            raise SystemExit(f"BLOCKED_SOURCE_GAP: 2026-03-11 close missing for {sym}")
        marks_0311[sym] = c
    p = Portfolio(cash=cash, holdings=holdings, benchmark_shares=benchmark_shares)
    starting_value = p.total_value(marks_0311)
    p.peak = starting_value
    return p, starting_value


def apply_corporate_actions(
    portfolio: Portfolio,
    date: pd.Timestamp,
    price_book: PriceBook,
    corp_action_rows: List[Dict[str, Any]],
) -> None:
    """Record yfinance-verified splits as audit-only events.

    yfinance Open/Close with `auto_adjust=False` is, in practice, still
    split-adjusted retroactively for ETFs (verified empirically against VUG
    2026-04-21 6:1 split: yfinance returns ~$82 on 04-20 and 04-22 with no
    discontinuity). Because our fill and mark prices come from yfinance,
    share counts derived as `target_dollars / yf_open_price` are already in
    post-split coordinates. Applying the split to shares again would
    double-count it. We therefore RECORD the split for the audit table but
    do NOT mutate share counts. This is a deliberate, documented choice; a
    Stooq-only simulator would do the opposite (raw prices need explicit
    split application).
    """
    for sym in list(portfolio.holdings.keys()):
        ratio = price_book.get_yf_split(sym, date)
        if ratio and ratio != 1.0 and ratio != 0.0:
            h = portfolio.holdings[sym]
            corp_action_rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "symbol": sym,
                "raw_open": "",
                "raw_close": "",
                "adjusted_open": "",
                "adjusted_close": "",
                "split_ratio": ratio,
                "corporate_action_source": "yfinance.Ticker.history Stock Splits",
                "price_source": "yfinance",
                "used_for_fill": "no",
                "used_for_mark": "no",
                "notes": (f"Split RECORDED for {sym} ratio={ratio} (audit-only — yfinance "
                          f"prices already split-adjusted; not multiplying shares). "
                          f"Held shares={h.shares:.6f} entry_price={h.entry_price:.4f}"),
            })


def execute_action(
    portfolio: Portfolio,
    date: pd.Timestamp,
    action: Dict[str, Any],
    price_book: PriceBook,
    fills_rows: List[Dict[str, Any]],
    gaps_rows: List[Dict[str, Any]],
    fill_decision_source: str = "",
) -> None:
    """Apply a single decisions.json action at next-session open price."""
    sym = action["symbol"]
    act = action["action"]
    fill_price, source = price_book.get_open(sym, date)
    fill_rule = "next_session_open"
    fallback = ""

    if fill_price is None:
        # Fallback: same-day close if open unavailable
        c, csrc = price_book.get_close(sym, date)
        if c is not None:
            fill_price = c
            source = csrc + "_close_fallback"
            fill_rule = "next_open_unavailable_fallback_close"
            fallback = "open_to_close"
        else:
            # Hard block this row
            gaps_rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "symbol": sym,
                "missing_source": "verified_open_and_close",
                "required_for": f"{act} fill",
                "attempted_lookup": "yfinance + stooq_via_pipeline",
                "fallback_used": "none",
                "can_continue_without_guessing": "yes — block this fill",
                "blocker": "BLOCKED_NO_VERIFIED_PRICE",
            })
            fills_rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "symbol": sym,
                "action": act,
                "decision_source": fill_decision_source,
                "decision_inputs_source": "raw pipeline artifact",
                "target_notional": float(action.get("dollars", 0.0)),
                "target_shares": float(action.get("shares", 0.0)),
                "fill_price_rule": "BLOCKED",
                "fill_price": "",
                "filled_shares": 0,
                "filled_notional": 0,
                "cash_after_fill": round(portfolio.cash, 2),
                "position_after_fill": "",
                "source_evidence": "MISSING — yfinance and Stooq both unavailable",
                "notes": "BLOCKED_NO_VERIFIED_PRICE",
            })
            return

    # Cross-check disagreement
    pct = price_book.disagreement_pct(date, sym, "open")
    if pct is not None and abs(pct) > CROSS_CHECK_THRESHOLD_PCT:
        gaps_rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "symbol": sym,
            "missing_source": "",
            "required_for": f"{act} fill",
            "attempted_lookup": "yfinance vs stooq",
            "fallback_used": "yfinance_primary",
            "can_continue_without_guessing": "yes",
            "blocker": f"CROSS_CHECK_DISAGREEMENT_{pct:.2f}pct",
        })

    if act == "BUY":
        # Independent sizing: use the strategy's target_dollars / verified open
        target_dollars = float(action.get("dollars", 0.0))
        if target_dollars <= 0:
            return
        if target_dollars > portfolio.cash:
            target_dollars = max(0.0, portfolio.cash)
        shares = target_dollars / fill_price if fill_price > 0 else 0.0
        if shares <= 0:
            return
        notional = shares * fill_price
        portfolio.cash -= notional
        h = portfolio.holdings.get(sym)
        if h is None:
            portfolio.holdings[sym] = Holding(
                symbol=sym, shares=shares, entry_price=fill_price,
                entry_date=date.strftime("%Y-%m-%d"),
                asset_class=action.get("asset_class", "equity"),
            )
        else:
            # Average cost basis
            total_cost = h.shares * h.entry_price + notional
            h.shares += shares
            h.entry_price = total_cost / h.shares if h.shares > 0 else fill_price
        fills_rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "symbol": sym,
            "action": act,
            "decision_source": f"daily/{date.strftime('%Y-%m-%d')}/decisions.json",
            "decision_inputs_source": "raw pipeline artifact",
            "target_notional": round(target_dollars, 2),
            "target_shares": float(action.get("shares", 0.0)),
            "fill_price_rule": fill_rule,
            "fill_price": round(fill_price, 6),
            "filled_shares": round(shares, 6),
            "filled_notional": round(notional, 2),
            "cash_after_fill": round(portfolio.cash, 2),
            "position_after_fill": round(portfolio.holdings[sym].shares, 6),
            "source_evidence": f"{source}; decisions.json action",
            "notes": fallback or "",
        })

    elif act == "SELL":
        h = portfolio.holdings.get(sym)
        if h is None or h.shares <= 0:
            fills_rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "symbol": sym, "action": act,
                "decision_source": fill_decision_source,
                "decision_inputs_source": "raw pipeline artifact",
                "target_notional": float(action.get("dollars", 0.0)),
                "target_shares": float(action.get("shares", 0.0)),
                "fill_price_rule": fill_rule,
                "fill_price": round(fill_price, 6),
                "filled_shares": 0,
                "filled_notional": 0,
                "cash_after_fill": round(portfolio.cash, 2),
                "position_after_fill": 0,
                "source_evidence": f"{source}; SELL skipped — not held in independent sim",
                "notes": "NOT_HELD_INDEPENDENT_SIM_DIVERGENCE",
            })
            return
        sell_shares = h.shares  # full liquidation per production SELL semantics
        proceeds = sell_shares * fill_price
        portfolio.cash += proceeds
        del portfolio.holdings[sym]
        fills_rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "symbol": sym, "action": act,
            "decision_source": f"daily/{date.strftime('%Y-%m-%d')}/decisions.json",
            "decision_inputs_source": "raw pipeline artifact",
            "target_notional": float(action.get("dollars", 0.0)),
            "target_shares": float(action.get("shares", 0.0)),
            "fill_price_rule": fill_rule,
            "fill_price": round(fill_price, 6),
            "filled_shares": round(sell_shares, 6),
            "filled_notional": round(proceeds, 2),
            "cash_after_fill": round(portfolio.cash, 2),
            "position_after_fill": 0,
            "source_evidence": f"{source}; decisions.json action",
            "notes": fallback or "",
        })


# ---- Driver ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-date", default=str(SIM_LAST_DAY.date()),
                        help="Last simulated trading day (inclusive)")
    parser.add_argument("--no-prefetch", action="store_true",
                        help="Skip yfinance prefetch (use cache only)")
    args = parser.parse_args()
    end_date = pd.Timestamp(args.end_date)

    starting_state = json.loads(
        (WORK / "2026-03-11_portfolio_state.json").read_text()
    )

    # 1. Build the price book — fetch a wide window so we have everything.
    price_book = PriceBook(PRICE_CACHE)
    universe = sorted(set([h["symbol"] for h in starting_state["holdings"]] + [SPY_BENCHMARK]))

    # We don't yet know the full universe of symbols that will be bought during
    # the simulation. Strategy: expand price book on demand as decisions.json
    # introduces new symbols. Start with what we have plus pre-fetch SPY.
    if not args.no_prefetch:
        for sym in universe:
            price_book.fetch(sym, START_AS_OF, end_date)

    # 2. Sweep through trading days. For each, also pull stooq cross-check.
    spy_cal = price_book.trading_day_calendar(START_AS_OF, end_date)
    if not spy_cal:
        raise SystemExit("BLOCKED_SOURCE_GAP: SPY trading-day calendar empty")
    sim_days = [d for d in spy_cal if d >= SIM_FIRST_DAY and d <= end_date]
    print(f"Simulator window: {sim_days[0].date()} .. {sim_days[-1].date()} "
          f"({len(sim_days)} trading days)")

    # 3. Initialize portfolio at 03-11 EOD
    portfolio, starting_value = build_starting_portfolio(starting_state, price_book)
    print(f"Starting EOD-2026-03-11: cash=${portfolio.cash:,.2f}, "
          f"holdings={len(portfolio.holdings)}, "
          f"total_value=${starting_value:,.2f}")

    # 4. Tables
    daily_rows: List[Dict[str, Any]] = []
    fills_rows: List[Dict[str, Any]] = []
    price_rows: List[Dict[str, Any]] = []
    gaps_rows: List[Dict[str, Any]] = []
    corp_rows: List[Dict[str, Any]] = []

    # 5. Iterate trading days
    last_known_marks: Dict[str, float] = {}
    for sym in list(portfolio.holdings.keys()) + [SPY_BENCHMARK]:
        c, _src = price_book.get_close(sym, START_AS_OF)
        last_known_marks[sym] = c

    trading_day_set = set(sim_days)
    for day in sim_days:
        date_str = day.strftime("%Y-%m-%d")

        # Pull decisions.json for this day, with non-trading-day fallback
        dec, dec_source = find_decisions_for_trading_day(day, trading_day_set)

        # Pull stooq cross-check via daily/<D>/prices.parquet for D-1's close
        # (the night phase writes EOD-(D-1) prices to daily/<D>/prices.parquet).
        # And via daily/<D+1>/prices.parquet to cross-check D's close.
        # We attach both for full coverage.
        prev_day_str = (day - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        for src_date_str, target_date in [(date_str, day - pd.Timedelta(days=1)),
                                          ((day + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), day)]:
            pdf = get_daily_prices_parquet(src_date_str)
            if pdf is not None:
                price_book.attach_stooq(target_date, pdf)

        # Expand price book for any new symbols introduced by today's decisions.
        # Fetch the FULL simulation window so corporate actions (splits,
        # dividends) on later dates are captured. Trigger yfinance fetch
        # if NO yf data exists for symbol (stooq-only is not enough — we need
        # yfinance Stock Splits column for corporate-action detection).
        if dec is not None and not args.no_prefetch:
            for a in dec.get("actions", []):
                sym = a["symbol"]
                rows = price_book._df[(price_book._df["symbol"] == sym)]
                has_yf = (rows["yf_close"].notna().any() if len(rows) > 0 else False)
                if not has_yf:
                    price_book.fetch(sym, day - pd.Timedelta(days=2), end_date)

        # Apply corporate actions for the day to held positions
        apply_corporate_actions(portfolio, day, price_book, corp_rows)

        # Execute actions at next-session open (= day open)
        actions = dec.get("actions", []) if dec else []
        starting_cash = portfolio.cash
        held_syms_before = list(portfolio.holdings.keys())
        # pre-day mark using prior-day close (last_known_marks)
        try:
            starting_total = portfolio.cash + sum(
                h.shares * last_known_marks.get(h.symbol, 0.0)
                for h in portfolio.holdings.values()
            )
        except Exception:
            starting_total = portfolio.cash

        for action in actions:
            execute_action(portfolio, day, action, price_book, fills_rows, gaps_rows,
                           fill_decision_source=dec_source)

        # Compute end-of-day marks for all held + benchmark
        eod_marks: Dict[str, float] = {}
        source_gap_today = 0
        for sym in list(portfolio.holdings.keys()) + [SPY_BENCHMARK]:
            c, csrc = price_book.get_close(sym, day)
            if c is None:
                # Use last-known mark, mark gap
                fallback_mark = last_known_marks.get(sym)
                if fallback_mark is None:
                    raise SystemExit(
                        f"BLOCKED_SOURCE_GAP: no close for {sym} on {date_str} and no prior mark"
                    )
                eod_marks[sym] = fallback_mark
                source_gap_today = 1
                gaps_rows.append({
                    "date": date_str,
                    "symbol": sym,
                    "missing_source": "verified_close",
                    "required_for": "EOD mark",
                    "attempted_lookup": "yfinance + stooq",
                    "fallback_used": "carry_forward_prior_close",
                    "can_continue_without_guessing": "yes (prior verified close)",
                    "blocker": "",
                })
            else:
                eod_marks[sym] = c
                last_known_marks[sym] = c

            # Record price + corp action source row
            yf_o, _ = price_book.get_open(sym, day)
            st_o, _ = (price_book._df[(price_book._df["symbol"] == sym)
                                       & (price_book._df["date"] == day)].iloc[0]["stooq_open"]
                       if not price_book._df[(price_book._df["symbol"] == sym)
                                              & (price_book._df["date"] == day)].empty
                       else None), None
            yf_c, _ = price_book.get_close(sym, day)
            split = price_book.get_yf_split(sym, day)
            price_rows.append({
                "date": date_str,
                "symbol": sym,
                "raw_open": yf_o,
                "raw_close": yf_c,
                "adjusted_open": yf_o,
                "adjusted_close": yf_c,
                "split_ratio": split if split else "",
                "corporate_action_source": "yfinance" if split else "",
                "price_source": "yfinance",
                "used_for_fill": "yes" if (yf_o is not None) else "no",
                "used_for_mark": "yes" if (yf_c is not None) else "no",
                "notes": "",
            })

        ending_total = portfolio.total_value(eod_marks)
        portfolio.peak = max(portfolio.peak, ending_total)
        drawdown = (ending_total / portfolio.peak - 1.0) if portfolio.peak > 0 else 0.0
        benchmark_value = portfolio.benchmark_shares * eod_marks[SPY_BENCHMARK]
        ending_cash = portfolio.cash
        ending_holdings_value = sum(h.shares * eod_marks[h.symbol] for h in portfolio.holdings.values())

        regime = (dec.get("regime") if dec else "")
        sim_buys = sum(1 for f in fills_rows if f["date"] == date_str and f["action"] == "BUY")
        sim_sells = sum(1 for f in fills_rows if f["date"] == date_str and f["action"] == "SELL")
        n_actions = len(actions)
        avail_artifacts = []
        for k in ["decisions.json", "trades.jsonl", "morning_execution.json",
                  "portfolio_state.json", "prices.parquet"]:
            try:
                s3().head_object(Bucket=S3_BUCKET, Key=f"daily/{date_str}/{k}")
                avail_artifacts.append(k)
            except Exception:
                pass

        daily_rows.append({
            "date": date_str,
            "is_trading_day": "yes",
            "starting_cash": round(starting_cash, 2),
            "starting_positions_value": round(starting_total - starting_cash, 2),
            "starting_total_value": round(starting_total, 2),
            "regime": regime,
            "available_source_artifacts": ",".join(avail_artifacts),
            "simulated_actions_count": n_actions,
            "simulated_buys_count": sim_buys,
            "simulated_sells_count": sim_sells,
            "ending_cash": round(ending_cash, 2),
            "ending_positions_value": round(ending_holdings_value, 2),
            "ending_total_value": round(ending_total, 2),
            "benchmark_value": round(benchmark_value, 2),
            "peak": round(portfolio.peak, 2),
            "drawdown": round(drawdown, 6),
            "source_gap": source_gap_today,
            "notes": "",
        })

    # 6. Write tables
    fields = {
        "DAILY_INDEPENDENT_RESIMULATION.tsv": [
            "date", "is_trading_day", "starting_cash", "starting_positions_value",
            "starting_total_value", "regime", "available_source_artifacts",
            "simulated_actions_count", "simulated_buys_count", "simulated_sells_count",
            "ending_cash", "ending_positions_value", "ending_total_value",
            "benchmark_value", "peak", "drawdown", "source_gap", "notes",
        ],
        "SIMULATED_ACTIONS_AND_FILLS.tsv": [
            "date", "symbol", "action", "decision_source", "decision_inputs_source",
            "target_notional", "target_shares", "fill_price_rule", "fill_price",
            "filled_shares", "filled_notional", "cash_after_fill",
            "position_after_fill", "source_evidence", "notes",
        ],
        "PRICE_AND_CORPORATE_ACTION_SOURCES.tsv": [
            "date", "symbol", "raw_open", "raw_close", "adjusted_open", "adjusted_close",
            "split_ratio", "corporate_action_source", "price_source",
            "used_for_fill", "used_for_mark", "notes",
        ],
        "SOURCE_GAPS_AND_BLOCKERS.tsv": [
            "date", "symbol", "missing_source", "required_for", "attempted_lookup",
            "fallback_used", "can_continue_without_guessing", "blocker",
        ],
    }

    def _write(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})

    # Append corp_rows into PRICE_AND_CORPORATE_ACTION_SOURCES.tsv as separate rows
    price_rows.extend(corp_rows)

    _write(TABLES / "DAILY_INDEPENDENT_RESIMULATION.tsv",
           daily_rows, fields["DAILY_INDEPENDENT_RESIMULATION.tsv"])
    _write(TABLES / "SIMULATED_ACTIONS_AND_FILLS.tsv",
           fills_rows, fields["SIMULATED_ACTIONS_AND_FILLS.tsv"])
    _write(TABLES / "PRICE_AND_CORPORATE_ACTION_SOURCES.tsv",
           price_rows, fields["PRICE_AND_CORPORATE_ACTION_SOURCES.tsv"])
    _write(TABLES / "SOURCE_GAPS_AND_BLOCKERS.tsv",
           gaps_rows, fields["SOURCE_GAPS_AND_BLOCKERS.tsv"])

    # Summary
    print("\n=== SIMULATION SUMMARY ===")
    print(f"Trading days simulated: {len(daily_rows)}")
    print(f"Total simulated fills: {len(fills_rows)}")
    print(f"Source gaps: {len(gaps_rows)}")
    print(f"Corporate actions: {len(corp_rows)}")
    last_row = daily_rows[-1]
    print(f"Final ending_total_value: ${last_row['ending_total_value']:,.2f} "
          f"(peak ${last_row['peak']:,.2f}, drawdown {last_row['drawdown']:.4%})")
    print(f"Final benchmark_value: ${last_row['benchmark_value']:,.2f}")

    price_book.persist()


if __name__ == "__main__":
    main()
