"""feeds/prices.py — the rebuilt price feed (Yahoo v8 chart raw endpoint PRIMARY).

Why this exists
---------------
The live ``src/steps/ingest_prices.py`` used **stooq as PRIMARY**. Stooq is
bot-blocked from AWS IPs, so from the Lambda it 404s all 64 symbols; prices only
survived because the yfinance fallback carried the whole universe every night,
and the vol complex (VVIX/SKEW) came from the same dead stooq-index source
(``fetch_stooq_index``) so ``vvix_value`` / ``skew_value`` were degraded.

This module makes the **Yahoo v8 chart raw endpoint** the primary source
(confirmed working from AWS), with stooq/AV/yfinance as *explicit, named*
fallbacks. Output is coerced to the ``OHLCVBar`` contract (coerce-or-RAISE at the
boundary), never a bare-except swallow.

Source order (per symbol)
-------------------------
    1. yahoo_v8     — raw ``query1.finance.yahoo.com/v8/finance/chart`` JSON
    2. yfinance     — the yfinance library (kept as a resilient secondary)
    3. alphavantage — critical symbols only (rate-limited)
    4. stooq        — last resort (AWS-IP-blocked, hence demoted from primary)

Each fallback is logged BY NAME. A transport/HTTP failure in one source falls
through to the next; a *contract* violation (a source returned rows that cannot
be coerced to the OHLCV dtypes) RAISES ``OHLCVContractError`` and does not fall
through — a wrong-dtype feed is a loud bug, not a quiet degrade.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from io import StringIO
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .contracts import OHLCVContractError, coerce_to_ohlcv, concat_ohlcv, empty_ohlcv

logger = logging.getLogger(__name__)

# Source identifiers (used for logging, blocking levers, and served-source
# accounting). Order here is the fallback order.
SOURCE_YAHOO_V8 = "yahoo_v8"
SOURCE_YFINANCE = "yfinance"
SOURCE_ALPHAVANTAGE = "alphavantage"
SOURCE_STOOQ = "stooq"

# Symbols worth spending an Alpha Vantage rate-limited call on if the free
# sources all miss. (Same critical set the live path used.)
DEFAULT_CRITICAL_SYMBOLS = frozenset(
    {"SPY", "QQQ", "IWM", "TLT", "IEF", "HYG", "LQD", "GLD", "VIXY"}
)

_YAHOO_CHART_BASE = "https://query1.finance.yahoo.com/v8/finance/chart/"
# A browser-ish UA; Yahoo's public chart endpoint 429s an empty/py UA.
_YAHOO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
}


def _yahoo_range_for_lookback(lookback_days: int) -> str:
    """Pick the smallest Yahoo ``range`` token that safely covers ``lookback_days``
    (with buffer for weekends/holidays)."""
    if lookback_days <= 5:
        return "5d"
    if lookback_days <= 25:
        return "1mo"
    if lookback_days <= 80:
        return "3mo"
    if lookback_days <= 170:
        return "6mo"
    if lookback_days <= 360:
        return "1y"
    if lookback_days <= 720:
        return "2y"
    return "5y"


def fetch_yahoo_v8_daily(
    symbol: str,
    lookback_days: int = 365,
    timeout: float = 12.0,
    max_retries: int = 3,
) -> pd.DataFrame:
    """PRIMARY: fetch daily OHLCV from the Yahoo v8 chart raw endpoint.

    Parses the JSON directly (no ``yfinance`` dependency). Returns an
    OHLCV-contracted frame (``coerce_to_ohlcv``) filtered to the last
    ``lookback_days``. Transport/HTTP/parse-shape failures return an EMPTY frame
    (so the caller can fall back); a genuine dtype-contract violation propagates
    as ``OHLCVContractError``.

    HTTP 429 (Yahoo rate-limit — a real risk when a single Lambda public IP fires
    64 requests in a burst) is retried with exponential backoff up to
    ``max_retries`` before giving up to the fallback chain.
    """
    rng = _yahoo_range_for_lookback(lookback_days)
    # ^-prefixed index symbols (^VVIX/^SKEW/^VIX) must be percent-encoded.
    enc = symbol.replace("^", "%5E")
    url = f"{_YAHOO_CHART_BASE}{enc}?range={rng}&interval=1d"

    payload = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=_YAHOO_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < max_retries:
                backoff = 1.5 * (2 ** (attempt - 1))
                logger.info(
                    "yahoo_v8: 429 for %s (attempt %d/%d), backing off %.1fs",
                    symbol, attempt, max_retries, backoff,
                )
                time.sleep(backoff)
                continue
            logger.warning("yahoo_v8: HTTP %s for %s: %s", exc.code, symbol, exc)
            return empty_ohlcv()
        except (urllib.error.URLError, TimeoutError,
                json.JSONDecodeError, ValueError) as exc:
            logger.warning("yahoo_v8: transport/parse failure for %s: %s", symbol, exc)
            return empty_ohlcv()
    if payload is None:
        logger.warning("yahoo_v8: exhausted retries for %s", symbol)
        return empty_ohlcv()

    chart = (payload or {}).get("chart") or {}
    err = chart.get("error")
    if err:
        logger.warning("yahoo_v8: API error for %s: %s", symbol, err)
        return empty_ohlcv()
    results = chart.get("result") or []
    if not results:
        logger.warning("yahoo_v8: no result rows for %s", symbol)
        return empty_ohlcv()

    r = results[0]
    timestamps = r.get("timestamp") or []
    meta = r.get("meta") or {}
    quote_blocks = ((r.get("indicators") or {}).get("quote") or [{}])
    quote = quote_blocks[0] if quote_blocks else {}
    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []

    if not timestamps or not closes:
        logger.warning("yahoo_v8: empty series for %s", symbol)
        return empty_ohlcv()

    # Convert epoch -> exchange-local trading DATE (tz-naive). Daily bars are
    # stamped at the session open; adding the exchange gmtoffset then taking the
    # date yields the correct trading day regardless of UTC rollover.
    gmtoffset = int(meta.get("gmtoffset") or 0)

    records = []
    n = len(timestamps)
    for i in range(n):
        c = closes[i] if i < len(closes) else None
        o = opens[i] if i < len(opens) else None
        h = highs[i] if i < len(highs) else None
        low = lows[i] if i < len(lows) else None
        v = volumes[i] if i < len(volumes) else None
        # Drop incomplete bars (Yahoo emits a null-close row for an in-progress
        # session). These are not contract violations — they are simply not
        # settled bars yet.
        if c is None or o is None or h is None or low is None or v is None:
            continue
        d = pd.Timestamp(timestamps[i] + gmtoffset, unit="s").normalize()
        records.append(
            {
                "date": d,
                "symbol": symbol,
                "open": o,
                "high": h,
                "low": low,
                "close": c,
                "volume": v,
            }
        )

    if not records:
        logger.warning("yahoo_v8: all bars incomplete for %s", symbol)
        return empty_ohlcv()

    raw = pd.DataFrame.from_records(records)
    # Contract boundary — coerce-or-RAISE. A dtype problem here is a loud bug.
    df = coerce_to_ohlcv(raw).sort_values("date")

    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=lookback_days)
    df = df[df["date"] >= cutoff]
    return df.reset_index(drop=True)


def fetch_yfinance_daily(symbol: str, lookback_days: int = 365) -> pd.DataFrame:
    """FALLBACK 1: daily OHLCV via the yfinance library."""
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not available for daily fallback on %s", symbol)
        return empty_ohlcv()

    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period="2y", interval="1d", auto_adjust=False)
    except Exception as exc:  # noqa: BLE001 — transport-layer; fall through
        logger.warning("yfinance: fetch failed for %s: %s", symbol, exc)
        return empty_ohlcv()

    if history is None or len(history) == 0:
        logger.warning("yfinance: no daily history for %s", symbol)
        return empty_ohlcv()

    history = history.reset_index()
    history.columns = [str(c).lower() for c in history.columns]
    if "date" not in history.columns and "datetime" in history.columns:
        history = history.rename(columns={"datetime": "date"})
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(history.columns):
        logger.warning(
            "yfinance: missing columns for %s: %s",
            symbol, sorted(required - set(history.columns)),
        )
        return empty_ohlcv()

    history["symbol"] = symbol
    df = coerce_to_ohlcv(history).sort_values("date")
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=lookback_days)
    return df[df["date"] >= cutoff].reset_index(drop=True)


def fetch_alphavantage_daily(symbol: str, api_key: str) -> pd.DataFrame:
    """FALLBACK 2: daily OHLCV via Alpha Vantage (critical symbols; rate-limited)."""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": api_key,
        "outputsize": "full",
    }
    try:
        response = requests_get(url, params=params, timeout=15)
        data = response.json()
    except Exception as exc:  # noqa: BLE001 — transport-layer; fall through
        logger.warning("alphavantage: fetch failed for %s: %s", symbol, exc)
        return empty_ohlcv()

    ts = data.get("Time Series (Daily)") if isinstance(data, dict) else None
    if not ts:
        if isinstance(data, dict) and ("Note" in data or "Information" in data):
            logger.warning("alphavantage: rate limited for %s", symbol)
        else:
            logger.warning("alphavantage: no data for %s", symbol)
        return empty_ohlcv()

    records = []
    for date_str, values in ts.items():
        records.append(
            {
                "date": date_str,
                "symbol": symbol,
                "open": values["1. open"],
                "high": values["2. high"],
                "low": values["3. low"],
                "close": values["4. close"],
                "volume": values["5. volume"],
            }
        )
    df = coerce_to_ohlcv(pd.DataFrame(records)).sort_values("date")
    return df.tail(365).reset_index(drop=True)


def fetch_stooq_daily(symbol: str, lookback_days: int = 365) -> pd.DataFrame:
    """FALLBACK 3: daily OHLCV via stooq CSV. AWS-IP-blocked (why it is demoted)."""
    stooq_symbol = f"{symbol}.US"
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
    try:
        response = requests_get(url, timeout=(3, 8))
        response.raise_for_status()
        text = response.text
    except Exception as exc:  # noqa: BLE001 — transport-layer; fall through
        logger.warning("stooq: fetch failed for %s: %s", symbol, exc)
        return empty_ohlcv()

    if len(text) < 50 or "No data" in text or "Exceeded" in text:
        logger.warning("stooq: no data / blocked for %s", symbol)
        return empty_ohlcv()

    try:
        df = pd.read_csv(StringIO(text))
    except Exception as exc:  # noqa: BLE001 — malformed body; treat as miss
        logger.warning("stooq: unparseable CSV for %s: %s", symbol, exc)
        return empty_ohlcv()

    df.columns = df.columns.str.lower()
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        logger.warning("stooq: missing columns for %s", symbol)
        return empty_ohlcv()

    df["symbol"] = symbol
    out = coerce_to_ohlcv(df).sort_values("date")
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=lookback_days)
    return out[out["date"] >= cutoff].reset_index(drop=True)


# --- vol complex (VVIX / SKEW) -------------------------------------------------

def fetch_vol_index(symbol: str, lookback_days: int = 365) -> pd.DataFrame:
    """Fetch a CBOE vol-complex index (``^VVIX`` / ``^SKEW`` / ``^VIX``) via the
    Yahoo v8 endpoint — the working replacement for the dead ``fetch_stooq_index``.

    Returns a ``date, symbol, close`` frame (indices have no meaningful volume).
    Empty on failure.
    """
    df = fetch_yahoo_v8_daily(symbol, lookback_days=lookback_days)
    if len(df) == 0:
        logger.warning("vol_index: yahoo_v8 empty for %s", symbol)
        return pd.DataFrame(columns=["date", "symbol", "close"])
    return df[["date", "symbol", "close"]].reset_index(drop=True)


def latest_vol_index(symbol: str) -> Optional[float]:
    """Latest close for a vol-complex index, or ``None`` if unavailable."""
    df = fetch_vol_index(symbol, lookback_days=30)
    if len(df) == 0:
        return None
    return float(df.sort_values("date")["close"].iloc[-1])


# --- orchestration -------------------------------------------------------------

@dataclass
class IngestReport:
    """Per-run accounting so a diag can SEE which source served each symbol and
    where fallbacks fired — the 'fallback taken visibly' evidence."""

    served_by: Dict[str, str] = field(default_factory=dict)      # symbol -> source
    fallback_events: List[str] = field(default_factory=list)      # human-readable
    failed_symbols: List[str] = field(default_factory=list)
    source_counts: Dict[str, int] = field(default_factory=dict)
    blocked_sources: List[str] = field(default_factory=list)

    def note_served(self, symbol: str, source: str) -> None:
        self.served_by[symbol] = source
        self.source_counts[source] = self.source_counts.get(source, 0) + 1


def _fetch_one(
    symbol: str,
    alphavantage_key: Optional[str],
    lookback_days: int,
    critical_symbols: frozenset,
    blocked: set,
    report: IngestReport,
) -> pd.DataFrame:
    """Run the fallback chain for one symbol, logging each fallback by name."""

    # 1) PRIMARY: Yahoo v8
    if SOURCE_YAHOO_V8 not in blocked:
        df = fetch_yahoo_v8_daily(symbol, lookback_days)
        if len(df) > 0:
            report.note_served(symbol, SOURCE_YAHOO_V8)
            return df
    else:
        logger.info("[FEEDS] %s: %s blocked (diag), skipping primary",
                    symbol, SOURCE_YAHOO_V8)

    # 2) FALLBACK: yfinance
    if SOURCE_YFINANCE not in blocked:
        logger.info("[FEEDS] %s: primary miss -> FALLBACK %s", symbol, SOURCE_YFINANCE)
        report.fallback_events.append(f"{symbol}: yahoo_v8 -> yfinance")
        df = fetch_yfinance_daily(symbol, lookback_days)
        if len(df) > 0:
            report.note_served(symbol, SOURCE_YFINANCE)
            return df
    else:
        logger.info("[FEEDS] %s: %s blocked (diag), skipping", symbol, SOURCE_YFINANCE)

    # 3) FALLBACK: Alpha Vantage (critical only)
    if (
        SOURCE_ALPHAVANTAGE not in blocked
        and symbol in critical_symbols
        and alphavantage_key
    ):
        logger.info("[FEEDS] %s: FALLBACK %s (critical)", symbol, SOURCE_ALPHAVANTAGE)
        report.fallback_events.append(f"{symbol}: -> alphavantage")
        df = fetch_alphavantage_daily(symbol, alphavantage_key)
        time.sleep(1)  # AV rate limit
        if len(df) > 0:
            report.note_served(symbol, SOURCE_ALPHAVANTAGE)
            return df

    # 4) FALLBACK: stooq (AWS-IP-blocked last resort)
    if SOURCE_STOOQ not in blocked:
        logger.info("[FEEDS] %s: FALLBACK %s (last resort)", symbol, SOURCE_STOOQ)
        report.fallback_events.append(f"{symbol}: -> stooq")
        df = fetch_stooq_daily(symbol, lookback_days)
        if len(df) > 0:
            report.note_served(symbol, SOURCE_STOOQ)
            return df
    else:
        logger.info("[FEEDS] %s: %s blocked (diag), skipping", symbol, SOURCE_STOOQ)

    report.failed_symbols.append(symbol)
    return empty_ohlcv()


def run_with_report(
    universe: Sequence,
    alphavantage_key: Optional[str] = None,
    lookback_days: int = 365,
    blocked_sources: Optional[Sequence[str]] = None,
    critical_symbols: frozenset = DEFAULT_CRITICAL_SYMBOLS,
) -> tuple[pd.DataFrame, IngestReport]:
    """Ingest the whole universe and return ``(ohlcv_df, report)``.

    ``blocked_sources`` is a diag lever: naming a source (e.g. ``["stooq"]`` or
    ``["yahoo_v8"]``) skips it, so a reality-test can prove the fallback path
    fires without a real outage. Output is OHLCV-contracted (concat re-asserts
    the contract; no raw ``pd.concat``).
    """
    blocked = {s for s in (blocked_sources or [])}
    report = IngestReport(blocked_sources=sorted(blocked))

    symbols = [u["symbol"] if isinstance(u, dict) else u for u in universe]
    logger.info(
        "[FEEDS] ingesting %d symbols (primary=%s, blocked=%s)",
        len(symbols), SOURCE_YAHOO_V8, sorted(blocked) or "none",
    )

    frames: List[pd.DataFrame] = []
    for i, symbol in enumerate(symbols):
        if i > 0 and i % 10 == 0:
            logger.info("[FEEDS] progress %d/%d", i, len(symbols))
        # Gentle pacing so a burst of 64 requests from the single Lambda public
        # IP does not trip Yahoo's rate limiter. ~0.15s * 64 ≈ 10s << 900s.
        if i > 0 and SOURCE_YAHOO_V8 not in blocked:
            time.sleep(0.15)
        df = _fetch_one(
            symbol, alphavantage_key, lookback_days, critical_symbols, blocked, report
        )
        if len(df) > 0:
            frames.append(df)

    result = concat_ohlcv(frames)
    logger.info(
        "[FEEDS] done: %d rows, %d/%d symbols non-empty (by source: %s); failed=%s",
        len(result),
        result["symbol"].nunique() if len(result) else 0,
        len(symbols),
        report.source_counts,
        report.failed_symbols[:10],
    )
    return result, report


def run(
    universe: Sequence,
    alphavantage_key: Optional[str] = None,
    lookback_days: int = 365,
) -> pd.DataFrame:
    """Public API mirroring the live ``ingest_prices.run`` signature: returns the
    combined OHLCV frame for the whole universe (Yahoo v8 primary + fallbacks)."""
    df, _report = run_with_report(
        universe, alphavantage_key=alphavantage_key, lookback_days=lookback_days
    )
    return df


# Thin ``requests``-style shim kept local so the module has no hard dependency on
# ``requests`` for its PRIMARY path (Yahoo v8 uses urllib). yfinance/AV/stooq
# fallbacks do use ``requests`` when present.
def requests_get(url, params=None, timeout=15):  # pragma: no cover - thin shim
    import requests
    return requests.get(url, params=params, timeout=timeout)
