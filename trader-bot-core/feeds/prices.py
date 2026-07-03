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

import http.cookiejar
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from io import StringIO
from typing import Dict, List, Optional, Sequence, Tuple

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

_YAHOO_HOSTS = ("query1.finance.yahoo.com", "query2.finance.yahoo.com")
_YAHOO_CHART_BASE = "https://query1.finance.yahoo.com/v8/finance/chart/"
_YAHOO_CRUMB_URL = "https://query1.finance.yahoo.com/v1/test/getcrumb"
# Cookie-seeding endpoints. fc.yahoo.com 404s but still sets the consent/A3
# cookie; finance.yahoo.com is the fallback seed.
_YAHOO_COOKIE_SEEDS = (
    "https://fc.yahoo.com/",
    "https://finance.yahoo.com/quote/SPY",
)
# A browser-ish UA; Yahoo's public chart endpoint 429s an empty/py UA.
_YAHOO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

# Per-container Yahoo session (cookie jar + crumb). Yahoo rate-limits (HTTP 429)
# and rejects (HTTP 401) UNCOOKIED requests to the v8 chart endpoint from
# datacenter/AWS IPs — this is the same AWS-IP block class that made stooq
# unusable, and it is why a *bare* v8 GET fails from the Lambda. The fix is the
# cookie+crumb handshake the yfinance library does internally: seed a cookie,
# fetch a crumb, then send both on every chart request. We do it here with
# urllib (no yfinance dependency). Cached module-globally so the handshake runs
# once per warm container, not once per symbol.
_YAHOO_OPENER: Optional[urllib.request.OpenerDirector] = None
_YAHOO_CRUMB: Optional[str] = None


def _build_yahoo_session(timeout: float) -> Tuple[urllib.request.OpenerDirector, Optional[str]]:
    """Seed a cookie jar and fetch a crumb. Returns (opener, crumb|None)."""
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

    for seed in _YAHOO_COOKIE_SEEDS:
        try:
            req = urllib.request.Request(seed, headers=_YAHOO_HEADERS)
            opener.open(req, timeout=timeout).read()
        except urllib.error.HTTPError:
            # fc.yahoo.com returns 404 but the Set-Cookie still lands in the jar.
            pass
        except (urllib.error.URLError, TimeoutError) as exc:
            logger.info("yahoo_v8: cookie seed %s failed: %s", seed, exc)
            continue
        if len(jar):
            break

    crumb = None
    try:
        req = urllib.request.Request(_YAHOO_CRUMB_URL, headers=_YAHOO_HEADERS)
        crumb = opener.open(req, timeout=timeout).read().decode("utf-8").strip()
        if not crumb or len(crumb) > 32 or "<" in crumb:
            crumb = None  # got an HTML error page, not a crumb token
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        logger.info("yahoo_v8: crumb fetch failed: %s", exc)
        crumb = None
    return opener, crumb


def yahoo_handshake_probe(timeout: float = 15.0) -> dict:
    """Surgical diagnostic: run the cookie+crumb handshake and a single direct SPY
    chart fetch, returning the internals (cookie count, whether a crumb was
    obtained, the chart HTTP status). Answers 'is the raw v8 endpoint reachable
    from here with a proper crumb, or is this a hard IP block?' without log-diving."""
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    seed_status = {}
    for seed in _YAHOO_COOKIE_SEEDS:
        try:
            req = urllib.request.Request(seed, headers=_YAHOO_HEADERS)
            opener.open(req, timeout=timeout).read()
            seed_status[seed] = "ok"
        except urllib.error.HTTPError as exc:
            seed_status[seed] = f"HTTP {exc.code}"
        except Exception as exc:  # noqa: BLE001
            seed_status[seed] = f"{type(exc).__name__}"
    cookie_names = sorted({c.name for c in jar})

    crumb, crumb_status = None, None
    try:
        req = urllib.request.Request(_YAHOO_CRUMB_URL, headers=_YAHOO_HEADERS)
        crumb = opener.open(req, timeout=timeout).read().decode("utf-8").strip()
        crumb_status = "ok"
    except urllib.error.HTTPError as exc:
        crumb_status = f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001
        crumb_status = f"{type(exc).__name__}"

    chart_status, chart_rows = None, None
    try:
        params = {"range": "5d", "interval": "1d"}
        if crumb:
            params["crumb"] = crumb
        url = f"{_YAHOO_CHART_BASE}SPY?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers=_YAHOO_HEADERS)
        payload = json.loads(opener.open(req, timeout=timeout).read().decode("utf-8"))
        res = (payload.get("chart") or {}).get("result") or []
        chart_status = "ok"
        chart_rows = len((res[0].get("timestamp") or [])) if res else 0
    except urllib.error.HTTPError as exc:
        chart_status = f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001
        chart_status = f"{type(exc).__name__}"

    return {
        "cookie_seed_status": seed_status,
        "cookie_names": cookie_names,
        "cookie_count": len(cookie_names),
        "crumb_status": crumb_status,
        "crumb_obtained": bool(crumb and len(crumb) <= 32 and "<" not in crumb),
        "crumb_len": len(crumb) if crumb else 0,
        "direct_spy_chart_status": chart_status,
        "direct_spy_chart_rows": chart_rows,
    }


def _ensure_yahoo_session(timeout: float, force: bool = False):
    """Return the cached (opener, crumb), building it if missing or ``force``."""
    global _YAHOO_OPENER, _YAHOO_CRUMB
    if force or _YAHOO_OPENER is None:
        _YAHOO_OPENER, _YAHOO_CRUMB = _build_yahoo_session(timeout)
        logger.info("yahoo_v8: session (re)built, crumb=%s",
                    "present" if _YAHOO_CRUMB else "MISSING")
    return _YAHOO_OPENER, _YAHOO_CRUMB


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

    payload = None
    for attempt in range(1, max_retries + 1):
        # Force a fresh cookie+crumb handshake on the retry after a 401/429.
        opener, crumb = _ensure_yahoo_session(timeout, force=(attempt > 1))
        params = {"range": rng, "interval": "1d"}
        if crumb:
            params["crumb"] = crumb
        # Alternate query1/query2 across attempts (yfinance does the same — the
        # two hosts have independent rate-limit buckets).
        host = _YAHOO_HOSTS[(attempt - 1) % len(_YAHOO_HOSTS)]
        url = (f"https://{host}/v8/finance/chart/{enc}"
               f"?{urllib.parse.urlencode(params)}")
        try:
            req = urllib.request.Request(url, headers=_YAHOO_HEADERS)
            with opener.open(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            # 401 = crumb/cookie rejected; 429 = rate-limited. Both are defeated
            # by rebuilding the session (next loop forces a fresh handshake).
            if exc.code in (401, 429) and attempt < max_retries:
                backoff = 1.5 * (2 ** (attempt - 1))
                logger.info(
                    "yahoo_v8: HTTP %s for %s (attempt %d/%d), rehandshake+backoff %.1fs",
                    exc.code, symbol, attempt, max_retries, backoff,
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
    """Fetch a CBOE vol-complex index (``^VVIX`` / ``^SKEW`` / ``^VIX``) — the
    resilient replacement for the dead ``fetch_stooq_index``.

    Yahoo v8 raw PRIMARY, yfinance FALLBACK (each logged by name). The fallback
    matters: the Yahoo v8 raw endpoint is 429-rate-limited from AWS datacenter
    IPs, so a Yahoo-only vol source would leave ``vvix``/``skew`` degraded on the
    Lambda exactly as the dead stooq source did — the fallback keeps them nonzero.

    Returns a ``date, symbol, close`` frame (indices have no meaningful volume).
    Empty on total failure.
    """
    df = fetch_yahoo_v8_daily(symbol, lookback_days=lookback_days)
    if len(df) == 0:
        logger.info("vol_index: yahoo_v8 empty for %s -> FALLBACK yfinance", symbol)
        df = fetch_yfinance_daily(symbol, lookback_days=lookback_days)
    if len(df) == 0:
        logger.warning("vol_index: no source returned %s", symbol)
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
    # Runtime-blocked sources: augmented by the circuit-breaker below so a dead
    # primary (e.g. Yahoo hard-429ing this IP) cannot burn the whole 900s budget
    # across 64 symbols before every one of them falls back anyway.
    runtime_blocked = set(blocked)
    report = IngestReport(blocked_sources=sorted(blocked))

    symbols = [u["symbol"] if isinstance(u, dict) else u for u in universe]
    logger.info(
        "[FEEDS] ingesting %d symbols (primary=%s, blocked=%s)",
        len(symbols), SOURCE_YAHOO_V8, sorted(blocked) or "none",
    )

    yahoo_consecutive_fail = 0
    yahoo_break_threshold = 8

    frames: List[pd.DataFrame] = []
    for i, symbol in enumerate(symbols):
        if i > 0 and i % 10 == 0:
            logger.info("[FEEDS] progress %d/%d", i, len(symbols))
        # Gentle pacing so a burst of 64 requests from the single Lambda public
        # IP does not trip Yahoo's rate limiter. ~0.15s * 64 ≈ 10s << 900s.
        if i > 0 and SOURCE_YAHOO_V8 not in runtime_blocked:
            time.sleep(0.15)
        df = _fetch_one(
            symbol, alphavantage_key, lookback_days, critical_symbols,
            runtime_blocked, report,
        )
        if len(df) > 0:
            frames.append(df)

        # Circuit-breaker: if Yahoo primary is enabled but keeps missing, stop
        # hammering it after `yahoo_break_threshold` consecutive misses and route
        # the remaining symbols straight to the fallback chain (logged, honest).
        if SOURCE_YAHOO_V8 not in runtime_blocked:
            if report.served_by.get(symbol) == SOURCE_YAHOO_V8:
                yahoo_consecutive_fail = 0
            else:
                yahoo_consecutive_fail += 1
                if yahoo_consecutive_fail >= yahoo_break_threshold:
                    runtime_blocked.add(SOURCE_YAHOO_V8)
                    report.fallback_events.append(
                        f"CIRCUIT-BREAK: yahoo_v8 disabled after "
                        f"{yahoo_break_threshold} consecutive misses; "
                        f"remaining symbols use fallback chain"
                    )
                    logger.warning(
                        "[FEEDS] CIRCUIT-BREAK: yahoo_v8 disabled after %d "
                        "consecutive misses at symbol #%d (%s); remaining -> fallback",
                        yahoo_break_threshold, i, symbol,
                    )

    report.blocked_sources = sorted(runtime_blocked)
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
