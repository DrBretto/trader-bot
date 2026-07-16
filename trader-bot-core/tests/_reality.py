"""Reality readers + canary predicates for the three-tier reality suite (P7).

LOAD-BEARING RULE (the whole reason this suite exists):
    No function here asserts a value it planted. Every reader pulls from an
    INDEPENDENT reality — the live S3 bucket ``investment-system-data``, the live
    Yahoo v8 chart feed, the live GDELT v2 files, or a DATED REAL RECORDING
    committed under ``tests/fixtures/``. The canary predicates are pure functions
    over a ``reader``. The live/external tests pass the real reader, so a green is
    only possible when independent reality agrees. The fault-injection tests pass
    a ``FrozenStoreReader`` / degraded input, which MUST drive the canary RED —
    that is how we prove the canary catches the real failure modes (the frozen
    OHLCV store / frozen ``mu`` / empty publish that went undetected for two
    weeks), never a planted green.

The old 570-test suite is discarded as the acceptance surface precisely because
it did the opposite: it mocked every input and asserted the failure value
(``gdelt_doc_count == 0``) as correct, so a two-week frozen production line
coexisted with a green suite.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- path bootstrap: the core package dir + the parent repo (some core modules
#     import ``src.utils.*``). Mirrors how the deployed brain resolves imports. ---
_CORE = Path(__file__).resolve().parents[1]          # trader-bot-core/
_REPO = _CORE.parent                                 # trader-bot/
for _p in (str(_CORE), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lines.ledger import (  # noqa: E402
    CACHE_KEY as CANON_CACHE_KEY,
    LEDGER_PREFIX as CANON_CLEAN_PREFIX,
    MANIFEST_KEY as CANON_MANIFEST_KEY,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# --- literal reality endpoints (independent of the production feed code) ---
S3_BUCKET = "investment-system-data"
YAHOO_HOSTS = ("query1.finance.yahoo.com", "query2.finance.yahoo.com")
GDELT_V2_BASE = "http://data.gdeltproject.org/gdeltv2"
_BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
               "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# Canon equity-ledger (the LINE) — import the production pointer instead of
# duplicating a prefix. A canon promotion must move watchdog and canary together.
SHADOW_TS_KEY = "dashboard/shadow_timeseries.json"


class FeedUnreachable(RuntimeError):
    """The independent reality feed could not be reached. Raised (never
    swallowed into a green) so an unreachable feed reports as an ERROR, not a
    pass — the opposite of the theater being removed."""


# --------------------------------------------------------------------------- #
# S3 client (default credential chain: honours AWS_PROFILE / AWS_REGION, exactly
# like the deployed Lambda execution role does).
# --------------------------------------------------------------------------- #
def s3_client():
    import boto3
    return boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))


# --------------------------------------------------------------------------- #
# Independent Yahoo v8 chart reader — deliberately NOT feeds.prices, so the
# external cross-check is an independent path from the code that filled S3.
# --------------------------------------------------------------------------- #
# Yahoo v8 politeness — the public endpoint rate-limits (429) under rapid
# sequential load. We alternate query1/query2 (independent rate buckets) per call
# and throttle to a minimum inter-call spacing so a full-universe sweep completes.
_YAHOO_MIN_INTERVAL = 2.0
_yahoo_state = {"n": 0, "last": 0.0}


def yahoo_v8_daily(symbol: str, rng: str = "1mo", *, timeout: float = 15.0,
                   retries: int = 3) -> List[Tuple[str, float]]:
    """Return [(YYYY-MM-DD, close), ...] for a symbol from the live Yahoo v8
    chart endpoint. Per-call host rotation + politeness throttle + exponential
    backoff on 429. Raises FeedUnreachable on total failure (never returns an
    empty list masquerading as data)."""
    enc = urllib.parse.quote(symbol, safe="")
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        # throttle: keep at least _YAHOO_MIN_INTERVAL between any two calls.
        dt = time.time() - _yahoo_state["last"]
        if dt < _YAHOO_MIN_INTERVAL:
            time.sleep(_YAHOO_MIN_INTERVAL - dt)
        _yahoo_state["n"] += 1
        host = YAHOO_HOSTS[(_yahoo_state["n"] + attempt) % len(YAHOO_HOSTS)]
        url = f"https://{host}/v8/finance/chart/{enc}?range={rng}&interval=1d"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _BROWSER_UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.load(r)
            _yahoo_state["last"] = time.time()
            return _parse_yahoo_v8(payload)
        except Exception as e:  # noqa: BLE001 — surfaced, not swallowed
            last_err = e
            _yahoo_state["last"] = time.time()
            time.sleep(min(3.0 * (2 ** attempt), 12.0))
    raise FeedUnreachable(f"Yahoo v8 unreachable for {symbol}: {last_err}")


def _parse_yahoo_v8(payload: dict) -> List[Tuple[str, float]]:
    import datetime as _dt
    res = payload["chart"]["result"][0]
    ts = res["timestamp"]
    gmt = int(res.get("meta", {}).get("gmtoffset", 0) or 0)
    closes = res["indicators"]["quote"][0]["close"]
    out: List[Tuple[str, float]] = []
    for i in range(len(ts)):
        c = closes[i]
        if c is None:                                    # drop in-progress bars
            continue
        d = _dt.datetime.fromtimestamp(int(ts[i]) + gmt, _dt.timezone.utc).date().isoformat()
        out.append((d, float(c)))
    if not out:
        raise FeedUnreachable("Yahoo v8 payload carried no complete bars")
    return out


# A rate-limit circuit-breaker: once the live feed 429-bans this IP, stop
# hammering it for the rest of the run (the carried CL-299705 self-DoS fix).
_yahoo_breaker = {"tripped": False}
_recorded_cache: Optional[Dict[str, Dict[str, float]]] = None


def _recorded_closes() -> Dict[str, Dict[str, float]]:
    """{symbol: {date: close}} from every committed dated real Yahoo recording
    under fixtures/ (independent DATED REAL captures — not planted by any test)."""
    global _recorded_cache
    if _recorded_cache is not None:
        return _recorded_cache
    out: Dict[str, Dict[str, float]] = {}
    for p in FIXTURES.glob("yahoo_v8_*_recorded.json"):
        try:
            rec = json.loads(p.read_text())
            closes = _parse_yahoo_v8(rec["payload"])
            out.setdefault(rec["symbol"], {}).update({d: c for d, c in closes})
        except Exception:  # noqa: BLE001
            continue
    _recorded_cache = out
    return out


def yahoo_close_on(symbol: str, date: str, rng: str = "3mo",
                   *, allow_recorded: bool = True) -> Optional[float]:
    """The independent Yahoo settled close for ``symbol`` on ``date``. Prefers the
    LIVE v8 feed; on rate-limit/unreachable falls back to a committed DATED REAL
    recording (CL-299705 fix — a cross-check must never become the outage it
    tests for). Returns None if neither the live window nor a recording carries
    that date. Raises FeedUnreachable only if the live feed is down AND no
    recording covers the symbol."""
    if not _yahoo_breaker["tripped"]:
        try:
            for d, c in yahoo_v8_daily(symbol, rng=rng):
                if d == date:
                    return c
            return None                              # live reached, date absent
        except FeedUnreachable:
            _yahoo_breaker["tripped"] = True          # trip: stop hammering
    # live unreachable → dated recording
    if allow_recorded:
        rec = _recorded_closes().get(symbol, {})
        if date in rec:
            return rec[date]
    raise FeedUnreachable(f"live Yahoo rate-limited and no recording for {symbol}@{date}")


# --------------------------------------------------------------------------- #
# Independent GDELT v2 reader — re-fetch the raw GKG files and re-count records,
# an independent path from the ingest that wrote the stored aggregate.
# --------------------------------------------------------------------------- #
_GDELT_HOURS_D4 = ("000000", "060000", "120000", "180000")


def gdelt_v2_gkg_record_count(date: str, hours: Tuple[str, ...] = _GDELT_HOURS_D4,
                              *, timeout: float = 60.0) -> Dict[str, int]:
    """Re-fetch the GDELT v2 GKG zips for the sampled hours of ``date``
    (YYYY-MM-DD) and count records INDEPENDENTLY (raw line count, not the
    ingest's parser). Returns {hours_fetched, records}. Raises FeedUnreachable if
    no hour could be fetched."""
    ymd = date.replace("-", "")
    total = 0
    fetched = 0
    last_err: Optional[Exception] = None
    for hh in hours:
        stamp = f"{ymd}{hh}"
        url = f"{GDELT_V2_BASE}/{stamp}.gkg.csv.zip"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _BROWSER_UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                blob = r.read()
            zf = zipfile.ZipFile(io.BytesIO(blob))
            raw = zf.read(zf.namelist()[0])
            total += sum(1 for line in raw.split(b"\n") if line.strip())
            fetched += 1
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    if fetched == 0:
        raise FeedUnreachable(f"GDELT v2 GKG unreachable for {date}: {last_err}")
    return {"hours_fetched": fetched, "records": total}


# --------------------------------------------------------------------------- #
# The live S3 reader — TODAY's real production output. Every method reads S3;
# nothing is planted.
# --------------------------------------------------------------------------- #
class LiveS3Reader:
    def __init__(self, client=None, bucket: str = S3_BUCKET):
        self._s3 = client or s3_client()
        self.bucket = bucket
        self._settled_dates: Optional[List[str]] = None

    # ---- raw helpers -------------------------------------------------------
    def get_bytes(self, key: str) -> bytes:
        try:
            return self._s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except Exception as e:  # noqa: BLE001
            raise FeedUnreachable(f"S3 get {key} failed: {type(e).__name__}: {e}")

    def get_json(self, key: str) -> dict:
        return json.loads(self.get_bytes(key))

    def exists(self, key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:  # noqa: BLE001
            return False

    # ---- OHLCV substrate ---------------------------------------------------
    def settled_dates(self) -> List[str]:
        """All daily/<D>/ folders that carry a SETTLED prices.parquet bar file,
        oldest→newest. This is the S3-observable OHLCV substrate."""
        if self._settled_dates is not None:
            return self._settled_dates
        dates = set()
        pg = self._s3.get_paginator("list_objects_v2")
        for page in pg.paginate(Bucket=self.bucket, Prefix="daily/", Delimiter="/"):
            for cp in page.get("CommonPrefixes", []) or []:
                d = cp["Prefix"].split("/")[-2]
                if len(d) == 10 and d[4] == "-" and self.exists(f"daily/{d}/prices.parquet"):
                    dates.add(d)
        self._settled_dates = sorted(dates)
        return self._settled_dates

    def latest_settled_date(self) -> str:
        ds = self.settled_dates()
        if not ds:
            raise FeedUnreachable("no settled daily/<D>/prices.parquet in S3")
        return ds[-1]

    def read_prices(self, date: str):
        import pandas as pd
        blob = self.get_bytes(f"daily/{date}/prices.parquet")
        return pd.read_parquet(io.BytesIO(blob))

    def store_watermark(self, symbol: str = "SPY") -> str:
        """Max settled bar date for ``symbol`` inside the latest published
        prices.parquet — the S3-observable OHLCV store watermark the freshness
        gate keys on. Reads real bar dates, never folder labels."""
        import pandas as pd
        df = self.read_prices(self.latest_settled_date())
        s = df[df["symbol"] == symbol]
        if s.empty:
            raise FeedUnreachable(f"{symbol} absent from latest prices.parquet")
        return pd.to_datetime(s["date"]).max().strftime("%Y-%m-%d")

    # ---- forecast fingerprint ---------------------------------------------
    def inference_fingerprint(self, date: str) -> Tuple[bytes, List[float]]:
        """(canonical prediction bytes, numeric vector) for inference.json.

        Volatile metadata such as date and ``recorded_at`` is deliberately
        excluded. Otherwise a frozen prediction gets a fresh timestamp and the
        byte-identity guard reports a false rotation.
        """
        raw = self.get_bytes(f"daily/{date}/inference.json")
        doc = json.loads(raw)
        mu = doc.get("mu")
        if isinstance(mu, dict) and mu:
            clean_mu = {
                str(symbol): float(value)
                for symbol, value in mu.items()
                if isinstance(value, (int, float))
            }
            canonical = json.dumps(
                {"mu": clean_mu}, sort_keys=True, separators=(",", ":")
            ).encode()
            return canonical, [clean_mu[s] for s in sorted(clean_mu)]

        # Legacy artifact compatibility during the transition to clean-core mu
        # records. This branch is also metadata-free.
        health = doc.get("asset_health") or []
        vec: List[float] = []
        for h in health:
            if isinstance(h, dict):
                for k in ("health_score", "health", "score", "value"):
                    if isinstance(h.get(k), (int, float)):
                        vec.append(float(h[k])); break
            elif isinstance(h, (int, float)):
                vec.append(float(h))
        # regime probabilities are a second moving component of the forecast.
        reg = doc.get("regime")
        if isinstance(reg, dict) and isinstance(reg.get("probs"), dict):
            vec.extend(float(v) for v in reg["probs"].values())
        canonical = json.dumps(
            {"asset_health": health, "regime_probs":
             reg.get("probs", {}) if isinstance(reg, dict) else {}},
            sort_keys=True, separators=(",", ":"), default=str,
        ).encode()
        return canonical, vec

    # ---- canon line (the ONLY source of truth for performance) -------------
    def canon_cache_rows(self) -> List[dict]:
        rows = []
        for line in self.get_bytes(CANON_CACHE_KEY).decode().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        rows.sort(key=lambda r: r["date"])
        return rows

    def canon_terminal(self) -> dict:
        rows = self.canon_cache_rows()
        if not rows:
            raise FeedUnreachable("canon ledger cache is empty")
        return rows[-1]

    def portfolio_state(self, date: str) -> dict:
        return self.get_json(f"daily/{date}/portfolio_state.json")

    def shadow_timeseries(self) -> dict:
        return self.get_json(SHADOW_TS_KEY)


# --------------------------------------------------------------------------- #
# Fault-injection reader — a REAL reader whose OHLCV watermark is deliberately
# frozen to a stale date, so the freshness / store-advance canaries MUST go red.
# Everything else delegates to the live reader (real S3).
# --------------------------------------------------------------------------- #
class FrozenStoreReader:
    """Wraps a live reader but reports a frozen OHLCV watermark (default the
    2026-06-10 seed the June freeze stuck at) while a NEWER settled bar genuinely
    exists in S3 — the exact ISSUE-01 frozen-substrate signature."""

    def __init__(self, live: LiveS3Reader, frozen_date: str = "2026-06-10"):
        self._live = live
        self.frozen_date = frozen_date

    def store_watermark(self, symbol: str = "SPY") -> str:
        return self.frozen_date

    def latest_settled_date(self) -> str:
        return self._live.latest_settled_date()

    def __getattr__(self, name):
        return getattr(self._live, name)


# --------------------------------------------------------------------------- #
# Calendar + freshness verdict (reuse the PRODUCTION gate logic verbatim).
# --------------------------------------------------------------------------- #
def expected_settled_day(today: Optional[str] = None) -> str:
    from decide.freshness_gate import _latest_settled_trading_day
    return _latest_settled_trading_day(today)


def reconstruct_canon_terminal(d1: Optional[str] = None) -> dict:
    """Independently RE-EXECUTE the P4/P6 replay marking machinery
    (``Book.value = cash + Σ shares × settled close``) over the real post-split
    settled window, on a fresh in-memory ledger sourcing the real substrate. This
    is the recompute: it reproduces the canon line from live holdings × settled
    marks — nothing is planted. Returns
    {date, recomputed_value, terminal_selected}.
    """
    import boto3
    from replay import seed_canon as SC
    from replay._fake_s3 import FakeS3
    from lines.ledger import EquityLedger

    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    ohlcv = SC.prepare_substrate()
    window = SC.post_split_window(ohlcv, d1)
    reg_fn = SC.recorded_regime_fn(window, s3=s3)
    prod = EquityLedger(s3, prefix=SC.PROD_PREFIX)
    proof = EquityLedger(FakeS3(), prefix=SC.CLEAN_PREFIX)
    SC.copy_presplit(prod, proof, SC.SPLIT_DATE)
    res = SC.seed_canon_by_replay(proof, ohlcv, d1=d1, regime_fn=reg_fn)
    term = res.days[-1]
    return {"date": term.date, "recomputed_value": float(res.terminal_leaf["value"]),
            "terminal_selected": list(term.canon_selected)}


def freshness_verdict(reader, *, today: Optional[str] = None,
                      inject_frozen: bool = False) -> dict:
    """Run the PRODUCTION freshness gate (store.ohlcv_store._freshness_gate_verdict)
    against the reader's S3-observed watermark. When inject_frozen, the reader's
    watermark is the stale seed while a newer settled bar exists in S3 → the gate
    MUST return stale=True (the frozen-mu catch)."""
    from store.ohlcv_store import _freshness_gate_verdict
    run_date = expected_settled_day(today)
    ohlcv_max = reader.store_watermark()
    newest_settled = reader.latest_settled_date()      # real newest settled bar in S3
    if inject_frozen:
        fresh = {"max_available_bar": newest_settled, "bars_added": 0,
                 "gap": [newest_settled], "gap_settled": [newest_settled]}
    else:
        # honest live snapshot: the store reached the newest settled bar present.
        fresh = {"max_available_bar": ohlcv_max, "bars_added": 0,
                 "gap": [], "gap_settled": []}
    return _freshness_gate_verdict(run_date, ohlcv_max, fresh)
