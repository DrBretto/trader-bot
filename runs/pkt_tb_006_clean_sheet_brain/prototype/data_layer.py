"""PKT-TB-006 SYN-1 prototype — data layer (BUILD_SPEC §2.1/§2.3/§2.4, §15 steps 2-3).

Implements:
  - DiskCachedS3Cache : drop-in for src/utils/three_line_replay/replay_engine.py S3Cache,
                        backed by a disk cache under cache/s3/.
  - SnapshotStore     : pulls the replay-window daily/<D>/ artifacts (2026-01-30..2026-06-10).
  - load_deep_history(): deep panels from training/data/*.parquet.
  - fetch_ohlcv_deep(): full-history daily OHLCV (raw + adj) for the 64-symbol universe,
                        yfinance bulk primary, Stooq per-symbol fallback. Split events recorded.
  - fetch_cboe()      : S1 CBOE index histories (clean date/close series).
  - fetch_fred()      : S3 FRED series, full history, with visible_from publication-lag column.
  - fetch_cot()       : S2 CFTC TFF (Socrata gpe5-46if) net positioning %OI, keyed by
                        publication Friday; visible_from = next trading day after publication.

Write surface: everything lands under runs/pkt_tb_006_clean_sheet_brain/prototype/cache/.
No production paths are touched. All look-ahead discipline is encoded as data
(visible_from columns) — no feature logic lives here.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

PROTO = Path(__file__).resolve().parent
REPO = PROTO.parents[2]  # runs/pkt_tb_006_clean_sheet_brain/prototype -> repo root
CACHE = PROTO / "cache"
BUCKET = "investment-system-data"
REGION = "us-east-1"
AWS_PROFILE = os.environ.get("AWS_PROFILE", "personal")

WINDOW_START = "2026-01-30"
WINDOW_END = "2026-06-10"
SNAPSHOT_FILES = [
    "prices.parquet",
    "context.parquet",
    "features.parquet",
    "portfolio_state.json",
    "morning_prices.parquet",  # present only on days the morning run wrote it
]

CBOE_INDICES = ["VIX", "VIX9D", "VIX3M", "VVIX", "SKEW", "COR3M", "VXN"]
CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{idx}_History.csv"

# S3 FRED edge series + context-rebuild series (BUILD_SPEC §2.3 / §2.4).
FRED_SERIES = [
    "NFCI", "STLFSI4", "T10YIE", "DFII10", "BAMLH0A0HYM2", "ICSA",
    "DGS2", "DGS3MO", "DGS10", "VIXCLS", "DCOILWTICO", "DEXUSEU",
]
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_SECRET_ID = "investment-system/fred-key"

# Publication-lag rules (BUILD_SPEC §2.3): weekly indices keyed to the named weekday
# FOLLOWING the observation date; ICSA 1-week lag; daily series lagged 1 day.
# Encoded as visible_from = first decision date that may see the row.
_WEEKDAY = {"WED": 2, "THU": 3}
FRED_LAG_RULES: Dict[str, Any] = {
    "NFCI": ("following_weekday", "WED"),
    "STLFSI4": ("following_weekday", "THU"),
    "ICSA": ("days", 7),
    # all remaining (daily) series: ("days", 1) — applied as default
}

COT_DATASET = "gpe5-46if"  # "TFF - Futures Only" on publicreporting.cftc.gov (Socrata)
COT_URL = f"https://publicreporting.cftc.gov/resource/{COT_DATASET}.json"
COT_MARKETS = {
    "es": "E-MINI S&P 500",
    "ust10y": "UST 10Y NOTE",
    "vix": "VIX FUTURES",
}

UNIVERSE_CSV = REPO / "config" / "universe.csv"
DEEP_FEATURES = REPO / "training" / "data" / "asset_features_history.parquet"
DEEP_CONTEXT = REPO / "training" / "data" / "historical_context.parquet"
DEEP_GDELT = REPO / "training" / "data" / "historical_gdelt.parquet"

CROSSCHECK_SEED = 4242  # registered slippage seed reused for determinism


# --------------------------------------------------------------------------- utils

def _boto_session():
    import boto3
    return boto3.Session(profile_name=AWS_PROFILE, region_name=REGION)


def universe_symbols() -> List[str]:
    df = pd.read_csv(UNIVERSE_CSV)
    return df["symbol"].astype(str).str.strip().tolist()


def _next_trading_day(ts: pd.Timestamp) -> pd.Timestamp:
    """Next US business day strictly after ts (federal-holiday calendar; NYSE-specific
    holidays like Good Friday are NOT in this calendar — approximation, flagged in report)."""
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from pandas.tseries.offsets import CustomBusinessDay
    bday = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    return (pd.Timestamp(ts).normalize() + bday).normalize()


def _following_weekday(d: pd.Timestamp, weekday: int) -> pd.Timestamp:
    """First date with .weekday()==weekday strictly after d."""
    d = pd.Timestamp(d).normalize()
    delta = (weekday - d.weekday() - 1) % 7 + 1
    return d + pd.Timedelta(days=delta)


# --------------------------------------------------------------------------- S3 layer

class DiskCachedS3Cache:
    """Drop-in replacement for replay_engine.S3Cache backed by a disk cache.

    Interface parity with S3Cache: get / get_json / get_parquet / get_csv /
    list_daily_dates. Reads hit (memory -> disk -> S3); S3 fetches are persisted
    to cache_dir/<key>. Works fully offline once the window is synced
    (pass s3_client=None).
    """

    def __init__(self, s3_client=None, bucket: str = BUCKET,
                 cache_dir: Path = CACHE / "s3"):
        self.s3 = s3_client
        self.bucket = bucket
        self.cache_dir = Path(cache_dir)
        self._mem: Dict[str, bytes] = {}

    def _disk_path(self, key: str) -> Path:
        return self.cache_dir / key

    def get(self, key: str) -> bytes:
        if key in self._mem:
            return self._mem[key]
        p = self._disk_path(key)
        if p.exists():
            data = p.read_bytes()
        else:
            if self.s3 is None:
                raise FileNotFoundError(f"{key} not in disk cache and no s3_client")
            data = self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        self._mem[key] = data
        return data

    def get_json(self, key: str) -> Any:
        return json.loads(self.get(key).decode())

    def get_parquet(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(io.BytesIO(self.get(key)))

    def get_csv(self, key: str) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(self.get(key)))

    def list_daily_dates(self) -> List[str]:
        dates = set()
        daily_dir = self.cache_dir / "daily"
        if daily_dir.exists():
            for d in daily_dir.iterdir():
                if d.is_dir() and d.name.startswith("2026-"):
                    dates.add(d.name)
        if self.s3 is not None:
            paginator = self.s3.get_paginator("list_objects_v2")
            for p in paginator.paginate(Bucket=self.bucket, Prefix="daily/", Delimiter="/"):
                for cp in p.get("CommonPrefixes", []) or []:
                    d = cp["Prefix"].split("/")[-2]
                    if d.startswith("2026-"):
                        dates.add(d)
        return sorted(dates)


class SnapshotStore:
    """Downloads + disk-caches the replay-window daily/<D>/ artifacts."""

    def __init__(self, s3_client=None, bucket: str = BUCKET,
                 cache_dir: Path = CACHE / "s3"):
        if s3_client is None:
            s3_client = _boto_session().client("s3")
        self.cache = DiskCachedS3Cache(s3_client, bucket, cache_dir)
        self.s3 = s3_client
        self.bucket = bucket

    def _list_window_keys(self, start: str, end: str) -> Dict[str, List[str]]:
        """Map date-dir -> list of files present in S3 within [start, end]."""
        paginator = self.s3.get_paginator("list_objects_v2")
        out: Dict[str, List[str]] = {}
        for page in paginator.paginate(Bucket=self.bucket, Prefix="daily/"):
            for obj in page.get("Contents", []) or []:
                parts = obj["Key"].split("/")
                if len(parts) != 3:
                    continue
                d, fname = parts[1], parts[2]
                if start <= d <= end and fname in SNAPSHOT_FILES:
                    out.setdefault(d, []).append(fname)
        return out

    def sync(self, start: str = WINDOW_START, end: str = WINDOW_END,
             verbose: bool = True) -> Dict[str, Any]:
        """Pull every SNAPSHOT_FILES artifact for daily/<D>/ in window; skip-existing.
        Returns a manifest {date: {file: 'cached'|'downloaded'|'absent_in_s3'}}."""
        window = self._list_window_keys(start, end)
        manifest: Dict[str, Any] = {}
        n_dl = n_skip = 0
        for d in sorted(window):
            manifest[d] = {}
            for fname in SNAPSHOT_FILES:
                key = f"daily/{d}/{fname}"
                p = self.cache._disk_path(key)
                if fname not in window[d]:
                    manifest[d][fname] = "absent_in_s3"
                    continue
                if p.exists():
                    manifest[d][fname] = "cached"
                    n_skip += 1
                    continue
                self.cache.get(key)
                manifest[d][fname] = "downloaded"
                n_dl += 1
        (self.cache.cache_dir / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
        (self.cache.cache_dir / "manifest.json").write_text(
            json.dumps({"start": start, "end": end, "dates": manifest}, indent=1))
        if verbose:
            n_prices = sum(1 for d in manifest if manifest[d].get("prices.parquet") in ("cached", "downloaded"))
            print(f"SnapshotStore.sync: {len(manifest)} date dirs, {n_prices} with prices.parquet; "
                  f"{n_dl} downloaded, {n_skip} already cached")
        return manifest


# --------------------------------------------------------------------------- deep history

def load_deep_history(include_gdelt: bool = True) -> Dict[str, pd.DataFrame]:
    """Tidy deep panels from training/data/ (read-only).

    Returns:
      asset_features: [date, symbol, close, 12 engineered cols] 2014-08-29..2026-06-05
      context:        [date, 9 context cols] 2014-12-10..2026-02-03
      gdelt:          [date, gdelt_* cols] (reference only)
    """
    out: Dict[str, pd.DataFrame] = {}
    af = pd.read_parquet(DEEP_FEATURES)
    af["date"] = pd.to_datetime(af["date"])
    out["asset_features"] = af.sort_values(["symbol", "date"]).reset_index(drop=True)

    cx = pd.read_parquet(DEEP_CONTEXT)
    if "date" not in cx.columns:
        cx = cx.reset_index().rename(columns={cx.index.name or "index": "date"})
    cx["date"] = pd.to_datetime(cx["date"])
    out["context"] = cx.sort_values("date").reset_index(drop=True)

    if include_gdelt and DEEP_GDELT.exists():
        gd = pd.read_parquet(DEEP_GDELT)
        if "date" not in gd.columns:
            gd = gd.reset_index().rename(columns={gd.index.name or "index": "date"})
        gd["date"] = pd.to_datetime(gd["date"])
        out["gdelt"] = gd.sort_values("date").reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- OHLCV deep

def _fetch_stooq_symbol(sym: str, timeout: int = 30) -> Optional[pd.DataFrame]:
    url = f"https://stooq.com/q/d/l/?s={sym.lower()}.US&i=d"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200 or not r.text or r.text.strip().lower().startswith(("<", "no data")):
            return None
        df = pd.read_csv(io.StringIO(r.text))
        if "Date" not in df.columns or len(df) == 0:
            return None
        df = df.rename(columns={c: c.lower() for c in df.columns})
        df["date"] = pd.to_datetime(df["date"])
        df["adj_close"] = np.nan  # stooq daily CSV is unadjusted-for-dividends; no adj series
        df["source"] = "stooq"
        return df[["date", "open", "high", "low", "close", "adj_close", "volume", "source"]]
    except Exception:
        return None


def fetch_ohlcv_deep(start: str = "2014-01-01", out_dir: Path = CACHE / "ohlcv",
                     snapshot_cache: Optional[DiskCachedS3Cache] = None,
                     force: bool = False) -> Dict[str, Any]:
    """Full-history daily OHLCV for the 64-symbol universe.

    Primary: yfinance bulk download, auto_adjust=False (raw split-adjusted OHLC + Adj Close
    -> both close and adj_close stored; adj_factor = adj_close/close).
    Fallback: Stooq per symbol. Writes cache/ohlcv/<SYMBOL>.parquet and
    cache/ohlcv/coverage_report.json (first/last/rows, splits per symbol, source,
    S3 cross-check results).

    HONESTY NOTE recorded in the report: yfinance prices are retroactively
    SPLIT-adjusted (VUG 6:1 on 2026-04-20), while S3 daily snapshots are NOT
    retro-adjusted (harness applies the split itself). Split events per symbol are
    written so the feature builder can align panels with snapshots.
    """
    import yfinance as yf

    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = universe_symbols()
    coverage: Dict[str, Any] = {}
    need = [s for s in symbols if force or not (out_dir / f"{s}.parquet").exists()]

    if need:
        bulk = yf.download(need, start=start, auto_adjust=False, actions=False,
                           group_by="ticker", threads=True, progress=False)
        for sym in need:
            df = None
            try:
                sub = bulk[sym].dropna(how="all") if len(need) > 1 else bulk.dropna(how="all")
                if len(sub) > 0:
                    df = sub.reset_index()
                    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
                    df = df.rename(columns={"adj_close": "adj_close"})
                    df["source"] = "yfinance"
                    df = df[["date", "open", "high", "low", "close", "adj_close", "volume", "source"]]
            except Exception:
                df = None
            if df is None or len(df) == 0:
                df = _fetch_stooq_symbol(sym)
                if df is not None:
                    df = df[df["date"] >= pd.Timestamp(start)]
            if df is None or len(df) == 0:
                coverage[sym] = {"status": "MISSING", "source": None}
                continue
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            df = df.sort_values("date").reset_index(drop=True)
            df["adj_factor"] = df["adj_close"] / df["close"]
            df.to_parquet(out_dir / f"{sym}.parquet", index=False)

    # splits per symbol (yfinance .splits) — needed for snapshot alignment
    splits_all: Dict[str, Dict[str, float]] = {}
    for sym in symbols:
        try:
            sp = yf.Ticker(sym).splits
            sp = sp[sp.index >= pd.Timestamp(start).tz_localize(sp.index.tz)] if len(sp) else sp
            splits_all[sym] = {str(ts.date()): float(v) for ts, v in sp.items()}
        except Exception as e:
            splits_all[sym] = {"_error": str(e)}
        time.sleep(0.05)

    for sym in symbols:
        p = out_dir / f"{sym}.parquet"
        if not p.exists():
            coverage.setdefault(sym, {"status": "MISSING", "source": None})
            continue
        df = pd.read_parquet(p)
        coverage[sym] = {
            "status": "OK",
            "source": str(df["source"].iloc[0]),
            "first": str(df["date"].min().date()),
            "last": str(df["date"].max().date()),
            "rows": int(len(df)),
            "splits_since_start": splits_all.get(sym, {}),
        }

    crosscheck = cross_check_ohlcv_vs_s3(out_dir=out_dir, snapshot_cache=snapshot_cache)
    report = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "start_requested": start,
        "universe_size": len(symbols),
        "missing_symbols": [s for s in symbols if coverage.get(s, {}).get("status") != "OK"],
        "split_adjustment_note": (
            "yfinance OHLC is retroactively split-adjusted; S3 daily snapshots are NOT "
            "(harness applies VUG 6:1 of 2026-04-20 itself). Use splits_since_start to "
            "align this panel with snapshot closes before the split effective date."),
        "s3_crosscheck": crosscheck,
        "symbols": coverage,
    }
    (out_dir / "coverage_report.json").write_text(json.dumps(report, indent=1))
    return report


def cross_check_ohlcv_vs_s3(n: int = 5, seed: int = CROSSCHECK_SEED,
                            out_dir: Path = CACHE / "ohlcv",
                            snapshot_cache: Optional[DiskCachedS3Cache] = None) -> Dict[str, Any]:
    """Compare OHLCV close vs S3 snapshot close for n random (symbol, date-D) pairs,
    2026-02..2026-06, where D's OHLC lives in daily/<D+1>/prices.parquet (we take the
    max date inside each sampled snapshot, which IS D for that dir)."""
    cache = snapshot_cache or DiskCachedS3Cache(s3_client=None)
    daily_dir = cache.cache_dir / "daily"
    dirs = sorted(d.name for d in daily_dir.iterdir()
                  if d.is_dir() and "2026-02-01" <= d.name <= "2026-06-10"
                  and (d / "prices.parquet").exists())
    rng = np.random.default_rng(seed)
    symbols = universe_symbols()
    pairs = []
    for i in rng.choice(len(dirs), size=min(n, len(dirs)), replace=False):
        snap = dirs[int(i)]
        sym = symbols[int(rng.integers(0, len(symbols)))]
        pairs.append((snap, sym))

    results = []
    for snap, sym in pairs:
        prices = cache.get_parquet(f"daily/{snap}/prices.parquet")
        # snapshot dates carry a 04:00 UTC time component — normalize to midnight
        prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
        d_max = prices["date"].max()  # = D, the day before the snapshot dir date
        row = prices[(prices["symbol"] == sym) & (prices["date"] == d_max)]
        rec: Dict[str, Any] = {"snapshot_dir": snap, "symbol": sym, "date_D": str(d_max.date())}
        p = out_dir / f"{sym}.parquet"
        if len(row) == 0 or not p.exists():
            rec["status"] = "no_data"
            results.append(rec)
            continue
        s3_close = float(row["close"].iloc[0])
        ours = pd.read_parquet(p, columns=["date", "close"])
        ours["date"] = pd.to_datetime(ours["date"]).dt.normalize()
        m = ours[ours["date"] == d_max]
        if len(m) == 0:
            rec["status"] = "date_missing_in_ohlcv"
            results.append(rec)
            continue
        our_close = float(m["close"].iloc[0])
        diff = abs(our_close - s3_close) / s3_close
        rec.update({"s3_close": s3_close, "ohlcv_close": our_close,
                    "rel_diff": round(diff, 6),
                    "status": "MATCH" if diff <= 0.005 else "MISMATCH_GT_0.5pct"})
        results.append(rec)
    return {"seed": seed, "pairs": results}


# --------------------------------------------------------------------------- CBOE S1

def fetch_cboe(out_dir: Path = CACHE / "cboe", force: bool = False) -> Dict[str, Any]:
    """S1: clean (date, close) series per CBOE index. Features computed downstream."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {}
    for idx in CBOE_INDICES:
        p = out_dir / f"{idx}.parquet"
        if p.exists() and not force:
            df = pd.read_parquet(p)
        else:
            r = requests.get(CBOE_URL.format(idx=idx), timeout=60)
            if r.status_code != 200:
                report[idx] = {"status": f"HTTP {r.status_code}"}
                continue
            raw = pd.read_csv(io.StringIO(r.text))
            raw.columns = [str(c).strip().lower().lstrip("﻿") for c in raw.columns]
            # most files: DATE,OPEN,HIGH,LOW,CLOSE; VVIX/SKEW: DATE,<IDX>
            value_col = "close" if "close" in raw.columns else idx.lower()
            if "date" not in raw.columns or value_col not in raw.columns:
                report[idx] = {"status": f"unexpected columns {list(raw.columns)[:8]}"}
                continue
            df = raw[["date", value_col]].rename(columns={value_col: "close"})
            df["date"] = pd.to_datetime(df["date"])
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna().sort_values("date").reset_index(drop=True)
            df.to_parquet(p, index=False)
        report[idx] = {"status": "OK", "first": str(df["date"].min().date()),
                       "last": str(df["date"].max().date()), "rows": int(len(df))}
    (out_dir / "coverage_report.json").write_text(json.dumps(report, indent=1))
    return report


# --------------------------------------------------------------------------- FRED S3

def _fred_api_key() -> str:
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key
    sm = _boto_session().client("secretsmanager")
    secret = json.loads(sm.get_secret_value(SecretId=FRED_SECRET_ID)["SecretString"])
    return secret["api_key"] if isinstance(secret, dict) else str(secret)


def _fred_visible_from(series: str, dates: pd.Series) -> pd.Series:
    rule = FRED_LAG_RULES.get(series, ("days", 1))
    if rule[0] == "days":
        return pd.to_datetime(dates) + pd.Timedelta(days=rule[1])
    if rule[0] == "following_weekday":
        wd = _WEEKDAY[rule[1]]
        return pd.to_datetime(dates).map(lambda d: _following_weekday(d, wd))
    raise ValueError(rule)


def fetch_fred(out_dir: Path = CACHE / "fred", force: bool = False) -> Dict[str, Any]:
    """S3: full-history FRED series with [date, value, visible_from].
    visible_from encodes the §2.3 publication-lag rules; nothing else applied here."""
    out_dir.mkdir(parents=True, exist_ok=True)
    key = _fred_api_key()
    report: Dict[str, Any] = {}
    for series in FRED_SERIES:
        p = out_dir / f"{series}.parquet"
        if p.exists() and not force:
            df = pd.read_parquet(p)
        else:
            r = requests.get(FRED_URL, params={
                "series_id": series, "api_key": key, "file_type": "json",
                "observation_start": "1900-01-01", "limit": 100000}, timeout=60)
            if r.status_code != 200:
                report[series] = {"status": f"HTTP {r.status_code}: {r.text[:120]}"}
                continue
            obs = r.json().get("observations", [])
            df = pd.DataFrame(obs)[["date", "value"]]
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"].replace(".", np.nan), errors="coerce")
            df = df.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)
            df["visible_from"] = _fred_visible_from(series, df["date"])
            df.to_parquet(p, index=False)
        report[series] = {"status": "OK", "first": str(df["date"].min().date()),
                          "last": str(df["date"].max().date()), "rows": int(len(df)),
                          "lag_rule": str(FRED_LAG_RULES.get(series, ("days", 1)))}
    (out_dir / "coverage_report.json").write_text(json.dumps(report, indent=1))
    return report


# --------------------------------------------------------------------------- COT S2

def fetch_cot(out_dir: Path = CACHE / "cot", force: bool = False) -> Dict[str, Any]:
    """S2: CFTC TFF futures-only (Socrata gpe5-46if) for ES / UST10Y / VIX futures.

    Columns per market: report_date (Tuesday positions), publication (Friday 15:30 ET,
    report_date + 3 days — the documented TFF release schedule), visible_from (next
    trading day strictly after publication; US-federal-holiday business calendar),
    lev_net_pct_oi, asset_mgr_net_pct_oi, open_interest, plus raw long/short %OI legs.
    Keyed by PUBLICATION, never report_date (the classic COT leak).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    select = ",".join([
        "report_date_as_yyyy_mm_dd", "contract_market_name", "market_and_exchange_names",
        "open_interest_all",
        "pct_of_oi_lev_money_long", "pct_of_oi_lev_money_short",
        "pct_of_oi_asset_mgr_long", "pct_of_oi_asset_mgr_short",
    ])
    report: Dict[str, Any] = {}
    for slug, market in COT_MARKETS.items():
        p = out_dir / f"{slug}.parquet"
        if p.exists() and not force:
            df = pd.read_parquet(p)
        else:
            r = requests.get(COT_URL, params={
                "$select": select,
                "$where": f"contract_market_name='{market}'",
                "$order": "report_date_as_yyyy_mm_dd",
                "$limit": 50000}, timeout=120)
            if r.status_code != 200:
                report[slug] = {"status": f"HTTP {r.status_code}: {r.text[:120]}"}
                continue
            raw = pd.DataFrame(r.json())
            if len(raw) == 0:
                report[slug] = {"status": "EMPTY"}
                continue
            raw["report_date"] = pd.to_datetime(raw["report_date_as_yyyy_mm_dd"])
            for c in ["open_interest_all", "pct_of_oi_lev_money_long", "pct_of_oi_lev_money_short",
                      "pct_of_oi_asset_mgr_long", "pct_of_oi_asset_mgr_short"]:
                raw[c] = pd.to_numeric(raw[c], errors="coerce")
            # historical renames give duplicate rows per report_date — keep largest-OI row
            raw = (raw.sort_values(["report_date", "open_interest_all"])
                      .drop_duplicates("report_date", keep="last"))
            df = pd.DataFrame({
                "report_date": raw["report_date"],
                "market": market,
                "open_interest": raw["open_interest_all"],
                "lev_long_pct_oi": raw["pct_of_oi_lev_money_long"],
                "lev_short_pct_oi": raw["pct_of_oi_lev_money_short"],
                "am_long_pct_oi": raw["pct_of_oi_asset_mgr_long"],
                "am_short_pct_oi": raw["pct_of_oi_asset_mgr_short"],
            })
            df["lev_net_pct_oi"] = df["lev_long_pct_oi"] - df["lev_short_pct_oi"]
            df["asset_mgr_net_pct_oi"] = df["am_long_pct_oi"] - df["am_short_pct_oi"]
            # publication: Tuesday report -> Friday ~15:30 ET (report_date + 3 days)
            df["publication"] = df["report_date"] + pd.Timedelta(days=3, hours=15, minutes=30)
            df["visible_from"] = df["publication"].map(_next_trading_day)
            df = df.sort_values("report_date").reset_index(drop=True)
            df.to_parquet(p, index=False)
        report[slug] = {"status": "OK", "market": market,
                        "first": str(pd.Timestamp(df["report_date"].min()).date()),
                        "last": str(pd.Timestamp(df["report_date"].max()).date()),
                        "rows": int(len(df))}
    (out_dir / "coverage_report.json").write_text(json.dumps(report, indent=1))
    return report


# --------------------------------------------------------------------------- main

def main() -> None:
    print("== SnapshotStore: replay-window S3 artifacts ==")
    store = SnapshotStore()
    store.sync()

    print("\n== Deep history panels ==")
    deep = load_deep_history()
    for k, v in deep.items():
        print(f"  {k}: {len(v)} rows, {v['date'].min().date()} -> {v['date'].max().date()}")

    print("\n== CBOE S1 ==")
    for idx, r in fetch_cboe().items():
        print(f"  {idx}: {r}")

    print("\n== FRED S3 ==")
    for s, r in fetch_fred().items():
        print(f"  {s}: {r}")

    print("\n== COT S2 (TFF) ==")
    for m, r in fetch_cot().items():
        print(f"  {m}: {r}")

    print("\n== OHLCV deep (yfinance bulk, Stooq fallback) ==")
    rep = fetch_ohlcv_deep(snapshot_cache=store.cache)
    print(f"  universe={rep['universe_size']}, missing={rep['missing_symbols']}")
    ok = [s for s, c in rep["symbols"].items() if c.get("status") == "OK"]
    firsts = sorted((rep["symbols"][s]["first"], s) for s in ok)
    print(f"  ok={len(ok)}; earliest first-date {firsts[0]}, latest first-date {firsts[-1]}")
    print("  cross-check vs S3 snapshots:")
    for rec in rep["s3_crosscheck"]["pairs"]:
        print(f"    {rec}")


if __name__ == "__main__":
    main()
