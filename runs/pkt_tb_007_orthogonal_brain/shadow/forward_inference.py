"""PKT-TB-007 dual forward shadow — forward panel + deploy-organ inference.

Reuses (imports):
  - TB-006 feature_store.FeatureStore  — the panel builder, instantiated with
    cache/store re-pointed at shadow/state (write surface stays in shadow/).
    Decision dates that have not yet traded (the live night) are appended to
    the price calendar as NaN rows: the X window for date D uses rows
    D-63..D-1 only, so a NaN bar AT D is exactly the training convention.
  - TB-006 features_gdelt.build_gdelt_features + gdelt_backfill (dirs
    re-pointed) — forward GDELT days, the trained gdelt_available masking.
  - TB-006 data_layer.fetch_cboe (out_dir parameterized).
  - TB-007 organs_007 (build_cast_xs / sector_class_ids / M1_XCOLS /
    m3_design / m4_design / m4_predict / ols_predict), folds (seeds).

Copied minimally (noted in shadow_lib header):
  - trailing_pct (precompute_nightly_007.py)
  - evt_ctl / disp_lags / cal_dummies blocks (make_targets_007.main is a
    monolith) -> build_p7_extras()

Inference parity: on first run (and on demand) the pipeline recomputes the
overlap dates 2026-06-05..2026-06-09 and asserts M1 mu / disp_z / p_exceed
against the frozen prototype store/nightly_007 files. A parity failure
ABORTS the run — the shadow never writes forecasts from a divergent
pipeline.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from shadow_lib import (CACHE_CBOE, CACHE_COT, CACHE_DAILY, CACHE_FRED,
                        CACHE_OHLCV, CBOE_STALE_DAYS, GDELT_DAILY_DIR,
                        GDELT_RECORDS_DIR, ORGAN_DIR, PROTO7,
                        SHADOW_PANEL_START, SPLIT_ALERT_RET, STORE, TB6_PROTO,
                        log_line, sha12, write_json)

REPO = PROTO7.parents[2]
MODELS = PROTO7 / "models_out_007"
PANEL_NAME = "panel_shadow.npz"
PARITY_DATES = ["2026-06-05", "2026-06-08", "2026-06-09"]
# disp_z parity uses complete-month dates only: the frozen reference panel's
# truncated final month (June 2026) carries a month-end (tom) labeling
# artifact at 2026-06-09/10 that the forward true-calendar convention
# corrects (see build_p7_extras).
PARITY_DATES_DISPZ = ["2026-05-20", "2026-05-27", "2026-05-28"]
PARITY_TOL = {"mu": 2e-3, "disp_z": 5e-2, "p_exceed": 5e-3, "q": 5e-2}


# ---------------------------------------------------------------- seed caches
def ensure_seed_caches(logf: Optional[Path] = None) -> None:
    """One-time copy of the frozen TB-006 caches into shadow/state (the
    shadow's write surface; TB-006 stays read-only)."""
    import shutil
    seeds = [
        (TB6_PROTO / "cache" / "ohlcv", CACHE_OHLCV),
        (TB6_PROTO / "cache" / "cboe", CACHE_CBOE),
        (TB6_PROTO / "cache" / "fred", CACHE_FRED),
        (TB6_PROTO / "cache" / "cot", CACHE_COT),
        (TB6_PROTO / "gdelt_cache" / "daily", GDELT_DAILY_DIR),
    ]
    for src, dst in seeds:
        if dst.exists() and any(dst.iterdir()):
            continue
        log_line(f"seeding cache {dst.name} from {src}", logf)
        shutil.copytree(src, dst, dirs_exist_ok=True)
    GDELT_RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    STORE.mkdir(parents=True, exist_ok=True)
    llm = TB6_PROTO / "store" / "llm_features.parquet"
    if llm.exists() and not (STORE / "llm_features.parquet").exists():
        shutil.copy2(llm, STORE / "llm_features.parquet")


# ---------------------------------------------------------------- ohlcv splice
def extend_ohlcv(daily_date: str, logf: Optional[Path] = None) -> List[str]:
    """Upsert rows from daily/<D>/prices.parquet (raw S3 bars through D-1)
    into the shadow ohlcv store. adj_close := close for new rows (no future-
    dividend back-adjustment — recorded limitation). Returns ALERT lines for
    suspicious overnight moves (possible unadjusted splits)."""
    alerts: List[str] = []
    src = CACHE_DAILY / daily_date / "prices.parquet"
    if not src.exists():
        return alerts
    px = pd.read_parquet(src)
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    for sym, grp in px.groupby("symbol"):
        p = CACHE_OHLCV / f"{sym}.parquet"
        if not p.exists():
            continue                      # not in the 64-symbol universe
        cur = pd.read_parquet(p)
        cur["date"] = pd.to_datetime(cur["date"]).dt.normalize()
        last = cur["date"].max()
        new = grp[grp["date"] > last].sort_values("date")
        if new.empty:
            continue
        last_close = float(cur.loc[cur["date"] == last, "close"].iloc[-1])
        for _, r in new.iterrows():
            c = float(r["close"])
            if last_close > 0 and abs(c / last_close - 1.0) > SPLIT_ALERT_RET:
                alerts.append(
                    f"ALERT possible split/bad bar {sym} {r['date'].date()} "
                    f"close {last_close:.2f} -> {c:.2f} (NOT auto-adjusted)")
            last_close = c
        add = pd.DataFrame({
            "date": new["date"], "open": new["open"].astype(float),
            "high": new["high"].astype(float), "low": new["low"].astype(float),
            "close": new["close"].astype(float),
            "adj_close": new["close"].astype(float),
            "volume": new["volume"].astype(float),
            "source": "s3_daily", "adj_factor": 1.0})
        # ISSUE-01 ROOT (F-D1): pandas 2.1.x (the Lambda image) raises a block-
        # consolidation ValueError when concatenating datetime64 columns of
        # DIFFERENT resolution — the seed store's `date` is datetime64[ms] while
        # the freshly-parsed S3 daily bars are datetime64[ns] (and volume is
        # int64 vs float64). Newer pandas (laptop repro) tolerates it, which is
        # why the store advanced locally but the raise was swallowed in-Lambda,
        # freezing the panel at 2026-06-10. Coerce the new rows to the store's
        # existing dtypes so the splice can never raise on a resolution mismatch.
        for col in add.columns:
            if col in cur.columns and add[col].dtype != cur[col].dtype:
                add[col] = add[col].astype(cur[col].dtype)
        out = pd.concat([cur, add], ignore_index=True)
        out.to_parquet(p, index=False)
    for a in alerts:
        log_line(a, logf)
    return alerts


# ---------------------------------------------------------------- gdelt + cboe
def gdelt_forward(logf: Optional[Path] = None) -> int:
    """Fetch missing GDELT UTC days (last cached + 1 .. yesterday UTC) into
    the shadow gdelt cache, then rebuild gdelt_features.parquet. Fail-soft:
    a failed day stays missing and the trained gdelt_available staleness
    masking covers it."""
    import gdelt_backfill as GB
    import features_gdelt as FG
    have = sorted(f.stem for f in GDELT_DAILY_DIR.glob("*.json"))
    if not have:
        raise RuntimeError("gdelt daily cache empty — seed missing")
    last = dt.datetime.strptime(have[-1], "%Y%m%d").date()
    yday = dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)
    n_new = 0
    if last < yday:
        GB.RECORDS_DIR = str(GDELT_RECORDS_DIR)
        GB.DAILY_DIR = str(GDELT_DAILY_DIR)
        GB.LOG_PATH = str(GDELT_RECORDS_DIR.parent / "backfill_log.txt")
        # ISSUE-08: the manifest is a module constant on the baked (read-only
        # /var/task) gdelt_cache; without this override open(MANIFEST_PATH,"a")
        # raises OSError: Read-only file system and every GDELT day FAILs. Redirect
        # it under the writable state tree alongside RECORDS/DAILY/LOG.
        GB.MANIFEST_PATH = str(GDELT_RECORDS_DIR.parent / "manifest.jsonl")
        start = (last + dt.timedelta(days=1)).strftime("%Y-%m-%d")
        end = yday.strftime("%Y-%m-%d")
        log_line(f"gdelt forward fetch {start}..{end}", logf)
        try:
            GB.run_phase(start, end, density=4, workers=2)
        except Exception as e:                                  # noqa: BLE001
            log_line(f"ALERT gdelt fetch failed ({type(e).__name__}: {e}); "
                     f"staleness masking will cover", logf)
        n_new = len(list(GDELT_DAILY_DIR.glob("*.json"))) - len(have)
    # rebuild the tidy feature parquet (deterministic full reload)
    FG.build_gdelt_features(daily_dir=GDELT_DAILY_DIR,
                            out_path=STORE / "gdelt_features.parquet")
    return n_new


def cboe_forward(logf: Optional[Path] = None) -> None:
    """Refresh the CBOE index histories (full-history CSVs). Fail-soft: on
    failure the old parquets stay; disp_z staleness nulling covers it."""
    try:
        from data_layer import fetch_cboe
        fetch_cboe(out_dir=CACHE_CBOE, force=True)
    except Exception as e:                                      # noqa: BLE001
        log_line(f"ALERT cboe refresh failed ({type(e).__name__}: {e}); "
                 f"disp_z will be nulled when stale > {CBOE_STALE_DAYS}d",
                 logf)


def cboe_last_date() -> Optional[dt.date]:
    p = CACHE_CBOE / "VIX.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    return pd.to_datetime(df["date"]).max().date()


# ---------------------------------------------------------------- panel build
class ShadowFeatureStore:
    """TB-006 FeatureStore with cache/store re-pointed at shadow/state and
    not-yet-traded decision dates appended as NaN calendar rows."""

    def __init__(self, future_dates: List[str]):
        from feature_store import FeatureStore
        fs = FeatureStore(proto_dir=TB6_PROTO)      # dicts/universe from TB6
        fs.cache = CACHE_OHLCV.parent               # state/cache
        fs.store = STORE
        self.fs = fs
        self.future_dates = sorted(future_dates)
        # wrap _load_prices to append the NaN rows
        orig_load = fs._load_prices

        def load_with_future():
            orig_load()
            extra = [pd.Timestamp(d) for d in self.future_dates
                     if pd.Timestamp(d) not in fs.cal]
            if not extra:
                return
            assert all(t > fs.cal.max() for t in extra), \
                "future decision dates must extend the calendar"
            n_extra = len(extra)
            fs.cal = fs.cal.append(pd.DatetimeIndex(extra))
            for k in fs.px:
                pad = np.full((n_extra, fs.px[k].shape[1]), np.nan)
                fs.px[k] = np.vstack([fs.px[k], pad])

        fs._load_prices = load_with_future

    def build(self, end: str) -> dict:
        return self.fs.build(start=SHADOW_PANEL_START, end=end,
                             out_name=PANEL_NAME, verbose=False)


def build_panel(pending: List[str], logf: Optional[Path] = None) -> dict:
    end = max(pending)
    sfs = ShadowFeatureStore(future_dates=pending)
    meta = sfs.build(end=end)
    log_line(f"panel_shadow built: {meta['n_dates']} dates "
             f"{meta['first']}..{meta['last']}", logf)
    return meta


# ---------------------------------------------------------------- p7 extras
def build_p7_extras(panel: dict) -> dict:
    """COPIED MINIMALLY from make_targets_007.main (evt_ctl, disp_lags,
    cal_dummies blocks — the monolith offers no importable boundary).
    Only what M3/M4 deploy inference consumes."""
    from organs_007 import SUPPORT
    dates = np.asarray(panel["dates"]).astype(str)
    symbols = [str(s) for s in panel["symbols"]]
    buckets = [str(b) for b in panel["buckets"]]
    C = panel["close_px"].astype(np.float64)        # [N,S] close AT D (NaN fwd)
    O = panel["open_px"].astype(np.float64)
    N, S = C.shape
    dts = pd.DatetimeIndex(pd.to_datetime(dates))

    # cal_dummies — conventions from make_targets_007.main, with ONE
    # forward correction: tom = first 2 / last 2 trading days of each month.
    # The frozen builder computed last-2 from observed panel rows, which is
    # only correct for COMPLETE months (its final truncated month carried an
    # artifact: 2026-06-09/10 labeled month-end). Forward, the month's
    # remaining trading days are projected from the known NYSE calendar so
    # the dummy matches the training distribution (true first2/last2).
    month_key = np.asarray(dts.year * 100 + dts.month)
    tom = np.zeros(N)
    last_seen = dts.max()
    proj = _project_month_trading_days(last_seen)
    for key in np.unique(month_key):
        idxs = np.nonzero(month_key == key)[0]
        tom[list(idxs[:2])] = 1.0                  # first2 always observed
        month_dates = list(dts[idxs])
        if key == last_seen.year * 100 + last_seen.month:
            month_dates += proj                    # complete the live month
        last2 = set(sorted(month_dates)[-2:])
        for i in idxs:
            if dts[i] in last2:
                tom[i] = 1.0
    # fw = FOMC week, calendar imported from make_targets_007 (ends
    # 2026-06-17; later FOMC weeks read 0 — recorded limitation)
    import make_targets_007 as MT
    fomc = pd.to_datetime(pd.Series(MT.FOMC_DATES))
    fomc_weeks = {(d.isocalendar()[0], d.isocalendar()[1]) for d in fomc}
    fw = np.array([1.0 if (d.isocalendar()[0], d.isocalendar()[1])
                   in fomc_weeks else 0.0 for d in dts])

    def is_opex_week(d: pd.Timestamp) -> bool:
        first = d.replace(day=1)
        fridays = [first + pd.Timedelta(days=i) for i in range(31)
                   if (first + pd.Timedelta(days=i)).month == d.month
                   and (first + pd.Timedelta(days=i)).dayofweek == 4]
        f3 = fridays[2]
        return f3.isocalendar()[:2] == d.isocalendar()[:2]

    ow = np.array([1.0 if is_opex_week(d) else 0.0 for d in dts])
    cal_dummies = np.column_stack([tom, fw, ow])

    # disp_lags — VERBATIM from make_targets_007.main: daily realized
    # dispersion delta_t over SUPPORT open(t-1)->open(t) returns, HAR lags
    # of the shifted series (lags only need data through D-1; the live
    # night's NaN open AT D never enters)
    sup_idx = np.array([symbols.index(s) for s in SUPPORT])
    delta = np.full(N, np.nan)
    for k in range(1, N):
        with np.errstate(invalid="ignore", divide="ignore"):
            r1 = O[k, sup_idx] / O[k - 1, sup_idx] - 1.0
        ok = np.isfinite(r1)
        if ok.sum() >= 8:
            d_ = r1[ok] - r1[ok].mean()
            sd = d_.std(ddof=1)
            if sd > 0:
                delta[k] = sd
    ds = pd.Series(delta)
    lag1 = ds.shift(1)
    lag5 = ds.shift(1).rolling(5, min_periods=3).mean()
    lag21 = ds.shift(1).rolling(21, min_periods=11).mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        disp_lags = np.log(np.column_stack([lag1, lag5, lag21]))

    # evt_ctl: bucket abnormal-vol control from close-px bucket returns
    bucket_map = json.loads((PROTO7 / "dicts" / "bucket_map.json").read_text())
    K = len(buckets)
    with np.errstate(invalid="ignore", divide="ignore"):
        r1c = C[1:] / C[:-1] - 1.0
    r1c = np.vstack([np.full((1, S), np.nan), r1c])
    br = np.full((N, K), np.nan)
    for kb, b in enumerate(buckets):
        js = [symbols.index(s) for s in bucket_map[b] if s in symbols]
        if js:
            with np.errstate(invalid="ignore"):
                br[:, kb] = np.nanmean(r1c[:, js], axis=1)
    brf = pd.DataFrame(br).shift(1)                # through D-1
    sd21 = brf.rolling(21, min_periods=11).std() * np.sqrt(252)
    sd252 = brf.rolling(252, min_periods=126).std() * np.sqrt(252)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.log((sd21 / sd252).to_numpy())
    evt_ctl = np.stack([ratio, sd21.to_numpy()], axis=2)

    return {"cal_dummies": cal_dummies, "disp_lags": disp_lags,
            "evt_ctl": evt_ctl,
            "y_disp": np.full(N, np.nan), "y_evt": np.full((N, K), np.nan)}


def _project_month_trading_days(last_seen: pd.Timestamp) -> list:
    """Remaining NYSE trading days of last_seen's month (weekdays minus the
    rule-based NYSE holidays). Unscheduled closures are alert-level."""
    from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday,
                                        Holiday, USLaborDay,
                                        USMartinLutherKingJr, USMemorialDay,
                                        USPresidentsDay, USThanksgivingDay,
                                        nearest_workday)

    class NYSE(AbstractHolidayCalendar):
        rules = [Holiday("NewYears", month=1, day=1,
                         observance=nearest_workday),
                 USMartinLutherKingJr, USPresidentsDay, GoodFriday,
                 USMemorialDay,
                 Holiday("Juneteenth", month=6, day=19,
                         observance=nearest_workday),
                 Holiday("July4", month=7, day=4,
                         observance=nearest_workday),
                 USLaborDay, USThanksgivingDay,
                 Holiday("Christmas", month=12, day=25,
                         observance=nearest_workday)]

    month_end = (last_seen + pd.offsets.MonthEnd(0)).normalize()
    if last_seen.normalize() >= month_end:
        return []
    days = pd.bdate_range(last_seen.normalize() + pd.Timedelta(days=1),
                          month_end)
    hols = set(NYSE().holidays(days[0], days[-1])) if len(days) else set()
    return [d for d in days if d not in hols]


# ---------------------------------------------------------------- helpers
def trailing_pct(series: np.ndarray, window: int = 252, minp: int = 64):
    """COPIED from precompute_nightly_007 (its module pulls torch via main)."""
    s = pd.Series(series)
    return s.rolling(window + 1, min_periods=minp).apply(
        lambda w: float((w.iloc[-1] > w.iloc[:-1]).mean()), raw=False) \
        .to_numpy()


def _sklearn_multiclass_compat(obj) -> None:
    """PKT-TB-012 cross-version compat: the frozen M4 LogisticRegression was
    pickled under sklearn >= 1.8 (where ``multi_class`` was removed); the Lambda
    image runs the newest py3.11 build (1.7.2), whose ``LogisticRegression.predict``
    still references ``self.multi_class``. Restore the attribute on any
    LogisticRegression in the loaded model. M4 is a BINARY classifier, so
    ``multi_class`` is a no-op for ``predict_proba`` (binary is always the sigmoid
    path) — the result is identical to 1.8.0, and the parity self-check validates
    it. The pickle bytes are untouched, so the frozen model_sha is unchanged.
    """
    try:
        from sklearn.linear_model import LogisticRegression as _LR
    except Exception:                                       # noqa: BLE001
        return

    def _walk(o):
        if isinstance(o, _LR):
            if not hasattr(o, "multi_class"):
                o.multi_class = "auto"
        elif isinstance(o, dict):
            for v in o.values():
                _walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                _walk(v)
    _walk(obj)


def model_sha() -> str:
    import hashlib
    h = hashlib.sha256()
    for p in [MODELS / "m1_cast" / "seed_4242.pt",
              MODELS / "m1_cast" / "seed_4243.pt",
              MODELS / "m4_evt_a" / "model.pkl",
              MODELS / "m3_disp" / "coefs.npz"]:
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- inference
def run_inference(pending: List[str], logf: Optional[Path] = None,
                  parity_check: bool = True) -> Dict[str, dict]:
    """M1 + disp_z + M4-A deploy inference over the shadow panel; writes
    organ_inputs/<D>.json for pending dates; returns per-date forecast
    records (full-universe mu) for the forecast ledger. ABORTS on parity
    failure."""
    import pickle
    import torch
    import folds as F
    import organs_007 as G

    z = np.load(STORE / PANEL_NAME, allow_pickle=False)
    panel = {k: z[k] for k in z.files}
    dates = np.asarray(panel["dates"]).astype(str)
    symbols = [str(s) for s in panel["symbols"]]
    sm = panel["symbol_mask"].astype(bool)
    sup_j = np.array([symbols.index(s) for s in G.SUPPORT])
    N = len(dates)
    didx = {d: i for i, d in enumerate(dates)}
    for d in pending:
        if d not in didx:
            raise RuntimeError(f"pending date {d} missing from shadow panel")

    # ---- M1 (2-seed mean) — mirrors precompute_nightly_007.main ----------
    xf = [str(c) for c in panel["X_features"]]
    cols10 = [xf.index(c) for c in G.M1_XCOLS]
    X10 = np.ascontiguousarray(panel["X"][:, :, :, cols10])
    sec_ids, cls_ids = G.sector_class_ids(REPO / "config" / "universe.csv")
    sec_t = torch.from_numpy(sec_ids)
    cls_t = torch.from_numpy(cls_ids)
    mus = []
    for seed in F.CAST_OOF_SEEDS_007:
        model = G.build_cast_xs(seed)
        model.load_state_dict(torch.load(
            MODELS / "m1_cast" / f"seed_{seed}.pt", weights_only=True))
        model.eval()
        mu = np.zeros((N, len(symbols)), dtype=np.float64)
        with torch.no_grad():
            for c0 in range(0, N, 64):
                sl = slice(c0, min(c0 + 64, N))
                m, _, _ = model(torch.from_numpy(X10[sl]),
                                torch.from_numpy(panel["scalars"][sl]),
                                sec_t, cls_t, torch.from_numpy(sm[sl]))
                mu[sl] = m.numpy()
        mus.append(mu)
    mu_m1 = np.mean(np.stack(mus), axis=0)
    q_m1 = trailing_pct(np.nanstd(mu_m1[:, sup_j], axis=1))

    # ---- M3 disp_z + M4-A — mirrors precompute_nightly_007.main ----------
    p7 = build_p7_extras(panel)
    z3 = np.load(MODELS / "m3_disp" / "coefs.npz")
    X_full, _, _, _ = G.m3_design(
        {"T": panel["T"], "T_cols": panel["T_cols"]}, p7)
    fc = G.ols_predict(z3["coef_full"], np.nan_to_num(X_full, nan=0.0))
    fz = pd.Series(fc)
    mz_ = fz.shift(1).rolling(252, min_periods=60).mean()
    sz_ = fz.shift(1).rolling(252, min_periods=60).std()
    disp_z = ((fz - mz_) / sz_).to_numpy()
    # honest degradation: stale CBOE => disp_z null (B-disp constant)
    cb_last = cboe_last_date()
    stale_after = (None if cb_last is None else
                   cb_last + dt.timedelta(days=CBOE_STALE_DAYS))

    with (MODELS / "m4_evt_a" / "model.pkl").open("rb") as fh:
        m4 = pickle.load(fh)
    _sklearn_multiclass_compat(m4)
    designs = G.m4_design({k: panel[k] for k in
                           ["B", "B_cols", "gdelt_available"]}, p7, dates)
    p4 = G.m4_predict(m4["fitted"], designs["A"], np.arange(N))
    bucket_sym = m4["bucket_sym"]
    p_sym = np.zeros((N, len(symbols)))
    cnt = np.zeros(len(symbols))
    for k in range(bucket_sym.shape[0]):
        p_sym[:, bucket_sym[k]] += p4[:, k][:, None]
        cnt[bucket_sym[k]] += 1
    cnt[cnt == 0] = 1
    p_sym /= cnt[None, :]

    # ---- parity self-check vs the frozen prototype nightly files ---------
    if parity_check:
        _parity(dates, didx, symbols, sm, sup_j, mu_m1, q_m1, disp_z, p_sym,
                logf)

    # ---- emit organ files + forecast records for pending dates -----------
    msha = model_sha()
    out: Dict[str, dict] = {}
    ORGAN_DIR.mkdir(parents=True, exist_ok=True)
    man_common = {
        "schema": "organ_inputs_007.v1",
        "shadow": "PKT-TB-007 dual forward shadow",
        "code_sha": G.code_sha(), "model_sha": msha,
        "seeds": {"m1": F.CAST_OOF_SEEDS_007, "master": F.MASTER_SEED_007},
        "roster": ["M1"], "m4_variant_shipped": "M4-A",
        "emission_policy": "shadow emits M1 + disp_z + p_exceed only; the "
                           "frozen book genomes decide what is listened to",
    }
    for d in pending:
        i = didx[d]
        ddate = pd.Timestamp(d).date()
        dz = disp_z[i]
        if stale_after is not None and ddate > stale_after:
            dz = np.nan
            log_line(f"ALERT cboe stale (last {cb_last}); disp_z nulled "
                     f"for {d}", logf)
        organs = {"M1": {
            "mu": {s: round(float(mu_m1[i, j]), 6)
                   for s, j in zip(G.SUPPORT, sup_j) if sm[i, j]},
            "q": round(float(np.nan_to_num(q_m1[i], nan=0.0)), 4)}}
        doc = {"date": d, "schema_version": "organ_inputs_007.v1",
               "organs": organs,
               "disp_z": (round(float(dz), 6) if np.isfinite(dz) else None),
               "p_exceed": {s: round(float(p_sym[i, j]), 4)
                            for s, j in zip(G.SUPPORT, sup_j) if sm[i, j]},
               "manifest": man_common}
        write_json(ORGAN_DIR / f"{d}.json", doc)

        valid = sm[i] & np.isfinite(mu_m1[i])
        mu_map = {symbols[j]: round(float(mu_m1[i, j]), 6)
                  for j in np.nonzero(valid)[0]}
        order = sorted(mu_map, key=lambda s: -mu_map[s])
        feat_sha = sha12(panel["X"][i].tobytes()
                         + panel["scalars"][i].tobytes())
        out[d] = {"date": d,
                  "recorded_at": dt.datetime.now(dt.timezone.utc)
                  .isoformat(timespec="seconds"),
                  "mu": mu_map,
                  "rank": {s: r + 1 for r, s in enumerate(order)},
                  "n_valid": int(valid.sum()),
                  "q": round(float(np.nan_to_num(q_m1[i], nan=0.0)), 4),
                  "feature_sha": feat_sha, "model_sha": msha,
                  "code_sha": G.code_sha()}
    log_line(f"inference done for {len(pending)} pending dates", logf)
    return out


def _parity(dates, didx, symbols, sm, sup_j, mu_m1, q_m1, disp_z, p_sym,
            logf) -> None:
    from organs_007 import SUPPORT
    ref_dir = PROTO7 / "store" / "nightly_007"
    checked = 0
    worst = {"mu": 0.0, "disp_z": 0.0, "p_exceed": 0.0, "q": 0.0}
    for d in PARITY_DATES:
        ref_p = ref_dir / f"{d}.json"
        if d not in didx or not ref_p.exists():
            continue
        ref = json.loads(ref_p.read_text())
        i = didx[d]
        for s, j in zip(SUPPORT, sup_j):
            if s in ref["organs"]["M1"]["mu"] and sm[i, j]:
                worst["mu"] = max(worst["mu"], abs(
                    ref["organs"]["M1"]["mu"][s] - float(mu_m1[i, j])))
            if s in (ref.get("p_exceed") or {}):
                worst["p_exceed"] = max(worst["p_exceed"], abs(
                    ref["p_exceed"][s] - float(p_sym[i, j])))
        worst["q"] = max(worst["q"], abs(
            ref["organs"]["M1"]["q"] - float(np.nan_to_num(q_m1[i]))))
        checked += 1
    for d in PARITY_DATES_DISPZ:                  # complete-month dates only
        ref_p = ref_dir / f"{d}.json"
        if d not in didx or not ref_p.exists():
            continue
        ref = json.loads(ref_p.read_text())
        i = didx[d]
        if ref.get("disp_z") is not None and np.isfinite(disp_z[i]):
            worst["disp_z"] = max(worst["disp_z"],
                                  abs(ref["disp_z"] - float(disp_z[i])))
    if checked == 0:
        log_line("parity check skipped (no overlap dates in panel)", logf)
        return
    fails = {k: v for k, v in worst.items() if v > PARITY_TOL[k]}
    log_line(f"parity vs store/nightly_007 on {checked} dates: "
             + ", ".join(f"{k} max|d|={v:.2e}" for k, v in worst.items()),
             logf)
    if fails:
        raise RuntimeError(
            f"PARITY FAILURE vs frozen prototype nightly files: {fails} "
            f"(tol {PARITY_TOL}) — shadow refuses to write forecasts from "
            f"a divergent pipeline")
