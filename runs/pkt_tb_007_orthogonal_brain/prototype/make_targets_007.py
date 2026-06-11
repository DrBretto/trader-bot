"""PKT-TB-007 — one-time panel work (BUILD_SPEC_007 §6.1).

Builds store/panel_007.npz with the NEW slices on the TB-006 panel's date
spine (read-only import; targets regenerated here, never inputs):

  y_rot_raw  [N,S]  open(D+5)->open(D+21) return minus per-day cross-sectional
                    mean over valid symbols (M2 target, raw)
  y_rot_rank [N,S]  per-day Gaussian-rank transform of y_rot_raw
  y_disp     [N]    log of forward 5d realized cross-sectional dispersion:
                    sd over the SUPPORT TIER SET (13 names, §2.2) of
                    open(D)->open(D+5) demeaned (over support) returns
  disp_lags  [N,3]  HAR lags of the daily realized dispersion series
                    delta_t = sd over support of demeaned open(t-1)->open(t)
                    returns; lag1 = delta_{D-1}, lag5/lag21 = trailing means
                    ending D-1 (all known by D-1 close: open(D-1) prints
                    before D-1's close)
  cal_dummies[N,3]  turn_of_month, fomc_week, opex_week (M3 calendar block)
  y_evt      [N,27] 1{|y5_bucket| > trailing 80th pct of |y5_bucket| for that
                    bucket}; threshold trailing 252 panel days ending D-6
                    (h+1 lag so only realized target windows enter), min 126
  evt_thresh [N,27] the trailing thresholds
  evt_ctl    [N,27,2] trailing bucket abnormal-vol control: [log(sd21/sd252),
                    sd21_ann] of bucket daily close returns, ending D-1
  m6_feat    [N,S,8] M6 GAP-GRU feature block (§1.6) from cache/ohlcv,
                    trailing-252d z per symbol (shift 1, TB-006 X convention),
                    value rows are "through D-1" when sliced as a sequence
                    ending at panel row D-1
  m2_slow    [N,S,4] raw slow per-symbol block at D-1: return_63d, vol_63d,
                    drawdown_63d, rel_strength_63d (recomputed from
                    cache/ohlcv adj_close, deep-history conventions)
  r1f_raw    [N,S]  open(D)->open(D+1) raw return (M6 falsifier IC + M5 tally)

FOMC dates are a builder-supplied scheduled-meeting calendar (decision days,
2020 emergency moves included) — provenance flagged in the slice manifest and
the validation ledger.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROTO = Path(__file__).resolve().parent
TB6 = PROTO.parents[1] / "pkt_tb_006_clean_sheet_brain" / "prototype"
sys.path.insert(0, str(PROTO))

from statistics import NormalDist
_ND = NormalDist()

SUPPORT = ["ITA", "SOXX", "XRT", "TLT", "AGG", "MUB", "FXE", "USO", "FXI",
           "RSP", "IYR", "SHY", "KRE"]          # TILT_CORE + TILT_COND, frozen
H5, H21 = 5, 21
EVT_PCT = 0.80
EVT_TRAIL = 252
EVT_MINP = 126
EVT_LAG = 6                                     # h + 1

# scheduled FOMC decision dates (second meeting day; 2020 emergency actions
# included). Builder-supplied calendar — provenance flagged.
FOMC_DATES = """
2015-01-28 2015-03-18 2015-04-29 2015-06-17 2015-07-29 2015-09-17 2015-10-28 2015-12-16
2016-01-27 2016-03-16 2016-04-27 2016-06-15 2016-07-27 2016-09-21 2016-11-02 2016-12-14
2017-02-01 2017-03-15 2017-05-03 2017-06-14 2017-07-26 2017-09-20 2017-11-01 2017-12-13
2018-01-31 2018-03-21 2018-05-02 2018-06-13 2018-08-01 2018-09-26 2018-11-08 2018-12-19
2019-01-30 2019-03-20 2019-05-01 2019-06-19 2019-07-31 2019-09-18 2019-10-30 2019-12-11
2020-01-29 2020-03-03 2020-03-15 2020-04-29 2020-06-10 2020-07-29 2020-09-16 2020-11-05 2020-12-16
2021-01-27 2021-03-17 2021-04-28 2021-06-16 2021-07-28 2021-09-22 2021-11-03 2021-12-15
2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21 2022-11-02 2022-12-14
2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13
2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18
2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10
2026-01-28 2026-03-18 2026-04-29 2026-06-17
""".split()


def gaussian_rank(y: np.ndarray) -> np.ndarray:
    out = np.full_like(y, np.nan, dtype=np.float64)
    for i in range(y.shape[0]):
        row = y[i]
        m = np.isfinite(row)
        n = int(m.sum())
        if n < 2:
            continue
        r = pd.Series(row[m]).rank(method="average").to_numpy()
        out[i, m] = [_ND.inv_cdf(v / (n + 1)) for v in r]
    return out


def load_panel6() -> dict:
    z = np.load(TB6 / "store" / "panel.npz", allow_pickle=False)
    return {k: z[k] for k in ["dates", "symbols", "buckets", "open_px",
                              "close_px", "symbol_mask", "y5_bucket"]}


def trailing_z(df: pd.DataFrame, window: int = 252, minp: int = 126):
    sh = df.shift(1)
    m = sh.rolling(window, min_periods=minp).mean()
    sd = sh.rolling(window, min_periods=minp).std().replace(0.0, np.nan)
    return (df - m) / sd


def main():
    t0 = time.time()
    p6 = load_panel6()
    dates = np.asarray(p6["dates"]).astype(str)
    symbols = [str(s) for s in p6["symbols"]]
    buckets = [str(b) for b in p6["buckets"]]
    O = p6["open_px"].astype(np.float64)        # [N,S] open AT D
    C = p6["close_px"].astype(np.float64)
    sm = p6["symbol_mask"].astype(bool)
    N, S = O.shape
    sup_idx = np.array([symbols.index(s) for s in SUPPORT])

    # ---------------- y_rot (M2) ------------------------------------------
    y_rot_raw = np.full((N, S), np.nan)
    for k in range(N - H21):
        with np.errstate(invalid="ignore", divide="ignore"):
            rr = O[k + H21] / O[k + H5] - 1.0
        rr = np.where(sm[k], rr, np.nan)
        mu = np.nanmean(rr)
        y_rot_raw[k] = rr - mu
    y_rot_rank = gaussian_rank(y_rot_raw)

    # ---------------- forward 1d raw return (M6 / M5) ---------------------
    r1f_raw = np.full((N, S), np.nan)
    for k in range(N - 1):
        with np.errstate(invalid="ignore", divide="ignore"):
            rr = O[k + 1] / O[k] - 1.0
        r1f_raw[k] = np.where(sm[k], rr, np.nan)

    # ---------------- y_disp (M3) + HAR lags ------------------------------
    y_disp = np.full(N, np.nan)
    for k in range(N - H5):
        with np.errstate(invalid="ignore", divide="ignore"):
            r5 = O[k + H5, sup_idx] / O[k, sup_idx] - 1.0
        ok = np.isfinite(r5)
        if ok.sum() >= 8:
            d = r5[ok] - r5[ok].mean()
            sd = d.std(ddof=1)
            if sd > 0:
                y_disp[k] = np.log(sd)
    # daily realized dispersion series delta_t (open(t-1)->open(t), support)
    delta = np.full(N, np.nan)
    for k in range(1, N):
        with np.errstate(invalid="ignore", divide="ignore"):
            r1 = O[k, sup_idx] / O[k - 1, sup_idx] - 1.0
        ok = np.isfinite(r1)
        if ok.sum() >= 8:
            d = r1[ok] - r1[ok].mean()
            sd = d.std(ddof=1)
            if sd > 0:
                delta[k] = sd
    ds = pd.Series(delta)
    lag1 = ds.shift(1)                          # delta_{D-1}
    lag5 = ds.shift(1).rolling(5, min_periods=3).mean()
    lag21 = ds.shift(1).rolling(21, min_periods=11).mean()
    disp_lags = np.log(np.column_stack([lag1, lag5, lag21]))   # log-space HAR

    # ---------------- calendar dummies (M3) --------------------------------
    dts = pd.DatetimeIndex(pd.to_datetime(dates))
    month_key = dts.year * 100 + dts.month
    tom = np.zeros(N)
    # last 2 / first 2 trading days of each month (panel = trading calendar)
    mk = np.asarray(month_key)
    for key in np.unique(mk):
        idxs = np.nonzero(mk == key)[0]
        sel = list(idxs[:2]) + list(idxs[-2:])
        tom[sel] = 1.0
    fomc = pd.to_datetime(pd.Series(FOMC_DATES))
    fomc_weeks = {(d.isocalendar()[0], d.isocalendar()[1]) for d in fomc}
    fw = np.array([1.0 if (d.isocalendar()[0], d.isocalendar()[1]) in fomc_weeks
                   else 0.0 for d in dts])
    # opex week = Mon-Fri week containing the 3rd Friday of D's month
    def is_opex_week(d: pd.Timestamp) -> bool:
        first = d.replace(day=1)
        fridays = [first + pd.Timedelta(days=i) for i in range(31)
                   if (first + pd.Timedelta(days=i)).month == d.month
                   and (first + pd.Timedelta(days=i)).dayofweek == 4]
        f3 = fridays[2]
        return f3.isocalendar()[:2] == d.isocalendar()[:2]
    ow = np.array([1.0 if is_opex_week(d) else 0.0 for d in dts])
    cal_dummies = np.column_stack([tom, fw, ow])

    # ---------------- y_evt (M4) -------------------------------------------
    y5b = p6["y5_bucket"].astype(np.float64)    # [N,27] 5d market-removed
    K = y5b.shape[1]
    ab = np.abs(y5b)
    thr = np.full((N, K), np.nan)
    abf = pd.DataFrame(ab)
    thr_df = abf.shift(EVT_LAG).rolling(EVT_TRAIL, min_periods=EVT_MINP) \
        .quantile(EVT_PCT)
    thr = thr_df.to_numpy()
    y_evt = np.where(np.isfinite(ab) & np.isfinite(thr),
                     (ab > thr).astype(np.float64), np.nan)

    # bucket abnormal-vol control from close-px bucket daily returns
    bucket_map = json.loads((PROTO / "dicts" / "bucket_map.json").read_text())
    with np.errstate(invalid="ignore", divide="ignore"):
        r1c = C[1:] / C[:-1] - 1.0
    r1c = np.vstack([np.full((1, S), np.nan), r1c])
    br = np.full((N, K), np.nan)
    for kb, b in enumerate(buckets):
        js = [symbols.index(s) for s in bucket_map[b] if s in symbols]
        if js:
            with np.errstate(invalid="ignore"):
                br[:, kb] = np.nanmean(r1c[:, js], axis=1)
    brf = pd.DataFrame(br).shift(1)             # through D-1
    sd21 = brf.rolling(21, min_periods=11).std() * np.sqrt(252)
    sd252 = brf.rolling(252, min_periods=126).std() * np.sqrt(252)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.log((sd21 / sd252).to_numpy())
    evt_ctl = np.stack([ratio, sd21.to_numpy()], axis=2)

    # ---------------- M6 features + M2 slow block (cache/ohlcv) ------------
    cache = TB6 / "cache" / "ohlcv"
    spy = pd.read_parquet(cache / "SPY.parquet")
    cal = pd.DatetimeIndex(pd.to_datetime(spy["date"]).dt.normalize()) \
        .unique().sort_values()
    pos = cal.searchsorted(dts)                 # panel date -> calendar row
    n_cal = len(cal)
    raw = {k: np.full((n_cal, S), np.nan) for k in
           ["open", "high", "low", "close", "adj_close", "volume"]}
    for j, sym in enumerate(symbols):
        df = pd.read_parquet(cache / f"{sym}.parquet")
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.set_index("date").reindex(cal)
        for k in raw:
            raw[k][:, j] = df[k].to_numpy(dtype=np.float64)
    a = pd.DataFrame(raw["adj_close"])
    o, h, l, c, v = (pd.DataFrame(raw[k]) for k in
                     ["open", "high", "low", "close", "volume"])
    with np.errstate(invalid="ignore", divide="ignore"):
        adj_open = o * (a / c)                  # dividend/split-consistent open
    on_ret = adj_open / a.shift(1) - 1.0        # overnight  (gap_open_pct twin)
    intra = a / adj_open - 1.0                  # intraday
    omi5 = (on_ret - intra).rolling(5, min_periods=3).sum()
    gap_up = on_ret > 0
    filled = pd.DataFrame(np.where(gap_up, (l.to_numpy() <= c.shift(1).to_numpy()),
                                   (h.to_numpy() >= c.shift(1).to_numpy())).astype(float),
                          index=on_ret.index, columns=on_ret.columns) \
        .where(on_ret.notna())
    gap_fill = filled.rolling(21, min_periods=11).mean()
    r1a = a.pct_change(1)
    dollar = (v * c).replace(0.0, np.nan)
    amihud = (r1a.abs() / dollar).rolling(21, min_periods=11).mean()
    range_pct = (h - l) / c
    vz = (v - v.rolling(21).mean()) / v.rolling(21).std().replace(0.0, np.nan)

    feats = [on_ret, intra, omi5, gap_fill, amihud, on_ret.copy(), range_pct, vz]
    m6 = np.full((n_cal, S, 8), np.nan, dtype=np.float32)
    for fi, fdf in enumerate(feats):
        z = trailing_z(fdf)
        m6[:, :, fi] = z.to_numpy(dtype=np.float32)
    m6 = np.nan_to_num(m6, nan=0.0, posinf=0.0, neginf=0.0)
    m6_feat = m6[pos]                           # rows aligned to panel dates
    # note: m6_feat[k] holds features valued AT panel date k; sequences for a
    # decision at row k must end at row k-1 (slice rows k-10..k-1).

    # M2 slow raw block at D-1
    ret63 = a.pct_change(63)
    vol63 = r1a.rolling(63).std() * np.sqrt(252)
    pk63 = a.rolling(63, min_periods=1).max()
    dd63 = (a - pk63) / pk63
    spy_j = symbols.index("SPY")
    rs63 = ret63.sub(ret63.iloc[:, spy_j], axis=0)
    slow = np.stack([ret63.to_numpy(), vol63.to_numpy(), dd63.to_numpy(),
                     rs63.to_numpy()], axis=2)
    m2_slow = slow[pos - 1]                     # value through D-1 close
    m2_slow = np.nan_to_num(m2_slow, nan=0.0, posinf=0.0, neginf=0.0) \
        .astype(np.float32)

    out = PROTO / "store" / "panel_007.npz"
    np.savez_compressed(
        out, dates=p6["dates"], symbols=p6["symbols"], buckets=p6["buckets"],
        y_rot_raw=y_rot_raw.astype(np.float32),
        y_rot_rank=y_rot_rank.astype(np.float32),
        y_disp=y_disp.astype(np.float32),
        disp_lags=np.asarray(disp_lags, dtype=np.float32),
        cal_dummies=cal_dummies.astype(np.float32),
        y_evt=y_evt.astype(np.float32), evt_thresh=thr.astype(np.float32),
        evt_ctl=np.asarray(evt_ctl, dtype=np.float32),
        m6_feat=m6_feat, m2_slow=m2_slow,
        r1f_raw=r1f_raw.astype(np.float32),
        delta_disp=delta.astype(np.float32))
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    manifest = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "builder": "make_targets_007.py", "code_sha": sha,
        "source_panel": str(TB6 / "store" / "panel.npz"),
        "n_dates": int(N), "first": str(dates[0]), "last": str(dates[-1]),
        "support_set": SUPPORT,
        "slices": ["y_rot_raw", "y_rot_rank", "y_disp", "disp_lags",
                   "cal_dummies", "y_evt", "evt_thresh", "evt_ctl", "m6_feat",
                   "m2_slow", "r1f_raw", "delta_disp"],
        "evt_threshold": {"pct": EVT_PCT, "trailing_days": EVT_TRAIL,
                          "min_periods": EVT_MINP, "lag_td": EVT_LAG,
                          "note": "trailing window/minp/lag are builder forced "
                                  "choices (spec gives pct only) — ledgered"},
        "fomc_calendar": {"n_dates": len(FOMC_DATES),
                          "provenance": "builder-supplied scheduled decision "
                                        "dates (2020 emergency moves included) "
                                        "— flagged, not exchange-sourced"},
        "m6_note": "cols overnight_ret_1d and gap_open_pct are numerically "
                   "identical by construction (spec lists both) — disclosed",
        "wall_clock_s": round(time.time() - t0, 1)}
    (PROTO / "store" / "panel_007_meta.json").write_text(
        json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in
                      ["n_dates", "first", "last", "wall_clock_s"]}))
    # quick sanity prints
    print("y_rot finite frac:", float(np.isfinite(y_rot_raw).mean()))
    print("y_disp finite frac:", float(np.isfinite(y_disp).mean()))
    print("y_evt base rate (finite):",
          float(np.nanmean(y_evt)), "finite frac:",
          float(np.isfinite(y_evt).mean()))
    print("m6_feat nonzero frac:", float((m6_feat != 0).mean()))


if __name__ == "__main__":
    main()
