"""PKT-TB-006 SYN-1 prototype — GDELT G1-G5 feature panel (BUILD_SPEC §2.2).

Reads gdelt_cache/daily/<YYYYMMDD>.json per-day aggregates (written by
gdelt_backfill.py) and emits ONE tidy parquet store/gdelt_features.parquet with
one row per GDELT UTC day d and visible_from = d+1 (calendar). Missing days are
skipped and reported (never silently interpolated).

Feature families (PROPOSAL_DATA_EDGE §1.2, constructions verbatim where given):
  G1  per-bucket mention share + trailing-90d z      g1_share_<b>, g1_z_<b>
  G2  per-country Goldstein + conflict share         g2_goldstein_<cc>, g2_conflict_<cc>
  G3  global tone mean/std/neg-share/polarity        g3_tone_mean/std/neg_share/polarity
      + per-bucket tone + cross-bucket dispersion    g3_tone_<b>, g3_tone_dispersion
  G4  per-bucket burst z (per-file-normalized count) g4_burst_z_<b>
      theme-novelty JSD vs trailing-90d mean dist    g4_theme_novelty, g4_theme_novelty_21
      doc-count surprise (per-file-normalized)       g4_doc_surprise
  G5  HHI locations / orgs + US share                g5_hhi_loc, g5_hhi_org, g5_us_share
  +   g1_share_entropy (info-health field_dispersion), density, n_files_gkg,
      n_records, n_fin_records, gdelt_available=1.

NO-LOOK-AHEAD (binding):
  - trailing-90d z statistics use days < d ONLY (rolling window shifted by 1 day,
    over a calendar-daily reindex so gaps don't smuggle future days into windows).
  - JSD reference distribution = mean of the previous <=90 PRESENT days' theme
    distributions, never including day d.
  - visible_from = d + 1 calendar day; the feature-store join uses it.

Density break: 2 files/day pre-2023 vs 4/day after. Count-based features are
normalized per GKG file (count / n_files_gkg) BEFORE z-scoring so the break does
not masquerade as a burst; `density` is also emitted as a column so models can
see the regime.

Z windows: 90 calendar days, min_periods=30 (forced builder choice, logged in
validation_looks.jsonl).
"""
from __future__ import annotations

import datetime as dt
import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROTO = Path(__file__).resolve().parent
GDELT_DAILY = PROTO / "gdelt_cache" / "daily"
DICTS = PROTO / "dicts"
STORE = PROTO / "store"
OUT_PARQUET = STORE / "gdelt_features.parquet"

Z_WINDOW = 90          # calendar days, trailing, day d excluded
Z_MIN_PERIODS = 30     # forced choice (spec gives window only) — logged
JSD_MIN_DAYS = 30      # min present days in the trailing deque before JSD is emitted
NOVELTY_SMOOTH = 21    # 21d smoothed novelty (trailing mean incl. d — smoothing of a
                       # known-at-d feature, not a stat from the future)

COUNTRIES = ["US", "CN", "JP", "BR", "IN", "RU", "EU"]


def gdelt_buckets() -> List[str]:
    """The frozen bucket set = distinct values of dicts/theme_to_sector.json."""
    t2s = json.loads((DICTS / "theme_to_sector.json").read_text())
    return sorted(set(t2s.values()))


def _hhi(counts: Dict[str, float]) -> float:
    tot = float(sum(counts.values()))
    if tot <= 0:
        return np.nan
    return float(sum((v / tot) ** 2 for v in counts.values()))


def load_daily_raw(daily_dir: Path = GDELT_DAILY) -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[str]]:
    """Load every daily JSON -> (raw frame, theme_dists aligned to frame rows, missing days).

    raw frame columns: identity + per-bucket counts/tone + globals + events.
    theme_dists[i] = {theme: prob} for row i (top-500 distribution, normalized).
    """
    buckets = gdelt_buckets()
    files = sorted(daily_dir.glob("*.json"))
    rows: List[Dict[str, Any]] = []
    theme_dists: List[Dict[str, float]] = []
    for f in files:
        d = json.loads(f.read_text())
        day = pd.Timestamp(dt.datetime.strptime(d["date"], "%Y%m%d").date())
        n_gkg = len(d.get("files", {}).get("gkg_ok", [])) or np.nan
        row: Dict[str, Any] = {
            "gdelt_date": day,
            "visible_from": day + pd.Timedelta(days=1),
            "density": d.get("density"),
            "n_files_gkg": n_gkg,
            "n_records": d.get("n_records"),
            "n_fin_records": d.get("n_fin_records"),
            "g3_tone_mean": d.get("tone_mean"),
            "g3_tone_std": d.get("tone_std"),
            "g3_neg_share": d.get("neg_share"),
            "g3_polarity": d.get("polarity_mean"),
            "g5_hhi_loc": _hhi(d.get("location_country_counts", {})),
            "g5_hhi_org": _hhi(d.get("org_counts", {})),
            "g5_us_share": d.get("us_share"),
        }
        bk = d.get("buckets", {})
        for b in buckets:
            info = bk.get(b, {})
            row[f"_count_{b}"] = info.get("mention_count", 0) or 0
            row[f"g3_tone_{b}"] = info.get("tone_mean", np.nan)
        ev = (d.get("events") or {}).get("per_country", {})
        for cc in COUNTRIES:
            info = ev.get(cc, {})
            row[f"g2_goldstein_{cc.lower()}"] = info.get("goldstein_mw_mean", np.nan)
            row[f"g2_conflict_{cc.lower()}"] = info.get("conflict_share", np.nan)
        rows.append(row)
        tc = d.get("theme_counts", {}) or {}
        tot = float(sum(tc.values()))
        theme_dists.append({k: v / tot for k, v in tc.items()} if tot > 0 else {})

    raw = pd.DataFrame(rows).sort_values("gdelt_date").reset_index(drop=True)
    # missing-day report over the observed span
    if len(raw):
        full = pd.date_range(raw["gdelt_date"].min(), raw["gdelt_date"].max(), freq="D")
        missing = sorted(set(full) - set(raw["gdelt_date"]))
    else:
        missing = []
    return raw, theme_dists, [str(m.date()) for m in missing]


def _trailing_z(series: pd.Series, dates: pd.Series,
                window: int = Z_WINDOW, min_periods: int = Z_MIN_PERIODS) -> pd.Series:
    """Trailing-`window`-calendar-day z with stats from days < d only.

    Reindexes onto the full calendar so a 90-row window never spans more than
    90 calendar days across gaps; missing days contribute nothing (NaN-skipped)."""
    s = pd.Series(series.values, index=pd.DatetimeIndex(dates.values))
    cal = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
    shifted = cal.shift(1)
    mean = shifted.rolling(window, min_periods=min_periods).mean()
    std = shifted.rolling(window, min_periods=min_periods).std()
    z = (cal - mean) / std.replace(0.0, np.nan)
    return pd.Series(z.reindex(pd.DatetimeIndex(dates.values)).values, index=series.index)


def _jsd(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Jensen-Shannon divergence (natural log; in [0, ln 2]) over union support."""
    if not p or not q:
        return np.nan
    keys = set(p) | set(q)
    pv = np.array([p.get(k, 0.0) for k in keys])
    qv = np.array([q.get(k, 0.0) for k in keys])
    pv = pv / pv.sum()
    qv = qv / qv.sum()
    m = 0.5 * (pv + qv)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.where(pv > 0, pv * np.log(pv / m), 0.0).sum()
        kl_qm = np.where(qv > 0, qv * np.log(qv / m), 0.0).sum()
    return float(0.5 * kl_pm + 0.5 * kl_qm)


def _theme_novelty(raw: pd.DataFrame, theme_dists: List[Dict[str, float]],
                   window: int = Z_WINDOW, min_days: int = JSD_MIN_DAYS) -> pd.Series:
    """JSD(today's top-500 dist || mean of trailing <=90 PRESENT days' dists), days < d.

    Maintained as a running sum over a deque keyed by gdelt_date so expiry is by
    calendar age, not row count."""
    out = np.full(len(raw), np.nan)
    dq: deque = deque()              # (date, dist)
    ref_sum: Dict[str, float] = {}   # sum of dist vectors currently in dq
    for i, (day, dist) in enumerate(zip(raw["gdelt_date"], theme_dists)):
        # expire entries older than `window` calendar days
        while dq and (day - dq[0][0]).days > window:
            _, old = dq.popleft()
            for k, v in old.items():
                nv = ref_sum.get(k, 0.0) - v
                if nv <= 1e-12:
                    ref_sum.pop(k, None)
                else:
                    ref_sum[k] = nv
        if len(dq) >= min_days and dist:
            n = len(dq)
            ref = {k: v / n for k, v in ref_sum.items()}
            out[i] = _jsd(dist, ref)
        if dist:
            dq.append((day, dist))
            for k, v in dist.items():
                ref_sum[k] = ref_sum.get(k, 0.0) + v
    return pd.Series(out, index=raw.index)


def build_gdelt_features(daily_dir: Path = GDELT_DAILY,
                         out_path: Optional[Path] = OUT_PARQUET) -> pd.DataFrame:
    raw, theme_dists, missing = load_daily_raw(daily_dir)
    buckets = gdelt_buckets()
    df = raw.copy()

    # --- G1: per-bucket mention share + trailing-90d z -----------------------
    count_cols = [f"_count_{b}" for b in buckets]
    counts = df[count_cols].to_numpy(dtype=float)
    total = counts.sum(axis=1)
    total[total <= 0] = np.nan
    shares = counts / total[:, None]
    for j, b in enumerate(buckets):
        df[f"g1_share_{b}"] = shares[:, j]
    for b in buckets:
        df[f"g1_z_{b}"] = _trailing_z(df[f"g1_share_{b}"], df["gdelt_date"])

    # field dispersion (info-health): entropy of today's bucket-share vector
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(shares > 0, shares * np.log(shares), 0.0), axis=1)
    df["g1_share_entropy"] = np.where(np.isnan(total), np.nan, ent)

    # --- G3: cross-bucket tone dispersion ------------------------------------
    tone_cols = [f"g3_tone_{b}" for b in buckets]
    df["g3_tone_dispersion"] = df[tone_cols].std(axis=1)

    # --- G4: burst z (per-file-normalized counts), novelty, doc surprise -----
    nf = df["n_files_gkg"].to_numpy(dtype=float)
    for j, b in enumerate(buckets):
        per_file = counts[:, j] / nf
        df[f"g4_burst_z_{b}"] = _trailing_z(pd.Series(per_file, index=df.index),
                                            df["gdelt_date"])
    df["g4_doc_surprise"] = _trailing_z(df["n_records"] / df["n_files_gkg"],
                                        df["gdelt_date"])
    df["g4_theme_novelty"] = _theme_novelty(df, theme_dists)
    nov = pd.Series(df["g4_theme_novelty"].values,
                    index=pd.DatetimeIndex(df["gdelt_date"].values))
    cal = nov.reindex(pd.date_range(nov.index.min(), nov.index.max(), freq="D"))
    sm = cal.rolling(NOVELTY_SMOOTH, min_periods=7).mean()
    df["g4_theme_novelty_21"] = sm.reindex(pd.DatetimeIndex(df["gdelt_date"].values)).values

    df["gdelt_available"] = 1
    df = df.drop(columns=count_cols)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        report = {
            "generated": dt.datetime.now().isoformat(timespec="seconds"),
            "rows": int(len(df)),
            "first": str(df["gdelt_date"].min().date()),
            "last": str(df["gdelt_date"].max().date()),
            "n_buckets": len(buckets),
            "buckets": buckets,
            "n_columns": int(df.shape[1]),
            "z_window_days": Z_WINDOW,
            "z_min_periods": Z_MIN_PERIODS,
            "missing_days_skipped": missing,
            "n_missing_days": len(missing),
        }
        (out_path.parent / "gdelt_features_report.json").write_text(json.dumps(report, indent=1))
    return df


if __name__ == "__main__":
    df = build_gdelt_features()
    rep = json.loads((STORE / "gdelt_features_report.json").read_text())
    print(f"gdelt_features.parquet: {rep['rows']} rows x {rep['n_columns']} cols, "
          f"{rep['first']} -> {rep['last']}, {rep['n_missing_days']} missing days skipped")
    fam = {"g1_share": 0, "g1_z": 0, "g2_": 0, "g3_": 0, "g4_": 0, "g5_": 0}
    for c in df.columns:
        for k in fam:
            if c.startswith(k):
                fam[k] += 1
    print("family column counts:", fam)
