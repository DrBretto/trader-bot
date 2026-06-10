"""PKT-TB-006 SYN-1 prototype — LLM organ feature emission (BUILD_SPEC §3 feature list).

Reads the per-day artifacts at store/llm/<YYYY-MM-DD>.json and emits one tidy
date-indexed parquet (store/llm_features.parquet) with, per GDELT day d:

  llm_sent_<bucket>, llm_conf_<bucket>, llm_sal_<bucket>   x27 buckets
  llm_sent_<bucket>_ema3, llm_sent_<bucket>_d1             x27
  llm_risk_appetite, llm_rates_pressure, llm_geopol_risk
  llm_event_<flag>                                          x12 (global 0/1)
  llm_status_ok, llm_available
  visible_from = d + 1 (downstream join key; BUILD_SPEC §2 conventions)

sal merge: llm_sal_<b> = clamp(LLM sal + sal_prior[b], 0, 1) — the F3 seen-cache decay
re-registers persistent stories' salience without re-billing (forced gap-fill, logged).

Grid: 2024-01-02 .. last artifact date. Rows with no artifact (incl. everything before
the realized Tier-2 start) are all-zero with llm_available=0. ema3/d1 are computed over
available days only (NaN-masked), then zero-filled.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from llm_organ import BUCKETS, EVENT_FLAGS, STORE_DIR

PROTO = Path(__file__).resolve().parent
OUT_PATH = PROTO / "store" / "llm_features.parquet"
GRID_START = dt.date(2024, 1, 2)


def load_artifacts(store_dir: Path = STORE_DIR) -> dict:
    arts = {}
    if store_dir.exists():
        for p in sorted(store_dir.glob("*.json")):
            with open(p) as f:
                a = json.load(f)
            arts[dt.date.fromisoformat(a["date"])] = a
    return arts


def build_features(store_dir: Path = STORE_DIR, grid_start: dt.date = GRID_START,
                   grid_end: dt.date | None = None) -> pd.DataFrame:
    arts = load_artifacts(store_dir)
    if not arts:
        raise RuntimeError(f"no artifacts under {store_dir}")
    if grid_end is None:
        grid_end = max(arts)
    idx = pd.date_range(grid_start, grid_end, freq="D")

    cols: dict = {}
    avail = np.array([d.date() in arts for d in idx], dtype=float)
    status_ok = np.array(
        [1.0 if (d.date() in arts and arts[d.date()]["llm_status"] == "ok") else 0.0
         for d in idx])
    cols["llm_available"] = avail
    cols["llm_status_ok"] = status_ok

    def per_day(fn):
        return np.array([fn(arts[d.date()]) if d.date() in arts else 0.0 for d in idx])

    for b in BUCKETS:
        sent = per_day(lambda a, b=b: a["llm"]["buckets"][b]["sent"])
        conf = per_day(lambda a, b=b: a["llm"]["buckets"][b]["conf"])
        sal = per_day(lambda a, b=b: min(
            1.0, a["llm"]["buckets"][b]["sal"] + a.get("sal_prior", {}).get(b, 0.0)))
        cols[f"llm_sent_{b}"] = sent
        cols[f"llm_conf_{b}"] = conf
        cols[f"llm_sal_{b}"] = sal
        # ema3/d1 over AVAILABLE days only; zero elsewhere
        s = pd.Series(np.where(avail > 0, sent, np.nan), index=idx)
        ema3 = s.ewm(span=3, ignore_na=True).mean()
        d1 = s.ffill().diff()
        cols[f"llm_sent_{b}_ema3"] = np.where(avail > 0, ema3.fillna(0.0), 0.0)
        cols[f"llm_sent_{b}_d1"] = np.where(avail > 0, d1.fillna(0.0), 0.0)

    # n_clusters: feature_store's ih_n_clusters_z reads this column (driver-gap
    # fix, logged as finding: the store guarded on its absence and silently
    # zeroed the z; the artifact carries n_clusters per day). NaN on
    # artifact-absent days so the store's trailing-observation z is unpolluted
    # (the store nan_to_nums after the z).
    cols["n_clusters"] = np.array(
        [float(arts[d.date()].get("n_clusters", 0)) if d.date() in arts else np.nan
         for d in idx])

    cols["llm_risk_appetite"] = per_day(lambda a: a["llm"]["global"]["risk_appetite"])
    cols["llm_rates_pressure"] = per_day(lambda a: a["llm"]["global"]["rates_pressure"])
    cols["llm_geopol_risk"] = per_day(lambda a: a["llm"]["global"]["geopolitical_risk"])

    for fl in EVENT_FLAGS:
        cols[f"llm_event_{fl}"] = per_day(
            lambda a, fl=fl: 1.0 if any(e["flag"] == fl for e in a["llm"]["event_flags"])
            else 0.0)

    df = pd.DataFrame(cols, index=idx)
    df.index.name = "date"
    df["visible_from"] = df.index + pd.Timedelta(days=1)
    return df


def main() -> None:
    df = build_features()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH)
    n_avail = int(df["llm_available"].sum())
    print(f"wrote {OUT_PATH}: {df.shape[0]} rows x {df.shape[1]} cols, "
          f"{n_avail} available days "
          f"({df.index.min().date()} .. {df.index.max().date()})")


if __name__ == "__main__":
    main()
