"""Fast offline unit checks for the PKT-TB-006 data layer (BUILD_SPEC §15 step 3).

All tests read the local cache only — no network, no AWS. Run:
  .venv/bin/python -m pytest runs/pkt_tb_006_clean_sheet_brain/prototype/tests -q
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROTO))

import data_layer as dl  # noqa: E402

WINDOW_START, WINDOW_END = dl.WINDOW_START, dl.WINDOW_END


# ----------------------------------------------------------------- universe

def test_universe_is_64_symbols():
    syms = dl.universe_symbols()
    assert len(syms) == 64
    assert len(set(syms)) == 64
    assert "SPY" in syms and "VUG" in syms


# ----------------------------------------------------------------- snapshot store

def test_snapshot_store_completeness():
    """Window dirs with prices.parquet ≈ 89 trading days. Observed shape: the
    pipeline writes Tue-Sat dirs (Saturday dir carries Friday close; no Monday
    dirs) and there is a snapshot gap 2026-05-11..2026-05-22 — so 87, within band."""
    daily = dl.CACHE / "s3" / "daily"
    have = sorted(d.name for d in daily.iterdir()
                  if d.is_dir() and WINDOW_START <= d.name <= WINDOW_END
                  and (d / "prices.parquet").exists())
    assert 80 <= len(have) <= 94, f"got {len(have)} prices dirs"
    # core artifacts travel together
    for d in have:
        assert (daily / d / "context.parquet").exists(), d
        assert (daily / d / "features.parquet").exists(), d
        assert (daily / d / "portfolio_state.json").exists(), d


def test_disk_cached_s3_cache_offline_interface():
    cache = dl.DiskCachedS3Cache(s3_client=None)
    df = cache.get_parquet(f"daily/{WINDOW_END}/prices.parquet")
    assert {"date", "symbol", "open", "high", "low", "close", "volume"} <= set(df.columns)
    assert df["symbol"].nunique() == 64
    ps = cache.get_json(f"daily/{WINDOW_END}/portfolio_state.json")
    assert isinstance(ps, dict)
    assert WINDOW_END in cache.list_daily_dates()


# ----------------------------------------------------------------- OHLCV vs S3

def test_ohlcv_s3_crosscheck_five_pairs():
    res = dl.cross_check_ohlcv_vs_s3(n=5, seed=dl.CROSSCHECK_SEED,
                                     snapshot_cache=dl.DiskCachedS3Cache(s3_client=None))
    assert len(res["pairs"]) == 5
    bad = [r for r in res["pairs"] if r.get("status") != "MATCH"]
    assert not bad, f"cross-check failures: {bad}"


def test_ohlcv_coverage_and_vug_split_recorded():
    rep = json.loads((dl.CACHE / "ohlcv" / "coverage_report.json").read_text())
    assert rep["universe_size"] == 64
    assert rep["missing_symbols"] == []
    ok = [s for s, c in rep["symbols"].items() if c["status"] == "OK"]
    assert len(ok) == 64
    for s in ok:
        assert rep["symbols"][s]["last"] >= "2026-06-05", s
    # the honesty anchor: VUG 6:1 split must be on record for the feature builder
    vug_splits = rep["symbols"]["VUG"]["splits_since_start"]
    assert any(v == 6.0 and k.startswith("2026-04-2") for k, v in vug_splits.items()), vug_splits


# ----------------------------------------------------------------- FRED

@pytest.mark.parametrize("series", dl.FRED_SERIES)
def test_fred_visible_from_monotone_and_lagged(series):
    df = pd.read_parquet(dl.CACHE / "fred" / f"{series}.parquet")
    assert {"date", "value", "visible_from"} <= set(df.columns)
    df = df.sort_values("date")
    assert (pd.to_datetime(df["visible_from"]) > pd.to_datetime(df["date"])).all()
    assert pd.to_datetime(df["visible_from"]).is_monotonic_increasing


def test_fred_weekly_lag_rules():
    nfci = pd.read_parquet(dl.CACHE / "fred" / "NFCI.parquet")
    assert (pd.to_datetime(nfci["visible_from"]).dt.weekday == 2).all()  # Wednesday
    stl = pd.read_parquet(dl.CACHE / "fred" / "STLFSI4.parquet")
    assert (pd.to_datetime(stl["visible_from"]).dt.weekday == 3).all()  # Thursday
    icsa = pd.read_parquet(dl.CACHE / "fred" / "ICSA.parquet")
    lag = pd.to_datetime(icsa["visible_from"]) - pd.to_datetime(icsa["date"])
    assert (lag == pd.Timedelta(days=7)).all()


# ----------------------------------------------------------------- COT

@pytest.mark.parametrize("slug", list(dl.COT_MARKETS))
def test_cot_visible_from_monotone_and_publication_keyed(slug):
    df = pd.read_parquet(dl.CACHE / "cot" / f"{slug}.parquet")
    assert {"report_date", "publication", "visible_from",
            "lev_net_pct_oi", "asset_mgr_net_pct_oi"} <= set(df.columns)
    df = df.sort_values("report_date")
    pub = pd.to_datetime(df["publication"])
    vis = pd.to_datetime(df["visible_from"])
    rep = pd.to_datetime(df["report_date"])
    assert (pub > rep).all()           # publication strictly after report Tuesday
    assert (vis > pub).all()           # visible only after publication
    assert vis.is_monotonic_increasing
    # report dates are Tuesdays except holiday shifts; publication = +3 days
    assert (rep.dt.weekday == 1).mean() > 0.95
    assert ((pub - rep.dt.normalize()) == pd.Timedelta(days=3, hours=15, minutes=30)).all()
    # depth per proposal: TFF starts 2006
    assert str(rep.min().date()) <= "2006-09-01"
    assert len(df) > 950


# ----------------------------------------------------------------- deep history

def test_deep_history_loads():
    deep = dl.load_deep_history()
    af = deep["asset_features"]
    assert af["symbol"].nunique() == 64
    assert str(af["date"].min().date()) == "2014-08-29"
    assert "close" in af.columns
    cx = deep["context"]
    assert "spy_return_1d" in cx.columns and "credit_spread_proxy" in cx.columns
