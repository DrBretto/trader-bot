"""PKT-TB-006 SYN-1 prototype — mandatory alignment tests (BUILD_SPEC §10.4).

Run BEFORE any member trains:

  (a) 20 random (symbol, date) pairs in the replay window: feature-store close ==
      S3 snapshot close, split handling proven (VUG 6:1 applied once, never twice);
      tolerance 0.5%. VUG pre- and post-split pairs are force-included.
  (b) GDELT visible_from join: no decision date D joins a GDELT day d with d >= D;
      parquet invariant visible_from == gdelt_date + 1 day.
  (c) COT publication-key: visible_from strictly after publication (report Tuesday
      + 3d, Friday 15:30 ET); a decision date BETWEEN report date and publication
      joins the PREVIOUS week's report, never the unpublished one.
  (d) trailing-z no-look-ahead: perturbing all prices from a cutoff onward leaves
      every model INPUT (X, scalars, T, B, Z, masks) for decision dates <= cutoff
      bit-identical; same property unit-tested for the GDELT trailing-90d z.
  (e) target no-overlap: y5(D) is a function of open(D) and open(D+5) only
      (plus the same-day cross-section); scrambling every other day's opens leaves
      it unchanged; perturbing open(D+5) changes it.
  (f) live-date feature row == rebuilt-from-deep-history row on shared columns.
      Tolerance 2e-5 (NOT 1e-6) on price-derived columns: the deep parquet was
      computed from an earlier yfinance snapshot whose adj-close basis differs in
      the last float digit (verified ~7e-7 relative on closes) and the store holds
      float32; sources are identical in kind. Rates compared with a 1-observation
      publication-lag allowance: deep history joins FRED by observation date
      (no publication lag), the store joins by visible_from (BUILD_SPEC-binding),
      so on rate-change days the deep row can be one observation fresher.

Usage:  .venv/bin/python -m pytest tests/test_alignment.py -v
   or:  .venv/bin/python tests/test_alignment.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROTO))

from data_layer import DiskCachedS3Cache  # noqa: E402
from feature_store import FeatureStore, _asof_values  # noqa: E402
from features_gdelt import _trailing_z  # noqa: E402

STORE = PROTO / "store"
SEED = 4242
REPLAY_LO, REPLAY_HI = "2026-02-02", "2026-06-09"


def _panel():
    return np.load(STORE / "panel.npz", allow_pickle=False)


# --------------------------------------------------------------------- (a)

def test_a_close_vs_s3_snapshots():
    p = _panel()
    dates = [str(x) for x in p["dates"]]
    symbols = [str(x) for x in p["symbols"]]
    close = p["close_px"]
    cov = json.loads((PROTO / "cache" / "ohlcv" / "coverage_report.json").read_text())
    splits = {s: {k: float(v) for k, v in
                  cov["symbols"][s].get("splits_since_start", {}).items()
                  if not k.startswith("_")} for s in symbols}

    cache = DiskCachedS3Cache(s3_client=None)
    daily = cache.cache_dir / "daily"
    dirs = sorted(d.name for d in daily.iterdir() if (d / "prices.parquet").exists())

    def snapshot_close(sym: str, d: str) -> float:
        """d's close as the replay engine serves it on the NEXT night (earliest
        dir > d). KNOWN SUBSTRATE FACT (surfaced finding): 11/87 dirs carry a
        same-day preliminary row for the dir date itself whose close is an
        intraday-grade quote (e.g. SLV 2026-02-02 70.32 vs final 72.44); reading
        d from a dir > d always returns the final print."""
        for dd in dirs:
            if dd <= d:
                continue
            pr = cache.get_parquet(f"daily/{dd}/prices.parquet")
            pr["date"] = pd.to_datetime(pr["date"]).dt.normalize()
            row = pr[(pr["symbol"] == sym) & (pr["date"] == pd.Timestamp(d))]
            if len(row):
                return float(row["close"].iloc[0])
        raise AssertionError(f"{sym} {d} not in any later snapshot")

    rng = np.random.default_rng(SEED)
    pool = [d for d in dates if REPLAY_LO <= d <= REPLAY_HI]
    pairs = [(str(d), symbols[int(rng.integers(0, len(symbols)))])
             for d in rng.choice(pool, size=18, replace=False)]
    # force VUG pre- and post-split pairs (split handling must be PROVEN)
    pre = [d for d in pool if d < "2026-04-20"]
    post = [d for d in pool if d > "2026-04-22"]
    pairs += [(pre[len(pre) // 2], "VUG"), (post[len(post) // 2], "VUG")]

    failures, checked = [], []
    for d, sym in pairs:
        s3_close = snapshot_close(sym, d)
        k = dates.index(d)
        j = symbols.index(sym)
        factor = 1.0
        for sd, ratio in splits.get(sym, {}).items():
            if pd.Timestamp(sd) > pd.Timestamp(d):
                factor *= ratio
        ours = close[k, j] * factor
        rel = abs(ours - s3_close) / s3_close
        checked.append((sym, d, factor, rel))
        if rel > 0.005:
            failures.append((sym, d, ours, s3_close, rel))
    assert not failures, f"close mismatches > 0.5%: {failures}"
    assert any(f != 1.0 for _, _, f, _ in checked), "no split-reconciled pair checked"
    print(f"  (a) {len(checked)} pairs OK; max rel diff "
          f"{max(r for *_, r in checked):.2e}; split factors used: "
          f"{sorted({f for _, _, f, _ in checked})}")


# --------------------------------------------------------------------- (b)

def test_b_gdelt_visible_from():
    gd = pd.read_parquet(STORE / "gdelt_features.parquet")
    assert (pd.to_datetime(gd["visible_from"])
            == pd.to_datetime(gd["gdelt_date"]) + pd.Timedelta(days=1)).all()
    p = _panel()
    D = pd.to_datetime(p["dates"])
    joined = p["gdelt_day_joined"]
    bad = [(str(D[i].date()), str(joined[i])) for i in range(len(D))
           if str(joined[i]) != "" and pd.Timestamp(str(joined[i])) >= D[i]]
    assert not bad, f"GDELT rows joined to decision dates <= their GDELT day: {bad[:5]}"
    n = sum(1 for j in joined if j != "")
    print(f"  (b) {n} joined rows, all strictly d < D; visible_from == d+1 for "
          f"{len(gd)} parquet rows")


# --------------------------------------------------------------------- (c)

def test_c_cot_publication_key():
    n_checked = 0
    for slug in ["es", "ust10y", "vix"]:
        df = pd.read_parquet(PROTO / "cache" / "cot" / f"{slug}.parquet")
        rep = pd.to_datetime(df["report_date"])
        pub = pd.to_datetime(df["publication"])
        vis = pd.to_datetime(df["visible_from"])
        assert (pub > rep).all() and (vis > pub).all(), f"{slug}: key ordering broken"
        # decision date between report Tuesday and publication Friday must join
        # the PREVIOUS report, not this one
        df2 = df.copy()
        df2["visible_from"] = vis
        for i in [50, 200, len(df) - 10]:
            D = rep.iloc[i] + pd.Timedelta(days=2)        # Thursday after report
            j = _asof_values(pd.DatetimeIndex([D]), df2, ["report_date"])
            joined = pd.Timestamp(j["report_date"].iloc[0])
            assert joined < rep.iloc[i], (
                f"{slug}: D={D.date()} joined unpublished report {joined.date()}")
            assert (rep.iloc[i] - joined).days <= 14
            n_checked += 1
    print(f"  (c) 3 markets: visible_from > publication > report_date for all rows; "
          f"{n_checked} between-report joins all hit the prior week")


# --------------------------------------------------------------------- (d)

class _PerturbedStore(FeatureStore):
    def __init__(self, cutoff: str, factor: float = 1.7):
        super().__init__()
        self._cutoff = pd.Timestamp(cutoff)
        self._factor = factor

    def _load_prices(self):
        super()._load_prices()
        i = int(self.cal.searchsorted(self._cutoff))
        for k in self.px:
            self.px[k][i:] *= self._factor


def test_d_trailing_z_no_look_ahead(tmp_path=None):
    # GDELT trailing-90d z: perturb the future, past z unchanged
    rng = np.random.default_rng(7)
    days = pd.date_range("2023-01-01", periods=400, freq="D")
    s = pd.Series(rng.normal(size=400))
    z0 = _trailing_z(s, pd.Series(days))
    s2 = s.copy()
    s2.iloc[300:] += 50.0
    z1 = _trailing_z(s2, pd.Series(days))
    assert np.allclose(z0.iloc[:300].to_numpy(), z1.iloc[:300].to_numpy(),
                       equal_nan=True), "GDELT trailing z leaks future"

    # full input stack: perturb all prices from cutoff, inputs <= cutoff identical
    cutoff = "2025-07-01"
    a = FeatureStore().build(start="2024-06-02", end="2026-01-30",
                             out_name="_test_base.npz", verbose=False)
    b = _PerturbedStore(cutoff).build(start="2024-06-02", end="2026-01-30",
                                      out_name="_test_pert.npz", verbose=False)
    pa = np.load(STORE / "_test_base.npz", allow_pickle=False)
    pb = np.load(STORE / "_test_pert.npz", allow_pickle=False)
    sel = pd.to_datetime(pa["dates"]) <= pd.Timestamp(cutoff)
    n = int(sel.sum())
    assert n > 100
    for key in ["X", "scalars", "T", "B", "Z", "symbol_mask", "gdelt_available"]:
        ka, kb = pa[key][sel], pb[key][sel]
        assert np.array_equal(ka, kb), f"input {key} changed for dates <= cutoff"
    # and the perturbation DID reach the feature pipeline after the cutoff
    after = ~sel
    assert not np.array_equal(pa["X"][after], pb["X"][after]), \
        "perturbation never reached X (test is vacuous)"
    for f in ["_test_base.npz", "_test_pert.npz",
              "_test_base_meta.json", "_test_pert_meta.json"]:
        (STORE / f).unlink(missing_ok=True)
    print(f"  (d) inputs bit-identical for {n} decision dates <= {cutoff} under a "
          f"x1.7 price perturbation from {cutoff}; GDELT z past unchanged")


# --------------------------------------------------------------------- (e)

def test_e_target_no_overlap():
    p = _panel()
    O = p["open_px"].astype(np.float64)
    msk = p["symbol_mask"]
    y5 = p["y5_raw"]
    rng = np.random.default_rng(11)
    N = O.shape[0]
    H = 5
    for k in rng.integers(64, N - 10, size=6):
        k = int(k)
        # reconstruct from opens at D and D+5 ONLY
        r5 = O[k + H] / O[k] - 1.0
        r5 = np.where(msk[k], r5, np.nan)
        rec = r5 - np.nanmean(r5)
        assert np.allclose(rec, y5[k], atol=1e-10, equal_nan=True), \
            f"y5 at row {k} is not open(D)->open(D+5) minus cross-sec mean"
        # scramble every OTHER day's opens -> unchanged
        O2 = O * rng.uniform(2, 3, size=O.shape)
        O2[k], O2[k + H] = O[k], O[k + H]
        r5b = np.where(msk[k], O2[k + H] / O2[k] - 1.0, np.nan)
        assert np.allclose(r5b - np.nanmean(r5b), y5[k], atol=1e-10, equal_nan=True)
        # perturb open(D+5) -> must change
        O3 = O.copy()
        O3[k + H] *= 1.01
        r5c = np.where(msk[k], O3[k + H] / O3[k] - 1.0, np.nan)
        assert not np.allclose(r5c - np.nanmean(r5c), y5[k], atol=1e-6, equal_nan=True)
    # aux y1 horizon: open(D)->open(D+1)
    k = int(rng.integers(64, N - 10))
    r1 = np.where(msk[k], O[k + 1] / O[k] - 1.0, np.nan)
    z = (r1 - np.nanmean(r1)) / np.nanstd(r1)
    assert np.allclose(z, p["y1_z"][k], atol=1e-9, equal_nan=True)
    print("  (e) y5 == f(open(D), open(D+5)) only; y1 == next-day open return z; "
          "scramble-invariance + sensitivity proven on 6 sampled dates")


# --------------------------------------------------------------------- (f)

def test_f_live_row_vs_deep_history():
    p = _panel()
    dates = [str(x) for x in p["dates"]]
    symbols = [str(x) for x in p["symbols"]]
    T = p["T"]
    T_cols = list(p["T_cols"])
    af = pd.read_parquet(PROTO.parents[2] / "training" / "data"
                         / "asset_features_history.parquet")
    af["date"] = pd.to_datetime(af["date"])
    cx = pd.read_parquet(PROTO.parents[2] / "training" / "data"
                         / "historical_context.parquet")
    cx["date"] = pd.to_datetime(cx["date"])
    cx = cx.set_index("date")

    # T price-block columns vs deep engineered columns (same formulas, same basis)
    pairs = [("ret_1d", "return_1d"), ("ret_5d", "return_5d"),
             ("ret_21d", "return_21d"), ("ret_63d", "return_63d"),
             ("vol_21d", "vol_21d"), ("vol_63d", "vol_63d"),
             ("dd_21d", "drawdown_21d"), ("rs_21d", "rel_strength_21d")]
    rng = np.random.default_rng(SEED + 1)
    live_dates = [d for d in dates if "2025-09-01" <= d <= "2026-06-05"]
    sample = rng.choice(live_dates, size=8, replace=False)
    worst = 0.0
    n_cells = 0
    for d in sample:
        k = dates.index(d)
        dm1_idx = k - 1                                     # T row uses D-1 data
        dm1 = pd.Timestamp(dates[dm1_idx])
        deep = af[af["date"] == dm1].set_index("symbol")
        for j, sym in enumerate(symbols):
            if sym not in deep.index or not p["symbol_mask"][k, j]:
                continue
            for tc, dc in pairs:
                ours = float(T[k, j, T_cols.index(tc)])
                theirs = float(deep.loc[sym, dc])
                diff = abs(ours - theirs)
                worst = max(worst, diff)
                n_cells += 1
                assert diff <= 2e-5, (
                    f"{sym} {d} {tc}: store {ours} vs deep {theirs} (diff {diff:.2e})")
    # context columns (deep context ends 2026-02-03)
    ctx_pairs = [("ctx_spy_ret_1d", "spy_return_1d"), ("ctx_spy_ret_21d", "spy_return_21d"),
                 ("ctx_spy_vol_21d", "spy_vol_21d"),
                 ("ctx_credit_spread", "credit_spread_proxy"),
                 ("ctx_risk_off", "risk_off_proxy"), ("ctx_vixy_ret_21d", "vixy_return_21d")]
    rate_pairs = [("ctx_rate_2y", "rate_2y"), ("ctx_rate_10y", "rate_10y")]
    ctx_dates = [d for d in dates if "2025-09-01" <= d <= "2026-02-03"]
    worst_ctx, n_rate_lag = 0.0, 0
    for d in rng.choice(ctx_dates, size=8, replace=False):
        k = dates.index(d)
        dm1 = pd.Timestamp(dates[k - 1])
        if dm1 not in cx.index:
            continue
        for tc, dc in ctx_pairs:
            diff = abs(float(T[k, 0, T_cols.index(tc)]) - float(cx.loc[dm1, dc]))
            worst_ctx = max(worst_ctx, diff)
            assert diff <= 2e-5, f"{d} {tc}: diff {diff:.2e}"
        for tc, dc in rate_pairs:
            ours = float(T[k, 0, T_cols.index(tc)])
            deep_v = float(cx.loc[dm1, dc])
            if abs(ours - deep_v) > 1e-9:
                # documented divergence: deep joins FRED by observation date,
                # the store by visible_from (1-day publication lag) — the deep
                # row may be one observation fresher. Allow exactly that.
                n_rate_lag += 1
                assert abs(ours - deep_v) <= 0.30, (
                    f"{d} {tc}: {ours} vs {deep_v} — more than a 1-day rate move")
    print(f"  (f) {n_cells} price-feature cells: max |diff| {worst:.2e} (tol 2e-5, "
          f"float32 store + snapshot-basis drift, documented); context max |diff| "
          f"{worst_ctx:.2e}; rate publication-lag divergences: {n_rate_lag} "
          f"(deep=as-of-date vs store=visible_from, documented)")


if __name__ == "__main__":
    for fn in [test_a_close_vs_s3_snapshots, test_b_gdelt_visible_from,
               test_c_cot_publication_key, test_d_trailing_z_no_look_ahead,
               test_e_target_no_overlap, test_f_live_row_vs_deep_history]:
        print(f"RUN {fn.__name__}")
        fn()
    print("ALL ALIGNMENT TESTS PASSED")
