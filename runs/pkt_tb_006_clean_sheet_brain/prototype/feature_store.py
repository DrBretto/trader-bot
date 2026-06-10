"""PKT-TB-006 SYN-1 prototype — the FeatureStore (BUILD_SPEC §4, §10.2, §9.2).

Builds per-decision-date artifacts for BOTH regimes (deep/training 2015-02-18 ->
2026-06-10 and the replay window, which is inside that range) as ONE consolidated
``store/panel.npz`` + ``store/panel_meta.json``. Layout (N decision dates, S=64
symbols, K=27 buckets; all model inputs float32, price/target panels float64):

  INPUTS (all built from data through D-1 close; §1 "all inputs = data through D-1"):
    X        [N,S,63,14]  per-symbol CAST panel; each of the 14 features z-scored
                          per (symbol, feature) over the trailing 252 trading days
                          with stats from days < D (one stat pair per decision date,
                          applied to the whole 63-day window).
    scalars  [N,S,4]      cross-sectional z at D-1: return_1d, vol_21d,
                          rel_strength_21d, volume_z_21d.
    T        [N,S,56]     tabular row per symbol (T_cols).
    B        [N,K,26]     bucket event panel (B_cols); LLM columns mask-interacted.
    Z        [N,24]       executive context incl. the information-health block (Z_cols).
    symbol_mask [N,S]     symbol tradable at D (inception + 70 trading days, §2.1).
    gdelt_day_joined [N]  the GDELT UTC day actually joined ('' if none) — audit column.

  TARGETS (NEVER part of X/T/B/Z — §10.2):
    y5_raw   [N,S]  open(D)->open(D+5) return minus cross-sectional mean (valid symbols)
    y5_rank  [N,S]  per-day Gaussian-rank transform of y5_raw
    y1_z     [N,S]  next-day open(D)->open(D+1) return, cross-sectional z
    y5_bucket[N,K]  bucket 5d abnormal return: equal-weight member open(D)->open(D+5)
                    return minus the cross-sectional market mean (EventHead target)
    fwd_vol5 [N,S]  realized vol of close returns D+1..D+5 (annualized)   (RiskNet+ a)
    fwd_beta63 [N,S] beta-to-SPY over close returns D+1..D+63             (RiskNet+ b)
    fwd_book_vol5 [N] realized vol of the equal-weight 64-ETF book D+1..D+5 (RiskNet+ c)
    w_rec_score [N,S] raw Infotropy Transfer-B record score in [0,1] (§9.2)
    w_rec    [N,S]  clip(eps + score, eps, 1) at the default eps
    open_px / close_px [N,S] (float64) split-adjusted open/close AT date D — price
                    panels for targets + alignment tests, not features.

Price conventions (forced choices, logged in validation_looks.jsonl):
  - return/vol/drawdown/rel-strength/52w features use yfinance adj_close
    (dividend+split adjusted) — this is what training/data/asset_features_history
    was built from (verified to ~1e-6), so live rows == rebuilt-deep-history rows.
  - targets use yfinance open (split-adjusted, NOT dividend-adjusted) — split-
    consistent open->open returns; S3 snapshots are unadjusted, reconciled via the
    splits table in cache/ohlcv/coverage_report.json (alignment test a).
  - gap_open_pct / range_pct / volume use raw (split-adjusted) OHLCV.

LLM columns: read store/llm_features.parquet IF present (built in parallel);
otherwise zeros + llm_available=0 masks. Re-running build() after that parquet
lands merges the LLM columns; everything else is deterministic and identical.

Infotropy Transfer-B (§9.2): regime_stat = {21d realized vol, 21d drift}; shift =
|stat(t+5) - stat(t-1)| / sd_base, sd_base = trailing-252d sd (days < t) of the
6-day stat change; the two stats are averaged; x (1 - fraction_retraced within
h=5); clipped to [0,1]. eps is a PARAMETER (gene record_weight_eps, default 0.25).
"""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROTO = Path(__file__).resolve().parent
CACHE = PROTO / "cache"
DICTS = PROTO / "dicts"
STORE = PROTO / "store"

PANEL_START = "2015-02-18"
PANEL_END = "2026-06-10"
H = 5                       # member/executive horizon (registered constant)
Z_WINDOW = 252              # trailing z window for X (trading days)
Z_MIN_PERIODS = 126         # forced choice, logged
ENTRY_LAG_TD = 70           # symbols enter at inception + 70 trading days (§2.1)
GDELT_STALE_DAYS = 5        # calendar days; staler => gdelt_available=0 (forced, logged)
COT_FRESH_DAYS = 10         # calendar days since publication => cot_fresh
RECORD_WEIGHT_EPS = 0.25    # gene default (D4)

X_FEATURES = ["return_1d", "return_5d", "return_21d", "return_63d",
              "vol_21d", "vol_63d", "drawdown_21d", "drawdown_63d",
              "rel_strength_21d", "rel_strength_63d", "range_pct",
              "volume_z_21d", "gap_open_pct", "dist_from_52w_high"]

EVENT_FLAGS = [f"llm_event_{i:02d}" for i in range(12)]  # names resolve at LLM merge

# bucket27 -> GDELT-20 theme-bucket fallback (forced choice, logged): the frozen
# theme_to_sector.json has no country/semis granularity; nearest sleeves used.
GDELT_BUCKET_FALLBACK = {
    "semis": "tech", "china": "em_broad", "india": "em_broad", "brazil": "em_broad",
    "japan": "em_broad", "intl_dev": "em_broad", "europe": "em_broad",
}
# factor/style ETFs absent from bucket_map consume us_broad (BUILD_SPEC §3).
FACTOR_STYLE_BUCKET = "us_broad"


# ----------------------------------------------------------------- small helpers

def _asof_values(decision_dates: pd.DatetimeIndex, df: pd.DataFrame,
                 value_cols: List[str], on: str = "visible_from") -> pd.DataFrame:
    """As-of join: latest row with df[on] <= D for each decision date."""
    right = df.copy()
    right[on] = pd.to_datetime(right[on]).astype("datetime64[ns]")
    right = right.sort_values(on).reset_index(drop=True)
    left = pd.DataFrame({"D": pd.DatetimeIndex(decision_dates).astype("datetime64[ns]")})
    out = pd.merge_asof(left, right, left_on="D", right_on=on, direction="backward")
    out.index = decision_dates
    return out[[c for c in value_cols if c in out.columns] + [on]]


def _trailing_z_obs(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    """z of each observation vs the trailing `window` OBSERVATIONS strictly before it."""
    sh = s.shift(1)
    m = sh.rolling(window, min_periods=min_periods).mean()
    sd = sh.rolling(window, min_periods=min_periods).std()
    return (s - m) / sd.replace(0.0, np.nan)


def _fwd_window_stats(r: np.ndarray, lo: int, hi: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per row i: (sum, sumsq, count) of r over rows [i+lo, i+hi] (NaN-aware).

    r: [n] or [n,S]. Full-window requirement is enforced by the caller via count."""
    x = np.nan_to_num(r, nan=0.0)
    v = (~np.isnan(r)).astype(np.float64)
    cs = np.cumsum(x, axis=0)
    cs2 = np.cumsum(x * x, axis=0)
    cv = np.cumsum(v, axis=0)

    def rng(c):
        n = c.shape[0]
        out = np.full_like(c, np.nan, dtype=np.float64)
        pad = np.concatenate([np.zeros((1,) + c.shape[1:]), c], axis=0)  # pad[k]=sum<k
        for i in range(n):
            a, b = i + lo, i + hi
            if b >= n:
                continue
            out[i] = pad[b + 1] - pad[a]
        return out

    return rng(cs), rng(cs2), rng(cv)


def gaussian_rank(y: np.ndarray) -> np.ndarray:
    """Per-day Gaussian-rank transform of a [N,S] target (NaN preserved)."""
    from scipy.stats import norm, rankdata
    out = np.full_like(y, np.nan)
    for i in range(y.shape[0]):
        row = y[i]
        m = ~np.isnan(row)
        n = int(m.sum())
        if n < 2:
            continue
        rk = rankdata(row[m])               # 1..n, average ties
        out[i, m] = norm.ppf((rk - 0.5) / n)
    return out


# ----------------------------------------------------------------- the store

class FeatureStore:
    """Builds + persists the consolidated SYN-1 feature/target panel."""

    def __init__(self, proto_dir: Path = PROTO, eps: float = RECORD_WEIGHT_EPS):
        self.proto = Path(proto_dir)
        self.cache = self.proto / "cache"
        self.store = self.proto / "store"
        self.eps = eps
        self.universe = pd.read_csv(self.proto.parents[2] / "config" / "universe.csv")
        self.symbols = self.universe["symbol"].astype(str).str.strip().tolist()
        self.bucket_map: Dict[str, List[str]] = json.loads(
            (self.proto / "dicts" / "bucket_map.json").read_text())
        self.buckets = list(self.bucket_map.keys())
        self.sym_bucket = {s: b for b, syms in self.bucket_map.items() for s in syms}
        for s in self.symbols:
            self.sym_bucket.setdefault(s, FACTOR_STYLE_BUCKET)
        self.actor_map = json.loads((self.proto / "dicts" / "ACTOR_MAP.json").read_text())
        self.gdelt_theme_buckets = sorted(set(json.loads(
            (self.proto / "dicts" / "theme_to_sector.json").read_text()).values()))

    # ------------------------------------------------------------- price layer

    def _load_prices(self) -> None:
        """Aligned [n_cal, S] float64 panels on the SPY trading calendar."""
        spy = pd.read_parquet(self.cache / "ohlcv" / "SPY.parquet")
        cal = pd.DatetimeIndex(pd.to_datetime(spy["date"]).dt.normalize()).unique().sort_values()
        self.cal = cal
        n, S = len(cal), len(self.symbols)
        panels = {k: np.full((n, S), np.nan) for k in
                  ["open", "high", "low", "close", "adj_close", "volume"]}
        for j, sym in enumerate(self.symbols):
            df = pd.read_parquet(self.cache / "ohlcv" / f"{sym}.parquet")
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df = df.set_index("date").reindex(cal)
            for k in panels:
                panels[k][:, j] = df[k].to_numpy(dtype=np.float64)
        self.px = panels
        cov = json.loads((self.cache / "ohlcv" / "coverage_report.json").read_text())
        self.splits = {s: {k: float(v) for k, v in
                           cov["symbols"][s].get("splits_since_start", {}).items()
                           if not k.startswith("_")}
                       for s in self.symbols if cov["symbols"].get(s, {}).get("status") == "OK"}

    def _compute_features(self) -> np.ndarray:
        """F [n_cal, S, 14] raw (pre-z) features, deep-history conventions."""
        a = self.px["adj_close"]
        o, h, l, c, v = (self.px[k] for k in ["open", "high", "low", "close", "volume"])
        n, S = a.shape
        A = pd.DataFrame(a)
        r1 = A.pct_change(1)
        F = np.full((n, S, len(X_FEATURES)), np.nan)
        F[:, :, 0] = r1.to_numpy()
        F[:, :, 1] = A.pct_change(5).to_numpy()
        F[:, :, 2] = A.pct_change(21).to_numpy()
        F[:, :, 3] = A.pct_change(63).to_numpy()
        F[:, :, 4] = (r1.rolling(21).std() * np.sqrt(252)).to_numpy()
        F[:, :, 5] = (r1.rolling(63).std() * np.sqrt(252)).to_numpy()
        pk21 = A.rolling(21, min_periods=1).max()
        pk63 = A.rolling(63, min_periods=1).max()
        F[:, :, 6] = ((A - pk21) / pk21).to_numpy()
        F[:, :, 7] = ((A - pk63) / pk63).to_numpy()
        spy_j = self.symbols.index("SPY")
        F[:, :, 8] = F[:, :, 2] - F[:, spy_j:spy_j + 1, 2]
        F[:, :, 9] = F[:, :, 3] - F[:, spy_j:spy_j + 1, 3]
        F[:, :, 10] = (h - l) / c
        V = pd.DataFrame(v)
        vm = V.rolling(21).mean()
        vs = V.rolling(21).std()
        F[:, :, 11] = ((V - vm) / vs.replace(0.0, np.nan)).to_numpy()
        C = pd.DataFrame(c)
        F[:, :, 12] = (pd.DataFrame(o) / C.shift(1) - 1.0).to_numpy()
        hi52 = A.rolling(252, min_periods=63).max()
        F[:, :, 13] = (A / hi52 - 1.0).to_numpy()
        self.r1 = r1.to_numpy()           # adj daily returns (for targets/w_rec)
        return F

    # ----------------------------------------------------------- context layers

    def _cboe(self) -> pd.DataFrame:
        """S1 features on CBOE observation dates; visible_from = date + 1 day."""
        s = {}
        for idx in ["VIX", "VIX9D", "VIX3M", "VVIX", "SKEW", "COR3M", "VXN"]:
            df = pd.read_parquet(self.cache / "cboe" / f"{idx}.parquet")
            s[idx] = df.set_index(pd.to_datetime(df["date"]))["close"]
        f = pd.DataFrame(s).sort_index()
        out = pd.DataFrame(index=f.index)
        out["s1_term_slope"] = (f["VIX9D"] - f["VIX3M"]) / f["VIX"]
        out["s1_curvature"] = f["VIX"] - (f["VIX9D"] + f["VIX3M"]) / 2.0
        out["s1_cor3m"] = f["COR3M"]
        out["s1_cor3m_z"] = _trailing_z_obs(f["COR3M"].dropna(), 252, 126).reindex(f.index)
        out["s1_vxn_minus_vix"] = f["VXN"] - f["VIX"]
        out["s1_vvix"] = f["VVIX"]
        out["s1_vvix_z"] = _trailing_z_obs(f["VVIX"].dropna(), 252, 126).reindex(f.index)
        out["s1_skew_z"] = _trailing_z_obs(f["SKEW"].dropna(), 252, 126).reindex(f.index)
        out = out.reset_index().rename(columns={"index": "date"})
        out["visible_from"] = out["date"] + pd.Timedelta(days=1)   # "drop any row dated >= D"
        return out

    def _fred(self) -> pd.DataFrame:
        """S3 features keyed by each observation's visible_from (publication lag)."""
        def load(series):
            df = pd.read_parquet(self.cache / "fred" / f"{series}.parquet")
            df["date"] = pd.to_datetime(df["date"])
            df["visible_from"] = pd.to_datetime(df["visible_from"])
            return df.sort_values("date").reset_index(drop=True)

        frames = []
        nfci = load("NFCI"); nfci["fred_nfci"] = nfci["value"]
        stl = load("STLFSI4"); stl["fred_stlfsi4"] = stl["value"]
        t10 = load("T10YIE"); t10["fred_t10yie_d21"] = t10["value"].diff(21)
        dfi = load("DFII10")
        dfi["fred_dfii10"] = dfi["value"]; dfi["fred_dfii10_d63"] = dfi["value"].diff(63)
        hy = load("BAMLH0A0HYM2")
        hy["fred_hy_oas"] = hy["value"]; hy["fred_hy_oas_d5"] = hy["value"].diff(5)
        ic = load("ICSA"); ic["fred_icsa_z"] = _trailing_z_obs(ic["value"], 52, 26)
        for df, cols in [(nfci, ["fred_nfci"]), (stl, ["fred_stlfsi4"]),
                         (t10, ["fred_t10yie_d21"]), (dfi, ["fred_dfii10", "fred_dfii10_d63"]),
                         (hy, ["fred_hy_oas", "fred_hy_oas_d5"]), (ic, ["fred_icsa_z"])]:
            frames.append(df[["visible_from"] + cols])
        return frames

    def _fred_rates(self) -> pd.DataFrame:
        """DGS2/DGS10 (context rates), visible_from-keyed."""
        out = []
        for series, col in [("DGS2", "rate_2y"), ("DGS10", "rate_10y")]:
            df = pd.read_parquet(self.cache / "fred" / f"{series}.parquet")
            df["visible_from"] = pd.to_datetime(df["visible_from"])
            df[col] = df["value"]
            out.append(df[["visible_from", col]])
        return out

    def _cot(self) -> pd.DataFrame:
        """S2 1y-z of leveraged-fund net %OI per market, visible_from-keyed."""
        frames = []
        for slug in ["es", "ust10y", "vix"]:
            df = pd.read_parquet(self.cache / "cot" / f"{slug}.parquet")
            df["visible_from"] = pd.to_datetime(df["visible_from"])
            df = df.sort_values("report_date").reset_index(drop=True)
            df[f"cot_{slug}_lev_net_z"] = _trailing_z_obs(df["lev_net_pct_oi"], 52, 26)
            df[f"cot_{slug}_pub"] = pd.to_datetime(df["publication"])
            frames.append(df[["visible_from", f"cot_{slug}_lev_net_z", f"cot_{slug}_pub"]])
        return frames

    def _llm(self) -> Optional[pd.DataFrame]:
        p = self.store / "llm_features.parquet"
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        if "date" not in df.columns:        # features_llm writes date as the index
            df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").sort_index()

    # ------------------------------------------------------------------- build

    def build(self, start: str = PANEL_START, end: str = PANEL_END,
              out_name: str = "panel.npz", verbose: bool = True) -> Dict:
        t0 = time.time()
        self._load_prices()
        cal = self.cal
        F = self._compute_features()
        n_cal, S, K = len(cal), len(self.symbols), len(self.buckets)

        # trailing-252d z stats from days < D (per symbol x feature)
        Fdf = pd.DataFrame(F.reshape(n_cal, S * len(X_FEATURES)))
        sh = Fdf.shift(1)
        m = sh.rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).mean().to_numpy() \
            .reshape(n_cal, S, len(X_FEATURES))
        sd = sh.rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).std().to_numpy() \
            .reshape(n_cal, S, len(X_FEATURES)).copy()
        sd[sd == 0] = np.nan

        # decision dates
        d_mask = (cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))
        didx = np.where(d_mask)[0]
        didx = didx[didx >= 64]            # need a full 63-day window ending at D-1
        dates = cal[didx]
        N = len(didx)

        # ---- X / scalars --------------------------------------------------
        X = np.zeros((N, S, 63, 14), dtype=np.float32)
        for k, i in enumerate(didx):
            win = F[i - 63:i]                                  # rows D-63..D-1
            X[k] = np.nan_to_num((win - m[i]) / sd[i], nan=0.0,
                                 posinf=0.0, neginf=0.0).transpose(1, 0, 2)
        scalars = np.zeros((N, S, 4), dtype=np.float32)
        scal_feats = [0, 4, 8, 11]                             # ret1, vol21, rs21, volz21
        for k, i in enumerate(didx):
            for a, f in enumerate(scal_feats):
                row = F[i - 1, :, f]
                mu, sg = np.nanmean(row), np.nanstd(row)
                scalars[k, :, a] = np.nan_to_num((row - mu) / (sg if sg > 0 else np.nan), nan=0.0)

        # ---- symbol mask ---------------------------------------------------
        first_idx = np.full(S, n_cal, dtype=int)
        has = ~np.isnan(self.px["close"])
        for j in range(S):
            nz = np.nonzero(has[:, j])[0]
            if len(nz):
                first_idx[j] = nz[0]
        symbol_mask = np.zeros((N, S), dtype=bool)
        for k, i in enumerate(didx):
            symbol_mask[k] = has[i - 1] & ((i - 1 - first_idx) >= ENTRY_LAG_TD)

        # ---- context / CBOE / FRED / COT joins (visible_from discipline) ---
        cboe = self._cboe()
        cboe_cols = ["s1_term_slope", "s1_curvature", "s1_cor3m", "s1_cor3m_z",
                     "s1_vxn_minus_vix", "s1_vvix", "s1_vvix_z", "s1_skew_z"]
        cboe_j = _asof_values(dates, cboe, cboe_cols)

        fred_join = {}
        for fr in self._fred():
            cols = [c for c in fr.columns if c != "visible_from"]
            j = _asof_values(dates, fr, cols)
            for c in cols:
                fred_join[c] = j[c].to_numpy(dtype=np.float64)
        for fr in self._fred_rates():
            cols = [c for c in fr.columns if c != "visible_from"]
            j = _asof_values(dates, fr, cols)
            for c in cols:
                fred_join[c] = j[c].to_numpy(dtype=np.float64)

        cot_join, cot_pub = {}, {}
        for fr in self._cot():
            zc = [c for c in fr.columns if c.endswith("_lev_net_z")][0]
            pc = [c for c in fr.columns if c.endswith("_pub")][0]
            j = _asof_values(dates, fr, [zc, pc])
            cot_join[zc] = j[zc].to_numpy(dtype=np.float64)
            cot_pub[zc] = pd.to_datetime(j[pc])
        pub_age = (dates.values[:, None]
                   - np.stack([cot_pub[c].values for c in sorted(cot_pub)], axis=1))
        cot_fresh = (np.nanmax(pub_age.astype("timedelta64[D]").astype(float), axis=1)
                     <= COT_FRESH_DAYS).astype(np.float32)

        # context block from own panels (deep-history conventions, verified)
        spy_j = self.symbols.index("SPY")
        ctx = {
            "ctx_spy_ret_1d": F[:, spy_j, 0], "ctx_spy_ret_21d": F[:, spy_j, 2],
            "ctx_spy_vol_21d": F[:, spy_j, 4],
            "ctx_credit_spread": F[:, self.symbols.index("HYG"), 2]
                                 - F[:, self.symbols.index("IEF"), 2],
            "ctx_risk_off": F[:, spy_j, 2] - F[:, self.symbols.index("TLT"), 2],
            "ctx_vixy_ret_21d": F[:, self.symbols.index("VIXY"), 2],
        }
        ctx_at_D = {k: v[didx - 1] for k, v in ctx.items()}     # value at D-1
        ctx_at_D["ctx_rate_2y"] = fred_join["rate_2y"]
        ctx_at_D["ctx_rate_10y"] = fred_join["rate_10y"]
        ctx_at_D["ctx_yield_slope"] = fred_join["rate_10y"] - fred_join["rate_2y"]
        ctx_at_D["ctx_vix_term_slope"] = cboe_j["s1_term_slope"].to_numpy()
        ctx_at_D["ctx_vvix_z"] = cboe_j["s1_vvix_z"].to_numpy()
        ctx_at_D["ctx_skew_z"] = cboe_j["s1_skew_z"].to_numpy()

        # breadth (S0) at D-1
        c_adj = self.px["adj_close"]
        Adf = pd.DataFrame(c_adj)
        ma50 = Adf.rolling(50).mean().to_numpy()
        ma200 = Adf.rolling(200).mean().to_numpy()
        rsp_j = self.symbols.index("RSP")
        br = {}
        i1 = didx - 1
        with np.errstate(invalid="ignore"):
            br["br_adv_dec"] = np.nanmean(self.r1[i1] > 0, axis=1)
            br["br_pct_above_50d"] = np.nanmean(c_adj[i1] > ma50[i1], axis=1)
            br["br_pct_above_200d"] = np.nanmean(c_adj[i1] > ma200[i1], axis=1)
            br["br_dispersion"] = np.nanstd(self.r1[i1], axis=1)
            br["br_rsp_spy_21"] = F[i1, rsp_j, 2] - F[i1, spy_j, 2]

        # ---- GDELT join ----------------------------------------------------
        gd = pd.read_parquet(self.store / "gdelt_features.parquet")
        gd["visible_from"] = pd.to_datetime(gd["visible_from"])
        gd["gdelt_date"] = pd.to_datetime(gd["gdelt_date"])
        gd_cols = [c for c in gd.columns if c.startswith(("g1_", "g2_", "g3_", "g4_", "g5_"))]
        gj = _asof_values(dates, gd, gd_cols + ["gdelt_date"])
        gd_day = pd.to_datetime(gj["gdelt_date"])
        days_since = (dates.values - gd_day.values).astype("timedelta64[D]").astype(float)
        gd_avail = ((~gd_day.isna()).to_numpy() & (days_since <= GDELT_STALE_DAYS)) \
            .astype(np.float32)
        GJ = {c: np.where(gd_avail > 0, gj[c].to_numpy(dtype=np.float64), 0.0)
              for c in gd_cols}
        days_since_capped = np.where(np.isnan(days_since), 30.0, np.minimum(days_since, 30.0))

        # ---- LLM (lazy) ------------------------------------------------------
        llm = self._llm()
        llm_avail = np.zeros(N, dtype=np.float32)
        llm_status_ok = np.zeros(N, dtype=np.float32)
        n_clusters_z = np.zeros(N, dtype=np.float32)
        llm_axes = {k: np.zeros(N) for k in
                    ["llm_risk_appetite", "llm_rates_pressure", "llm_geopol_risk"]}
        llm_bucket = {b: {k: np.zeros(N) for k in ["sent", "conf", "sal", "sent_d1"]}
                      for b in self.buckets}
        llm_flags = {f: np.zeros(N) for f in EVENT_FLAGS}
        llm_merged = False
        if llm is not None:
            llm_merged = True
            sub = llm.reindex(dates)
            llm_avail = np.nan_to_num(sub.get("llm_available",
                                              pd.Series(0.0, index=dates)).to_numpy(), nan=0.0).astype(np.float32)
            llm_status_ok = np.nan_to_num(sub.get("llm_status_ok",
                                                  pd.Series(0.0, index=dates)).to_numpy(), nan=0.0).astype(np.float32)
            if "n_clusters" in sub.columns:
                n_clusters_z = np.nan_to_num(
                    _trailing_z_obs(llm["n_clusters"], 252, 30).reindex(dates).to_numpy(),
                    nan=0.0).astype(np.float32)
            for k in llm_axes:
                if k in sub.columns:
                    llm_axes[k] = np.nan_to_num(sub[k].to_numpy(), nan=0.0)
            for b in self.buckets:
                for k, pat in [("sent", f"llm_sent_{b}"), ("conf", f"llm_conf_{b}"),
                               ("sal", f"llm_sal_{b}"), ("sent_d1", f"llm_sent_{b}_d1")]:
                    if pat in sub.columns:
                        llm_bucket[b][k] = np.nan_to_num(sub[pat].to_numpy(), nan=0.0)
            flag_cols = sorted(c for c in sub.columns if c.startswith("llm_event_"))
            for i_f, c in enumerate(flag_cols[:12]):
                llm_flags[EVENT_FLAGS[i_f]] = np.nan_to_num(sub[c].to_numpy(), nan=0.0)

        # ---- T [N,S,56] ------------------------------------------------------
        T_cols = (["ret_1d", "ret_5d", "ret_21d", "ret_63d", "vol_21d", "vol_63d",
                   "dd_21d", "rs_21d"]
                  + ["ctx_spy_ret_1d", "ctx_spy_ret_21d", "ctx_spy_vol_21d",
                     "ctx_rate_2y", "ctx_rate_10y", "ctx_yield_slope",
                     "ctx_credit_spread", "ctx_risk_off", "ctx_vixy_ret_21d",
                     "ctx_vix_term_slope", "ctx_vvix_z", "ctx_skew_z"]
                  + ["s1_curvature", "s1_cor3m", "s1_cor3m_z", "s1_vxn_minus_vix", "s1_vvix"]
                  + ["fred_nfci", "fred_stlfsi4", "fred_t10yie_d21", "fred_dfii10",
                     "fred_dfii10_d63", "fred_hy_oas", "fred_hy_oas_d5", "fred_icsa_z"]
                  + ["cot_es_lev_net_z", "cot_ust10y_lev_net_z", "cot_vix_lev_net_z"]
                  + ["gd_g1_share", "gd_g1_z", "gd_g3_tone", "gd_g4_burst_z"]
                  + ["gd_novelty", "gd_novelty_21", "gd_hhi_loc", "gd_hhi_org"]
                  + ["llm_sent", "llm_conf", "llm_sal", "llm_sent_d1"]
                  + ["m_llm_available", "m_cot_fresh", "m_gdelt_available"]
                  + ["br_adv_dec", "br_pct_above_50d", "br_pct_above_200d",
                     "br_dispersion", "br_rsp_spy_21"])
        T = np.zeros((N, S, len(T_cols)), dtype=np.float32)
        price_feats = [0, 1, 2, 3, 4, 5, 6, 8]                  # F indices for price block
        for a, f in enumerate(price_feats):
            T[:, :, a] = np.nan_to_num(F[i1][:, :, f], nan=0.0)
        col = {c: T_cols.index(c) for c in T_cols}

        def bcast(name, vec):                                   # [N] -> all symbols
            T[:, :, col[name]] = np.nan_to_num(vec, nan=0.0).astype(np.float32)[:, None]

        for c in ["ctx_spy_ret_1d", "ctx_spy_ret_21d", "ctx_spy_vol_21d", "ctx_rate_2y",
                  "ctx_rate_10y", "ctx_yield_slope", "ctx_credit_spread", "ctx_risk_off",
                  "ctx_vixy_ret_21d", "ctx_vix_term_slope", "ctx_vvix_z", "ctx_skew_z"]:
            bcast(c, ctx_at_D[c])
        for c in ["s1_curvature", "s1_cor3m", "s1_cor3m_z", "s1_vxn_minus_vix", "s1_vvix"]:
            bcast(c, cboe_j[c].to_numpy())
        for c in ["fred_nfci", "fred_stlfsi4", "fred_t10yie_d21", "fred_dfii10",
                  "fred_dfii10_d63", "fred_hy_oas", "fred_hy_oas_d5", "fred_icsa_z"]:
            bcast(c, fred_join[c])
        for c in ["cot_es_lev_net_z", "cot_ust10y_lev_net_z", "cot_vix_lev_net_z"]:
            bcast(c, cot_join[c])
        # per-symbol GDELT bucket features
        for j, sym in enumerate(self.symbols):
            b27 = self.sym_bucket[sym]
            gb = b27 if b27 in self.gdelt_theme_buckets else GDELT_BUCKET_FALLBACK.get(b27, "us_broad")
            T[:, j, col["gd_g1_share"]] = np.nan_to_num(GJ[f"g1_share_{gb}"], nan=0.0).astype(np.float32)
            T[:, j, col["gd_g1_z"]] = np.nan_to_num(GJ[f"g1_z_{gb}"], nan=0.0).astype(np.float32)
            T[:, j, col["gd_g3_tone"]] = np.nan_to_num(GJ[f"g3_tone_{gb}"], nan=0.0).astype(np.float32)
            T[:, j, col["gd_g4_burst_z"]] = np.nan_to_num(GJ[f"g4_burst_z_{gb}"], nan=0.0).astype(np.float32)
            for k in ["sent", "conf", "sal", "sent_d1"]:
                T[:, j, col[f"llm_{k}"]] = (llm_bucket[b27][k] * llm_avail).astype(np.float32)
        for c, vec in [("gd_novelty", GJ["g4_theme_novelty"]),
                       ("gd_novelty_21", GJ["g4_theme_novelty_21"]),
                       ("gd_hhi_loc", GJ["g5_hhi_loc"]), ("gd_hhi_org", GJ["g5_hhi_org"])]:
            bcast(c, vec)
        bcast("m_llm_available", llm_avail)
        bcast("m_cot_fresh", cot_fresh)
        bcast("m_gdelt_available", gd_avail)
        for c in ["br_adv_dec", "br_pct_above_50d", "br_pct_above_200d",
                  "br_dispersion", "br_rsp_spy_21"]:
            bcast(c, br[c])

        # ---- B [N,27,26] -----------------------------------------------------
        B_cols = (["llm_sent_x", "llm_conf_x", "llm_sal_x", "llm_sent_d1_x", "llm_available"]
                  + ["g1_z", "g3_tone", "g3_tone_dispersion", "g2_goldstein", "g2_conflict",
                     "g4_burst_z", "g4_novelty", "g5_hhi_loc", "g5_hhi_org"]
                  + [f + "_x" for f in EVENT_FLAGS])
        B = np.zeros((N, K, len(B_cols)), dtype=np.float32)
        bcol = {c: B_cols.index(c) for c in B_cols}
        cc_for_bucket = {info["bucket"]: cc for cc, info in self.actor_map.items()}
        for kb, b in enumerate(self.buckets):
            for a, k in enumerate(["sent", "conf", "sal", "sent_d1"]):
                B[:, kb, a] = (llm_bucket[b][k] * llm_avail).astype(np.float32)
            B[:, kb, bcol["llm_available"]] = llm_avail
            gb = b if b in self.gdelt_theme_buckets else GDELT_BUCKET_FALLBACK.get(b, "us_broad")
            B[:, kb, bcol["g1_z"]] = np.nan_to_num(GJ[f"g1_z_{gb}"], nan=0.0)
            B[:, kb, bcol["g3_tone"]] = np.nan_to_num(GJ[f"g3_tone_{gb}"], nan=0.0)
            B[:, kb, bcol["g3_tone_dispersion"]] = np.nan_to_num(GJ["g3_tone_dispersion"], nan=0.0)
            B[:, kb, bcol["g4_burst_z"]] = np.nan_to_num(GJ[f"g4_burst_z_{gb}"], nan=0.0)
            B[:, kb, bcol["g4_novelty"]] = np.nan_to_num(GJ["g4_theme_novelty"], nan=0.0)
            B[:, kb, bcol["g5_hhi_loc"]] = np.nan_to_num(GJ["g5_hhi_loc"], nan=0.0)
            B[:, kb, bcol["g5_hhi_org"]] = np.nan_to_num(GJ["g5_hhi_org"], nan=0.0)
            cc = cc_for_bucket.get(b)
            if cc is not None:
                B[:, kb, bcol["g2_goldstein"]] = np.nan_to_num(
                    GJ[f"g2_goldstein_{cc.lower()}"], nan=0.0)
                B[:, kb, bcol["g2_conflict"]] = np.nan_to_num(
                    GJ[f"g2_conflict_{cc.lower()}"], nan=0.0)
            for f in EVENT_FLAGS:
                B[:, kb, bcol[f + "_x"]] = (llm_flags[f] * llm_avail).astype(np.float32)

        # ---- Z [N,24] ----------------------------------------------------------
        Z_cols = ["z_spy_ret_21", "z_spy_vol_21", "z_rate_2y", "z_rate_10y",
                  "z_yield_slope", "z_credit_spread", "z_risk_off", "z_vixy_ret_21",
                  "z_vix_term_slope", "z_skew_z",
                  "z_cor3m_z", "z_curvature", "z_vxn_minus_vix", "z_vvix_z",
                  "z_llm_risk_appetite", "z_llm_rates_pressure", "z_llm_geopol_risk",
                  "z_llm_status_ok",
                  "ih_gdelt_available", "ih_llm_status_ok", "ih_n_clusters_z",
                  "ih_theme_novelty", "ih_field_dispersion", "ih_days_since_last_gdelt"]
        Zv = np.zeros((N, len(Z_cols)), dtype=np.float32)
        zsrc = [ctx_at_D["ctx_spy_ret_21d"], ctx_at_D["ctx_spy_vol_21d"],
                ctx_at_D["ctx_rate_2y"], ctx_at_D["ctx_rate_10y"],
                ctx_at_D["ctx_yield_slope"], ctx_at_D["ctx_credit_spread"],
                ctx_at_D["ctx_risk_off"], ctx_at_D["ctx_vixy_ret_21d"],
                ctx_at_D["ctx_vix_term_slope"], ctx_at_D["ctx_skew_z"],
                cboe_j["s1_cor3m_z"].to_numpy(), cboe_j["s1_curvature"].to_numpy(),
                cboe_j["s1_vxn_minus_vix"].to_numpy(), cboe_j["s1_vvix_z"].to_numpy(),
                llm_axes["llm_risk_appetite"], llm_axes["llm_rates_pressure"],
                llm_axes["llm_geopol_risk"], llm_status_ok,
                gd_avail, llm_status_ok, n_clusters_z,
                np.nan_to_num(GJ["g4_theme_novelty"], nan=0.0),
                np.nan_to_num(GJ["g1_share_entropy"]
                              if "g1_share_entropy" in GJ else
                              np.where(gd_avail > 0, gj["g1_share_entropy"].to_numpy(), 0.0),
                              nan=0.0),
                days_since_capped]
        for a, v in enumerate(zsrc):
            Zv[:, a] = np.nan_to_num(np.asarray(v, dtype=np.float64), nan=0.0).astype(np.float32)

        # ---- targets (§10.2) — never inputs ---------------------------------
        O = self.px["open"]
        open_D = O[didx]
        close_D = self.px["close"][didx]
        r5 = np.full((N, S), np.nan)
        r1f = np.full((N, S), np.nan)
        for k, i in enumerate(didx):
            if i + H < n_cal:
                r5[k] = O[i + H] / O[i] - 1.0
            if i + 1 < n_cal:
                r1f[k] = O[i + 1] / O[i] - 1.0
        r5 = np.where(symbol_mask, r5, np.nan)
        r1f = np.where(symbol_mask, r1f, np.nan)
        xs_mean5 = np.nanmean(r5, axis=1, keepdims=True)
        y5_raw = r5 - xs_mean5
        y5_rank = gaussian_rank(y5_raw)
        mu1 = np.nanmean(r1f, axis=1, keepdims=True)
        sg1 = np.nanstd(r1f, axis=1, keepdims=True)
        sg1[sg1 == 0] = np.nan
        y1_z = (r1f - mu1) / sg1

        y5_bucket = np.full((N, K), np.nan)
        for kb, b in enumerate(self.buckets):
            js = [self.symbols.index(s) for s in self.bucket_map[b] if s in self.symbols]
            if js:
                with np.errstate(invalid="ignore"):
                    y5_bucket[:, kb] = np.nanmean(r5[:, js], axis=1) - xs_mean5[:, 0]

        # RiskNet+ targets
        r = self.r1                                        # adj close-to-close returns
        sm, s2, cnt = _fwd_window_stats(r, 1, H)
        var5 = (s2 - sm * sm / np.maximum(cnt, 1)) / np.maximum(cnt - 1, 1)
        fwd_vol5 = np.where(cnt >= H, np.sqrt(np.maximum(var5, 0)) * np.sqrt(252), np.nan)[didx]
        fwd_vol5 = np.where(symbol_mask, fwd_vol5, np.nan)

        spy_r = r[:, self.symbols.index("SPY")]
        sx, sxx, cx = _fwd_window_stats(spy_r, 1, 63)
        beta = np.full((n_cal, S), np.nan)
        sy, syy, cy = _fwd_window_stats(r, 1, 63)
        sxy = _fwd_window_stats(r * spy_r[:, None], 1, 63)[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            cov = sxy - sy * sx[:, None] / 63.0
            varx = sxx - sx * sx / 63.0
            beta = np.where((cy >= 63) & (cx[:, None] >= 63), cov / varx[:, None], np.nan)
        fwd_beta63 = np.where(symbol_mask, beta[didx], np.nan)

        with np.errstate(invalid="ignore"):
            book_r = np.nanmean(r, axis=1)
        bs, bs2, bc = _fwd_window_stats(book_r, 1, H)
        bvar = (bs2 - bs * bs / np.maximum(bc, 1)) / np.maximum(bc - 1, 1)
        fwd_book_vol5 = np.where(bc >= H, np.sqrt(np.maximum(bvar, 0)) * np.sqrt(252), np.nan)[didx]

        # ---- Infotropy Transfer-B w_rec (§9.2) -------------------------------
        w_rec_score = self._w_rec_score(didx)
        w_rec_score = np.where(symbol_mask, w_rec_score, np.nan)
        w_rec = np.clip(self.eps + np.nan_to_num(w_rec_score, nan=0.0), self.eps, 1.0)
        w_rec = np.where(np.isnan(w_rec_score), np.nan, w_rec)

        # ---- persist ---------------------------------------------------------
        self.store.mkdir(parents=True, exist_ok=True)
        out = self.store / out_name
        np.savez(out,
                 dates=np.array([str(d.date()) for d in dates]),
                 symbols=np.array(self.symbols), buckets=np.array(self.buckets),
                 X=X, scalars=scalars, T=T, B=B, Z=Zv,
                 T_cols=np.array(T_cols), B_cols=np.array(B_cols), Z_cols=np.array(Z_cols),
                 X_features=np.array(X_FEATURES),
                 symbol_mask=symbol_mask,
                 gdelt_available=gd_avail, llm_available=llm_avail,
                 gdelt_day_joined=np.array(
                     ["" if pd.isna(d) else str(pd.Timestamp(d).date()) for d in gd_day]),
                 y5_raw=y5_raw, y5_rank=y5_rank, y1_z=y1_z, y5_bucket=y5_bucket,
                 fwd_vol5=fwd_vol5, fwd_beta63=fwd_beta63, fwd_book_vol5=fwd_book_vol5,
                 w_rec_score=w_rec_score, w_rec=w_rec,
                 open_px=open_D, close_px=close_D)
        wall = time.time() - t0
        meta = {
            "generated": dt.datetime.now().isoformat(timespec="seconds"),
            "panel": str(out.name), "n_dates": int(N),
            "first": str(dates[0].date()), "last": str(dates[-1].date()),
            "n_symbols": S, "n_buckets": K,
            "shapes": {k: list(v) for k, v in {
                "X": X.shape, "scalars": scalars.shape, "T": T.shape, "B": B.shape,
                "Z": Zv.shape}.items()},
            "T_cols": T_cols, "B_cols": B_cols, "Z_cols": Z_cols,
            "X_features": X_FEATURES,
            "targets": ["y5_raw", "y5_rank", "y1_z", "y5_bucket", "fwd_vol5",
                        "fwd_beta63", "fwd_book_vol5", "w_rec_score", "w_rec",
                        "open_px", "close_px"],
            "target_note": "target arrays are NEVER inputs; open_px/close_px are "
                           "price panels AT D for targets/tests only",
            "llm_merged": llm_merged,
            "llm_merge_path": "re-run FeatureStore().build() after "
                              "store/llm_features.parquet exists; all non-LLM columns "
                              "are deterministic and reproduce exactly",
            "record_weight_eps": self.eps,
            "conventions": {
                "inputs_through": "D-1 close",
                "x_zscore": f"per (symbol,feature), trailing {Z_WINDOW} td, "
                            f"min_periods {Z_MIN_PERIODS}, stats from days < D",
                "price_basis": "features from adj_close (matches deep history); "
                               "targets from split-adjusted opens",
                "gdelt_join": f"visible_from <= D, stale > {GDELT_STALE_DAYS}d masked",
                "cot_join": "publication-keyed visible_from <= D",
                "symbol_entry": f"inception + {ENTRY_LAG_TD} trading days",
                "bucket27_to_gdelt20_fallback": GDELT_BUCKET_FALLBACK,
                "factor_style_bucket": FACTOR_STYLE_BUCKET,
            },
            "wall_clock_seconds": round(wall, 1),
            "size_mb": round(out.stat().st_size / 1e6, 1),
        }
        meta_name = Path(out_name).stem + "_meta.json"   # panel.npz -> panel_meta.json
        (self.store / meta_name).write_text(json.dumps(meta, indent=1))
        if verbose:
            print(f"panel.npz: {N} dates {meta['first']}..{meta['last']}, "
                  f"X{X.shape} T{T.shape} B{B.shape} Z{Zv.shape}; "
                  f"{meta['size_mb']} MB, {wall:.0f}s")
        return meta

    # ------------------------------------------------------- Transfer-B weights

    def _w_rec_score(self, didx: np.ndarray) -> np.ndarray:
        """Raw record score in [0,1] per (decision date, symbol) — §9.2."""
        r = pd.DataFrame(self.r1)
        vol21 = (r.rolling(21).std() * np.sqrt(252))
        drift21 = r.rolling(21).mean()
        a = self.px["adj_close"]
        n_cal, S = a.shape
        scores = np.full((len(didx), S), np.nan)
        norm_parts = []
        for stat in (vol21, drift21):
            sv = stat.to_numpy()
            # sd_base: trailing 252d sd of the 6-day stat change, days < t
            d6 = stat.diff(H + 1)
            sd_base = d6.shift(1).rolling(252, min_periods=126).std().to_numpy().copy()
            sd_base[sd_base == 0] = np.nan
            shift = np.full((n_cal, S), np.nan)
            valid = np.arange(n_cal)
            ok = (valid + H < n_cal) & (valid - 1 >= 0)
            idx = valid[ok]
            shift[idx] = np.abs(sv[idx + H] - sv[idx - 1])
            norm_parts.append(shift / sd_base)
        norm = np.nanmean(np.stack(norm_parts), axis=0)

        move = np.diff(a, axis=0, prepend=np.full((1, S), np.nan))      # close_t - close_{t-1}
        fwd = np.full((n_cal, S), np.nan)
        fwd[:-H] = a[H:] - a[:-H]                                       # close_{t+5} - close_t
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.clip(-np.sign(move) * fwd / np.abs(move), 0.0, 1.0)
        frac = np.where(np.abs(move) < 1e-12, 1.0, frac)
        score_full = np.clip(norm * (1.0 - frac), 0.0, 1.0)
        scores = score_full[didx]
        return scores

    def w_rec(self, score: np.ndarray, eps: Optional[float] = None) -> np.ndarray:
        eps = self.eps if eps is None else eps
        w = np.clip(eps + np.nan_to_num(score, nan=0.0), eps, 1.0)
        return np.where(np.isnan(score), np.nan, w)


def _log_forced_choices() -> None:
    """Append build-time forced choices to validation_looks.jsonl (BUILD_SPEC header)."""
    ledger = PROTO / "validation_looks.jsonl"
    seen = set()
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            try:
                seen.add(json.loads(line).get("what"))
            except Exception:
                pass
    entries = [
        {"ts": dt.datetime.now().isoformat(timespec="seconds"), "component": "feature_store",
         "kind": "forced_choice", "what": w, "provenance": p}
        for w, p in [
            ("GDELT theme buckets = 20 (frozen theme_to_sector.json) not the spec's "
             "'13 sleeves' estimate; G1 emits 20 share + 20 z columns",
             "dicts/ frozen before model selection; BUILD_SPEC §2.2 dictionary-freeze rule"),
            ("bucket27->gdelt20 fallback map for theme-features: semis->tech, "
             "china/india/brazil/japan/intl_dev/europe->em_broad",
             "frozen dict has no country/semis theme buckets; G2 covers countries in B"),
            ("factor/style ETFs (MTUM QUAL VLUE USMV VIG SCHD VUG VTV) consume us_broad",
             "BUILD_SPEC §3 bucket table (factor/style consume us_broad)"),
            ("z-stat min_periods: 126 (price 252d), 30 (GDELT 90d), 26 (weekly 52w)",
             "spec gives windows only; min_periods needed for early panel"),
            ("gdelt_available masked 0 when joined day > 5 calendar days stale",
             "spec defines visible_from but not staleness; 5d cap chosen"),
            ("w_rec: the two regime stats (21d vol, 21d drift) averaged after "
             "per-stat sd_base normalization; sd_base = trailing 252d sd of 6d stat change",
             "§9.2 names both stats without a combiner; mean is the neutral choice"),
            ("X/T price features from adj_close (dividend+split adjusted) — verified "
             "identical basis to training/data/asset_features_history (~1e-6); targets "
             "from split-adjusted opens",
             "§10.4(e) live==deep consistency requires the deep basis; targets must be "
             "split-consistent tradable returns"),
        ]
    ]
    with ledger.open("a") as f:
        for e in entries:
            if e["what"] not in seen:
                f.write(json.dumps(e) + "\n")


if __name__ == "__main__":
    fs = FeatureStore()
    meta = fs.build()
    _log_forced_choices()
    print(json.dumps({k: meta[k] for k in
                      ["n_dates", "first", "last", "size_mb", "wall_clock_seconds",
                       "llm_merged"]}, indent=1))
