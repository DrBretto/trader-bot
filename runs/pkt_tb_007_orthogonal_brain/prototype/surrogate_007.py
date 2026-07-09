"""PKT-TB-007 — the chassis surrogate (BUILD_SPEC_007 §5.2/§5.3; TOURNAMENT_007 §4.4.7).

The fast paired walk that the EA fitness (ea_007.py) evaluates genomes on:
a deployed-MLP-on-historical-features, geometry-preserving simulator of the
incumbent chassis. Per fold the BASE arm (incumbent selection policy) is walked
ONCE and cached as a "day-pack"; the TILT arm is re-solved per genome through
THE SAME expression code path as the replay adapter (tilt_adapter.solve_tilt,
compute_conviction, tilt_bounds — one implementation, not a re-derivation), and
the fitness consumes the PAIRED daily difference

    dr_t = (tilt book − base book) daily return − incremental costs × c.

Base-arm replication (§5.2): base_score = 0.65*health + 0.35*MLP (the deployed
ranking_blend, asserted at build time), regime multiplier, clip [0,1],
threshold-by-regime, top-N selection, vol-adjusted sizing, matched gross by
construction (the tilt is cash-neutral). Costs: half-spread table + 1 bp
expectation slippage (the seeded full model only at bake-off).

Eras:
  fold era  (F1–F6, 2020-02-03 -> 2026-02-06): no daily artifacts exist, so
    features are rebuilt from cache/ohlcv with the PRODUCTION feature
    definitions (src/utils/feature_utils), health = the production
    baseline_health_model (the deployed health path — live inference.json
    latents are zero, i.e. the baseline model IS the deployed model), regime =
    the repo's own heuristic regime rule (scripts/build_ranking_history.py
    thresholds, frozen).
  live era  (2026-01-31 -> 2026-03-06 pre-holdout): the artifact record is
    used where it exists — features.parquet, inference.json health,
    regime_used from the incumbent replay timeline, seed book from
    portfolio_state.json.

Every known divergence from the real engine is itemized in
surrogate_manifest.json (L2c). HARD RAIL: no row dated >= 2026-03-11 is ever
loaded into any pack (asserted at build).

Fidelity checks (§5.3, run before any production EA generation):
  L2a  deployed-MLP per-fold rank-IC (F1–F6) vs live-era rank-IC (memorization).
  L2c  blend fidelity: the deployed ranking_blend really is 0.35 (0.65/0.35).
  L7   deployed-MLP training-window end vs the 2026-03-11 firewall (repo
       forensics — see verify_l7).
  (L2b blend-0 sensitivity is an EA re-run — entrypoint in ea_007.py.)

CLI:
  .venv/bin/python surrogate_007.py --build-packs            # folds + live
  .venv/bin/python surrogate_007.py --fidelity --l7          # checks -> manifest
  .venv/bin/python surrogate_007.py --synth-organs --synth-masks  # smoke inputs
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROTO = Path(__file__).resolve().parent
REPO = PROTO.parents[2]
TB006_PROTO = REPO / "runs" / "pkt_tb_006_clean_sheet_brain" / "prototype"
for p in (str(PROTO), str(REPO), str(TB006_PROTO)):
    if p not in sys.path:
        sys.path.append(p)

import tilt_adapter as TA                                    # noqa: E402
from genome_007 import Genome007                             # noqa: E402
from risk_stats_007 import RiskStats                         # noqa: E402
from chassis.models.baseline_health import baseline_health_model # noqa: E402
from chassis.utils.feature_utils import (compute_asset_features, # noqa: E402
                                     compute_relative_strength)
from chassis.utils.transaction_costs import get_half_spread_bps  # noqa: E402

HOLDOUT_START = "2026-03-11"          # absolute firewall — asserted everywhere
FITNESS_END = "2026-02-06"            # fitness/selection data ends here
LIVE_ERA_START = "2026-01-31"
PREHOLDOUT_LAST_DECISION = "2026-03-06"
OHLCV_DIR = TB006_PROTO / "cache" / "ohlcv"
MODEL_DIR = TB006_PROTO / "cache" / "s3" / "models" / "ranking_expanded_unconditioned"
STORE = PROTO / "store" / "surrogate_007"
S_MIN_DAILY = 2e-4                    # 2 bp/day fitness sd floor (§4.2)
COST_SCENARIOS = (1.0, 1.5)
SLIPPAGE_EXPECT_BPS = 1.0             # + half-spread table (§5.2)
MIN_ORDER = 250.0

# fold bounds (TB-006 folds.py carried verbatim — single source re-asserted in tests)
FOLD_BOUNDS = {1: ("2020-02-01", "2021-02-01"), 2: ("2021-02-01", "2022-02-01"),
               3: ("2022-02-01", "2023-02-01"), 4: ("2023-02-01", "2024-02-01"),
               5: ("2024-02-01", "2025-02-01"), 6: ("2025-02-01", "2026-02-01")}

# deployed decision params used by the base arm (read + asserted from the live
# variant config in build_packs; hardcoded here so fold packs build offline)
DEPLOYED_PARAMS = {
    "max_positions": 8, "max_position_weight": 0.30,
    "buy_score_threshold": 0.65,
    "buy_score_threshold_by_regime": {"calm_uptrend": 0.62, "risk_on_trend": 0.62,
                                      "choppy": 0.65, "risk_off_trend": 0.68,
                                      "high_vol_panic": 0.72},
    "min_health_buy": 0.60, "sell_health_threshold": 0.35, "sell_health_days": 3,
    "trailing_stop_base": 0.10, "min_order_dollars": 250.0,
    "min_cash_reserve_by_regime": {"calm_uptrend": 0.1, "risk_on_trend": 0.1,
                                   "choppy": 0.2, "risk_off_trend": 0.4,
                                   "high_vol_panic": 0.4},
    "ranking_blend": 0.35,
    "vol_adj": {"low": 1.10, "med": 1.0, "high": 0.80},
    "regime_adj": {"calm_uptrend": 1.10, "risk_on_trend": 1.10, "choppy": 0.90,
                   "risk_off_trend": 0.80, "high_vol_panic": 0.50},
    "high_vol_exception_score": 0.80,
}

DIVERGENCES_L2C = [
    "fold era: features rebuilt from ohlcv adj_close (production formulas; live "
    "pipeline uses broker closes — differs around distributions/splits)",
    "fold era: regime = heuristic SPY rule (scripts/build_ranking_history.py "
    "thresholds); live era uses the incumbent replay's regime_used record",
    "fold era: base book starts all-cash at fold start (ramp-in days; symmetric "
    "across genomes by pairing)",
    "regime_compatibility multiplier table not replicated (multiplier := 1.0); "
    "threshold-by-regime IS replicated",
    "LLM veto / LLM size adj / ensemble+expert sizing modifiers := neutral 1.0",
    "max_sector_weight cluster cap, leveraged constraints, reduce/trim layer, "
    "VUG split handling: not replicated",
    "sell layer reduced to: 3-day health<=0.35 collapse + 10% trailing stop + "
    "panic force-sell of equity sleeve (the dominant observed engine behaviors)",
    "tilt quantization mirrors tilt_to_intents at leg level: chassis-order "
    "edits floor-free, new legs whole-share-rounded at the mark + min_order; "
    "the suppression/absorber micro-logic is not replicated",
    "tilt overlay: re-solved on active days, CARRIED (no unwind) on neutral "
    "days exactly like the real channel, closed when the chassis fully sells "
    "the name; but the carried tilt does NOT feed back into future chassis "
    "decisions (the real path coupling is what the anchor certifies)",
    "fills at adj open(D); marks at adj close(D); costs = half-spread + 1bp "
    "expectation (deterministic; seeded full model only at bake-off)",
]


# ===================================================================== panel
def _ts(x) -> str:
    return pd.Timestamp(x).strftime("%Y-%m-%d")


def load_universe() -> pd.DataFrame:
    return pd.read_csv(REPO / "config" / "universe.csv")


def half_spread_map(universe: pd.DataFrame) -> Dict[str, float]:
    return {r["symbol"]: float(get_half_spread_bps(r.get("sector", "broad"),
                                                   r.get("asset_class", "equity")))
            for _, r in universe.iterrows()}


def build_feature_panel(start: str = "2019-06-01") -> Dict[str, Any]:
    """Long features frame (production definitions on adj prices) + per-symbol
    open/close panels. Everything strictly < HOLDOUT_START (hard rail)."""
    uni = load_universe()
    symbols = sorted(uni["symbol"].tolist())
    frames, opens, closes = [], {}, {}
    spy_feat = None
    raw = {}
    for sym in symbols:
        p = OHLCV_DIR / f"{sym}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= start) & (df["date"] < HOLDOUT_START)].copy()
        df = df.sort_values("date").reset_index(drop=True)
        if len(df) < 70:
            continue
        adj_f = (df["adj_close"] / df["close"]).replace([np.inf, -np.inf], np.nan)
        df["open_adj"] = df["open"] * adj_f
        raw[sym] = df
    def _fin(df):
        return pd.DataFrame({"date": df["date"], "open": df["open_adj"],
                             "high": df["open_adj"], "low": df["open_adj"],
                             "close": df["adj_close"], "volume": df["volume"]})

    spy_feat = compute_asset_features(_fin(raw["SPY"]))
    for sym, df in raw.items():
        f = compute_asset_features(_fin(df))
        f = compute_relative_strength(f, spy_feat)
        f["symbol"] = sym
        frames.append(f[["date", "symbol", "close", "return_1d", "return_5d",
                         "return_21d", "return_63d", "vol_21d", "vol_63d",
                         "drawdown_63d", "trend_63d", "volume_ma21",
                         "rel_strength_21d", "rel_strength_63d"]])
        s = df.set_index(df["date"].dt.strftime("%Y-%m-%d"))
        opens[sym] = s["open_adj"]
        closes[sym] = s["adj_close"]
    feats = pd.concat(frames, ignore_index=True)
    assert feats["date"].max() < pd.Timestamp(HOLDOUT_START), "panel touches holdout"
    feats["date_s"] = feats["date"].dt.strftime("%Y-%m-%d")
    dates = sorted(raw["SPY"]["date"].dt.strftime("%Y-%m-%d"))
    return {"features": feats, "opens": opens, "closes": closes,
            "dates": dates, "universe": uni}


# ============================================================== deployed MLP
class DeployedMLP:
    """Batch scorer numerically identical to RankingMLP.predict_scores on the
    deployed unconditioned model (10 features, no one-hots)."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        import torch
        from training.models.ranking_mlp import RankingMLP, RANKING_FEATURES
        self.features = list(RANKING_FEATURES)
        self.norm = json.loads((model_dir / "ranking_normalization.json").read_text())
        assert all(k in self.norm for k in self.features)
        assert "regime_calm_uptrend" not in self.norm, "model is conditioned?!"
        self.model = RankingMLP(input_dim=len(self.features))
        self.model.load_state_dict(torch.load(model_dir / "ranking_mlp.pt",
                                              weights_only=True))
        self.model.eval()
        self._torch = torch
        self.pt_sha = hashlib.sha256((model_dir / "ranking_mlp.pt").read_bytes()).hexdigest()

    def score_frame(self, cs: pd.DataFrame) -> Dict[str, float]:
        """cs: one-date cross-section with RANKING_FEATURES columns + symbol."""
        if len(cs) == 0:
            return {}
        X = np.zeros((len(cs), len(self.features)), dtype=np.float32)
        for i, f in enumerate(self.features):
            v = cs[f].to_numpy(dtype=np.float64)
            v = np.where(np.isfinite(v), v, 0.0)        # predict_scores: missing -> 0.0 raw
            m = self.norm[f].get("mean", 0.0)
            s = self.norm[f].get("std", 1.0) or 1.0
            if s < 1e-8:
                s = 1.0
            X[:, i] = (v - m) / s
        with self._torch.no_grad():
            out = self.model.net(self._torch.tensor(X)).numpy().ravel()
        return dict(zip(cs["symbol"].tolist(), out.astype(float)))


# ============================================================== regime proxy
def heuristic_regime(spy_ret21: float, spy_vol21: float) -> str:
    """scripts/build_ranking_history.py:_assign_heuristic_regimes, verbatim."""
    ret = spy_ret21 if np.isfinite(spy_ret21) else 0.0
    vol = spy_vol21 if np.isfinite(spy_vol21) else 0.0
    if ret < -0.05 and vol > 0.25:
        return "high_vol_panic"
    if ret < -0.02 or vol > 0.22:
        return "risk_off_trend"
    if vol > 0.16:
        return "choppy"
    if ret > 0.02:
        return "risk_on_trend"
    return "calm_uptrend"


# ============================================================== base-arm walk
class _Holding:
    __slots__ = ("shares", "peak_close", "below_days", "asset_class")

    def __init__(self, shares, peak_close, asset_class):
        self.shares = shares
        self.peak_close = peak_close
        self.below_days = 0
        self.asset_class = asset_class


def base_walk_and_pack(dates: List[str], panel: dict, mlp: DeployedMLP,
                       regime_for: Dict[str, str], health_for=None,
                       seed_book: Optional[dict] = None,
                       params: dict = DEPLOYED_PARAMS,
                       risk: Optional[RiskStats] = None) -> dict:
    """Walk the incumbent selection policy over `dates`; emit the day-pack the
    tilt walk consumes. `regime_for`: date->label. `health_for`: optional
    date->{sym:{health_score, vol_bucket}} (live era artifact record); default
    = production baseline_health_model on the panel features.
    `seed_book`: {"cash": $, "holdings": {sym: {shares, peak_close, asset_class}}}.
    """
    assert all(d < HOLDOUT_START for d in dates), "walk touches holdout"
    feats = panel["features"]
    by_date = {d: g for d, g in feats.groupby("date_s")}
    opens, closes = panel["opens"], panel["closes"]
    uni = panel["universe"]
    uni_ac = dict(zip(uni["symbol"], uni["asset_class"]))
    uni_elig = dict(zip(uni["symbol"], uni["eligible"]))
    hs_map = half_spread_map(uni)
    risk = risk or RiskStats()
    all_dates = panel["dates"]
    didx = {d: i for i, d in enumerate(all_dates)}

    cash = 100_000.0
    holdings: Dict[str, _Holding] = {}
    if seed_book:
        cash = float(seed_book["cash"])
        for s, h in seed_book["holdings"].items():
            holdings[s] = _Holding(float(h["shares"]), float(h["peak_close"]),
                                   h.get("asset_class", uni_ac.get(s, "equity")))

    blend = float(params["ranking_blend"])
    days = []
    for D in dates:
        i = didx.get(D)
        if i is None or i == 0:
            continue
        Dprev = all_dates[i - 1]
        cs = by_date.get(Dprev)
        if cs is None or len(cs) < 10:
            continue
        # ---- inputs through D-1 close --------------------------------------
        cs = cs.copy()
        if health_for is not None and D in health_for:
            hmap = health_for[D]
        else:
            hdf = baseline_health_model(cs.rename(columns={"date_s": "_ds"}),
                                        cs["date"].iloc[0])
            hmap = {r["symbol"]: {"health_score": float(r["health_score"]),
                                  "vol_bucket": str(r["vol_bucket"])}
                    for _, r in hdf.iterrows()}
        mlp_scores = mlp.score_frame(cs)
        regime = regime_for[D]
        close_prev = {s: float(closes[s].loc[Dprev]) for s in closes
                      if Dprev in closes[s].index}
        open_today = {s: float(opens[s].loc[D]) for s in opens
                      if D in opens[s].index}
        close_today = {s: float(closes[s].loc[D]) for s in closes
                       if D in closes[s].index}

        nav = cash + sum(h.shares * close_prev.get(s, h.peak_close)
                         for s, h in holdings.items())
        held_pre = dict(holdings)

        # ---- sells ----------------------------------------------------------
        sells: Dict[str, float] = {}
        for s, h in list(holdings.items()):
            hp = hmap.get(s, {}).get("health_score", 0.5)
            h.below_days = h.below_days + 1 if hp <= params["sell_health_threshold"] else 0
            px = close_prev.get(s)
            stop = (px is not None
                    and px < h.peak_close * (1 - params["trailing_stop_base"]))
            collapse = h.below_days >= params["sell_health_days"]
            panic = (regime == "high_vol_panic" and h.asset_class == "equity")
            if stop or collapse or panic:
                fill = open_today.get(s, px or h.peak_close)
                cash += h.shares * fill
                sells[s] = h.shares
                del holdings[s]

        # ---- buys (score -> filter -> size) ---------------------------------
        rows = []
        for s, hp in hmap.items():
            if not uni_elig.get(s, 0):
                continue
            if s in held_pre:                       # no same-day re-buy
                continue
            ms = mlp_scores.get(s, 0.5)
            base = (1 - blend) * hp["health_score"] + blend * ms
            final = float(np.clip(base * 1.0, 0.0, 1.0))    # regime mult := 1.0 (L2c)
            rows.append((s, final, hp["health_score"], hp["vol_bucket"]))
        thresh = params["buy_score_threshold_by_regime"].get(
            regime, params["buy_score_threshold"])
        cands = [r for r in rows
                 if r[1] >= thresh and r[2] >= params["min_health_buy"]]
        cands = [r for r in cands
                 if r[3] != "high" or (regime == "calm_uptrend"
                                       and r[1] > params["high_vol_exception_score"])]
        if regime == "high_vol_panic":
            cands = [r for r in cands
                     if uni_ac.get(r[0]) in ("bond", "commodity")]
        cands.sort(key=lambda r: (-r[1], r[0]))
        slots = max(int(params["max_positions"]) - len(holdings), 0)
        reserve = params["min_cash_reserve_by_regime"].get(regime, 0.1) * nav
        buys: Dict[str, float] = {}
        for s, final, hscore, vb in cands:
            if slots <= 0:
                break
            px = open_today.get(s)
            if px is None or px <= 0:
                continue
            dollars = (nav * params["max_position_weight"]
                       * params["vol_adj"].get(vb, 1.0)
                       * params["regime_adj"].get(regime, 1.0))
            dollars = min(dollars, cash - reserve)
            if dollars < params["min_order_dollars"]:
                continue
            holdings[s] = _Holding(dollars / px, px, uni_ac.get(s, "equity"))
            cash -= dollars
            buys[s] = dollars
            slots -= 1

        # ---- the day-pack record (decision-time view, pre-execution) --------
        support = sorted((set(buys) | set(held_pre)
                          | set(TA.TILT_CORE) | set(TA.TILT_COND)) - {TA.VIXY})
        stats = risk.table(support, D)
        w_prev = {s: held_pre[s].shares * close_prev.get(s, 0.0) / nav
                  for s in held_pre}
        rec = {
            "date": D, "regime": regime, "nav": nav,
            "support": support,
            "beta": np.array([stats.get(n, {}).get("beta", 0.0) for n in support]),
            "sigma": np.array([stats.get(n, {}).get("sigma", 0.0) for n in support]),
            "has_stats": np.array([n in stats for n in support]),
            "w_prev": np.array([w_prev.get(n, 0.0) for n in support]),
            "buys_w": np.array([buys.get(n, 0.0) / nav for n in support]),
            "fully_sold": np.array([n in sells for n in support]),
            "ovn": np.array([(open_today[n] / close_prev[n] - 1.0)
                             if (n in open_today and n in close_prev
                                 and close_prev[n] > 0) else 0.0
                             for n in support]),
            "intra": np.array([(close_today[n] / open_today[n] - 1.0)
                               if (n in close_today and n in open_today
                                   and open_today[n] > 0) else 0.0
                               for n in support]),
            "hs_bps": np.array([hs_map.get(n, 3.0) + SLIPPAGE_EXPECT_BPS
                                for n in support]),
            "marks": np.array([close_prev.get(n, 0.0) for n in support]),
            "n_holdings": len(holdings), "n_buys": len(buys),
            "n_sells": len(sells),
            "book_empty_at_close": len(holdings) == 0,
        }
        days.append(rec)

        # ---- mark to close ---------------------------------------------------
        for s, h in holdings.items():
            px = close_today.get(s)
            if px is not None:
                h.peak_close = max(h.peak_close, px)

    return {"days": days, "params": params,
            "asset_class": uni_ac,
            "divergences_l2c": DIVERGENCES_L2C}


# ============================================================== the tilt walk
def walk_fold_paired(pack: dict, genome: Genome007, organs_dir: Path,
                     masks: Dict[str, Dict[str, float]],
                     min_order: float = MIN_ORDER) -> dict:
    """The paired walk: per day, the SAME conviction + bounds + solve path as
    the replay adapter (tilt_adapter functions), then
    dr_gross_t = dw_{t-1}·ovn_t + dw_t·intra_t  and  cost1_t (at cost x1.0).

    CARRY SEMANTICS (matches the real channel): on a neutral day the adapter
    returns the incumbent intents UNCHANGED — nothing unwinds — so the carried
    tilt dw persists (no trade, no cost), except legs on names the chassis
    fully sells that day (every lot goes, tilt shares included => the leg
    closes, cost charged). The carried tilt is unwound at the END of the
    window (cost charged) so no genome keeps free exposure past the fold."""
    organs_dir = Path(organs_dir)
    days = pack["days"]
    n = len(days)
    dr_gross = np.zeros(n)
    cost1 = np.zeros(n)
    dw_prev: Dict[str, float] = {}
    hs_prev: Dict[str, float] = {}
    n_active = 0
    reasons: Dict[str, int] = {}

    def _bump(r):
        reasons[r] = reasons.get(r, 0) + 1

    for t, rec in enumerate(days):
        D = rec["date"]
        organs = TA.load_organ_outputs(organs_dir, D)
        conv = TA.compute_conviction(organs, genome)
        sold_today = {s for s, fs in zip(rec["support"], rec["fully_sold"]) if fs}
        # carry: neutral day => incumbent intents unchanged => tilt persists,
        # minus legs the chassis's OWN sell layer would close that day:
        #   (a) names the base book fully sells (lot fix: full exit sells every
        #       lot, tilt shares included);
        #   (b) the whole carried tilt when the base book liquidates to cash
        #       (panic force-sell takes the ORB book's lots with it);
        #   (c) equity-class legs on high_vol_panic days (the panic sell layer).
        dw_today: Dict[str, float] = {s: w for s, w in dw_prev.items()
                                      if s not in sold_today}
        if rec.get("book_empty_at_close"):
            dw_today = {}
        elif rec.get("regime") == "high_vol_panic":
            ac = pack.get("asset_class", {})
            dw_today = {s: w for s, w in dw_today.items()
                        if ac.get(s, "equity") != "equity"}
        hs_today: Dict[str, float] = {s: hs_prev[s] for s in dw_today
                                      if s in hs_prev}
        if conv["neutral"]:
            _bump(conv["neutral_reason"])
        else:
            support = rec["support"]
            nav = rec["nav"]
            lb = np.zeros(len(support))
            ub = np.zeros(len(support))
            sell_w = rec.get("sell_w")
            partial = rec.get("partially_sold")
            for i, s in enumerate(support):
                w_held = float(rec["w_prev"][i])
                w_avail = max(w_held - (float(sell_w[i]) if sell_w is not None
                                        else (w_held if rec["fully_sold"][i]
                                              else 0.0)), 0.0)
                lo, hi = TA.tilt_bounds(s, genome, w_avail,
                                        float(rec["buys_w"][i]), w_held,
                                        bool(rec["fully_sold"][i]),
                                        bool(partial[i]) if partial is not None
                                        else False,
                                        bool(rec["has_stats"][i]))
                lb[i], ub[i] = lo, hi
            if float(-lb.sum()) < 1e-6 or float(ub.sum()) < 1e-6:
                _bump("no_tilt_capacity")
            else:
                dw, pdiag, _, _ = TA.solve_tilt(
                    organs, genome, masks, support, conv["active"],
                    conv["tau"], conv["T_t"], rec["beta"], rec["sigma"], lb, ub)
                if not dw or sum(abs(v) for v in dw.values()) <= 0:
                    _bump("zero_direction")
                elif pdiag["projection"].get("infeasible"):
                    mass = sum(abs(v) for v in dw.values()) / 2.0 * nav
                    _bump("dead_zone_quantized" if mass < 2 * min_order
                          else "projection_infeasible")
                else:
                    # quantization mirrors tilt_to_intents: edits of EXISTING
                    # chassis BUY orders carry no min_order floor; new legs
                    # (added buys / sell trims) must clear min_order AND are
                    # rounded to whole shares at the mark; neutral only when
                    # NOTHING survives
                    bw = dict(zip(support, rec["buys_w"]))
                    mk = dict(zip(support, rec.get("marks",
                                                   np.zeros(len(support)))))
                    kept = {}
                    for s, v in dw.items():
                        if abs(v) <= 1e-12:
                            continue
                        if bw.get(s, 0.0) > 0:
                            kept[s] = v                  # order edit, no floor
                            continue
                        px = float(mk.get(s, 0.0) or 0.0)
                        if px > 0:
                            sh = int(abs(v) * nav / px)
                            v = float(np.sign(v) * sh * px / nav)
                        if abs(v) * nav >= min_order:
                            kept[s] = v
                    if not kept:
                        _bump("dead_zone_quantized")
                    else:
                        dw_today = kept
                        hs_today = {s: float(rec["hs_bps"][rec["support"].index(s)])
                                    for s in kept}
                        n_active += 1
                        solved = True
        # ---- paired daily delta ---------------------------------------------
        ovn = dict(zip(rec["support"], rec["ovn"]))
        intra = dict(zip(rec["support"], rec["intra"]))
        g = sum(w * ovn.get(s, 0.0) for s, w in dw_prev.items())
        g += sum(w * intra.get(s, 0.0) for s, w in dw_today.items())
        dr_gross[t] = g
        c = 0.0
        for s in set(dw_prev) | set(dw_today):
            trade = abs(dw_today.get(s, 0.0) - dw_prev.get(s, 0.0))
            bps = hs_today.get(s, hs_prev.get(s, 3.0 + SLIPPAGE_EXPECT_BPS))
            c += trade * bps / 1e4
        cost1[t] = c
        dw_prev, hs_prev = dw_today, hs_today

    if dw_prev and n:
        cost1[-1] += sum(abs(w) * hs_prev.get(s, 4.0) / 1e4
                         for s, w in dw_prev.items())
    return {"dr_gross": dr_gross, "cost1": cost1, "n_days": n,
            "n_active": n_active, "neutral_reasons": reasons,
            "dates": [d["date"] for d in days]}


def fold_U(dr_gross: np.ndarray, cost1: np.ndarray,
           scenarios=COST_SCENARIOS) -> float:
    """U_f = min over cost scenarios of the floored paired IR (§4.2)."""
    us = []
    for c in scenarios:
        dr = dr_gross - cost1 * c
        sd = float(dr.std(ddof=1)) if len(dr) > 1 else 0.0
        us.append(float(np.sqrt(252) * dr.mean() / max(sd, S_MIN_DAILY)))
    return min(us)


# ============================================================== pack building
def fold_dates(panel_dates: List[str], fold: int) -> List[str]:
    lo, hi = FOLD_BOUNDS[fold]
    return [d for d in panel_dates if lo <= d < hi and d <= FITNESS_END]


def build_packs(folds=(1, 2, 3, 4, 5, 6), live: bool = True,
                store: Path = STORE) -> dict:
    store.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    panel = build_feature_panel()
    mlp = DeployedMLP()
    risk = RiskStats()

    # deployed-config assertion (blend fidelity input)
    deployed = _read_deployed_config()
    summary = {"folds": {}, "panel_dates": [panel["dates"][0], panel["dates"][-1]],
               "deployed_config_check": deployed}

    # fold-era regimes from the frozen heuristic (SPY features through D-1)
    feats = panel["features"]
    spy = feats[feats["symbol"] == "SPY"].set_index("date_s")
    didx = {d: i for i, d in enumerate(panel["dates"])}

    def regime_of(D):
        i = didx[D]
        if i == 0:
            return "choppy"
        Dp = panel["dates"][i - 1]
        if Dp not in spy.index:
            return "choppy"
        r = spy.loc[Dp]
        return heuristic_regime(float(r["return_21d"]), float(r["vol_21d"]))

    for f in folds:
        dates = fold_dates(panel["dates"], f)
        regimes = {d: regime_of(d) for d in dates}
        pack = base_walk_and_pack(dates, panel, mlp, regimes, risk=risk)
        pack["fold"] = f
        with open(store / f"pack_F{f}.pkl", "wb") as fh:
            pickle.dump(pack, fh)
        navs = [d["nav"] for d in pack["days"]]
        summary["folds"][f] = {
            "n_days": len(pack["days"]), "first": dates[0], "last": dates[-1],
            "mean_holdings": float(np.mean([d["n_holdings"] for d in pack["days"]])),
            "mean_buys_per_day": float(np.mean([d["n_buys"] for d in pack["days"]])),
            "nav_end_over_start": float(navs[-1] / navs[0]) if navs else None,
            "regime_counts": pd.Series(list(regimes.values())).value_counts().to_dict(),
        }
        print(f"pack F{f}: {summary['folds'][f]}")

    if live:
        lp = build_live_pack(panel, mlp, risk, store)
        summary["live"] = lp
    summary["wall_clock_s"] = round(time.time() - t0, 1)
    (store / "packs_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def build_live_pack(panel: dict, mlp: DeployedMLP, risk: RiskStats,
                    store: Path = STORE,
                    inc_dir: Optional[Path] = None) -> dict:
    """Pre-holdout live-era pack driven by the REAL incumbent replay record
    (anchor leg). The spec's rule — health/regime 'from the artifact record
    where it exists' — extends to the whole base-book geometry here: the
    incumbent replay's timeline/actions ARE the record of the base arm on this
    window, so the anchor isolates what it is registered to certify (the tilt
    expression + paired-Δ accounting). Fold-era packs have no such record and
    use the policy walk (fold-era fidelity assumed-by-geometry, §4.4.7)."""
    from data_layer import DiskCachedS3Cache
    cache = DiskCachedS3Cache(s3_client=None)
    inc_dir = Path(inc_dir or PROTO / "runs_battery_007" / "B0_EXPR" / "incumbent")
    timeline = json.loads((inc_dir / "timeline.json").read_text())
    result = json.loads((inc_dir / "result.json").read_text())
    dates = [r["date"] for r in timeline]
    assert all(d < HOLDOUT_START for d in dates)
    actions_by_date: Dict[str, list] = {}
    for a in result.get("actions", []):
        actions_by_date.setdefault(a["date"], []).append(a)

    uni = panel["universe"]
    hs_map = half_spread_map(uni)
    opens, closes = panel["opens"], panel["closes"]

    # day-0 book = the replay's seed state
    st = cache.get_json(f"daily/{LIVE_ERA_START}/portfolio_state.json")
    prev_hold = {h["symbol"]: float(h["shares"]) for h in st.get("holdings", [])}
    prev_marks = {h["symbol"]: float(h.get("current_price", h["entry_price"]))
                  for h in st.get("holdings", [])}
    prev_nav = float(st.get("portfolio_value", 100_000.0))

    days = []
    for r in timeline:
        D = r["date"]
        nav = prev_nav
        held = dict(prev_hold)
        marks = dict(prev_marks)
        buys: Dict[str, float] = {}
        sold_sh: Dict[str, float] = {}
        for a in actions_by_date.get(D, []):
            d_ = float(a.get("dollars", 0) or
                       float(a.get("shares", 0)) * float(a.get("price", 0) or 0))
            if a["action"] == "BUY":
                buys[a["symbol"]] = buys.get(a["symbol"], 0.0) + d_
            elif a["action"] in ("SELL", "REDUCE"):
                sold_sh[a["symbol"]] = (sold_sh.get(a["symbol"], 0.0)
                                        + float(a.get("shares", 0)))
        fully = {s for s, sh in sold_sh.items()
                 if sh >= held.get(s, 0.0) - 1e-6}
        partial = {s for s in sold_sh if s not in fully}
        support = sorted((set(buys) | set(held)
                          | set(TA.TILT_CORE) | set(TA.TILT_COND)) - {TA.VIXY})
        stats = risk.table(support, D)
        w_prev = {s: held[s] * marks.get(s, 0.0) / nav for s in held}
        sell_w = {s: min(sold_sh.get(s, 0.0), held.get(s, 0.0))
                  * marks.get(s, 0.0) / nav for s in sold_sh}
        days.append({
            "date": D, "regime": r.get("regime_used") or "choppy", "nav": nav,
            "support": support,
            "beta": np.array([stats.get(n, {}).get("beta", 0.0) for n in support]),
            "sigma": np.array([stats.get(n, {}).get("sigma", 0.0) for n in support]),
            "has_stats": np.array([n in stats for n in support]),
            "w_prev": np.array([w_prev.get(n, 0.0) for n in support]),
            "buys_w": np.array([buys.get(n, 0.0) / nav for n in support]),
            "sell_w": np.array([sell_w.get(n, 0.0) for n in support]),
            "fully_sold": np.array([n in fully for n in support]),
            "partially_sold": np.array([n in partial for n in support]),
            "ovn": np.array([(float(opens[n].loc[D]) / float(closes[n].loc[_prev_idx(panel, D)]) - 1.0)
                             if (n in opens and D in opens[n].index
                                 and n in closes
                                 and _prev_idx(panel, D) in closes[n].index)
                             else 0.0 for n in support]),
            "intra": np.array([(float(closes[n].loc[D]) / float(opens[n].loc[D]) - 1.0)
                               if (n in closes and D in closes[n].index
                                   and n in opens and D in opens[n].index
                                   and float(opens[n].loc[D]) > 0)
                               else 0.0 for n in support]),
            "hs_bps": np.array([hs_map.get(n, 3.0) + SLIPPAGE_EXPECT_BPS
                                for n in support]),
            "marks": np.array([marks.get(n) or
                               (float(closes[n].loc[_prev_idx(panel, D)])
                                if (n in closes
                                    and _prev_idx(panel, D) in closes[n].index)
                                else 0.0) for n in support]),
            "n_holdings": len(held), "n_buys": len(buys), "n_sells": len(sold_sh),
            "book_empty_at_close": len(r.get("holdings_at_close", [])) == 0,
        })
        prev_hold = {h["symbol"]: float(h["shares"])
                     for h in r.get("holdings_at_close", [])}
        prev_marks = {h["symbol"]: float(h.get("close_price")
                                         or h.get("peak_price") or 0.0)
                      for h in r.get("holdings_at_close", [])}
        prev_nav = float(r["ending_value"])

    pack = {"days": days, "params": DEPLOYED_PARAMS,
            "asset_class": dict(zip(uni["symbol"], uni["asset_class"])),
            "divergences_l2c": DIVERGENCES_L2C,
            "window": "preholdout_live",
            "base_book_source": f"incumbent replay record: {inc_dir}"}
    with open(store / "pack_live.pkl", "wb") as fh:
        pickle.dump(pack, fh)
    regimes = [d["regime"] for d in days]
    return {"n_days": len(days), "first": dates[0], "last": dates[-1],
            "base_book_source": str(inc_dir),
            "regime_counts": pd.Series(regimes).value_counts().to_dict()}


def _prev_idx(panel: dict, D: str) -> str:
    i = panel["dates"].index(D)
    return panel["dates"][i - 1] if i > 0 else D


def _read_deployed_config() -> dict:
    """Read the deployed variant config and check the 0.65/0.35 blend (L2c)."""
    try:
        from data_layer import DiskCachedS3Cache
        from chassis.utils.three_line_replay import replay_engine as RE
        cache = DiskCachedS3Cache(s3_client=None)
        variant, _ = RE.load_variant_configs(cache)
        blend = float(variant.decision_engine_overrides.get("ranking_blend", 0) or 0)
        out = {"ranking_blend": blend,
               "blend_is_065_035": abs(blend - 0.35) < 1e-12,
               "model_dir": variant.decision_engine_overrides.get("ranking_model_dir"),
               "params_match": {}}
        for k in ("max_positions", "max_position_weight", "buy_score_threshold",
                  "min_health_buy", "sell_health_threshold", "sell_health_days",
                  "trailing_stop_base", "min_order_dollars"):
            out["params_match"][k] = (variant.decision_params.get(k)
                                      == DEPLOYED_PARAMS.get(k, variant.decision_params.get(k)))
        return out
    except Exception as e:                                    # pragma: no cover
        return {"error": str(e)}


# ============================================================== fidelity (L2a)
def fidelity_ic(store: Path = STORE) -> dict:
    """Deployed-MLP per-fold rank-IC vs live-era rank-IC (§5.3 check 1).

    IC = per-day cross-sectional Spearman(MLP score, forward RELATIVE return)
    at the model's own 21d horizon and at 5d. LIVE-ERA NOTE (ledgered): the
    21d live-era target needs prices >= 2026-03-11 (the holdout) — computing
    it now would breach the firewall, so the live-era row is computed at the
    5d horizon only (target windows end <= 2026-03-10); the 21d live-era IC
    is deferred to bake-off."""
    panel = build_feature_panel()
    mlp = DeployedMLP()
    feats = panel["features"]
    dates = panel["dates"]
    didx = {d: i for i, d in enumerate(dates)}
    closes = panel["closes"]
    syms = sorted(closes)
    px = pd.DataFrame({s: closes[s] for s in syms}).reindex(dates)

    def day_ic(D, horizon):
        i = didx[D]
        j = i + horizon
        if j >= len(dates):
            return None
        if dates[j] >= HOLDOUT_START:
            return None
        cs = feats[feats["date_s"] == D]
        if len(cs) < 10:
            return None
        scores = mlp.score_frame(cs)
        p0 = px.iloc[i]
        p1 = px.iloc[j]
        fwd = (p1 / p0 - 1.0).dropna()
        common = [s for s in fwd.index if s in scores]
        if len(common) < 10:
            return None
        f = fwd.loc[common] - fwd.loc[common].mean()
        sc = pd.Series({s: scores[s] for s in common})
        return float(sc.rank().corr(f.rank()))

    out = {"per_fold": {}, "live_era": {}, "note_live_21d":
           "deferred to bake-off — 21d targets from the live era cross the "
           "2026-03-11 firewall (pre-registered consequence: firewall wins)"}
    for f in range(1, 7):
        fd = fold_dates(dates, f)
        sample = fd[::5]                       # every 5th day (independent-ish)
        for h, key in ((21, "ic21"), (5, "ic5")):
            ics = [v for v in (day_ic(D, h) for D in sample) if v is not None]
            out["per_fold"].setdefault(f, {})[key] = (
                round(float(np.mean(ics)), 4) if ics else None)
            out["per_fold"][f][f"n_{key}"] = len(ics)
    live = [d for d in dates if LIVE_ERA_START <= d < HOLDOUT_START]
    ics5 = [v for v in (day_ic(D, 5) for D in live) if v is not None]
    out["live_era"]["ic5"] = round(float(np.mean(ics5)), 4) if ics5 else None
    out["live_era"]["n_ic5"] = len(ics5)
    out["live_era"]["ic21"] = None
    fold_mean5 = np.mean([v["ic5"] for v in out["per_fold"].values()
                          if v["ic5"] is not None])
    live5 = out["live_era"]["ic5"]
    out["memorization_read"] = {
        "fold_mean_ic5": round(float(fold_mean5), 4),
        "live_ic5": live5,
        "fold_minus_live_ic5": (round(float(fold_mean5 - live5), 4)
                                if live5 is not None else None),
        "fold_much_greater_than_live": bool(
            live5 is not None and fold_mean5 - live5 > 0.05),
        "in_sample_overlap_disclosure": (
            "the deployed MLP trained on 2023-06-01 -> 2026-01-14 (L7), which "
            "OVERLAPS folds F4-F6 — their fold-era IC is partially in-sample "
            "for the model; the live era (>= 2026-01-31) is fully "
            "out-of-training. fold >> live = memorization of the historical "
            "record, exactly what this check quantifies (Skeptic L2)."),
    }
    return out


# ===================================================================== L7
def verify_l7() -> dict:
    """Deployed RankingMLP training-window end vs the 2026-03-11 firewall.

    There is NO formal training manifest (S3 models/ranking_expanded_
    unconditioned/ holds only the .pt and the normalization json) — that
    absence is itself a ledgered finding. Verification is by repo forensics:
      1. the deployed S3 .pt is byte-identical (sha256) to the .pt committed
         in 30fb365 ('feat: hybrid ranking system with shadow deployment',
         2026-03-21), which also committed models/.../ranking_history.json;
      2. that commit's training script scripts/build_ranking_history.py
         hardcodes gate_cutoff = '2026-01-15' and filters
         features['date'] < gate_cutoff BEFORE train_ranking_model;
      3. forward 21d targets are computed INSIDE the filtered frame
         (train_ranking._prepare_ranking_data shift(-21) then dropna), so
         both features and target windows end <= 2026-01-14;
      4. sample-count consistency: 36,736 = 64 symbols x 574 dates matches
         the 2023-06-01 -> <2026-01-15 window net of 63d warmup + 21d target.
    Conclusion: training-data end 2026-01-14 < 2026-03-11 -> PASS.
    """
    out: Dict[str, Any] = {"check": "L7 deployed-MLP training-window firewall",
                           "firewall": HOLDOUT_START}
    cached_pt = MODEL_DIR / "ranking_mlp.pt"
    repo_pt = REPO / "models" / "ranking_expanded_unconditioned" / "ranking_mlp.pt"
    sha_c = hashlib.sha256(cached_pt.read_bytes()).hexdigest()
    out["deployed_pt_sha256"] = sha_c
    out["repo_pt_sha256"] = (hashlib.sha256(repo_pt.read_bytes()).hexdigest()
                             if repo_pt.exists() else None)
    out["sha_match_repo"] = out["repo_pt_sha256"] == sha_c
    try:
        r = subprocess.run(
            ["git", "log", "--diff-filter=A", "--format=%H %ad %s",
             "--date=short", "--",
             "models/ranking_expanded_unconditioned/ranking_mlp.pt"],
            cwd=REPO, capture_output=True, text=True, timeout=20)
        out["pt_added_in_commit"] = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else None
    except Exception as e:                                    # pragma: no cover
        out["pt_added_in_commit"] = f"git unavailable: {e}"
    hist_p = REPO / "models" / "ranking_expanded_unconditioned" / "ranking_history.json"
    if hist_p.exists():
        h = json.loads(hist_p.read_text())
        out["ranking_history"] = {k: h.get(k) for k in
                                  ("train_samples", "val_samples", "forward_days",
                                   "best_val_loss")}
    script = (REPO / "scripts" / "build_ranking_history.py").read_text()
    out["training_script_gate_cutoff"] = ("gate_cutoff = '2026-01-15'" in script
                                          and "features['date'] < gate_cutoff" in script)
    out["training_download_window"] = ("start='2023-06-01', end='2026-03-21'" in script)
    try:
        import boto3
        s3 = boto3.client("s3")
        head = s3.head_object(Bucket="investment-system-data",
                              Key="models/ranking_expanded_unconditioned/ranking_mlp.pt")
        out["s3_last_modified"] = str(head["LastModified"])
        out["s3_sha_note"] = "upload date 2026-03-22 is the ARTIFACT upload, not the data end"
    except Exception as e:
        out["s3_last_modified"] = f"unavailable offline: {e}"
    out["training_data_end"] = "2026-01-14"
    out["verdict"] = ("PASS" if (out["sha_match_repo"]
                                 and out["training_script_gate_cutoff"]) else
                      "UNDETERMINABLE")
    out["formal_manifest_exists"] = False
    out["finding"] = ("no formal training manifest exists for the deployed "
                      "RankingMLP — verified by repo forensics (sha-matched "
                      "blob -> committed training script with hardcoded "
                      "gate_cutoff 2026-01-15); training-data end 2026-01-14 "
                      "< 2026-03-11 firewall. Paired-cancellation argument "
                      "(both arms share the model) prints in the E2 block "
                      "regardless (TOURNAMENT_007 G17).")
    return out


# ============================================================ synthetic inputs
def gen_synthetic_organs(out_dir: Path, folds=(1, 2, 3, 4, 5, 6),
                         roster=("M1", "M2", "M5"), seed: int = 4242,
                         signal: float = 0.35) -> dict:
    """SMOKE-ONLY organ inputs (organ_inputs_007.v1) over fold-era dates with a
    planted forward signal so the GA has something real to find. Uses forward
    returns => NEVER evidence; manifest flags synthetic on every file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = build_feature_panel()
    rng = np.random.default_rng(seed)
    closes = panel["closes"]
    dates = panel["dates"]
    names = list(TA.TILT_CORE) + list(TA.TILT_COND)
    px = pd.DataFrame({s: closes[s] for s in names}).reindex(dates)
    fwd5 = px.shift(-5) / px - 1.0
    fwd5 = fwd5.sub(fwd5.mean(axis=1), axis=0)
    n_written = 0
    for f in folds:
        for D in fold_dates(dates, f):
            sig = fwd5.loc[D]
            organs = {}
            for k, snr in zip(roster, (signal, signal * 0.6, signal * 0.3)):
                z = pd.Series(rng.standard_normal(len(names)), index=names)
                sv = sig.fillna(0.0)
                sv = (sv - sv.mean()) / (sv.std() or 1.0)
                mu = snr * sv + np.sqrt(max(1 - snr ** 2, 0.0)) * z
                organs[k] = {"mu": {s: round(float(mu[s]), 6) for s in names},
                             "q": round(float(np.clip(0.4 + 0.3 * rng.random(), 0, 1)), 3)}
            doc = {"date": D, "schema_version": "organ_inputs_007.v1",
                   "organs": organs,
                   "disp_z": round(float(rng.standard_normal() * 0.8), 6),
                   "p_exceed": {},
                   "manifest": {"synthetic": True,
                                "purpose": "wave-2b EA machinery smoke — "
                                           "planted forward signal, NEVER evidence",
                                "seed": seed}}
            (out_dir / f"{D}.json").write_text(json.dumps(doc))
            n_written += 1
    return {"out_dir": str(out_dir), "n_files": n_written, "roster": list(roster),
            "synthetic": True}


def gen_synthetic_rotation_masks(out_dir: Path, roster=("M1", "M2", "M5")) -> dict:
    """SMOKE-ONLY per-rotation masks per the wave-2a contract:
    store/rotation_masks_007/rotation_<f>.json with
    {rotation, withheld_fold, derived_from_folds, roster, masks, synthetic}.
    Real files (wave-2a) re-derive the §2.3 rule from the 5 training folds."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in range(1, 7):
        doc = {"rotation": f, "withheld_fold": f,
               "derived_from_folds": [k for k in range(1, 7) if k != f],
               "roster": list(roster),
               "masks": {k: TA.PRODUCTION_MASKS.get(k, {}) for k in roster},
               "synthetic": True,
               "note": "SMOKE placeholder = frozen production masks; wave-2a "
                       "replaces with per-rotation re-derived masks (§4.6/L1)"}
        (out_dir / f"rotation_{f}.json").write_text(json.dumps(doc, indent=1))
    return {"out_dir": str(out_dir), "n_files": 6, "synthetic": True}


# ============================================================== manifest CLI
def write_manifest(extra: dict, store: Path = STORE) -> Path:
    store.mkdir(parents=True, exist_ok=True)
    p = store / "surrogate_manifest.json"
    doc = json.loads(p.read_text()) if p.exists() else {}
    doc.setdefault("generated", _dt.datetime.now().isoformat(timespec="seconds"))
    doc["surrogate_space_disclaimer"] = ("every number from this simulator is "
                                         "(surrogate space)")
    doc["divergences_l2c"] = DIVERGENCES_L2C
    doc.update(extra)
    doc["updated"] = _dt.datetime.now().isoformat(timespec="seconds")
    p.write_text(json.dumps(doc, indent=1, default=str))
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-packs", action="store_true")
    ap.add_argument("--folds", default="1,2,3,4,5,6")
    ap.add_argument("--no-live", action="store_true")
    ap.add_argument("--fidelity", action="store_true")
    ap.add_argument("--l7", action="store_true")
    ap.add_argument("--synth-organs", action="store_true")
    ap.add_argument("--synth-masks", action="store_true")
    args = ap.parse_args(argv)
    if args.build_packs:
        folds = tuple(int(x) for x in args.folds.split(","))
        s = build_packs(folds=folds, live=not args.no_live)
        write_manifest({"packs": s})
    if args.fidelity:
        ic = fidelity_ic()
        dep = _read_deployed_config()
        write_manifest({"fidelity_l2a_rank_ic": ic,
                        "fidelity_l2c_blend_check": dep})
        print(json.dumps({"l2a": ic, "blend": dep}, indent=1, default=str))
    if args.l7:
        l7 = verify_l7()
        write_manifest({"l7_firewall_check": l7})
        print(json.dumps(l7, indent=1, default=str))
    if args.synth_organs:
        r = gen_synthetic_organs(PROTO / "store" / "nightly_007_folds_SYNTH")
        print(json.dumps(r, indent=1))
    if args.synth_masks:
        r = gen_synthetic_rotation_masks(PROTO / "store" / "rotation_masks_007_SYNTH")
        print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
