"""PKT-TB-007 — ORB-1 tilt adapter (BUILD_SPEC_007 §2; the ONE expression channel).

A Strategy.post_decision passed to replay_engine.run_variant. It EDITS the
chassis's own intents at the margin — never replaces them. Incumbent arm runs
with no adapter; one OS process per arm.

NEUTRAL RECOVERY (load-bearing, §2.5): T_t = 0 (B0 genome / zero conviction /
no organ file / tilt quantized away) => post_decision returns the incumbent's
intents object UNCHANGED — the same list, the same dicts, nothing mutated —
so the daily series is bit-for-bit the incumbent's (verified by B0-EXPR).

----------------------------------------------------------------------------
ORGAN INPUT CONTRACT (frozen here; organs plug in later — §6.2)

The adapter consumes ONLY precomputed per-date files (no model forward passes
inside the replay loop):  <nightly_dir>/<D>.json   with schema

  {
    "date": "YYYY-MM-DD",
    "schema_version": "organ_inputs_007.v1",
    "organs": {                       # per shipped organ
      "M1": {"mu": {"ITA": 0.42, ...},   # RAW per-symbol tilt scores; the
                                          # adapter rank_z's them over support;
                                          # must cover at least TILT_CORE+COND
             "q": 0.61},                  # gain scalar / self-confidence in [0,1]
      "M2": {...}, "M5": {...}
    },
    "disp_z": 1.3,                    # M3 forecast z-scored on its trailing
                                      # 252d window (precompute job); absent or
                                      # null => B-disp constant disp_t = 0.5
    "p_exceed": {"FXI": 0.31, ...},   # M4 per-symbol exceedance prob (already
                                      # bucket->symbol mapped); absent => no damp
    "manifest": {...}                 # provenance (code sha, seeds) — opaque here
  }

Missing file for D, or no active directional organ => NEUTRAL day.
All point-in-time discipline (visible_from, publication keys) lives in the
precompute that writes these files — never here.
----------------------------------------------------------------------------
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROTO = Path(__file__).resolve().parent
REPO = PROTO.parents[2]

import sys
for _p in (str(PROTO), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from genome_007 import Genome007                       # noqa: E402
from lot_fix_007 import aggregate_book                 # noqa: E402
from risk_stats_007 import RiskStats                   # noqa: E402
from src.utils.three_line_replay.strategies import Strategy, StrategyContext  # noqa: E402

# ---------------------------------------------------------------- frozen constants
# §2.2 support tiers (FROZEN — Phase-0 table, TOURNAMENT G15/A9)
TILT_CORE = ("ITA", "SOXX", "XRT", "TLT", "AGG", "MUB", "FXE", "USO", "FXI", "RSP")
TILT_COND = ("IYR", "SHY", "KRE")
VIXY = "VIXY"                       # excluded from support entirely
FXE_HARD_CAP = 0.0125               # FXE additionally capped 1.25% (ADV $14M)
DEFENSIVE = ("TLT", "AGG", "MUB", "SHY")
T_MAX = 0.08                        # NAV one-sided, FIXED — never a gene
EPS_SIGMA_FRAC = 0.10               # eps_sigma = 0.10 x Sum|dw|*sigma
A_CONV = 1.0                        # conviction 'a', FIXED
THETA_CONV = 0.5                    # conviction theta, FIXED
PROJ_MAX_ITER = 5
M4_BASE_RATE = 0.2
MIN_ORDER_DEFAULT = 250.0
_ND = NormalDist()

# §2.3 production per-organ per-name tier masks (FROZEN rule applied to the
# Phase-0 table). Cross-fit interface: pass a masks dict of the same shape to
# make_tilt_strategy to override (rotations re-derive via the frozen rule text).
PRODUCTION_MASKS: Dict[str, Dict[str, float]] = {
    "M1": {**{s: 1.0 for s in ("ITA", "SOXX", "TLT", "AGG", "MUB", "FXE",
                               "USO", "FXI", "RSP")},
           **{s: 0.5 for s in ("XRT", "IYR", "SHY")}, "KRE": 0.0},
    "M2": {"XRT": 1.0, "KRE": 0.5, "IYR": 0.5},
    # M5: masks not applied (its sleeve definition is its mask) => 1.0 inside
    # the support tiers; forced 0 outside CORE+COND like everything else.
    "M5": {s: 1.0 for s in TILT_CORE + TILT_COND},
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def load_organ_outputs(nightly_dir: Path, date: str) -> Optional[dict]:
    p = Path(nightly_dir) / f"{date}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


# ---------------------------------------------------------------- rank helpers
def rank_z(values: Dict[str, float], names: List[str]) -> Dict[str, float]:
    """Gaussian-rank z over `names` (names missing from values contribute
    nothing and get z=0)."""
    present = [n for n in names if n in values and np.isfinite(values[n])]
    if len(present) < 2:
        return {n: 0.0 for n in names}
    s = pd.Series({n: float(values[n]) for n in present})
    r = s.rank(method="average")
    n = len(present)
    z = {k: _ND.inv_cdf(float(rv) / (n + 1)) for k, rv in r.items()}
    return {nm: z.get(nm, 0.0) for nm in names}


def _spearman(a: Dict[str, float], b: Dict[str, float]) -> Optional[float]:
    common = sorted(set(a) & set(b))
    if len(common) < 3:
        return None
    sa = pd.Series([a[k] for k in common]).rank()
    sb = pd.Series([b[k] for k in common]).rank()
    if sa.std() == 0 or sb.std() == 0:
        return None
    return float(np.corrcoef(sa, sb)[0, 1])


# ---------------------------------------------------------------- conviction (§2.4)
def compute_conviction(organs: Optional[dict], genome: Genome007
                       ) -> Dict[str, Any]:
    """c_t = sigmoid((a*agree + b*disp + sum_k tau_k q_k - theta)/temp);
    T_t = tilt_gain * c_t * T_max. Returns the full diagnostic block.
    neutral=True when there is nothing to express."""
    out: Dict[str, Any] = {"c": 0.0, "T_t": 0.0, "agree": None, "disp": None,
                           "organ_q": {}, "tau": {}, "active": [],
                           "neutral": True, "neutral_reason": None}
    roster = sorted(genome.organ_trust)
    directional = [o for o in genome.shipped_directional]
    if organs is None:
        out["neutral_reason"] = "no_organ_file"
        return out
    od = organs.get("organs", {}) or {}
    active = [o for o in directional
              if o in od and isinstance(od[o].get("mu"), dict)
              and len(od[o]["mu"]) >= 2]
    out["active"] = active
    if not active:
        out["neutral_reason"] = "no_active_directional_organ"
        return out

    # tau over shipped directional organs (softmax of trust genes, §2.4),
    # renormalized over today's ACTIVE set
    logits = np.array([genome.organ_trust[o] for o in directional], dtype=float)
    e = np.exp(logits - logits.max())
    tau_dir = dict(zip(directional, e / e.sum()))
    tsum = sum(tau_dir[o] for o in active)
    tau = {o: (tau_dir[o] / tsum if tsum > 0 else 0.0) for o in active}
    out["tau"] = tau

    # q-sum over ALL shipped organs present (M4's trust gene = the weight on
    # its damp inside the q-sum, §4.1)
    logits_all = np.array([genome.organ_trust[o] for o in roster], dtype=float)
    ea = np.exp(logits_all - logits_all.max())
    tau_all = dict(zip(roster, ea / ea.sum()))
    qs = {o: float(np.clip(od[o].get("q", 0.0), 0.0, 1.0))
          for o in roster if o in od}
    out["organ_q"] = qs
    present = sorted(qs)
    wsum = sum(tau_all[o] for o in present)
    q_sum = (sum(tau_all[o] * qs[o] for o in present) / wsum) if wsum > 0 else 0.0

    # agree_t: mean pairwise rank-corr of active organs' mu vectors;
    # 1 active organ => agree = that organ's q
    if len(active) == 1:
        agree = qs.get(active[0], 0.0)
    else:
        cors = []
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                c = _spearman(od[active[i]]["mu"], od[active[j]]["mu"])
                if c is not None:
                    cors.append(c)
        agree = float(np.mean(cors)) if cors else 0.0
    out["agree"] = agree

    disp_z = organs.get("disp_z", None)
    disp = _sigmoid(float(disp_z)) if disp_z is not None else 0.5   # B-disp
    out["disp"] = disp
    out["disp_forecast_z"] = disp_z

    c = _sigmoid((A_CONV * agree + genome.disp_gain * disp + q_sum - THETA_CONV)
                 / genome.conviction_temp)
    out["c"] = c
    if c < genome.dead_zone or genome.tilt_gain <= 0.0:
        out["neutral_reason"] = ("dead_zone" if genome.tilt_gain > 0.0
                                 else "tilt_gain_zero")
        return out
    out["T_t"] = genome.tilt_gain * c * T_MAX
    out["neutral"] = False
    return out


# ---------------------------------------------------------------- direction (§2.4)
def combine_direction(organs: dict, genome: Genome007,
                      masks: Dict[str, Dict[str, float]],
                      support: List[str], active: List[str],
                      tau: Dict[str, float]
                      ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """s_i = sum_k tau_k * mask[k,i] * rank_z(mu_k)_i  over support.
    Masks forced 0 outside TILT_CORE+TILT_COND regardless of the table."""
    od = organs.get("organs", {})
    tiers = set(TILT_CORE) | set(TILT_COND)
    organ_mu_z: Dict[str, Dict[str, float]] = {}
    s = {n: 0.0 for n in support}
    for k in active:
        z = rank_z(od[k]["mu"], support)
        organ_mu_z[k] = z
        mk = masks.get(k, {})
        for n in support:
            m = float(mk.get(n, 0.0)) if n in tiers else 0.0
            s[n] += tau[k] * m * z[n]
    return s, organ_mu_z


def apply_m4_damp(dw: Dict[str, float], organs: dict, genome: Genome007
                  ) -> Dict[str, float]:
    """dw_i <- dw_i * (1 - eds * clip((p_hat - 0.2)/0.8, 0, 1)) (§2.4)."""
    if genome.event_damp_strength <= 0 or "M4" not in genome.organ_trust:
        return {}
    p = organs.get("p_exceed") or {}
    damp = {}
    for n in sorted(dw):
        if n in p:
            f = 1.0 - genome.event_damp_strength * float(
                np.clip((float(p[n]) - M4_BASE_RATE) / (1 - M4_BASE_RATE), 0, 1))
            if f < 1.0:
                dw[n] *= f
                damp[n] = round(f, 6)
    return damp


def apply_defensive_clip(dw: Dict[str, float], frac: float
                         ) -> Tuple[float, float]:
    """Clip the {TLT,AGG,MUB,SHY} share of Sum|dw|/2 to `frac`; excess
    redistributed pro-rata to non-defensive legs (signs kept). BEFORE
    projection (§2.4). Returns (share_pre, share_post)."""
    one_sided = sum(abs(v) for v in dw.values()) / 2.0
    if one_sided <= 0:
        return 0.0, 0.0
    dmass = sum(abs(dw[n]) for n in dw if n in DEFENSIVE)
    share = dmass / one_sided
    if share <= frac or dmass <= 0:
        return share, share
    scale = frac / share
    removed = dmass * (1 - scale)
    nd_mass = sum(abs(dw[n]) for n in dw if n not in DEFENSIVE)
    for n in sorted(dw):
        if n in DEFENSIVE:
            dw[n] *= scale
        elif nd_mass > 0:
            dw[n] *= (1.0 + removed / nd_mass)
    one_sided2 = sum(abs(v) for v in dw.values()) / 2.0
    dmass2 = sum(abs(dw[n]) for n in dw if n in DEFENSIVE)
    return share, (dmass2 / one_sided2 if one_sided2 > 0 else 0.0)


# ---------------------------------------------------------------- projection (§2.5)
def project_parity(d: np.ndarray, beta: np.ndarray, sigma: np.ndarray,
                   lb: np.ndarray, ub: np.ndarray,
                   max_iter: int = PROJ_MAX_ITER) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Project d onto {Sum x = 0} n {Sum x*beta = 0} n {|Sum x*sigma| <= eps},
    eps = EPS_SIGMA_FRAC * Sum|x|*sigma, with box bounds [lb, ub].

    Closed-form projection onto the two equalities, clip to box, iterate
    project->clip <= max_iter; then sigma-budget check — if violated, shrink
    the offending side pro-rata and re-project once (§2.5 verbatim)."""
    n = len(d)
    info: Dict[str, Any] = {"iters": 0, "clipped": [], "infeasible": False,
                            "sigma_shrunk": False}
    if n == 0:
        return np.zeros(0), info
    ones = np.ones(n)
    A = np.vstack([ones, beta])
    G = A @ A.T
    # degenerate beta (collinear with ones) -> drop the beta row
    degenerate = abs(np.linalg.det(G)) < 1e-12 * max(1.0, float(np.trace(G)) ** 2)
    if degenerate:
        A = ones.reshape(1, -1)
        G = A @ A.T
        info["beta_row_dropped"] = True

    def _proj(x):
        lam = np.linalg.solve(G, A @ x)
        return x - A.T @ lam

    def _repair(x):
        """Exact equality repair on the free (unclipped) set: distributes the
        residual over names with box slack so project->clip ends EXACT."""
        for _ in range(4):
            r = A @ x
            if np.all(np.abs(r) < 1e-12):
                return x, True
            tiny = 1e-12
            free = (x > lb + tiny) & (x < ub - tiny)
            if free.sum() <= A.shape[0]:
                return x, False
            Af = A[:, free]
            Gf = Af @ Af.T
            if abs(np.linalg.det(Gf)) < 1e-15:
                return x, False
            x = x.copy()
            x[free] -= Af.T @ np.linalg.solve(Gf, r)
            x = np.clip(x, lb, ub)
        return x, bool(np.all(np.abs(A @ x) < 1e-10))

    x = d.astype(float).copy()
    clipped: set = set()
    for it in range(max_iter):
        x = _proj(x)
        xc = np.clip(x, lb, ub)
        clipped |= set(np.nonzero(np.abs(xc - x) > 1e-12)[0].tolist())
        x = xc
        info["iters"] = it + 1
        if (abs(x.sum()) < 1e-10 and abs(float(x @ beta)) < 1e-10):
            break
    x, _ok = _repair(x)

    def _sigma_check(x):
        eps = EPS_SIGMA_FRAC * float(np.abs(x) @ sigma)
        return float(x @ sigma), eps

    # sigma-budget. BUILD_SPEC §2.5 prescribes ONE pro-rata shrink of the
    # offending side + a single re-projection — but the equality re-projection
    # redistributes mass and can re-violate the budget (logged finding in
    # validation_looks_007.jsonl). Completion: after the spec's shrink pass,
    # project onto the THREE-row affine system {Sum x=0, Sum x*beta=0,
    # Sum x*sigma=target-inside-budget} with box clipping, <=3 passes.
    sres, eps = _sigma_check(x)
    if abs(sres) > eps and eps > 0:
        # the spec's literal move first: shrink offending side, re-project once
        side = np.sign(sres)
        mask = (np.sign(x) == side)
        side_sum = float((x[mask] * sigma[mask]).sum())
        if abs(side_sum) > 1e-15:
            other = sres - side_sum
            f = float(np.clip((side * eps * 0.5 - other) / side_sum, 0.0, 1.0))
            x = x.copy()
            x[mask] *= f
            x = np.clip(_proj(x), lb, ub)
            x, _ok = _repair(x)
            info["sigma_shrunk"] = True
        sres, eps = _sigma_check(x)
    for _ in range(50):
        if not (abs(sres) > eps and eps > 0):
            break
        A3 = np.vstack([A, sigma])
        G3 = A3 @ A3.T
        if abs(np.linalg.det(G3)) < 1e-18 * max(1.0, float(np.trace(G3)) ** 3):
            break                          # sigma ~ collinear: cannot bind
        # aim the sigma row at ZERO (x=0 is always feasible, so POCS between
        # this affine set and the box converges; the 10% budget is the slack
        # that absorbs the box-clip perturbation)
        lam = np.linalg.solve(G3, A3 @ x)
        x = np.clip(x - A3.T @ lam, lb, ub)
        x, _ok = _repair(x)                  # restore the two equalities exactly
        info["sigma_shrink_passes"] = info.get("sigma_shrink_passes", 0) + 1
        sres, eps = _sigma_check(x)

    tol = 1e-9 + 1e-6 * float(np.abs(x).sum())
    cash_resid = float(x.sum())
    beta_resid = float(x @ beta)
    if abs(cash_resid) > tol or abs(beta_resid) > tol or (abs(sres) > eps + tol):
        info["infeasible"] = True
    info["clipped"] = sorted(clipped)
    info["cash_resid"] = cash_resid
    info["beta_resid"] = beta_resid
    info["sigma_resid"] = sres
    info["eps_sigma"] = eps
    return x, info


# ---------------------------------------------------------------- bounds (§2.2)
def tilt_bounds(sym: str, genome: Genome007, w_avail: float, w_buy: float,
                w_held: float, fully_sold: bool, partially_sold: bool,
                has_stats: bool) -> Tuple[float, float]:
    """Box bounds on dw for one support name.

    - fully-sold-today names: no tilt.
    - core: |dw| <= cap_core (FXE additionally 1.25%); cond: cap_conditional.
    - ballast tilt cap = 0; HELD ballast may be trimmed/topped only as
      projection funding legs within its existing position => box
      [-w_avail, +w_held]  (interpretation of 'within its existing position'
      for the top side = at most the current weight; logged as a finding).
    - no shorting: lower bound never exceeds what is actually reducible
      (held-and-not-yet-sold weight + today's chassis buy weight).
    - names with no point-in-time beta/sigma stats are frozen (cannot even
      fund) so the projection algebra stays exact.
    """
    if fully_sold or not has_stats:
        return 0.0, 0.0
    if sym in TILT_CORE:
        cap = min(genome.cap_core, FXE_HARD_CAP) if sym == FXE_SYM else genome.cap_core
        lo = -min(cap, w_avail + w_buy)
        hi = 0.0 if partially_sold else cap
        return lo, hi
    if sym in TILT_COND:
        cap = genome.cap_conditional
        lo = -min(cap, w_avail + w_buy)
        hi = 0.0 if partially_sold else cap
        return lo, hi
    # ballast
    if w_held > 0:
        return -w_avail, (0.0 if partially_sold else w_held)
    return 0.0, 0.0


FXE_SYM = "FXE"


# ---------------------------------------------------------------- intent edits (§2.5)
def tilt_to_intents(incumbent_intents: List[Dict[str, Any]],
                    dw: Dict[str, float], nav: float,
                    marks: Dict[str, float],
                    held_avail_shares: Dict[str, float],
                    universe_rows: Dict[str, dict],
                    min_order: float = MIN_ORDER_DEFAULT
                    ) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]]:
    """Convert projected dw to intent EDITS: scale chassis BUY dollars; add
    SELL trims of held names; add BUYs (b-extended). NEVER REDUCE; integer
    shares; min_order respected (legs under min_order dropped); residual cash
    imbalance absorbed by shrinking the largest leg on the heavier side.

    Returns (new_intents, edits_log); new_intents=None => nothing tradeable
    survived quantization (caller goes NEUTRAL, 'dead_zone_quantized').
    Incumbent intent dicts are NEVER mutated (edits are copies)."""
    log: Dict[str, Any] = {"scaled": [], "added": [], "suppressed": [],
                           "dropped_sub_min_order": [], "rounding_absorber": None}
    buy_idx: Dict[str, int] = {}
    for i, it in enumerate(incumbent_intents):
        if it.get("action") == "BUY":
            buy_idx.setdefault(it["symbol"], i)

    edited: Dict[int, Optional[Dict[str, Any]]] = {}   # idx -> new dict | None=suppress
    added_sells: List[Dict[str, Any]] = []
    added_buys: List[Dict[str, Any]] = []
    realized: List[Tuple[str, float, str]] = []        # (sym, cash-flow delta, kind)

    for sym in sorted(dw):
        w = dw[sym]
        leg = abs(w) * nav
        if leg <= 0:
            continue
        price = float(marks.get(sym, 0.0) or 0.0)
        if leg < min_order:
            log["dropped_sub_min_order"].append(sym)
            continue
        if price <= 0 or not np.isfinite(price):
            log["dropped_sub_min_order"].append(sym)
            continue
        uni = universe_rows.get(sym, {})
        if w > 0 and sym in buy_idx:
            i = buy_idx[sym]
            orig = incumbent_intents[i]
            orig_d = float(orig.get("dollars", 0) or
                           (float(orig.get("shares", 0)) * float(orig.get("price", 0) or 0)))
            p0 = float(orig.get("price", price) or price)
            new_d = orig_d + leg
            new_it = dict(orig)
            new_it["dollars"] = round(new_d, 2)
            new_it["shares"] = int(new_d / p0) if p0 > 0 else 0
            edited[i] = new_it
            realized.append((sym, new_d - orig_d, "scale_up"))
            log["scaled"].append({"symbol": sym, "from": round(orig_d, 2),
                                  "to": round(new_d, 2)})
        elif w < 0 and sym in buy_idx:
            i = buy_idx[sym]
            orig = incumbent_intents[i]
            orig_d = float(orig.get("dollars", 0) or
                           (float(orig.get("shares", 0)) * float(orig.get("price", 0) or 0)))
            p0 = float(orig.get("price", price) or price)
            new_d = orig_d - leg
            if new_d < min_order:
                edited[i] = None
                realized.append((sym, -orig_d, "suppress"))
                log["suppressed"].append({"symbol": sym, "dollars": round(orig_d, 2)})
            else:
                new_it = dict(orig)
                new_it["dollars"] = round(new_d, 2)
                new_it["shares"] = int(new_d / p0) if p0 > 0 else 0
                edited[i] = new_it
                realized.append((sym, new_d - orig_d, "scale_down"))
                log["scaled"].append({"symbol": sym, "from": round(orig_d, 2),
                                      "to": round(new_d, 2)})
        elif w > 0:
            shares = int(leg / price)
            if shares <= 0 or shares * price < min_order:
                log["dropped_sub_min_order"].append(sym)
                continue
            it = {"symbol": sym, "action": "BUY", "shares": shares,
                  "dollars": round(shares * price, 2), "price": round(price, 6),
                  "asset_class": uni.get("asset_class", "equity"),
                  "sector": uni.get("sector", "broad"),
                  "leverage_flag": int(uni.get("leverage_flag", 0) or 0),
                  "reason": "ORB1_TILT_BUY"}
            added_buys.append(it)
            realized.append((sym, shares * price, "add_buy"))
        else:
            avail = float(held_avail_shares.get(sym, 0.0))
            shares = min(int(leg / price), int(avail))
            if shares <= 0 or shares * price < min_order:
                log["dropped_sub_min_order"].append(sym)
                continue
            it = {"symbol": sym, "action": "SELL", "shares": float(shares),
                  "dollars": round(shares * price, 2), "price": round(price, 6),
                  "asset_class": uni.get("asset_class", "equity"),
                  "sector": uni.get("sector", "broad"),
                  "leverage_flag": int(uni.get("leverage_flag", 0) or 0),
                  "reason": "ORB1_TILT_TRIM"}
            added_sells.append(it)
            realized.append((sym, -(shares * price), "add_sell"))

    if not edited and not added_sells and not added_buys:
        return None, log

    # ---- rounding/imbalance absorber: shrink the largest leg on the heavier side
    imb = sum(d for _, d, _ in realized)        # >0: buys exceed sells
    if abs(imb) > 1e-6:
        if imb > 0:
            cands = ([(it["dollars"], "add", it) for it in added_buys] +
                     [(v["dollars"], "edit", v) for k, v in edited.items()
                      if v is not None and v.get("action") == "BUY"])
            if cands:
                cands.sort(key=lambda t: (-t[0], t[2]["symbol"]))
                _, kind, it = cands[0]
                p0 = float(it.get("price", 0) or 0)
                new_d = max(float(it["dollars"]) - imb, min_order)
                it["dollars"] = round(new_d, 2)
                if p0 > 0:
                    it["shares"] = (int(new_d / p0) if it["action"] == "BUY"
                                    else it["shares"])
                log["rounding_absorber"] = it["symbol"]
        else:
            cands = [(it["dollars"], it) for it in added_sells]
            if cands:
                cands.sort(key=lambda t: (-t[0], t[1]["symbol"]))
                _, it = cands[0]
                p0 = float(it.get("price", 0) or 0)
                new_d = max(float(it["dollars"]) + imb, 0.0)   # imb<0 shrinks
                sh = float(int(new_d / p0)) if p0 > 0 else 0.0
                if sh <= 0 or sh * p0 < min_order:
                    added_sells.remove(it)
                else:
                    it["shares"] = sh
                    it["dollars"] = round(sh * p0, 2)
                log["rounding_absorber"] = it["symbol"]

    new_intents: List[Dict[str, Any]] = []
    for i, it in enumerate(incumbent_intents):
        if i in edited:
            if edited[i] is not None:
                new_intents.append(edited[i])
        else:
            new_intents.append(it)
    new_intents.extend(sorted(added_sells, key=lambda t: t["symbol"]))
    new_intents.extend(sorted(added_buys, key=lambda t: t["symbol"]))
    return new_intents, log


# ---------------------------------------------------------------- shared solve
def solve_tilt(organs: dict, genome: Genome007,
               masks: Dict[str, Dict[str, float]],
               support: List[str], active: List[str],
               tau: Dict[str, float], T_t: float,
               beta_v: np.ndarray, sigma_v: np.ndarray,
               lb_v: np.ndarray, ub_v: np.ndarray
               ) -> Tuple[Dict[str, float], dict, dict, tuple]:
    """THE expression solve (§2.4 -> §2.5): combine direction -> demean over
    support -> scale to the tilt budget -> M4 damp -> defensive clip -> parity
    projection. Module-level so the replay adapter (post_decision) and the EA
    fitness walk (surrogate_007 / ea_007) share ONE implementation — the
    simulator matches the harness expression bit-for-bit by construction
    (BUILD_SPEC_007 §5.2; wave-2b consistency test asserts <=1e-9)."""
    s, organ_mu_z = combine_direction(organs, genome, masks, support,
                                      active, tau)
    sv = np.array([s[n] for n in support])
    sv = sv - sv.mean()                      # demeaned over support
    mass = np.abs(sv).sum()
    if mass <= 0:
        return {}, {}, organ_mu_z, (0.0, 0.0)
    dw = {n: float(sv[i] * (2 * T_t / mass)) for i, n in enumerate(support)}
    m4 = apply_m4_damp(dw, organs, genome)
    share_pre, share_post = apply_defensive_clip(dw, genome.defensive_fraction)
    d = np.array([dw[n] for n in support])
    x, pinfo = project_parity(d, beta_v, sigma_v, lb_v, ub_v)
    return ({n: float(x[i]) for i, n in enumerate(support)},
            {"m4_damp": m4, "projection": pinfo}, organ_mu_z,
            (share_pre, share_post))


# ---------------------------------------------------------------- marks guard
def guarded_marks(features_df, inputs_date: str, cache=None
                  ) -> Tuple[Dict[str, float], str]:
    """Latest close per symbol with rows dated == D dropped (TB-006 wiring
    adjudication #3 carried). Falls back to daily/D/prices.parquet (< D rows)
    when the features frame is entirely same-day."""
    marks: Dict[str, float] = {}
    src = "features"
    df = features_df
    if df is not None and len(df) > 0:
        ds = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df[ds < inputs_date]
    if df is None or len(df) == 0:
        src = "prices_fallback"
        if cache is None:
            return {}, "empty_no_cache"
        pr = cache.get_parquet(f"daily/{inputs_date}/prices.parquet")
        pr = pr[["symbol", "date", "close"]].copy()
        ds = pd.to_datetime(pr["date"]).dt.normalize().dt.strftime("%Y-%m-%d")
        df = pr[ds < inputs_date]
    df = df[["symbol", "date", "close"]].sort_values(["symbol", "date"])
    latest = df.groupby("symbol").tail(1)
    for _, row in latest.iterrows():
        if pd.notna(row["close"]):
            marks[str(row["symbol"])] = float(row["close"])
    return marks, src


# ---------------------------------------------------------------- the strategy
def make_tilt_strategy(genome: Genome007 | dict | str | Path,
                       nightly_dir: Path | str,
                       log_dir: Optional[Path | str] = None,
                       cache=None,
                       masks: Optional[Dict[str, Dict[str, float]]] = None,
                       universe_csv: Path | str = REPO / "config" / "universe.csv",
                       risk: Optional[RiskStats] = None,
                       min_order: float = MIN_ORDER_DEFAULT) -> Strategy:
    if isinstance(genome, (str, Path)):
        genome = Genome007.from_json(Path(genome))
    elif isinstance(genome, dict):
        genome = Genome007.from_dict(genome)
    masks = masks if masks is not None else PRODUCTION_MASKS
    nightly_dir = Path(nightly_dir)
    log_dir = Path(log_dir) if log_dir is not None else None
    risk = risk or RiskStats()
    uni = pd.read_csv(universe_csv)
    universe_rows = {r["symbol"]: dict(r) for _, r in uni.iterrows()}

    def _log(payload: dict) -> None:
        if log_dir is None:
            return
        d = log_dir / "expression_log"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{payload['date']}.json").write_text(
            json.dumps(payload, indent=1, sort_keys=True, default=float))
        with (log_dir / "expression_log.jsonl").open("a") as fh:
            fh.write(json.dumps(payload, sort_keys=True, default=float) + "\n")

    def _neutral(D: str, reason: str, conv: Optional[dict] = None) -> None:
        _log({"date": D, "channel": "tilt", "neutral": True,
              "neutral_reason": reason,
              "conviction": ({"c": conv["c"], "agree": conv["agree"],
                              "disp_forecast_z": conv.get("disp_forecast_z"),
                              "organ_q": conv["organ_q"]} if conv else None)})

    # the core direction->projection pipeline, reused by organ attribution —
    # a thin delegate to the module-level shared solve (solve_tilt), which the
    # EA fitness walk also calls (one implementation, by construction)
    def _pipeline(organs, active, tau, T_t, support, beta_v, sigma_v,
                  lb_v, ub_v) -> Tuple[Dict[str, float], dict, dict, tuple]:
        return solve_tilt(organs, genome, masks, support, active, tau, T_t,
                          beta_v, sigma_v, lb_v, ub_v)

    def post(ctx: StrategyContext, incumbent_intents: List[Dict[str, Any]]
             ) -> List[Dict[str, Any]]:
        D = ctx.inputs_date
        organs = load_organ_outputs(nightly_dir, D)
        conv = compute_conviction(organs, genome)
        if conv["neutral"]:
            _neutral(D, conv["neutral_reason"], conv)
            return incumbent_intents                       # UNCHANGED object
        T_t = conv["T_t"]
        active, tau = conv["active"], conv["tau"]

        # ---- book state (lot-aggregated, C5 site 1) ------------------------
        marks, mark_src = guarded_marks(ctx.features_df, D, cache=cache)
        held_shares, w_prev, nav, missing = aggregate_book(
            list(ctx.portfolio.positions), marks, float(ctx.portfolio.cash))
        if nav <= 0:
            _neutral(D, "nav_nonpositive", conv)
            return incumbent_intents

        # ---- chassis intent analysis ---------------------------------------
        buys_w: Dict[str, float] = {}
        sell_w: Dict[str, float] = {}
        sell_shares: Dict[str, float] = {}
        for it in incumbent_intents:
            sym = it.get("symbol")
            act = it.get("action")
            if act == "BUY":
                d_ = float(it.get("dollars", 0) or
                           (float(it.get("shares", 0)) * float(it.get("price", 0) or 0)))
                buys_w[sym] = buys_w.get(sym, 0.0) + d_ / nav
            elif act in ("SELL", "REDUCE"):
                sh = float(it.get("shares", held_shares.get(sym, 0.0)))
                if act == "REDUCE":
                    sh *= 0.5
                sh = min(sh, held_shares.get(sym, 0.0))
                sell_shares[sym] = sell_shares.get(sym, 0.0) + sh
                mk = marks.get(sym, 0.0)
                sell_w[sym] = sell_shares[sym] * mk / nav if mk else 0.0
        fully_sold = {s for s, sh in sell_shares.items()
                      if sh >= held_shares.get(s, 0.0) - 1e-9}
        partially_sold = {s for s in sell_shares if s not in fully_sold}

        # ---- support set (§2.2; VIXY excluded entirely) --------------------
        support = sorted((set(buys_w) | set(held_shares)
                          | set(TILT_CORE) | set(TILT_COND)) - {VIXY})
        stats = risk.table(support, D)
        beta_v = np.array([stats.get(n, {}).get("beta", 0.0) for n in support])
        sigma_v = np.array([stats.get(n, {}).get("sigma", 0.0) for n in support])
        lb, ub = [], []
        for n in support:
            w_held = w_prev.get(n, 0.0)
            w_avail = max(w_held - sell_w.get(n, 0.0), 0.0)
            lo, hi = tilt_bounds(n, genome, w_avail, buys_w.get(n, 0.0), w_held,
                                 n in fully_sold, n in partially_sold,
                                 n in stats)
            lb.append(lo); ub.append(hi)
        lb_v, ub_v = np.array(lb), np.array(ub)

        # zero funding capacity (e.g. all-cash / panic-force-sell book): the
        # cash-neutral overlay structurally cannot speak — honest neutral
        if float(-lb_v.sum()) < 1e-6 or float(ub_v.sum()) < 1e-6:
            _neutral(D, "no_tilt_capacity", conv)
            return incumbent_intents

        dw, pdiag, organ_mu_z, (dshare_pre, dshare_post) = _pipeline(
            organs, active, tau, T_t, support, beta_v, sigma_v, lb_v, ub_v)
        if not dw or sum(abs(v) for v in dw.values()) <= 0:
            _neutral(D, "zero_direction", conv)
            return incumbent_intents
        if pdiag["projection"].get("infeasible"):
            pi = dict(pdiag["projection"])
            pi["clipped"] = [support[i] for i in pi.get("clipped", [])]
            mo_ = float(ctx.variant_config.get("decision_params", {})
                        .get("min_order_dollars", min_order))
            mass = sum(abs(v) for v in dw.values()) / 2.0 * nav
            reason = ("dead_zone_quantized" if mass < 2 * mo_
                      else "projection_infeasible")
            _log({"date": D, "channel": "tilt", "neutral": True,
                  "neutral_reason": reason,
                  "residual_tilt_dollars": round(mass, 2),
                  "projection": pi,
                  "support_n": len(support),
                  "bounds_lo_sum": float(lb_v.sum()),
                  "bounds_hi_sum": float(ub_v.sum()),
                  "conviction": {"c": conv["c"], "agree": conv["agree"],
                                 "organ_q": conv["organ_q"]}})
            return incumbent_intents

        # ---- organ attribution (recompute with organ k zeroed, tau renorm) --
        attribution: Dict[str, Dict[str, float]] = {}
        if len(active) > 1:
            for k in active:
                rest = [o for o in active if o != k]
                ts = sum(tau[o] for o in rest)
                tau_k = {o: tau[o] / ts for o in rest} if ts > 0 else {}
                dw_k, _, _, _ = _pipeline(organs, rest, tau_k, T_t, support,
                                          beta_v, sigma_v, lb_v, ub_v)
                attribution[k] = {n: round(dw.get(n, 0.0) - dw_k.get(n, 0.0), 8)
                                  for n in support
                                  if abs(dw.get(n, 0.0) - dw_k.get(n, 0.0)) > 1e-8}
        elif active:
            attribution[active[0]] = {n: round(v, 8) for n, v in dw.items()
                                      if abs(v) > 1e-8}

        # ---- intent edits ----------------------------------------------------
        held_avail = {s: held_shares.get(s, 0.0) - sell_shares.get(s, 0.0)
                      for s in held_shares}
        mo = float(ctx.variant_config.get("decision_params", {})
                   .get("min_order_dollars", min_order))
        new_intents, edits = tilt_to_intents(incumbent_intents, dw, nav, marks,
                                             held_avail, universe_rows, mo)
        if new_intents is None:
            _neutral(D, "dead_zone_quantized", conv)
            return incumbent_intents

        # ---- book overlap diagnostic ----------------------------------------
        w_chassis = dict(w_prev)
        for s_, v in buys_w.items():
            w_chassis[s_] = w_chassis.get(s_, 0.0) + v
        for s_, v in sell_w.items():
            w_chassis[s_] = max(w_chassis.get(s_, 0.0) - v, 0.0)
        w_tilt = dict(w_chassis)
        for s_, v in dw.items():
            w_tilt[s_] = max(w_tilt.get(s_, 0.0) + v, 0.0)
        names = set(w_chassis) | set(w_tilt)
        smin = sum(min(w_chassis.get(n, 0.0), w_tilt.get(n, 0.0)) for n in names)
        smax = max(sum(w_chassis.values()), sum(w_tilt.values()), 1e-12)
        overlap = smin / smax

        pinfo = pdiag["projection"]
        _log({"date": D, "channel": "tilt", "neutral": False,
              "conviction": {"c": conv["c"], "agree": conv["agree"],
                             "disp_forecast_z": conv.get("disp_forecast_z"),
                             "organ_q": conv["organ_q"]},
              "trust": {k: round(v, 6) for k, v in tau.items()},
              "organ_mu": {k: {n: round(z, 4) for n, z in zz.items() if z != 0.0}
                           for k, zz in organ_mu_z.items()},
              "combined": {n: round(v, 6) for n, v in dw.items() if v != 0.0},
              "m4_damp": pdiag["m4_damp"],
              "budget": {"T_t": round(T_t, 6), "T_max": T_MAX,
                         "tilt_gain": genome.tilt_gain,
                         "defensive_share_pre": round(dshare_pre, 4),
                         "defensive_share_post": round(dshare_post, 4)},
              "projection": {"beta_resid": pinfo["beta_resid"],
                             "cash_resid": pinfo["cash_resid"],
                             "sigma_resid": pinfo["sigma_resid"],
                             "eps_sigma": pinfo["eps_sigma"],
                             "iters": pinfo["iters"],
                             "clipped": [support[i] for i in pinfo["clipped"]],
                             "infeasible": pinfo["infeasible"]},
              "tilt": {n: round(v, 8) for n, v in dw.items() if abs(v) > 1e-8},
              "organ_attribution": attribution,
              "intents_edits": edits,
              "marks_source": mark_src, "nav": round(nav, 2),
              "rllm": {"z": None, "g": None},
              "book_overlap_vs_chassis": round(overlap, 4),
              "parity_expost": {"gross_gap": None, "beta_gap_21d": None}})
        return new_intents

    return Strategy(
        name="orb1_tilt",
        description="ORB-1 post-decision tilt (PKT-TB-007): edits chassis "
                    "intents at the margin under the hard parity projection; "
                    "neutral => incumbent intents object unchanged",
        post_decision=post,
        params={"genome": genome.to_dict(), "genome_hash": genome.hash(),
                "nightly_dir": str(nightly_dir), "min_order": min_order,
                "masks": masks, "t_max": T_MAX},
    )
