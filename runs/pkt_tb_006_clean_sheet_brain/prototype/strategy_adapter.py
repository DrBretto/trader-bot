"""PKT-TB-006 — SYN-1 Strategy adapter for the three-line replay harness
(replay wiring deliverable #2).

make_syn1_strategy(genome, exec_weights_dir, nightly_dir, cache) returns a
src.utils.three_line_replay.strategies.Strategy whose post_decision:

  1. reads ctx (portfolio, NAV) — marks come from daily/D features with rows
     dated == D DROPPED (wiring adjudication #3: 11 window dirs carry a
     preliminary same-day row); if the guarded frame is empty (the quirk dirs
     are ENTIRELY same-day rows) it falls back to daily/D/prices.parquet rows
     dated < D — still strictly D-1-close information;
  2. loads the precomputed member opinions / solo books / exec inputs for
     ctx.inputs_date from store/nightly/<D>/;
  3. runs the deployed executive ensemble forward in PURE NUMPY (all
     exec_weights_dir/executive_seed*.pt seeds, outputs averaged) with live
     w_prev (portfolio positions x latest guarded closes / NAV) and an
     own-drawdown state threaded across days;
  4. applies the genome (EA live-gene semantics, mirroring ea.FitnessEngine
     .fold_utility): trust_prior + ledger trust-tilt (genome halflife/eps,
     D−h−1 lag) + conviction_temp + member_gate + feature_gate z-masking +
     event_weight_cap, then the rails — gross_target/cash_floor cap,
     max_symbol_weight, vol cap (trailing-vol-proxy sigma_hat, the executive's
     training convention), dd brake, abstain, no_trade_band;
  5. converts Δw to intents per wiring adjudication #4 (BUY and SELL only —
     NEVER REDUCE; SELL with explicit shares; intent price = latest guarded
     close; asset_class/sector/leverage_flag from config/universe.csv;
     min_order respected; full exits allowed below min_order);
  6. REPLACES the incumbent intents entirely and writes
     store/nightly/<D>/meta_decision.json + trade_intents.json.

Missing nightly artifacts for D (e.g. the 2026-05-11..05-22 snapshot gap)
=> emits NO intents (hold) and flags it in meta_decision.json.

Battery knobs (TOURNAMENT §4.3/§4.4):
  exec_mode    'learned' (default) | 'equal_trust' — the R08 executive bypass:
               tau = 1/M over ACTIVE members, deployment f fixed at 0.7, same
               genome rails (vol cap etc.); learned executive never loaded.
  sigma_source 'trailing21' (default; the trailing-vol-proxy columns the
               executive trained on — the landed/R01 behavior, now named) |
               'risknet' (E4 RiskNet+ heads) — selects the vol-cap sigma_hat
               column in risknet.parquet and the exec book-vol input key (R07).
Both are recorded in meta_decision.json per date and in the run manifest.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROTO = Path(__file__).resolve().parent
REPO = PROTO.parents[2]

import sys
for p in (str(PROTO), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import books as bk                      # noqa: E402
from ea import Genome, FEATURE_GATE_NAMES, FEATURE_GATE_Z_MAP, EVENT_MEMBER_IDX  # noqa: E402
from src.utils.three_line_replay.strategies import Strategy, StrategyContext  # noqa: E402

EPS_TAU = 0.05
N_MEMBERS = 3
LAG = bk.H + 1          # ledger / tilt lag: stats at D use u(D') with D' <= D-h-1
MIN_ORDER_DEFAULT = 250.0
FULL_EXIT_EPS = 1e-9

# R08 executive bypass (TOURNAMENT §4.3 meta-evaluator baseline): tau = 1/M over
# ACTIVE members, deployment f fixed, same genome rails — mirrors the
# e1_reads.E1WalkEngine.fold_daily_returns bypass walk.
EXEC_MODES = ("learned", "equal_trust")
EQUAL_TRUST_F = 0.7

# R07 sigma source (vol-cap sigma_hat + executive book-vol input). 'trailing21'
# = the trailing-vol proxy columns (the executive's training convention and the
# landed/R01 default behavior, now NAMED); 'risknet' = the E4 RiskNet+ heads.
# Maps mode -> (risknet.parquet sigma column, exec_inputs.json book-vol key).
SIGMA_SOURCES = {
    "trailing21": ("sigma_hat_proxy", "book_vol_hat_proxy"),
    "risknet": ("sigma_hat", "book_vol_hat_e4"),
}


# ------------------------------------------------------------------ exec weights
def load_exec_weights(exec_dir: Path) -> List[Dict[str, np.ndarray]]:
    """Load every executive_seed*.pt in exec_dir as plain numpy dicts."""
    import torch
    files = sorted(Path(exec_dir).glob("executive_seed*.pt"))
    if not files:
        raise FileNotFoundError(f"no executive_seed*.pt in {exec_dir}")
    out = []
    for f in files:
        sd = torch.load(f, weights_only=True)
        out.append({k: v.detach().numpy().astype(np.float64) for k, v in sd.items()})
    return out


def exec_trust_numpy(w: Dict[str, np.ndarray], x_trust: np.ndarray
                     ) -> tuple[np.ndarray, float]:
    """Pure-numpy mirror of executive.Executive.trust pre-softmax: x_trust [3,30]
    -> (raw logits s[3], model temperature T)."""
    e = np.tanh(x_trust @ w["phi.weight"].T + w["phi.bias"])      # [3,8]
    s = e @ w["v"] + w["b"][0] + w["b_m"]                          # [3]
    T = float(np.clip(np.exp(w["log_T"][0]), 0.1, 10.0))
    return s, T


def exec_sizing_numpy(w: Dict[str, np.ndarray], x_psi_pre_dd: np.ndarray,
                      dd: float) -> float:
    """Pure-numpy mirror of executive.Executive.sizing for ONE date.
    x_psi_pre_dd [35] with the drawdown slot (34) unset. Returns deployment f."""
    x_psi = x_psi_pre_dd.copy()
    x_psi[34] = dd
    hidden = np.tanh(x_psi @ w["psi1.weight"].T + w["psi1.bias"])  # [8]
    raw = float(hidden @ w["psi2.weight"][0] + w["psi2.bias"][0])
    return 1.0 / (1.0 + np.exp(-(float(w["psi_out_gain"][0]) * raw
                                 + float(w["psi_out_bias"][0])
                                 + float(w["f_sigmoid_bias"][0]))))


# ------------------------------------------------------------------ marks guard
def guarded_marks(features_df: pd.DataFrame, inputs_date: str,
                  cache=None) -> tuple[Dict[str, float], str]:
    """Latest close per symbol from daily/D inputs with rows dated == D dropped
    (adjudication #3). Falls back to daily/D/prices.parquet (< D rows) when the
    features frame is entirely same-day (the 11 quirk dirs).
    Returns (marks, source_tag)."""
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
        pr = pr[ds < inputs_date]
        df = pr
    df = df[["symbol", "date", "close"]].sort_values(["symbol", "date"])
    latest = df.groupby("symbol").tail(1)
    for _, row in latest.iterrows():
        if pd.notna(row["close"]):
            marks[str(row["symbol"])] = float(row["close"])
    return marks, src


# ------------------------------------------------------------------ tilt
def build_tilt_table(ledger_df: pd.DataFrame, u_std: float, genome: Genome,
                     members: List[str]) -> Dict[str, np.ndarray]:
    """date -> tilt[3]: lagged EWMA of eps-record-weighted counterfactual member
    utility (ea.FitnessEngine.fold_utility semantics; halflife/eps from genome)."""
    dates = np.asarray(sorted(ledger_df["date"].unique()))
    u = np.zeros((len(dates), N_MEMBERS))
    wr = np.zeros(len(dates))
    pos = {d: i for i, d in enumerate(dates)}
    for mi, m in enumerate(members):
        sub = ledger_df[ledger_df["member"] == m]
        for d, uu, ww in zip(sub["date"], sub["u_realized"], sub["w_rec_raw"]):
            u[pos[d], mi] = uu if np.isfinite(uu) else np.nan
            wr[pos[d]] = ww
    lam = 1.0 - 0.5 ** (1.0 / genome.trust_halflife_days)
    w_rec = np.clip(genome.record_weight_eps + wr, genome.record_weight_eps, 1.0)
    tilt = np.zeros((len(dates), N_MEMBERS))
    acc, wsum = np.zeros(N_MEMBERS), 0.0
    for i in range(len(dates)):
        tilt[i] = acc / wsum if wsum > 1e-9 else 0.0
        j = i - LAG
        if j >= 0 and np.all(np.isfinite(u[j])):
            acc = (1 - lam) * acc + lam * w_rec[j] * u[j]
            wsum = (1 - lam) * wsum + lam
    tilt = tilt / u_std
    return {str(d): tilt[i] for i, d in enumerate(dates)}


# ------------------------------------------------------------------ intents
def delta_to_intents(w_tgt: np.ndarray, w_prev: np.ndarray, nav: float,
                     symbols: List[str], marks: Dict[str, float],
                     held_shares: Dict[str, float], universe_rows: Dict[str, dict],
                     no_trade_band: float,
                     min_order: float = MIN_ORDER_DEFAULT) -> List[Dict[str, Any]]:
    """Δw -> BUY/SELL intent dicts (adjudication #4). NEVER emits REDUCE."""
    intents: List[Dict[str, Any]] = []
    for i, sym in enumerate(symbols):
        delta = float(w_tgt[i] - w_prev[i])
        if abs(delta) <= no_trade_band:
            continue
        price = marks.get(sym, 0.0)
        if not np.isfinite(price) or price <= 0:
            continue
        dollars = abs(delta) * nav
        uni = universe_rows.get(sym, {})
        base = {"symbol": sym, "price": round(price, 6),
                "asset_class": uni.get("asset_class", "equity"),
                "sector": uni.get("sector", "broad"),
                "leverage_flag": int(uni.get("leverage_flag", 0) or 0)}
        if delta > 0:
            if dollars < min_order:
                continue
            shares = int(dollars / price)
            if shares <= 0:
                continue
            intents.append({**base, "action": "BUY", "shares": shares,
                            "dollars": round(dollars, 2),
                            "reason": "SYN1_REBALANCE_BUY"})
        else:
            held = float(held_shares.get(sym, 0.0))
            if held <= 0:
                continue
            full_exit = w_tgt[i] <= FULL_EXIT_EPS
            if dollars < min_order and not full_exit:
                continue
            shares = held if full_exit else min(held, dollars / price)
            if shares <= 0:
                continue
            intents.append({**base, "action": "SELL", "shares": float(shares),
                            "dollars": round(shares * price, 2),
                            "reason": "SYN1_REBALANCE_EXIT" if full_exit
                                      else "SYN1_REBALANCE_TRIM"})
    return intents


# ------------------------------------------------------------------ strategy
def make_syn1_strategy(genome: Genome | dict | str | Path,
                       exec_weights_dir: Path | str,
                       nightly_dir: Path | str,
                       cache=None,
                       universe_csv: Path | str = REPO / "config" / "universe.csv",
                       min_order: float = MIN_ORDER_DEFAULT,
                       exec_mode: str = "learned",
                       sigma_source: str = "trailing21") -> Strategy:
    if exec_mode not in EXEC_MODES:
        raise ValueError(f"unknown exec_mode {exec_mode!r} (choices: {EXEC_MODES})")
    if sigma_source not in SIGMA_SOURCES:
        raise ValueError(f"unknown sigma_source {sigma_source!r} "
                         f"(choices: {tuple(SIGMA_SOURCES)})")
    sigma_col, bvh_key = SIGMA_SOURCES[sigma_source]
    if isinstance(genome, (str, Path)):
        genome = Genome.from_json(Path(genome))
    elif isinstance(genome, dict):
        genome = Genome.from_dict(genome)
    nightly_dir = Path(nightly_dir)
    # equal_trust bypasses the learned executive entirely — no weights needed
    seeds_w = ([] if exec_mode == "equal_trust"
               else load_exec_weights(Path(exec_weights_dir)))

    uni = pd.read_csv(universe_csv)
    symbols = [str(s).strip() for s in uni["symbol"]]
    universe_rows = {r["symbol"]: dict(r) for _, r in uni.iterrows()}
    S = len(symbols)

    ledger_df = pd.read_parquet(nightly_dir / "ledger.parquet")
    u_std = float(json.loads((nightly_dir / "ledger_meta.json").read_text())["u_std"])
    tilt_table = build_tilt_table(ledger_df, u_std, genome, list(bk.MEMBERS))

    zmask = np.ones(24)
    for gname, on in zip(FEATURE_GATE_NAMES, genome.feature_gate):
        if not on:
            zmask[FEATURE_GATE_Z_MAP[gname]] = 0.0
    mg = np.asarray(genome.member_gate, dtype=np.float64)
    if exec_mode == "equal_trust" and mg.sum() <= 0:
        raise ValueError("equal_trust bypass with all members gated off")
    capf = np.ones(N_MEMBERS)
    capf[EVENT_MEMBER_IDX] = genome.event_weight_cap

    state = {"peak_log_nav": None}

    def _write(out_dir: Path, name: str, payload) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / name).write_text(json.dumps(payload, indent=1))

    def post(ctx: StrategyContext, incumbent_intents: List[Dict[str, Any]]
             ) -> List[Dict[str, Any]]:
        D = ctx.inputs_date
        day_dir = nightly_dir / D
        meta: Dict[str, Any] = {"date": D, "strategy": "syn1",
                                "exec_mode": exec_mode,
                                "sigma_source": sigma_source, "flags": []}

        # ---- marks / NAV / w_prev (guarded; adjudication #3) ------------------
        marks, mark_src = guarded_marks(ctx.features_df, D, cache=cache)
        meta["marks_source"] = mark_src
        positions = list(ctx.portfolio.positions)
        held_shares = {p.symbol: float(p.shares) for p in positions}
        nav = float(ctx.portfolio.cash)
        missing_marks = []
        for p in positions:
            mk = marks.get(p.symbol)
            if mk is None:
                mk = p.last_close or p.entry_price
                missing_marks.append(p.symbol)
            nav += p.shares * mk
        if missing_marks:
            meta["flags"].append({"positions_marked_at_last_close": missing_marks})
        if nav <= 0:
            nav = float(ctx.decisions.get("_portfolio_value_passed", 0.0) or 0.0)
            meta["flags"].append("nav_from_portfolio_value_passed")
        meta["nav"] = round(nav, 2)
        w_prev = np.zeros(S)
        for p in positions:
            if p.symbol in symbols and nav > 0:
                mk = marks.get(p.symbol, p.last_close or p.entry_price)
                w_prev[symbols.index(p.symbol)] = p.shares * mk / nav

        # ---- drawdown state threaded across days ------------------------------
        log_nav = float(np.log(max(nav, 1e-9)))
        if state["peak_log_nav"] is None:
            state["peak_log_nav"] = log_nav
        state["peak_log_nav"] = max(state["peak_log_nav"], log_nav)
        dd = max(0.0, state["peak_log_nav"] - log_nav)
        meta["own_drawdown"] = round(dd, 6)

        # ---- nightly artifacts -------------------------------------------------
        needed = ["expert_opinions.parquet", "solo_books.parquet",
                  "risknet.parquet", "exec_inputs.json"]
        if not all((day_dir / f).exists() for f in needed):
            meta["flags"].append("missing_nightly_artifacts_hold")
            meta["intents"] = []
            _write(day_dir, "meta_decision.json", meta)
            _write(day_dir, "trade_intents.json",
                   {"date": D, "intents": [], "hold": True})
            return []                                      # HOLD: replace with nothing

        ops = pd.read_parquet(day_dir / "expert_opinions.parquet")
        sb = pd.read_parquet(day_dir / "solo_books.parquet")
        rk = pd.read_parquet(day_dir / "risknet.parquet")
        xin = json.loads((day_dir / "exec_inputs.json").read_text())
        mu = np.stack([ops[ops["member"] == m].set_index("symbol")
                       .loc[symbols, "mu"].to_numpy() for m in bk.MEMBERS])
        books = np.stack([sb[sb["member"] == m].set_index("symbol")
                          .loc[symbols, "w"].to_numpy() for m in bk.MEMBERS])
        sigma_hat = rk.set_index("symbol").loc[symbols, sigma_col].to_numpy()
        r = np.asarray(xin["r"], dtype=np.float64)             # [3,4]
        z = np.asarray(xin["z"], dtype=np.float64) * zmask     # [24] feature-gated
        c = np.asarray(xin["c"], dtype=np.float64)
        agree = np.asarray(xin["agree"], dtype=np.float64)
        g = np.asarray(xin["g"], dtype=np.float64)
        bvh = float(xin[bvh_key])
        tilt = tilt_table.get(D)
        if tilt is None:
            tilt = np.zeros(N_MEMBERS)
            meta["flags"].append("no_tilt_for_date")

        if exec_mode == "equal_trust":
            # ---- R08 §4.3 bypass: tau = 1/M over ACTIVE members, fixed f -------
            # (replaces the WHOLE learned trust/sizing stack; genome rails below
            # unchanged — mirrors e1_reads.fold_daily_returns bypass walk)
            tau_bar = mg / mg.sum()
            coef_bar = tau_bar.copy()
            f_bar = EQUAL_TRUST_F
        else:
            # ---- executive forward (numpy, seed-averaged) + genome modulation --
            zz = np.tile(z, (N_MEMBERS, 1))
            x_trust = np.concatenate([r, c[:, None], agree[:, None], zz], axis=1)  # [3,30]
            coef_seeds, f_seeds, tau_seeds = [], [], []
            for w in seeds_w:
                # trust pass first (psi needs tau-dependent inputs)
                s, T_model = exec_trust_numpy(w, x_trust)
                T = T_model * genome.conviction_temp
                logits = s + np.asarray(genome.trust_prior) + tilt
                ex_l = np.exp((logits - logits.max()) / T)
                tau = ex_l / ex_l.sum()
                tau = (1 - EPS_TAU) * tau + EPS_TAU / N_MEMBERS
                tau = tau * mg
                tau = tau / tau.sum() if tau.sum() > 1e-12 else np.zeros(N_MEMBERS)
                coef = tau * capf
                coef = coef / coef.sum() if coef.sum() > 1e-12 else np.zeros(N_MEMBERS)
                mu_blend = coef @ mu
                am = np.abs(mu_blend)
                ent = float(-(tau * np.log(tau + 1e-12)).sum())
                x_psi = np.zeros(35)
                x_psi[0:24] = z
                x_psi[24:27] = g
                x_psi[27:30] = [am.mean(), am.std(), am.max()]
                x_psi[30] = ent
                x_psi[31:33] = [r[:, 0].mean(), r[:, 1].mean()]
                x_psi[33] = bvh
                coef_seeds.append(coef)
                tau_seeds.append(tau)
                f_seeds.append(exec_sizing_numpy(w, x_psi, dd))
            coef_bar = np.mean(coef_seeds, axis=0)
            coef_bar = coef_bar / coef_bar.sum() if coef_bar.sum() > 1e-12 else coef_bar
            tau_bar = np.mean(tau_seeds, axis=0)
            f_bar = float(np.mean(f_seeds))
        w_unit = coef_bar @ books
        mu_blend = coef_bar @ mu
        am = np.abs(mu_blend)

        # ---- genome rails (ea.fold_utility order) ------------------------------
        rails = []
        w = f_bar * w_unit
        gross = float(w.sum())
        gmax = min(genome.gross_target, 1.0 - genome.cash_floor)
        if gross > gmax > 0:
            w = w * (gmax / gross)
            rails.append("gross_cap")
        if (w > genome.max_symbol_weight).any():
            w = np.minimum(w, genome.max_symbol_weight)
            rails.append("max_symbol_weight")
        vol = float(w @ sigma_hat)
        if vol > genome.vol_target_ann > 0:
            w = w * (genome.vol_target_ann / vol)
            rails.append("vol_cap")
        if dd > genome.dd_brake_threshold:
            brake = 1.0 - genome.dd_brake_strength * min(
                1.0, (dd - genome.dd_brake_threshold)
                / max(genome.dd_brake_threshold, 1e-6))
            w = w * brake
            rails.append("dd_brake")
        abstained = bool(am.mean() < genome.abstain_threshold)
        if abstained:
            rails.append("abstain_hold")
            intents: List[Dict[str, Any]] = []
            w_tgt = w_prev.copy()
        else:
            w_tgt = w
            intents = delta_to_intents(w_tgt, w_prev, nav, symbols, marks,
                                       held_shares, universe_rows,
                                       genome.no_trade_band, min_order)

        meta.update({
            "trust": {m: float(tau_bar[i]) for i, m in enumerate(bk.MEMBERS)},
            "blend_coef": {m: float(coef_bar[i]) for i, m in enumerate(bk.MEMBERS)},
            "tilt": {m: float(tilt[i]) for i, m in enumerate(bk.MEMBERS)},
            "deployment_fraction": f_bar,
            "n_exec_seeds": len(seeds_w),
            "gross_target_applied": round(float(w_tgt.sum()), 6),
            "est_book_vol": round(float(w_tgt @ sigma_hat), 6),
            "abstained": abstained, "rails": rails,
            "mu_blend_abs_mean": float(am.mean()),
            "genome_hash": _genome_hash(genome),
            "n_intents": len(intents),
            "intents": [{k: v for k, v in it.items()} for it in intents],
            "replaced_incumbent_intents": len(incumbent_intents),
        })
        _write(day_dir, "meta_decision.json", meta)
        _write(day_dir, "trade_intents.json", {"date": D, "intents": intents})
        return intents                                     # REPLACE incumbent

    return Strategy(
        name="syn1_brain",
        description="SYN-1 clean-sheet brain (PKT-TB-006): precomputed members + "
                    "numpy executive + genome rails; replaces incumbent intents",
        post_decision=post,
        params={"genome": genome.to_dict(), "exec_weights_dir": str(exec_weights_dir),
                "nightly_dir": str(nightly_dir), "min_order": min_order,
                "exec_mode": exec_mode, "sigma_source": sigma_source},
    )


def _genome_hash(genome: Genome) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(genome.to_dict(), sort_keys=True)
                          .encode()).hexdigest()[:12]
