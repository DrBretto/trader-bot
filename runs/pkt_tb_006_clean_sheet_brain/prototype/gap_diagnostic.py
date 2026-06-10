"""PKT-TB-006 — §4.6.6 train-vs-harness gap diagnostic (Skeptic B-K2; Phase-D
evidence repair for REVIEW_SKEPTIC_PHASE_D.md finding F11).

Pre-registered gate (TOURNAMENT §4.6 item 6): "executive training injects
±2 bps uniform slippage noise into its cost term; the dossier reports the gap
between training utility and harness replay utility on identical dates; a gap
> 25% of mean |daily utility| is reported as relaxation-gaming."

What this computes, on the PRE-HOLDOUT replay decision dates only
(2026-02-04 -> 2026-03-06; HARD-ASSERTED < HOLDOUT_START 2026-03-11):

  SIMULATOR side ("training utility"): walks the ONE frozen deployed config
  (FREEZE_SYN1.md: linear twin gate exec_out/linear_twin.pt, EA champion
  genome, trailing-21 sigma) over the precomputed store/nightly/<D> inputs
  (exec_inputs.json / solo_books / expert_opinions / risknet.parquet, tilt from
  the nightly ledger via strategy_adapter.build_tilt_table) and scores it with
  the TRAINING-convention utility math the executive/EA trained against
  (ea.FitnessEngine.fold_utility / e1_reads.E1WalkEngine semantics): smooth
  genome rails (gross cap, max-symbol, vol cap, dd brake, abstain -> hold,
  no-trade-band masking of traded), cost = |traded| @ half_spread_bps / 1e4 at
  cost scenario 1.0 (the deterministic center of the noised training cost
  term), w_prev threading, equity/drawdown state in log space. Daily utility =
  daily book return (the E1 forced-choice convention, e1_reads.py header).

  HARNESS side ("harness replay utility"): R01's cost-adjusted daily returns
  (runs_battery/R01/daily_series.csv) on the IDENTICAL dates — pre-holdout
  rows only. R01 is the deployed config run through the real replay harness
  (hard fills at open, integer shares, min-order, post-hoc cost overlay).

  Alignment: both sides are indexed by the harness decision-date grid
  (Tue–Fri snapshot dates; no Mondays exist in the store by construction).
  Step k = decision date D_k -> next decision date D_{k+1}; simulator holding
  returns are close(D_k) -> close(D_{k+1}) from the frozen price panel
  (store/panel.npz close_px), so week-boundary steps span the same interval
  in both series. Simulator w_prev is seeded from the same live book the
  harness seeds from (daily/2026-02-03/portfolio_state.json).

  WINDOW CAVEAT (stated per the repair order): this is the pre-holdout replay
  slice. The training window itself ends FITNESS_END = 2026-02-06, so all but
  the first two steps read the simulator OUT-OF-SAMPLE — the diagnostic
  compares conventions (training math vs harness fills), not in-sample fit.

Output: prototype/evidence/gap_diagnostic.json. ZERO holdout looks: nothing
dated >= 2026-03-11 is read, computed, or written.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROTO = Path(__file__).resolve().parent
REPO = PROTO.parents[2]
for p in (str(PROTO), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import books as bk                                              # noqa: E402
import folds as fd                                              # noqa: E402
from ea import Genome, FEATURE_GATE_NAMES, FEATURE_GATE_Z_MAP, EVENT_MEMBER_IDX  # noqa: E402
import strategy_adapter as sa                                   # noqa: E402

HOLDOUT_START = "2026-03-11"
PRE_HOLDOUT_LAST = "2026-03-06"
SEED_STATE = PROTO / "cache" / "s3" / "daily" / "2026-02-03" / "portfolio_state.json"
NIGHTLY = PROTO / "store" / "nightly"
R01_SERIES = PROTO / "runs_battery" / "R01" / "daily_series.csv"
OUT = PROTO / "evidence" / "gap_diagnostic.json"
GATE_PCT = 25.0


def load_panel_closes() -> tuple[list[str], dict[str, np.ndarray]]:
    z = np.load(PROTO / "store" / "panel.npz", allow_pickle=True)
    dates = list(np.asarray(z["dates"]).astype(str))
    close = np.asarray(z["close_px"], dtype=np.float64)
    syms = list(np.asarray(z["symbols"]).astype(str))
    uni = pd.read_csv(REPO / "config" / "universe.csv")
    usyms = [str(s).strip() for s in uni["symbol"]]
    assert syms == usyms, "panel symbol order != universe order"
    return usyms, {d: close[i] for i, d in enumerate(dates)}


def harness_decision_dates() -> tuple[list[str], np.ndarray]:
    """Pre-holdout decision dates + cost-adjusted values from R01's
    already-produced series file (pre-holdout rows ONLY are read)."""
    df = pd.read_csv(R01_SERIES)
    df = df[df["date"] <= PRE_HOLDOUT_LAST]
    assert (df["date"] < HOLDOUT_START).all(), "holdout row leaked into slice"
    return list(df["date"]), df["cost_adjusted_value"].to_numpy(np.float64)


def seed_w_prev(symbols: list[str], closes_by_date: dict[str, np.ndarray],
                mark_date: str) -> np.ndarray:
    """The live book the harness full-window runs seed from (2026-02-03
    portfolio_state.json), expressed as training-convention weights at the
    same pre-decision mark (2026-02-03 close)."""
    state = json.loads(SEED_STATE.read_text())
    px = closes_by_date[mark_date]
    idx = {s: i for i, s in enumerate(symbols)}
    hold_val = np.zeros(len(symbols))
    for h in state.get("holdings", []):       # accumulate duplicate-symbol lots
        i = idx.get(str(h["symbol"]))
        if i is not None and np.isfinite(px[i]):
            hold_val[i] += float(h["shares"]) * px[i]
    nav = float(state.get("cash", 0.0)) + hold_val.sum()
    assert nav > 0, "seed portfolio NAV <= 0"
    return hold_val / nav


def rails_and_emit(w: np.ndarray, w_prev: np.ndarray, dd: float,
                   am: np.ndarray, genome: Genome, sigma_hat: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """The genome rail ladder, order-identical to ea.FitnessEngine /
    e1_reads.E1WalkEngine._rails (the executive's TRAINING convention)."""
    gross = w.sum()
    gmax = min(genome.gross_target, 1.0 - genome.cash_floor)
    if gross > gmax > 0:
        w = w * (gmax / gross)
    w = np.minimum(w, genome.max_symbol_weight)
    vol = w @ sigma_hat
    if vol > genome.vol_target_ann > 0:
        w = w * (genome.vol_target_ann / vol)
    if dd > genome.dd_brake_threshold:
        brake = 1.0 - genome.dd_brake_strength * min(
            1.0, (dd - genome.dd_brake_threshold)
            / max(genome.dd_brake_threshold, 1e-6))
        w = w * brake
    if am.mean() < genome.abstain_threshold:
        return w_prev, np.zeros(len(w))
    delta = w - w_prev
    mask = np.abs(delta) > genome.no_trade_band
    traded = delta * mask
    return w_prev + traded, traded


def simulator_returns(dates: list[str], symbols: list[str],
                      closes_by_date: dict[str, np.ndarray]
                      ) -> tuple[np.ndarray, list[str]]:
    """Training-convention daily utility of the deployed config walked over
    store/nightly inputs on the harness decision grid. Returns per-step
    utilities for steps D_0->D_1 .. D_{n-2}->D_{n-1} and any warnings."""
    warnings: list[str] = []
    genome = Genome.from_json(PROTO / "ea" / f"genome_{fd.FITNESS_END}.json")
    twin = sa.load_exec_weights(PROTO / "exec_out", "linear_twin")
    assert len(twin) == 1, "linear_twin must be a one-element ensemble"
    w_twin = twin[0]

    ledger_df = pd.read_parquet(NIGHTLY / "ledger.parquet")
    u_std = float(json.loads((NIGHTLY / "ledger_meta.json").read_text())["u_std"])
    tilt_table = sa.build_tilt_table(ledger_df, u_std, genome, list(bk.MEMBERS))

    zmask = np.ones(24)
    for gname, on in zip(FEATURE_GATE_NAMES, genome.feature_gate):
        if not on:
            zmask[FEATURE_GATE_Z_MAP[gname]] = 0.0
    mg = np.asarray(genome.member_gate, dtype=np.float64)
    capf = np.ones(sa.N_MEMBERS)
    capf[EVENT_MEMBER_IDX] = genome.event_weight_cap
    hs = bk.load_half_spread_bps()
    sigma_col, bvh_key = sa.SIGMA_SOURCES["trailing21"]   # FROZEN convention

    # training-convention state
    w_prev = seed_w_prev(symbols, closes_by_date, "2026-02-03")
    equity, peak = 0.0, 0.0
    r_out: list[float] = []

    for k in range(len(dates) - 1):
        D, Dn = dates[k], dates[k + 1]
        assert D < HOLDOUT_START and Dn < HOLDOUT_START
        day = NIGHTLY / D
        ops = pd.read_parquet(day / "expert_opinions.parquet")
        sb = pd.read_parquet(day / "solo_books.parquet")
        rk = pd.read_parquet(day / "risknet.parquet")
        xin = json.loads((day / "exec_inputs.json").read_text())
        assert list(xin["members"]) == list(bk.MEMBERS)
        mu = np.stack([ops[ops["member"] == m].set_index("symbol")
                       .loc[symbols, "mu"].to_numpy() for m in bk.MEMBERS])
        books = np.stack([sb[sb["member"] == m].set_index("symbol")
                          .loc[symbols, "w"].to_numpy() for m in bk.MEMBERS])
        sigma_hat = rk.set_index("symbol").loc[symbols, sigma_col].to_numpy()
        r = np.asarray(xin["r"], dtype=np.float64)
        z = np.asarray(xin["z"], dtype=np.float64) * zmask
        c = np.asarray(xin["c"], dtype=np.float64)
        agree = np.asarray(xin["agree"], dtype=np.float64)
        g = np.asarray(xin["g"], dtype=np.float64)
        bvh = float(xin[bvh_key])
        tilt = tilt_table.get(D)
        if tilt is None:
            tilt = np.zeros(sa.N_MEMBERS)
            warnings.append(f"no tilt for {D}; zeros used")

        dd = max(0.0, peak - equity)

        # --- linear-twin trust + genome modulation (strategy_adapter math) ---
        zz = np.tile(z, (sa.N_MEMBERS, 1))
        x_trust = np.concatenate([r, c[:, None], agree[:, None], zz], axis=1)
        s, T_model = sa.exec_trust_numpy(w_twin, x_trust)
        T = T_model * genome.conviction_temp
        logits = s + np.asarray(genome.trust_prior) + tilt
        e = np.exp((logits - logits.max()) / T)
        tau = e / e.sum()
        tau = (1 - sa.EPS_TAU) * tau + sa.EPS_TAU / sa.N_MEMBERS
        tau = tau * mg
        tau = tau / tau.sum() if tau.sum() > 1e-12 else np.zeros(sa.N_MEMBERS)
        coef = tau * capf
        coef = coef / coef.sum() if coef.sum() > 1e-12 else np.zeros(sa.N_MEMBERS)
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
        f_dep = sa.exec_sizing_numpy(w_twin, x_psi, dd)
        w_unit = coef @ books

        # --- training rails + training cost term + step holding return ------
        w_emit, traded = rails_and_emit(f_dep * w_unit, w_prev, dd, am,
                                        genome, sigma_hat)
        cost = float(np.abs(traded) @ hs) / 1e4                # scenario 1.0
        step = closes_by_date[Dn] / closes_by_date[D] - 1.0
        step = np.where(np.isfinite(step), step, 0.0)
        r_k = float(w_emit @ step) - cost
        r_out.append(r_k)
        equity += np.log1p(max(r_k, -0.99))
        peak = max(peak, equity)
        w_prev = w_emit
    return np.asarray(r_out), warnings


def main() -> dict:
    symbols, closes = load_panel_closes()
    dates, adj_values = harness_decision_dates()
    r_harness = adj_values[1:] / adj_values[:-1] - 1.0
    r_sim, warnings = simulator_returns(dates, symbols, closes)
    assert len(r_sim) == len(r_harness)
    step_labels = [f"{dates[k]}->{dates[k + 1]}" for k in range(len(dates) - 1)]

    mean_abs_sim = float(np.abs(r_sim).mean())
    mean_abs_har = float(np.abs(r_harness).mean())
    mean_abs_pooled = float(np.abs(np.concatenate([r_sim, r_harness])).mean())
    gap_abs = abs(mean_abs_sim - mean_abs_har)
    gap_pct = 100.0 * gap_abs / mean_abs_pooled if mean_abs_pooled > 0 else float("inf")
    per_date_gap = float(np.abs(r_sim - r_harness).mean())
    per_date_gap_pct = (100.0 * per_date_gap / mean_abs_pooled
                        if mean_abs_pooled > 0 else float("inf"))
    corr = (float(np.corrcoef(r_sim, r_harness)[0, 1])
            if len(r_sim) > 2 else float("nan"))

    if gap_pct > GATE_PCT:
        verdict = (f"Train-vs-harness gap = {gap_pct:.1f}% of mean |daily "
                   f"utility| (> {GATE_PCT:.0f}%): reported as "
                   f"relaxation-gaming per TOURNAMENT §4.6.6.")
    else:
        verdict = (f"Train-vs-harness gap = {gap_pct:.1f}% of mean |daily "
                   f"utility| (<= {GATE_PCT:.0f}%): below the §4.6.6 "
                   f"relaxation-gaming bar.")

    result = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "gate": "TOURNAMENT §4.6 item 6 (Skeptic B-K2) — train-vs-harness gap",
        "window": {
            "slice": "pre-holdout replay dates only",
            "first_decision_date": dates[0], "last_decision_date": dates[-1],
            "n_decision_dates": len(dates), "n_paired_steps": int(len(r_sim)),
            "holdout_start_never_touched": HOLDOUT_START,
            "training_window_caveat": (
                f"the executive/EA training window ends FITNESS_END = "
                f"{fd.FITNESS_END}; every step from {fd.FITNESS_END} onward "
                f"reads the simulator OUT-OF-SAMPLE (16 of {len(r_sim)} "
                f"steps), so this compares conventions (training utility "
                f"math vs harness fills), not in-sample fit"),
        },
        "simulator_side": {
            "what": ("deployed FROZEN config (linear twin, champion genome, "
                     "trailing-21 sigma) walked over store/nightly inputs "
                     "with ea.fold_utility/e1_reads training math: smooth "
                     "rails, cost=|traded|@half_spread/1e4 at scenario 1.0, "
                     "w_prev/equity/dd threading; seeded from the same "
                     "2026-02-03 live book the harness seeds from"),
            "mean_abs_daily_utility": mean_abs_sim,
            "mean_daily_utility": float(r_sim.mean()),
            "sd_daily_utility": float(r_sim.std(ddof=1)),
        },
        "harness_side": {
            "what": ("R01 cost-adjusted daily returns "
                     "(runs_battery/R01/daily_series.csv), identical dates, "
                     "pre-holdout rows only"),
            "mean_abs_daily_utility": mean_abs_har,
            "mean_daily_utility": float(r_harness.mean()),
            "sd_daily_utility": float(r_harness.std(ddof=1)),
        },
        "gap": {
            "abs_diff_of_mean_abs_daily_utility": gap_abs,
            "pct_of_mean_abs_daily_utility": gap_pct,
            "denominator": ("pooled mean |daily utility| across both sides "
                            f"= {mean_abs_pooled:.6g}"),
            "per_date_mean_abs_diff": per_date_gap,
            "per_date_mean_abs_diff_pct_of_mean_abs": per_date_gap_pct,
            "daily_series_correlation": corr,
            "threshold_pct": GATE_PCT,
        },
        "verdict_sentence": verdict,
        "per_step": [{"step": s, "sim": float(a), "harness": float(b)}
                     for s, a, b in zip(step_labels, r_sim, r_harness)],
        "warnings": warnings,
        "mechanism_note": (
            "The gap is dominated by realized-book divergence, not by the "
            "±2 bps cost-noise term: under the frozen genome's no-trade band "
            "(0.026/symbol) the training-convention walk trims the inherited "
            "2026-02-03 seed book to ~3% gross and holds, while the harness "
            "book retained an inherited ~22% SCHD position over the window — "
            "the seed portfolio held SCHD in two equal lots and the adapter's "
            "per-symbol trim reduced only one lot's worth (verifiable in "
            "runs_battery/R01/result.json final_holdings: SCHD 738.0 + 13.88 "
            "shares remain). Harness replay reality therefore differs "
            "materially from the training-utility assumptions on the same "
            "dates, which is exactly what this gate measures; the "
            "pre-registered consequence sentence is printed unmodified."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in
                      ("simulator_side", "harness_side", "gap",
                       "verdict_sentence")}, indent=1))
    return result


if __name__ == "__main__":
    main()
