"""PKT-TB-007 — organ acceptance gates (TOURNAMENT_007 §4.4.1 verbatim bars,
BUILD_SPEC_007 §5.1 order) + M4-A/B decision (§4.4.8) + per-name OOF stats.

Every gate prints its power pair (P(pass|null) / P(pass|design-effect), the
pre-registered numbers) and appends to ledgers/gates.json and
validation_looks_007.jsonl. Gate-failers ship as battery challengers, never
in the verdict tilt (roster mapping mechanical, §4.4.1).

Outputs: gates/acceptance_007.json, gates/per_name_oof_stats.json,
ledgers/gates.json (append), stdout report.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as ss

PROTO = Path(__file__).resolve().parent
REPO = PROTO.parents[2]
TB6 = REPO / "runs" / "pkt_tb_006_clean_sheet_brain" / "prototype"
sys.path.insert(0, str(PROTO))

import folds as F                      # noqa: E402
from tilt_adapter import PRODUCTION_MASKS, TILT_CORE, TILT_COND  # noqa: E402

OOF_DIR = PROTO / "oof"
RUN_ROOT = PROTO.parent
MASTER_SEED = 4242

# pre-registered power pairs (TOURNAMENT §4.4.1, printed verbatim)
POWER_PAIRS = {
    "M1": {"p_pass_null": "2.3% (x~50% for the delta conjunct)",
           "p_pass_effect": "~99.9% at receipt 0.105",
           "note": "near-formality; LABELED as such — does not count as evidence"},
    "M2": {"p_pass_null": "5.0%",
           "p_pass_effect": "62% at IC 0.049 (no decay); 29% at half-decay 0.025",
           "note": "status OPEN (FM6): a fail at 29% power reads 'not "
                   "measurable yet', not 'rotation is dead'"},
    "M3": {"p_pass_null": "5%",
           "p_pass_effect": "moderate (unmeasured increment — first measurement)"},
    "M4": {"p_pass_null": "~5% x ~5%",
           "p_pass_effect": "~50% if true AUC 0.55; >90% if 0.58"},
    "M5": {"p_pass_null": "2.1%",
           "p_pass_effect": "low at n~15-25 — expected DISABLED"},
    "M6": {"p_pass_null": "~1%", "p_pass_effect": "~89% at true IC 0.03",
           "note": "forecast-altitude verdict line, never a roster gate"},
}


def ledger(entry: dict) -> None:
    entry = {"ts": dt.datetime.now().isoformat(timespec="seconds"),
             "phase": "C-wave2a", **entry}
    with (PROTO / "validation_looks_007.jsonl").open("a") as f:
        f.write(json.dumps(entry, default=float) + "\n")
    gl = PROTO / "ledgers" / "gates.json"
    cur = json.loads(gl.read_text()) if gl.exists() else []
    cur.append(entry)
    gl.write_text(json.dumps(cur, indent=1, default=float))


def load_oof(member: str, folds=range(1, 7)) -> dict:
    out = {}
    for f in folds:
        z = np.load(OOF_DIR / f"fold_{f}_{member}.npz", allow_pickle=True)
        out[f] = {k: z[k] for k in z.files if k != "manifest"}
        out[f]["manifest"] = json.loads(str(z["manifest"]))
        assert out[f]["manifest"].get("walk_forward", False)
    return out


def fold_day_index(dates, oof):
    """map each fold's OOF rows to panel row indices."""
    didx = {d: i for i, d in enumerate(dates)}
    return {f: np.array([didx[str(d)] for d in
                         np.asarray(o["dates"]).astype(str)])
            for f, o in oof.items()}


def windowed_ics(mu_by_fold, idx_by_fold, y, sm, step):
    from scipy.stats import spearmanr
    ics, fold_of = [], []
    for f in sorted(mu_by_fold):
        mu = mu_by_fold[f]
        ii = idx_by_fold[f]
        for j in range(0, len(ii), step):
            i = ii[j]
            m = sm[i] & np.isfinite(y[i]) & np.isfinite(mu[j])
            if m.sum() >= 8:
                ics.append(spearmanr(mu[j][m], y[i][m]).statistic)
                fold_of.append(f)
    return np.asarray(ics, dtype=float), np.asarray(fold_of)


def per_name_stats(mu_by_fold, idx_by_fold, y, sm, symbols, folds=range(1, 7)):
    """per-symbol per-fold time-series Spearman IC + t-based 90% CI."""
    from scipy.stats import spearmanr
    out = {}
    for j, s in enumerate(symbols):
        fold_ics = []
        for f in folds:
            if f not in mu_by_fold:
                continue
            mu = mu_by_fold[f][:, j]
            ii = idx_by_fold[f]
            ys = y[ii, j]
            m = np.isfinite(mu) & np.isfinite(ys) & sm[ii, j]
            if m.sum() >= 30:
                fold_ics.append(float(spearmanr(mu[m], ys[m]).statistic))
            else:
                fold_ics.append(np.nan)
        fi = np.asarray(fold_ics, dtype=float)
        ok = np.isfinite(fi)
        n = int(ok.sum())
        if n >= 3:
            mean = float(fi[ok].mean())
            se = float(fi[ok].std(ddof=1) / np.sqrt(n))
            tcrit = ss.t.ppf(0.95, n - 1)
            ci = [mean - tcrit * se, mean + tcrit * se]
        else:
            mean, ci = np.nan, [np.nan, np.nan]
        out[s] = {"fold_ics": [None if not np.isfinite(v) else round(v, 4)
                               for v in fi],
                  "pooled_mean": None if not np.isfinite(mean) else round(mean, 4),
                  "ci90": [None if not np.isfinite(c) else round(c, 4)
                           for c in ci],
                  "n_folds": n}
    return out


def mask_rule(stats_by_sym, n_folds_total):
    """FROZEN rule text (BUILD_SPEC §2.3): 1.0 if 90% CI fully > 0; 0.5 if
    >= all-but-one folds positive but CI spans 0; else 0; forced 0 outside
    CORE+COND."""
    support = set(TILT_CORE) | set(TILT_COND)
    out = {}
    for s, st in stats_by_sym.items():
        if s not in support:
            continue
        lo = st["ci90"][0]
        fis = [v for v in st["fold_ics"] if v is not None]
        n_pos = sum(1 for v in fis if v > 0)
        if lo is not None and lo > 0:
            out[s] = 1.0
        elif len(fis) >= 3 and n_pos >= len(fis) - 1 and n_pos >= 1:
            out[s] = 0.5
        else:
            out[s] = 0.0
    return out


def hac_cov(X1, u, lags=10):
    n, k = X1.shape
    Xu = X1 * u[:, None]
    S = Xu.T @ Xu / n
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        G = Xu[lag:].T @ Xu[:-lag] / n
        S += w * (G + G.T)
    XtX_inv = np.linalg.inv(X1.T @ X1 / n)
    return XtX_inv @ S @ XtX_inv / n


def cluster_boot_auc(y, p, day_ids, b=500, seed=MASTER_SEED, p2=None):
    """day-cluster bootstrap of AUC (and paired delta if p2 given)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(seed)
    days = np.unique(day_ids)
    rows_of = {d: np.nonzero(day_ids == d)[0] for d in days}
    aucs, deltas = [], []
    for _ in range(b):
        samp = rng.choice(days, size=len(days), replace=True)
        rows = np.concatenate([rows_of[d] for d in samp])
        yy = y[rows]
        if len(np.unique(yy)) < 2:
            continue
        a1 = roc_auc_score(yy, p[rows])
        aucs.append(a1)
        if p2 is not None:
            deltas.append(a1 - roc_auc_score(yy, p2[rows]))
    return (float(np.std(aucs, ddof=1)),
            float(np.std(deltas, ddof=1)) if p2 is not None else None)


def main():
    dates = None
    z6 = np.load(TB6 / "store" / "panel.npz", allow_pickle=False)
    p6 = {k: z6[k] for k in ["dates", "symbols", "y5_raw", "symbol_mask"]}
    z7 = np.load(PROTO / "store" / "panel_007.npz", allow_pickle=False)
    p7 = {k: z7[k] for k in z7.files}
    dates = np.asarray(p6["dates"]).astype(str)
    symbols = [str(s) for s in p6["symbols"]]
    sm = p6["symbol_mask"].astype(bool)
    y5 = p6["y5_raw"].astype(np.float64)
    exploit = {r["symbol"]: r for r in json.loads(
        (RUN_ROOT / "universe_exploitability.json").read_text())["per_symbol"]}
    mz = np.load(PROTO / "store" / "member_zero_scores.npz")["score"] \
        .astype(np.float64)
    results = {}
    pvals = {}

    # ===================== M1 =============================================
    oof1 = load_oof("m1_cast")
    idx1 = fold_day_index(dates, oof1)
    mu1 = {f: oof1[f]["mu"].astype(np.float64) for f in oof1}
    ics1, _ = windowed_ics(mu1, idx1, y5, sm, 5)
    m1_ic = float(np.nanmean(ics1))
    m1_se = float(np.nanstd(ics1, ddof=1) / np.sqrt(len(ics1)))
    # member zero weekly IC on identical days
    mz_by_fold = {f: mz[idx1[f]] for f in idx1}
    ics0, _ = windowed_ics(mz_by_fold, idx1, y5, sm, 5)
    mz_ic = float(np.nanmean(ics0))
    delta_ic = m1_ic - mz_ic
    pn1 = per_name_stats(mu1, idx1, y5, sm, symbols)
    core1 = [s for s, v in PRODUCTION_MASKS["M1"].items() if v == 1.0]
    gross = sum((pn1[s]["pooled_mean"] or 0.0) * exploit[s]["sigma_y5_bps"]
                for s in core1)
    rt = sum(exploit[s]["round_trip_bps"] for s in core1)
    ne_ratio1 = gross / rt if rt > 0 else np.nan
    p1 = float(1 - ss.norm.cdf(m1_ic / m1_se)) if m1_se > 0 else np.nan
    exist1 = (m1_ic >= 0.033) and (delta_ic > 0)
    mat1 = ne_ratio1 > 1.0
    results["M1"] = {
        "existence": {"weekly_rank_ic": m1_ic, "se": m1_se, "n_weeks": len(ics1),
                      "bar": 0.033, "member_zero_weekly_ic": mz_ic,
                      "ic_delta_vs_member_zero": delta_ic,
                      "p_one_sided": p1, "pass": bool(exist1)},
        "materiality": {"net_edge_ratio_masked_core": ne_ratio1,
                        "core_names": core1, "bar": 1.0, "pass": bool(mat1),
                        "convention": "sum(IC_s x sigma5_s)/sum(RT_s), "
                                      "Phase-0 per-symbol table units"},
        "power_pair": POWER_PAIRS["M1"],
        "PASS": bool(exist1 and mat1)}
    pvals["M1"] = p1

    # ===================== M2 (status OPEN) ================================
    oof2 = load_oof("m2_rot")
    idx2 = fold_day_index(dates, oof2)
    mu2 = {f: oof2[f]["mu"].astype(np.float64) for f in oof2}
    yrot = p7["y_rot_raw"].astype(np.float64)
    ics2, _ = windowed_ics(mu2, idx2, yrot, sm, 16)
    m2_ic = float(np.nanmean(ics2))
    m2_se = float(np.nanstd(ics2, ddof=1) / np.sqrt(len(ics2)))
    m2_t = m2_ic / m2_se if m2_se > 0 else np.nan
    pn2 = per_name_stats(mu2, idx2, yrot, sm, symbols)
    core2 = [s for s, v in PRODUCTION_MASKS["M2"].items() if v == 1.0]
    # sigma of the rotation window per name (pooled fold days), bps
    sig_rot = {}
    for s in core2:
        j = symbols.index(s)
        vals = np.concatenate([yrot[idx2[f], j] for f in idx2])
        sig_rot[s] = float(np.nanstd(vals)) * 1e4
    gross2 = sum((pn2[s]["pooled_mean"] or 0.0) * sig_rot[s] for s in core2)
    rt2 = sum(exploit[s]["round_trip_bps"] for s in core2)
    ne_ratio2 = gross2 / rt2 if rt2 > 0 else np.nan
    p2 = float(1 - ss.norm.cdf(m2_t)) if np.isfinite(m2_t) else np.nan
    exist2 = m2_ic >= 0.041
    mat2 = ne_ratio2 > 1.0
    results["M2"] = {
        "status": "OPEN (FM6)",
        "existence": {"rot_rank_ic_16d": m2_ic, "se": m2_se, "t": m2_t,
                      "n_windows": len(ics2), "bar": 0.041,
                      "registered_equivalence": "0.041 <=> t 1.64 at se 0.025",
                      "p_one_sided": p2, "pass": bool(exist2)},
        "materiality": {"net_edge_ratio_masked_core": ne_ratio2,
                        "core_names": core2, "sigma_rot_bps": sig_rot,
                        "bar": 1.0, "pass": bool(mat2),
                        "convention": "21d-window conversion: sigma = pooled "
                                      "sd of the rotation target per name"},
        "power_pair": POWER_PAIRS["M2"],
        "PASS": bool(exist2 and mat2)}
    pvals["M2"] = p2

    # ===================== M3 ==============================================
    import organs_007 as G
    X_full, lags_X, ydisp, okm = G.m3_design(
        {k: z6[k] for k in ["T", "T_cols"]}, p7)
    oof3 = load_oof("m3_disp")
    idx3 = fold_day_index(dates, oof3)
    pool_rows = np.concatenate([idx3[f] for f in sorted(idx3)])
    pr = pool_rows[okm[pool_rows]]
    Xp, yp = X_full[pr], ydisp[pr]
    X1 = np.column_stack([np.ones(len(Xp)), Xp])
    beta, *_ = np.linalg.lstsq(X1, yp, rcond=None)
    u = yp - X1 @ beta
    V = hac_cov(X1, u, lags=10)
    Rsel = np.arange(4, 14)               # vol-surface block + 3 dummies
    bR = beta[Rsel]
    VR = V[np.ix_(Rsel, Rsel)]
    wald = float(bR @ np.linalg.solve(VR, bR))
    df = len(Rsel)
    p3 = float(ss.chi2.sf(wald, df))
    n3 = len(yp)
    n3_eff = n3 / 2.69
    # OOF increment (supporting read)
    fc = np.concatenate([oof3[f]["forecast"] for f in sorted(oof3)])
    ft = np.concatenate([oof3[f]["twin_forecast"] for f in sorted(oof3)])
    yo = np.concatenate([oof3[f]["y"] for f in sorted(oof3)])
    okv = np.isfinite(yo)
    mse_full = float(np.mean((yo[okv] - fc[okv]) ** 2))
    mse_twin = float(np.mean((yo[okv] - ft[okv]) ** 2))
    var_y = float(np.var(yo[okv]))
    # trailing-21d-mean baseline (the B-disp constant)
    delta_log = np.log(pd.Series(p7["delta_disp"].astype(np.float64)))
    bdisp = delta_log.shift(1).rolling(21, min_periods=11).mean().to_numpy()
    bvals = np.concatenate([bdisp[idx3[f]] for f in sorted(idx3)])
    okb = okv & np.isfinite(bvals)
    mse_bdisp = float(np.mean((yo[okb] - bvals[okb]) ** 2))
    # materiality conversion sentence (factors printed)
    fz = pd.Series(fc)
    fz_z = ((fz - fz.shift(1).rolling(252, min_periods=60).mean())
            / fz.shift(1).rolling(252, min_periods=60).std()).to_numpy()
    disp_t = 1.0 / (1.0 + np.exp(-np.nan_to_num(fz_z, nan=0.0)))
    swing = float(np.nanstd(disp_t))
    dc_ddisp = 0.5 * 0.25 / 1.0          # b=0.5 x c(1-c)|c=0.5 / temp=1
    typ_gain, t_max = 0.5, 0.08
    delta_mean_daily = float(np.nanmean(p7["delta_disp"]))
    ic_honest = 0.05                      # G15 shrinkage sentence value
    conv_bp = 1e4 * (2 * swing * dc_ddisp * typ_gain * t_max) \
        * ic_honest * delta_mean_daily
    exist3 = p3 <= 0.05
    mat3 = conv_bp >= 0.2
    results["M3"] = {
        "existence": {"hac_wald_chi2": wald, "df": df, "p": p3, "bar": 0.05,
                      "n_days": n3, "n_eff_5d_overlap": n3_eff,
                      "pass": bool(exist3)},
        "oof_increment_supporting": {
            "mse_full": mse_full, "mse_lags_twin": mse_twin,
            "mse_trailing21_mean": mse_bdisp, "var_y": var_y,
            "oof_r2_full": 1 - mse_full / var_y,
            "oof_r2_twin": 1 - mse_twin / var_y},
        "materiality": {
            "conversion_bp_day": conv_bp, "bar": 0.2, "pass": bool(mat3),
            "factors": {"disp_t_swing_sd": swing, "dc_ddisp": dc_ddisp,
                        "typ_tilt_gain": typ_gain, "T_max": t_max,
                        "honest_core_ic": ic_honest,
                        "mean_daily_support_dispersion": delta_mean_daily},
            "sentence": f"forecastable dispersion swing (sd of squashed "
                        f"forecast {swing:.3f}) x gain swing "
                        f"({dc_ddisp:.3f} x {typ_gain} x {t_max}) converts to "
                        f"~{conv_bp:.2f} bp/day on the fold-pool instrument "
                        f"at honest core IC {ic_honest}"},
        "power_pair": POWER_PAIRS["M3"],
        "PASS": bool(exist3 and mat3),
        "fail_consequence": "dispersion term in c_t becomes the constant = "
                            "trailing-21d mean (arm B-disp)"}
    pvals["M3"] = p3

    # ===================== M4 + A/B decision ===============================
    from sklearn.metrics import roc_auc_score
    res4 = {}
    for member, tag in [("m4_evt_a", "A"), ("m4_evt_b", "B"),
                        ("m4_control", "CTL")]:
        o = load_oof(member)
        ii = fold_day_index(dates, o)
        res4[tag] = {"oof": o, "idx": ii}
    yevt = p7["y_evt"].astype(np.float64)

    def pool_p(tag, folds):
        ps, ys, dd = [], [], []
        for f in folds:
            o = res4[tag]["oof"][f]
            ii = res4[tag]["idx"][f]
            p = o["p"].reshape(-1)
            yv = yevt[ii].reshape(-1)
            day = np.repeat(ii, o["p"].shape[1])
            ok = np.isfinite(yv)
            ps.append(p[ok]); ys.append(yv[ok]); dd.append(day[ok])
        return np.concatenate(ps), np.concatenate(ys), np.concatenate(dd)

    pA, yA, dayA = pool_p("A", range(1, 7))
    pC, _, _ = pool_p("CTL", range(1, 7))
    aucA = float(roc_auc_score(yA, pA))
    aucC = float(roc_auc_score(yA, pC))
    seA, seD = cluster_boot_auc(yA, pA, dayA, p2=pC)
    dAC = aucA - aucC
    exist4 = (aucA >= 0.526) and (dAC > 1.64 * seD)
    p4 = float(1 - ss.norm.cdf((aucA - 0.5) / seA)) if seA > 0 else np.nan
    # A/B decision on F5-F6 pooled (ONE ledger look)
    pA56, y56, day56 = pool_p("A", [5, 6])
    pB56, _, _ = pool_p("B", [5, 6])
    aucA56 = float(roc_auc_score(y56, pA56))
    aucB56 = float(roc_auc_score(y56, pB56))
    _, seD56 = cluster_boot_auc(y56, pB56, day56, p2=pA56)
    dBA = aucB56 - aucA56
    ship_b = dBA > 2 * seD56
    ledger({"kind": "ledger_look", "family": "M4-A/B",
            "look": "TOURNAMENT 4.4.8 (the ONE look)",
            "delta_auc_B_minus_A_F5F6": dBA, "se_delta": seD56,
            "bar": 2 * seD56, "decision": "ship M4-B" if ship_b else "ship M4-A"})
    # materiality: avoided-loss conversion at the 20% base rate
    hi = pA > np.quantile(pA, 0.8)
    avoided = float(np.nanmean(np.abs(yA[hi] - yA.mean())))
    base_rate = float(np.nanmean(yA))
    results["M4"] = {
        "existence": {"pooled_auc_A": aucA, "bar": 0.526, "se_day_cluster": seA,
                      "auc_vol_control": aucC, "delta_vs_control": dAC,
                      "se_delta": seD, "delta_bar": 1.64 * seD,
                      "p_one_sided_auc": p4, "pass": bool(exist4)},
        "ab_decision": {"auc_A_F5F6": aucA56, "auc_B_F5F6": aucB56,
                        "delta_B_minus_A": dBA, "se_delta": seD56,
                        "bar_2se": 2 * seD56,
                        "ships": "M4-B" if ship_b else "M4-A",
                        "llm_columns_zero_printed": not ship_b},
        "materiality": {"base_rate": base_rate,
                        "top_quintile_exceed_rate_minus_base":
                            float(np.nanmean(yA[hi]) - base_rate),
                        "sentence": f"top-quintile p-hat days exceed at "
                                    f"{float(np.nanmean(yA[hi])):.3f} vs base "
                                    f"{base_rate:.3f} — avoided-loss "
                                    f"conversion at the 20% base rate",
                        "pass": bool(float(np.nanmean(yA[hi])) > base_rate)},
        "power_pair": POWER_PAIRS["M4"],
        "PASS": bool(exist4 and float(np.nanmean(yA[hi])) > base_rate)}
    pvals["M4"] = p4

    # ===================== M5 ==============================================
    eps = json.loads((PROTO / "store" / "m5_episodes.json").read_text())
    m5sig = np.load(PROTO / "store" / "m5_signal.npz")
    mu5 = m5sig["mu"].astype(np.float64)
    r1f = p7["r1f_raw"].astype(np.float64)
    fold_days = np.zeros(len(dates), dtype=bool)
    for f in range(1, 7):
        fold_days |= F.fold_date_mask(dates, f)
    tally = []
    half_spread = {s: exploit[s]["half_spread_bps"] for s in symbols}
    for e in eps["episodes"]:
        if not fold_days[e["start"]]:
            continue
        rows = np.arange(e["start"], min(e["end"] + 1, len(dates)))
        rows = rows[fold_days[rows]]
        if len(rows) == 0:
            continue
        pnl = float(np.nansum(mu5[rows] * np.nan_to_num(r1f[rows], nan=0.0)))
        gross_peak = float(np.max(np.sum(np.abs(mu5[rows]), axis=1)))
        sleeve = [s for s in G.M5_SLEEVES[e["complex"]][0]]
        rt_bps = float(np.mean([2 * half_spread[s] for s in sleeve]))
        conv_bps = 1e4 * pnl / gross_peak if gross_peak > 0 else np.nan
        tally.append({"complex": e["complex"],
                      "start_date": e["start_date"], "end_date": e["end_date"],
                      "n_days": len(rows), "pnl_unit_book": pnl,
                      "conv_bps_per_unit_gross": conv_bps,
                      "sleeve_rt_bps": rt_bps, "win": bool(pnl > 0)})
    n_ep = len(tally)
    n_win = sum(1 for t in tally if t["win"])
    p5 = float(ss.binom.sf(n_win - 1, n_ep, 0.5)) if n_ep else np.nan
    exist5 = (n_ep > 0) and (p5 <= 0.05)
    mean_conv = float(np.nanmean([t["conv_bps_per_unit_gross"]
                                  for t in tally])) if tally else np.nan
    mean_rt = float(np.nanmean([t["sleeve_rt_bps"] for t in tally])) \
        if tally else np.nan
    mat5 = np.isfinite(mean_conv) and mean_conv >= mean_rt
    d3 = "would have enabled under the TR bar (p<=0.10)" \
        if (np.isfinite(p5) and 0.05 < p5 <= 0.10) else None
    results["M5"] = {
        "existence": {"n_episodes_F1F6": n_ep, "n_wins": n_win,
                      "binomial_p_one_sided": p5, "bar": 0.05,
                      "pass": bool(exist5), "d3_vindication_line": d3},
        "materiality": {"mean_conv_bps_per_unit_gross": mean_conv,
                        "mean_sleeve_rt_bps": mean_rt, "pass": bool(mat5)},
        "tally": tally,
        "power_pair": POWER_PAIRS["M5"],
        "PASS": bool(exist5 and mat5),
        "expected": "DISABLED with tally printed (pre-registered)"}
    pvals["M5"] = p5

    # ===================== M6 (forecast altitude ONLY) ======================
    from scipy.stats import spearmanr
    oof6 = load_oof("m6_gru")
    idx6 = fold_day_index(dates, oof6)
    daily_ics, fold_means = [], {}
    for f in sorted(oof6):
        mu = oof6[f]["mu"].astype(np.float64)
        ii = idx6[f]
        fl = []
        for j, i in enumerate(ii):
            m = sm[i] & np.isfinite(r1f[i]) & np.isfinite(mu[j])
            if m.sum() >= 8:
                fl.append(spearmanr(mu[j][m], r1f[i][m]).statistic)
        fold_means[f] = float(np.nanmean(fl))
        daily_ics += fl
    daily_ics = np.asarray(daily_ics)
    m6_ic = float(np.nanmean(daily_ics))
    m6_se = float(np.nanstd(daily_ics, ddof=1) / np.sqrt(len(daily_ics)))
    m6_se_haircut = m6_se * 1.4
    n_pos_folds = sum(1 for v in fold_means.values() if v > 0)
    pass6 = (m6_ic >= 0.02) and (n_pos_folds >= 4)
    p6_ = float(1 - ss.norm.cdf(m6_ic / m6_se_haircut))
    results["M6"] = {
        "forecast_falsifier": {"pooled_1d_rank_ic": m6_ic, "se": m6_se,
                               "se_clustering_haircut_x1.4": m6_se_haircut,
                               "n_days": len(daily_ics),
                               "fold_means": fold_means,
                               "n_sign_positive_folds": n_pos_folds,
                               "bar": "IC >= 0.02 AND >= 4/6 folds positive",
                               "p_one_sided_haircut": p6_,
                               "PASS": bool(pass6)},
        "utility_line": "structurally unresolvable at this T_max — "
                        "forecast-altitude verdict only",
        "power_pair": POWER_PAIRS["M6"],
        "no_tilt_path": True, "no_organ_inputs_entry": True}

    # ===================== BH-FDR(10%) over the 5-test family ==============
    fam = {k: v for k, v in pvals.items() if np.isfinite(v)}
    ks = sorted(fam, key=lambda k: fam[k])
    m = len(ks)
    bh = {}
    for r, k in enumerate(ks, start=1):
        bh[k] = {"p": fam[k], "bh_threshold_10pct": 0.10 * r / m,
                 "rejected": fam[k] <= 0.10 * r / m}
    # enforce step-up monotonicity
    rejected = set()
    max_r = 0
    for r, k in enumerate(ks, start=1):
        if fam[k] <= 0.10 * r / m:
            max_r = r
    for r, k in enumerate(ks, start=1):
        if r <= max_r:
            rejected.add(k)
    for k in bh:
        bh[k]["rejected"] = k in rejected

    # ===================== roster (mechanical) ==============================
    ships_m4_variant = results["M4"]["ab_decision"]["ships"]
    roster = {
        "M1": "SHIPS" if results["M1"]["PASS"] else "CHALLENGER",
        "M2": "SHIPS" if results["M2"]["PASS"] else "CHALLENGER (OPEN)",
        "M3": ("GAIN ACTIVE" if results["M3"]["PASS"]
               else "B-disp constant (LOO control arm)"),
        "M4": (f"SHIPS ({ships_m4_variant})" if results["M4"]["PASS"]
               else f"CHALLENGER ({ships_m4_variant} trained)"),
        "M5": "ENABLED" if results["M5"]["PASS"] else "DISABLED (tally printed)",
        "M6": "forecast-altitude line only (never a roster member)",
    }
    out = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
           "gates": results, "bh_fdr_family": bh, "roster": roster,
           "shrinkage_sentence": "honest expected true core IC ~ 0.04-0.06, "
                                 "not the displayed per-name OOF values "
                                 "(inverse-Mills selection bound, G15)"}
    (PROTO / "gates").mkdir(exist_ok=True)
    (PROTO / "gates" / "acceptance_007.json").write_text(
        json.dumps(out, indent=1, default=float))
    # per-name OOF stats + mask-rule outputs (M2's printed REPORT-ONLY, D7)
    rule_m1 = mask_rule(pn1, 6)
    rule_m2 = mask_rule(pn2, 6)
    per_name = {
        "M1": {"stats": {s: pn1[s] for s in sorted(set(TILT_CORE) | set(TILT_COND))},
               "rule_output_new_oof": rule_m1,
               "production_mask_frozen_phase0": PRODUCTION_MASKS["M1"]},
        "M2": {"stats": {s: pn2[s] for s in sorted(set(TILT_CORE) | set(TILT_COND))},
               "rule_output_new_oof_REPORT_ONLY_D7": rule_m2,
               "production_mask_frozen_phase0_gbm_tags": PRODUCTION_MASKS["M2"]},
        "note": "production masks remain the FROZEN Phase-0 table (G15/A9); "
                "rule outputs from new OOF are printed report-only",
    }
    (PROTO / "gates" / "per_name_oof_stats.json").write_text(
        json.dumps(per_name, indent=1, default=float))
    for organ in ["M1", "M2", "M3", "M4", "M5"]:
        r = results[organ]
        ledger({"kind": "gate_result", "family": "acceptance",
                "organ": organ, "pass": r["PASS"],
                "power_pair": POWER_PAIRS[organ],
                "p_existence": pvals.get(organ),
                "roster": roster[organ]})
    ledger({"kind": "gate_result", "family": "forecast_battery",
            "organ": "M6",
            "pass": bool(results["M6"]["forecast_falsifier"]["PASS"]),
            "power_pair": POWER_PAIRS["M6"],
            "note": "forecast-altitude only; report row, not a roster gate"})
    ledger({"kind": "multiplicity", "family": "acceptance BH-FDR(10%)",
            "result": bh})

    print(json.dumps({"roster": roster}, indent=1))
    for organ in ["M1", "M2", "M3", "M4", "M5", "M6"]:
        print(f"\n=== {organ} ===")
        print(json.dumps(results[organ], indent=1, default=float)[:2400])


if __name__ == "__main__":
    main()
