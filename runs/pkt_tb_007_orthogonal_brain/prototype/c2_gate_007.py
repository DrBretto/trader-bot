"""PKT-TB-007 — C2 orthogonality gate (BUILD_SPEC_007 §4.4, TOURNAMENT §4.4.2).

Binding: pooled |rho| <= 0.7 for every directional pair in BOTH spaces
(G-signal: per-day cross-sectional Spearman of standardized signals over the
common 64-symbol space, abstention days excluded per pair with inclusion rate
printed; G-book: TB-006 fixed solo-book daily-return Pearson), pooled F1-F6
OOF. Member-zero (deployed RankingMLP) special row: M1-vs-Z0 pooled <= 0.8
binding, NO per-fold ceiling. Per-fold ceiling elsewhere is a REPLICATION
TRIGGER: a pair fires the remediation ladder iff |rho| > 0.8 in >= 2 of its
12 fold x space cells, OR one cell > 0.8 + 0.29 (= 2x fold-se).
G-scalar rows: M3 aggressiveness vs members' |tilt| proxy / regime scalar /
VIX (pooled <= 0.7, per-fold printed; regime row is coverage-limited to the
artifact-record era and disclosed as such). M6 rows REPORT-ONLY.

Output: designs/c2_gate_matrix.json (committed before any bake-off replay).
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROTO = Path(__file__).resolve().parent
REPO = PROTO.parents[2]
TB6 = REPO / "runs" / "pkt_tb_006_clean_sheet_brain" / "prototype"
sys.path.insert(0, str(PROTO))

import books                                # noqa: E402  (TB-006 §6 verbatim)
import folds as F                           # noqa: E402

OOF_DIR = PROTO / "oof"
SUPPORT = ["ITA", "SOXX", "XRT", "TLT", "AGG", "MUB", "FXE", "USO", "FXI",
           "RSP", "IYR", "SHY", "KRE"]
POOLED_BAR = 0.7
Z0_M1_BAR = 0.8
TRIGGER_CELL = 0.8
TRIGGER_BIG = 0.8 + 0.29
REGIME_RISK_SCALAR = {"high_vol_panic": 0.0, "risk_off_trend": 0.25,
                      "choppy": 0.5, "calm_uptrend": 0.75,
                      "risk_on_trend": 1.0}


def ledger(entry: dict) -> None:
    entry = {"ts": dt.datetime.now().isoformat(timespec="seconds"),
             "phase": "C-wave2a", **entry}
    with (PROTO / "validation_looks_007.jsonl").open("a") as f:
        f.write(json.dumps(entry, default=float) + "\n")


def load_member_signals():
    """Returns dict member -> {fold -> {idx (panel rows), mu_std [n,64],
    sigma [n,64], expressive [n] bool}}."""
    z6 = np.load(TB6 / "store" / "panel.npz", allow_pickle=False)
    dates = np.asarray(z6["dates"]).astype(str)
    didx = {d: i for i, d in enumerate(dates)}
    sm = z6["symbol_mask"].astype(bool)
    out = {"dates": dates, "sm": sm, "close_px": z6["close_px"],
           "symbols": [str(s) for s in z6["symbols"]]}

    def std_rows(mu):
        o = np.zeros_like(mu, dtype=np.float64)
        for i in range(mu.shape[0]):
            m = np.isfinite(mu[i])
            if m.sum() >= 2:
                sd = mu[i][m].std()
                o[i][m] = (mu[i][m] - mu[i][m].mean()) / (sd if sd > 1e-9 else 1.0)
        return o

    members = {}
    for name, oofm in [("M1", "m1_cast"), ("M2", "m2_rot"),
                       ("M4", "m4_evt_a"), ("M5", "m5_posz"),
                       ("M6", "m6_gru")]:
        members[name] = {}
        for f in range(1, 7):
            z = np.load(OOF_DIR / f"fold_{f}_{oofm}.npz", allow_pickle=True)
            idx = np.array([didx[str(d)] for d in
                            np.asarray(z["dates"]).astype(str)])
            mu = np.asarray(z["mu"], dtype=np.float64)
            sigma = np.asarray(z["sigma"], dtype=np.float64)
            if name == "M4":
                p = np.asarray(z["p"], dtype=np.float64)
                expressive = p.max(axis=1) > 0.2        # "no event mass" rule
            elif name == "M5":
                expressive = np.any(mu != 0, axis=1)
            else:
                expressive = np.ones(len(idx), dtype=bool)
            members[name][f] = {"idx": idx, "mu_std": std_rows(mu),
                                "sigma": sigma, "expressive": expressive}
    # member zero
    mz = np.load(PROTO / "store" / "member_zero_scores.npz")["score"] \
        .astype(np.float64)
    members["Z0"] = {}
    for f in range(1, 7):
        idx = members["M1"][f]["idx"]
        members["Z0"][f] = {"idx": idx, "mu_std": std_rows(mz[idx]),
                            "sigma": np.ones((len(idx), mz.shape[1])),
                            "expressive": np.ones(len(idx), dtype=bool)}
    out["members"] = members
    return out


def pair_signal(ma, mb, sm):
    """per-fold + pooled mean per-day Spearman; inclusion rate."""
    per_fold, pooled_vals = {}, []
    inc_n, tot_n = 0, 0
    for f in range(1, 7):
        A, B = ma[f], mb[f]
        assert np.array_equal(A["idx"], B["idx"])
        inc = A["expressive"] & B["expressive"]
        tot_n += len(inc)
        inc_n += int(inc.sum())
        vals = []
        for j in np.nonzero(inc)[0]:
            i = A["idx"][j]
            m = sm[i] & np.isfinite(A["mu_std"][j]) & np.isfinite(B["mu_std"][j])
            if m.sum() >= 8 and A["mu_std"][j][m].std() > 1e-9 \
                    and B["mu_std"][j][m].std() > 1e-9:
                r = spearmanr(A["mu_std"][j][m], B["mu_std"][j][m]).statistic
                if np.isfinite(r):
                    vals.append(r)
        per_fold[f] = float(np.mean(vals)) if vals else np.nan
        pooled_vals += vals
    pooled = float(np.mean(pooled_vals)) if pooled_vals else np.nan
    return pooled, per_fold, (inc_n / tot_n if tot_n else np.nan)


def book_returns(member, sm, close_px):
    """solo-book daily return series per fold day (TB-006 §6 rule; the book
    formed at D earns next-day close-to-close)."""
    N = close_px.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        r1 = close_px[1:] / close_px[:-1] - 1.0
    r1 = np.vstack([r1, np.full((1, close_px.shape[1]), np.nan)])
    # r1[d] = close(d+1)/close(d) - 1 (the return the day-D book earns next)
    out = {}
    for f in range(1, 7):
        A = member[f]
        w = books.solo_books(A["mu_std"], A["sigma"],
                             active=sm[A["idx"]])
        ret = np.array([np.nansum(w[j] * np.nan_to_num(r1[i], nan=0.0))
                        if w[j].sum() > 0 else np.nan
                        for j, i in enumerate(A["idx"])])
        out[f] = {"idx": A["idx"], "ret": ret, "expressive": A["expressive"]}
    return out


def pair_book(ba, bb):
    per_fold, pooled_a, pooled_b = {}, [], []
    inc_n, tot_n = 0, 0
    for f in range(1, 7):
        A, B = ba[f], bb[f]
        inc = A["expressive"] & B["expressive"] & np.isfinite(A["ret"]) \
            & np.isfinite(B["ret"])
        tot_n += len(inc)
        inc_n += int(inc.sum())
        a, b = A["ret"][inc], B["ret"][inc]
        if len(a) >= 20 and a.std() > 0 and b.std() > 0:
            per_fold[f] = float(np.corrcoef(a, b)[0, 1])
        else:
            per_fold[f] = np.nan
        pooled_a.append(a)
        pooled_b.append(b)
    a = np.concatenate(pooled_a)
    b = np.concatenate(pooled_b)
    pooled = float(np.corrcoef(a, b)[0, 1]) if len(a) >= 20 else np.nan
    return pooled, per_fold, (inc_n / tot_n if tot_n else np.nan)


def residualized_book_diag(booksig, close_px, sm, members_list):
    """REPORT-ONLY diagnostic: pairwise Pearson of solo-book daily returns
    after residualizing each member's series on the equal-weight
    active-universe daily return (the common long-only market factor).
    The registered G-book instrument (TB-006 §6 long-only unit-gross books)
    loads every book on market beta ~1; this strips that shared loading so
    the orthogonality of the SELECTIONS is visible. Never a gate input."""
    N = close_px.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        r1 = close_px[1:] / close_px[:-1] - 1.0
    r1 = np.vstack([r1, np.full((1, close_px.shape[1]), np.nan)])
    mkt = np.array([float(np.nanmean(np.where(sm[i], r1[i], np.nan)))
                    for i in range(N)])

    def resid_series(b):
        out = {}
        # pooled OLS beta per member over all fold days
        rets, ms = [], []
        for f in range(1, 7):
            ok = np.isfinite(b[f]["ret"]) & np.isfinite(mkt[b[f]["idx"]])
            rets.append(b[f]["ret"][ok])
            ms.append(mkt[b[f]["idx"]][ok])
        rr, mm = np.concatenate(rets), np.concatenate(ms)
        A = np.column_stack([np.ones(len(mm)), mm])
        coef, *_ = np.linalg.lstsq(A, rr, rcond=None)
        for f in range(1, 7):
            m_f = mkt[b[f]["idx"]]
            out[f] = b[f]["ret"] - (coef[0] + coef[1] * m_f)
        return out

    resid = {m: resid_series(booksig[m]) for m in members_list}
    rows = {}
    for i, a in enumerate(members_list):
        for b in members_list[i + 1:]:
            aa, bb = [], []
            for f in range(1, 7):
                A, B = booksig[a][f], booksig[b][f]
                inc = A["expressive"] & B["expressive"] \
                    & np.isfinite(resid[a][f]) & np.isfinite(resid[b][f])
                aa.append(resid[a][f][inc])
                bb.append(resid[b][f][inc])
            x, y = np.concatenate(aa), np.concatenate(bb)
            rows[f"{a}-{b}"] = (float(np.corrcoef(x, y)[0, 1])
                                if len(x) >= 20 and x.std() > 0
                                and y.std() > 0 else np.nan)
    return rows


def main():
    data = load_member_signals()
    sm = data["sm"]
    dates = data["dates"]
    members = data["members"]
    symbols = data["symbols"]
    sup_j = [symbols.index(s) for s in SUPPORT]

    directional = ["Z0", "M1", "M2", "M4", "M5"]
    pairs = [(a, b) for i, a in enumerate(directional)
             for b in directional[i + 1:]]

    booksig = {m: book_returns(members[m], sm, data["close_px"])
               for m in directional + ["M6"]}

    matrix = {}
    trigger_fired = []
    binding_breach = []
    for a, b in pairs:
        gs_p, gs_f, gs_inc = pair_signal(members[a], members[b], sm)
        gb_p, gb_f, gb_inc = pair_book(booksig[a], booksig[b])
        bar = Z0_M1_BAR if {a, b} == {"Z0", "M1"} else POOLED_BAR
        cells = [v for v in list(gs_f.values()) + list(gb_f.values())
                 if np.isfinite(v)]
        n_over = sum(1 for v in cells if abs(v) > TRIGGER_CELL)
        big = any(abs(v) > TRIGGER_BIG for v in cells)
        is_z0m1 = {a, b} == {"Z0", "M1"}
        fired = (not is_z0m1) and (n_over >= 2 or big)
        breach = (np.isfinite(gs_p) and abs(gs_p) > bar) or \
                 (np.isfinite(gb_p) and abs(gb_p) > bar)
        if fired:
            trigger_fired.append(f"{a}-{b}")
        if breach:
            binding_breach.append(f"{a}-{b}")
        matrix[f"{a}-{b}"] = {
            "signal": {"pooled": gs_p, "per_fold": gs_f,
                       "inclusion_rate": gs_inc},
            "book": {"pooled": gb_p, "per_fold": gb_f,
                     "inclusion_rate": gb_inc},
            "pooled_bar": bar,
            "binding_breach": bool(breach),
            "per_fold_cells_over_0.8": n_over,
            "replication_trigger_fired": bool(fired),
            "special_row": ("M1-vs-member-zero: pooled <= 0.8 binding, "
                            "no per-fold ceiling (per-fold se 0.146)")
            if is_z0m1 else None}

    # ---------------- M6 rows (REPORT-ONLY) --------------------------------
    m6_rows = {}
    for b in ["Z0", "M1", "M2", "M4", "M5"]:
        gs_p, gs_f, gs_inc = pair_signal(members["M6"], members[b], sm)
        gb_p, gb_f, _ = pair_book(booksig["M6"], booksig[b])
        m6_rows[f"M6-{b}"] = {"signal_pooled": gs_p, "signal_per_fold": gs_f,
                              "book_pooled": gb_p, "book_per_fold": gb_f,
                              "report_only": True}
    # raw-input overlap disclosure (TOURNAMENT G9): 3 shared columns
    z6 = np.load(TB6 / "store" / "panel.npz", allow_pickle=False)
    xf = [str(c) for c in z6["X_features"]]
    p7 = np.load(PROTO / "store" / "panel_007.npz", allow_pickle=False)
    m6f = p7["m6_feat"]
    shared = [("gap_open_pct", 5, xf.index("gap_open_pct")),
              ("range_pct", 6, xf.index("range_pct")),
              ("volume_z_21d", 7, xf.index("volume_z_21d"))]
    fold_rows = np.concatenate([members["M1"][f]["idx"] for f in range(1, 7)])
    overlap = {}
    for name, jm6, jx in shared:
        a = m6f[fold_rows - 1, :, jm6].reshape(-1)      # valued at D-1, like
        bvals = z6["X"][fold_rows, :, 62, jx].reshape(-1)   # X's last seq row
        ok = np.isfinite(a) & np.isfinite(bvals)
        overlap[name] = float(np.corrcoef(a[ok], bvals[ok])[0, 1])
    m6_rows["raw_input_overlap_with_M1"] = {
        "columns": [s[0] for s in shared],
        "value_correlation_m6_vs_m1_lastday": overlap,
        "disclosure": "M6 shares these 3 raw input columns with M1's fast "
                      "block — disclosed report-only per TOURNAMENT G9; M6 "
                      "never enters the tilt"}

    # ---------------- G-scalar rows -----------------------------------------
    # M3 aggressiveness = squashed expanding-z of the OOF dispersion forecast
    fc_by_fold, idx3 = {}, {}
    didx = {d: i for i, d in enumerate(dates)}
    for f in range(1, 7):
        z = np.load(OOF_DIR / f"fold_{f}_m3_disp.npz", allow_pickle=True)
        fc_by_fold[f] = np.asarray(z["forecast"], dtype=np.float64)
        idx3[f] = np.array([didx[str(d)] for d in
                            np.asarray(z["dates"]).astype(str)])
    fc = np.concatenate([fc_by_fold[f] for f in range(1, 7)])
    rows3 = np.concatenate([idx3[f] for f in range(1, 7)])
    fz = pd.Series(fc)
    fz_z = ((fz - fz.shift(1).rolling(252, min_periods=60).mean())
            / fz.shift(1).rolling(252, min_periods=60).std()).to_numpy()
    disp_t = 1.0 / (1.0 + np.exp(-np.nan_to_num(fz_z, nan=0.0)))
    fold_of_row = np.concatenate([[f] * len(idx3[f]) for f in range(1, 7)])

    def scalar_row(series_by_row, label, note=None):
        ok = np.isfinite(series_by_row) & np.isfinite(disp_t)
        per_fold = {}
        for f in range(1, 7):
            sel = ok & (fold_of_row == f)
            if sel.sum() >= 20:
                per_fold[f] = float(np.corrcoef(disp_t[sel],
                                                series_by_row[sel])[0, 1])
            else:
                per_fold[f] = np.nan
        pooled = float(np.corrcoef(disp_t[ok], series_by_row[ok])[0, 1]) \
            if ok.sum() >= 20 else np.nan
        return {"pooled": pooled, "per_fold": per_fold, "n": int(ok.sum()),
                "pooled_bar": POOLED_BAR,
                "binding_breach": bool(np.isfinite(pooled)
                                       and abs(pooled) > POOLED_BAR),
                "note": note}

    scalar_rows = {}
    # (a) members' |tilt| proxy: per-day sd over SUPPORT of standardized mu
    for m in ["M1", "M2", "M4", "M5"]:
        proxy = np.full(len(rows3), np.nan)
        row_of = {}
        for f in range(1, 7):
            for j, i in enumerate(members[m][f]["idx"]):
                row_of[i] = members[m][f]["mu_std"][j]
        for k, i in enumerate(rows3):
            if i in row_of:
                v = row_of[i][sup_j]
                if np.isfinite(v).sum() >= 8:
                    proxy[k] = np.nanstd(v)
        scalar_rows[f"M3_vs_{m}_tilt_proxy"] = scalar_row(
            proxy, m, note="pre-EA proxy: per-day sd over support of the "
                           "member's standardized mu (genes not yet evolved)")
    # (b) regime scalar (artifact-record era only — disclosed)
    labels = json.loads((PROTO / "regime_labels_preholdout.json").read_text())
    reg = np.array([REGIME_RISK_SCALAR.get(labels.get(str(dates[i])), np.nan)
                    for i in rows3])
    scalar_rows["M3_vs_regime_scalar"] = scalar_row(
        reg, "regime", note="regime labels exist only in the artifact record "
        "(2025-08-04..fitness end) — coverage-limited row, disclosed; scalar "
        "encoding panic=0 .. risk_on=1")
    # M5 vs regime
    m5mag = np.load(PROTO / "store" / "m5_signal.npz")["mag"].astype(np.float64)
    okr = np.isfinite(reg)
    m5r = m5mag[rows3]
    if okr.sum() >= 20 and np.nanstd(m5r[okr]) > 0:
        m5_reg = float(np.corrcoef(m5r[okr], reg[okr])[0, 1])
    else:
        m5_reg = np.nan
    scalar_rows["M5_vs_regime_scalar"] = {
        "pooled": m5_reg, "n": int(okr.sum()), "pooled_bar": POOLED_BAR,
        "binding_breach": bool(np.isfinite(m5_reg) and abs(m5_reg) > POOLED_BAR),
        "note": "coverage-limited (artifact-record era)"}
    # (c) VIX level
    vix = pd.read_parquet(TB6 / "cache" / "cboe" / "VIX.parquet")
    vix["visible_from"] = pd.to_datetime(vix["date"]) + pd.Timedelta(days=1)
    dts = pd.DatetimeIndex(pd.to_datetime([dates[i] for i in rows3]))
    vj = pd.merge_asof(pd.DataFrame({"D": dts}),
                       vix.sort_values("visible_from")[["visible_from", "close"]],
                       left_on="D", right_on="visible_from",
                       direction="backward")
    scalar_rows["M3_vs_VIX_level"] = scalar_row(
        vj["close"].to_numpy(dtype=np.float64), "VIX")

    # ---------------- gate verdict ----------------------------------------
    operating = {
        "pooled": "n_eff ~ 300, se ~ 0.06; P(breach | true rho 0.5) ~ 0.04%; "
                  "P(sneak-under | true 0.78) ~ 9%",
        "trigger": "double-breach P ~ 1.5%/pair at true 0.5; ~24% at true 0.6"}
    scalar_breach = [k for k, v in scalar_rows.items()
                     if v.get("binding_breach")]
    resid_rows = residualized_book_diag(booksig, data["close_px"], sm,
                                        directional)
    finding = {
        "kind": "measurement_validity_finding",
        "summary": "the registered G-book instrument (TB-006 §6 long-only "
                   "unit-gross solo books) saturates on the common market "
                   "factor: every directional pair pools at book rho ~0.9 "
                   "INCLUDING pairs whose signal-space rho is ~0.00-0.20, "
                   "and the pre-registration's printed operating "
                   "characteristics assumed true rho ~0.5 "
                   "(P(breach|true 0.5) ~ 0.04%). The breach pattern is "
                   "uniform across signal-orthogonal pairs => it identifies "
                   "the instrument's market-beta loading, not organ "
                   "redundancy.",
        "market_residualized_book_pearson_report_only": resid_rows,
        "ladder_disposition": "ESCALATED, no rungs burned: the ladder "
                              "(<=2 passes/pair, <=4 rungs system-wide) is "
                              "mechanically inapplicable to 10 simultaneous "
                              "instrument-driven breaches; rung remedies "
                              "(re-partition/residualize/horizon-shift/"
                              "demote) target organ redundancy, which the "
                              "signal space contradicts. Disposition is a "
                              "chair/orchestrator decision, surfaced as a "
                              "finding per executor rails.",
    }
    out = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "spaces": ["G-signal (per-day cross-sectional Spearman, standardized)",
                   "G-book (TB-006 fixed solo-book daily-return Pearson)"],
        "pooled_bar": POOLED_BAR, "member_zero_m1_bar": Z0_M1_BAR,
        "replication_trigger": ">=2 of 12 fold x space cells |rho| > 0.8, or "
                               "1 cell > 1.09",
        "operating_characteristics": operating,
        "directional_pairs": matrix,
        "g_scalar_rows": scalar_rows,
        "m6_rows_report_only": m6_rows,
        "binding_breaches": binding_breach,
        "scalar_binding_breaches": scalar_breach,
        "replication_triggers_fired": trigger_fired,
        "remediation_ladder_history": [],
        "measurement_validity_finding": finding,
        "GATE": ("PASS" if not binding_breach and not trigger_fired
                 and not scalar_breach else "REMEDIATION REQUIRED"),
    }
    (PROTO / "designs").mkdir(exist_ok=True)
    (PROTO / "designs" / "c2_gate_matrix.json").write_text(
        json.dumps(out, indent=1, default=float))
    ledger({"kind": "gate_result", "family": "C2",
            "binding_breaches": binding_breach,
            "triggers_fired": trigger_fired,
            "scalar_breaches": scalar_breach, "gate": out["GATE"]})
    if binding_breach or trigger_fired:
        ledger({"kind": "finding", "family": "C2",
                "component": "G-book instrument",
                "decision": finding["ladder_disposition"],
                "summary": finding["summary"],
                "signal_space_max_abs_pooled": max(
                    abs(v["signal"]["pooled"]) for v in matrix.values()
                    if np.isfinite(v["signal"]["pooled"])),
                "book_space_residualized_max_abs": max(
                    abs(v) for v in resid_rows.values() if np.isfinite(v)),
                "provenance": "TOURNAMENT §4.4.2 operating characteristics "
                              "(printed expectation true rho ~0.5) + TB-006 "
                              "BUILD_SPEC §6 long-only book rule"})
    print(json.dumps({k: out[k] for k in
                      ["binding_breaches", "replication_triggers_fired",
                       "scalar_binding_breaches", "GATE"]}, indent=1))
    print("\npooled matrix:")
    for k, v in matrix.items():
        print(f"  {k:8s} signal {v['signal']['pooled']:+.3f} "
              f"(inc {v['signal']['inclusion_rate']:.2f})  "
              f"book {v['book']['pooled']:+.3f}")
    for k, v in m6_rows.items():
        if "signal_pooled" in v:
            print(f"  {k:8s} signal {v['signal_pooled']:+.3f}  "
                  f"book {v['book_pooled']:+.3f}  [report-only]")
    print("\nmarket-residualized book Pearson (report-only diagnostic):")
    for k, v in resid_rows.items():
        print(f"  {k:8s} {v:+.3f}")


if __name__ == "__main__":
    main()
