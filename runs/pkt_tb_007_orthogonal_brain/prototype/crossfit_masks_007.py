"""PKT-TB-007 — L1 cross-fit per-rotation re-derivation (TOURNAMENT §4.4.6).

For each of the 6 EA rotations (rotation r holds fold r out), re-derive from
the OTHER 5 folds only, using the frozen rule texts:
  (i)   per-organ tier masks (BUILD_SPEC §2.3 rule text; "all but at most one
        fold positive" transcribes 5/6 to the 5-fold case — documented),
  (ii)  all five acceptance-gate decisions (TOURNAMENT §4.4.1 bars verbatim)
        => that rotation's roster (M4 A/B on F5-F6 ∩ training folds),
  (iii) approximate C2 (pooled-only, no ladder): pooled |rho| > 0.7 in either
        space (0.8 for Z0-M1) demotes the junior of the pair
        (seniority M1 > M2 > M4 > M3 > M5).

Output: store/rotation_masks_007.json (combined, audit copy) + the EA
contract files store/rotation_masks_007/rotation_<f>.json
({rotation, withheld_fold, derived_from_folds, roster, masks}) and
store/roster_007.json (production roster = acceptance-gate passers among
{M1,M2,M4,M5}, post-C2; mechanical map, TOURNAMENT §4.4.1). The PRODUCTION
run keeps the frozen Phase-0 masks (G15/A9).
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

import folds as F                                    # noqa: E402
import gates_007 as GA                               # noqa: E402
import organs_007 as G                               # noqa: E402
import c2_gate_007 as C2                             # noqa: E402
from tilt_adapter import PRODUCTION_MASKS            # noqa: E402

SENIORITY = ["M1", "M2", "M4", "M3", "M5"]


def main():
    z6 = np.load(TB6 / "store" / "panel.npz", allow_pickle=False)
    p6 = {k: z6[k] for k in ["dates", "symbols", "y5_raw", "symbol_mask",
                             "T", "T_cols", "close_px"]}
    z7 = np.load(PROTO / "store" / "panel_007.npz", allow_pickle=False)
    p7 = {k: z7[k] for k in z7.files}
    dates = np.asarray(p6["dates"]).astype(str)
    symbols = [str(s) for s in p6["symbols"]]
    sm = p6["symbol_mask"].astype(bool)
    y5 = p6["y5_raw"].astype(np.float64)
    yrot = p7["y_rot_raw"].astype(np.float64)
    r1f = p7["r1f_raw"].astype(np.float64)
    exploit = {r["symbol"]: r for r in json.loads(
        (PROTO.parent / "universe_exploitability.json").read_text())
        ["per_symbol"]}
    mz = np.load(PROTO / "store" / "member_zero_scores.npz")["score"] \
        .astype(np.float64)

    oof1 = GA.load_oof("m1_cast")
    oof2 = GA.load_oof("m2_rot")
    oof3 = GA.load_oof("m3_disp")
    oof4a = GA.load_oof("m4_evt_a")
    oof4b = GA.load_oof("m4_evt_b")
    oof4c = GA.load_oof("m4_control")
    idx1 = GA.fold_day_index(dates, oof1)
    idx2 = GA.fold_day_index(dates, oof2)
    idx3 = GA.fold_day_index(dates, oof3)
    idx4 = GA.fold_day_index(dates, oof4a)
    mu1 = {f: oof1[f]["mu"].astype(np.float64) for f in oof1}
    mu2 = {f: oof2[f]["mu"].astype(np.float64) for f in oof2}
    yevt = p7["y_evt"].astype(np.float64)
    eps = json.loads((PROTO / "store" / "m5_episodes.json").read_text())
    m5sig = np.load(PROTO / "store" / "m5_signal.npz")
    mu5_all = m5sig["mu"].astype(np.float64)
    X_full, lags_X, ydisp, okm = G.m3_design(
        {k: z6[k] for k in ["T", "T_cols"]}, p7)
    sigdata = C2.load_member_signals()
    members_sig = sigdata["members"]
    booksig = {m: C2.book_returns(members_sig[m], sm, p6["close_px"])
               for m in ["Z0", "M1", "M2", "M4", "M5"]}
    from sklearn.metrics import roc_auc_score

    rotations = {}
    for r in range(1, 7):
        tf = [f for f in range(1, 7) if f != r]
        rot = {"held_out_fold": r, "training_folds": tf}

        # ---- masks (frozen rule on 5-fold per-name stats) ------------------
        pn1 = GA.per_name_stats(mu1, idx1, y5, sm, symbols, folds=tf)
        pn2 = GA.per_name_stats(mu2, idx2, yrot, sm, symbols, folds=tf)
        masks = {"M1": GA.mask_rule(pn1, 5), "M2": GA.mask_rule(pn2, 5),
                 "M5": PRODUCTION_MASKS["M5"]}
        rot["masks"] = masks

        # ---- acceptance decisions on the 5 training folds ------------------
        gates = {}
        sub1 = {f: mu1[f] for f in tf}
        sub_idx1 = {f: idx1[f] for f in tf}
        ics1, _ = GA.windowed_ics(sub1, sub_idx1, y5, sm, 5)
        mz_sub = {f: mz[idx1[f]] for f in tf}
        ics0, _ = GA.windowed_ics(mz_sub, sub_idx1, y5, sm, 5)
        ic1 = float(np.nanmean(ics1))
        d1 = ic1 - float(np.nanmean(ics0))
        core1 = [s for s, v in masks["M1"].items() if v == 1.0]
        if core1:
            gross = sum((pn1[s]["pooled_mean"] or 0.0)
                        * exploit[s]["sigma_y5_bps"] for s in core1)
            rt = sum(exploit[s]["round_trip_bps"] for s in core1)
            ne1 = gross / rt if rt > 0 else np.nan
        else:
            ne1 = np.nan
        gates["M1"] = {"weekly_ic": ic1, "delta_vs_z0": d1,
                       "net_edge_ratio": ne1,
                       "pass": bool(ic1 >= 0.033 and d1 > 0
                                    and np.isfinite(ne1) and ne1 > 1)}

        sub2 = {f: mu2[f] for f in tf}
        sub_idx2 = {f: idx2[f] for f in tf}
        ics2, _ = GA.windowed_ics(sub2, sub_idx2, yrot, sm, 16)
        ic2 = float(np.nanmean(ics2))
        core2 = [s for s, v in masks["M2"].items() if v == 1.0]
        if core2:
            gr2 = 0.0
            rt2 = 0.0
            for s in core2:
                j = symbols.index(s)
                vals = np.concatenate([yrot[idx2[f], j] for f in tf])
                gr2 += (pn2[s]["pooled_mean"] or 0.0) * float(np.nanstd(vals)) * 1e4
                rt2 += exploit[s]["round_trip_bps"]
            ne2 = gr2 / rt2 if rt2 > 0 else np.nan
        else:
            ne2 = np.nan
        gates["M2"] = {"rot_ic": ic2, "net_edge_ratio": ne2,
                       "pass": bool(ic2 >= 0.041 and np.isfinite(ne2)
                                    and ne2 > 1)}

        rows = np.concatenate([idx3[f] for f in tf])
        rows = rows[okm[rows]]
        Xp, yp = X_full[rows], ydisp[rows]
        X1 = np.column_stack([np.ones(len(Xp)), Xp])
        beta, *_ = np.linalg.lstsq(X1, yp, rcond=None)
        u = yp - X1 @ beta
        V = GA.hac_cov(X1, u, lags=10)
        Rsel = np.arange(4, 14)
        bR = beta[Rsel]
        wald = float(bR @ np.linalg.solve(V[np.ix_(Rsel, Rsel)], bR))
        p3 = float(ss.chi2.sf(wald, len(Rsel)))
        fc = np.concatenate([oof3[f]["forecast"] for f in tf])
        fz = pd.Series(fc)
        fz_z = ((fz - fz.shift(1).rolling(252, min_periods=60).mean())
                / fz.shift(1).rolling(252, min_periods=60).std()).to_numpy()
        disp_t = 1.0 / (1.0 + np.exp(-np.nan_to_num(fz_z, nan=0.0)))
        swing = float(np.nanstd(disp_t))
        conv = 1e4 * (2 * swing * 0.125 * 0.5 * 0.08) * 0.05 \
            * float(np.nanmean(p7["delta_disp"]))
        gates["M3"] = {"hac_wald_p": p3, "conv_bp_day": conv,
                       "pass": bool(p3 <= 0.05 and conv >= 0.2)}

        def pool4(oofm, folds):
            ps, ys, dd = [], [], []
            for f in folds:
                p = oofm[f]["p"].reshape(-1)
                yv = yevt[idx4[f]].reshape(-1)
                day = np.repeat(idx4[f], oofm[f]["p"].shape[1])
                ok = np.isfinite(yv)
                ps.append(p[ok]); ys.append(yv[ok]); dd.append(day[ok])
            return (np.concatenate(ps), np.concatenate(ys),
                    np.concatenate(dd))
        pA, yA, dayA = pool4(oof4a, tf)
        pC, _, _ = pool4(oof4c, tf)
        aucA = float(roc_auc_score(yA, pA))
        aucC = float(roc_auc_score(yA, pC))
        seA, seD = GA.cluster_boot_auc(yA, pA, dayA, b=200, p2=pC)
        hi = pA > np.quantile(pA, 0.8)
        mat4 = float(np.nanmean(yA[hi])) > float(np.nanmean(yA))
        pass4 = (aucA >= 0.526) and ((aucA - aucC) > 1.64 * seD) and mat4
        ab_folds = [f for f in (5, 6) if f in tf]
        if ab_folds:
            pa56, ya56, day56 = pool4(oof4a, ab_folds)
            pb56, _, _ = pool4(oof4b, ab_folds)
            aA = float(roc_auc_score(ya56, pa56))
            aB = float(roc_auc_score(ya56, pb56))
            _, seD56 = GA.cluster_boot_auc(ya56, pb56, day56, b=200, p2=pa56)
            ships = "M4-B" if (aB - aA) > 2 * seD56 else "M4-A"
        else:
            ships = "M4-A"
        gates["M4"] = {"auc": aucA, "auc_control": aucC,
                       "delta_bar": 1.64 * seD, "pass": bool(pass4),
                       "variant": ships}

        fold_days = np.zeros(len(dates), dtype=bool)
        for f in tf:
            fold_days |= F.fold_date_mask(dates, f)
        wins, n_ep = 0, 0
        for e in eps["episodes"]:
            if not fold_days[e["start"]]:
                continue
            rws = np.arange(e["start"], min(e["end"] + 1, len(dates)))
            rws = rws[fold_days[rws]]
            if len(rws) == 0:
                continue
            pnl = float(np.nansum(mu5_all[rws]
                                  * np.nan_to_num(r1f[rws], nan=0.0)))
            n_ep += 1
            wins += int(pnl > 0)
        p5 = float(ss.binom.sf(wins - 1, n_ep, 0.5)) if n_ep else np.nan
        gates["M5"] = {"episodes": n_ep, "wins": wins, "p": p5,
                       "pass": bool(n_ep and p5 <= 0.05)}
        rot["gates"] = gates

        # ---- approximate C2 (pooled-only, 5 folds, no ladder) --------------
        # DEVIATION (ledgered, see designs/c2_gate_matrix.json
        # measurement_validity_finding): the registered G-book instrument
        # (TB-006 §6 long-only unit-gross books) saturates on the common
        # market factor (every pair ~0.9 incl. signal-rho~0.00 pairs), a
        # fold-INVARIANT artifact carrying no rotation-specific information.
        # Rotation demotion therefore keys on the signal space; book-space
        # breaches are recorded with the artifact flag, never silently
        # dropped. Disposition mirrors the production C2 escalation.
        roster = [m for m in ["M1", "M2", "M4", "M5"] if gates[m]["pass"]]
        c2 = {}
        demoted = []
        live = ["Z0"] + roster
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                gs_p, _, _ = _pair_signal_folds(members_sig[a],
                                                members_sig[b], sm, tf)
                gb_p, _ = _pair_book_folds(booksig[a], booksig[b], tf)
                bar = 0.8 if {a, b} == {"Z0", "M1"} else 0.7
                breach = bool(np.isfinite(gs_p) and abs(gs_p) > bar)
                book_breach = bool(np.isfinite(gb_p) and abs(gb_p) > bar)
                c2[f"{a}-{b}"] = {
                    "signal": gs_p, "book": gb_p, "bar": bar,
                    "breach": breach,
                    "book_breach_instrument_artifact": book_breach}
                if breach:
                    junior = max((a, b),
                                 key=lambda m: SENIORITY.index(m)
                                 if m in SENIORITY else -1)
                    if junior in roster:
                        demoted.append(junior)
        roster = [m for m in roster if m not in demoted]
        rot["c2_pooled_only"] = c2
        rot["c2_demotions"] = demoted
        rot["roster"] = {"directional": roster,
                         "M3": "gain" if gates["M3"]["pass"] else "B-disp",
                         "M4_variant": gates["M4"]["variant"],
                         "M5": "enabled" if gates["M5"]["pass"] else "disabled"}
        rotations[f"rotation_{r}"] = rot
        print(f"rotation {r}: roster {roster} M3="
              f"{rot['roster']['M3']} demotions {demoted}", flush=True)

    out = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
           "rule_text_provenance": "BUILD_SPEC_007 §2.3 + TOURNAMENT §4.4.1/"
                                   "§4.4.6 (frozen); 5/6-folds-positive "
                                   "transcribed as all-but-one of 5",
           "production_masks_frozen_phase0": PRODUCTION_MASKS,
           "shrinkage_sentence": "honest expected true core IC ~ 0.04-0.06 "
                                 "(inverse-Mills selection bound, G15)",
           "rotations": rotations}
    (PROTO / "store" / "rotation_masks_007.json").write_text(
        json.dumps(out, indent=1, default=float))
    GA.ledger({"kind": "build_artifact", "component": "crossfit_masks",
               "decision": "store/rotation_masks_007.json written",
               "provenance": "TOURNAMENT §4.4.6 frozen rule text"})
    GA.ledger({"kind": "deviation", "component": "crossfit_masks per-rotation C2",
               "decision": "rotation demotion keys on SIGNAL space; "
                           "book-space breaches recorded with "
                           "instrument-artifact flag (fold-invariant market-"
                           "beta loading of the long-only solo-book rule; "
                           "demoting on it zeroes every rotation roster "
                           "mechanically)",
               "provenance": "designs/c2_gate_matrix.json "
                             "measurement_validity_finding; mirrors the "
                             "production C2 escalation disposition"})

    # ---- EA contract files (ea_007.py header; wave-2a contract) -----------
    rot_dir = PROTO / "store" / "rotation_masks_007"
    rot_dir.mkdir(exist_ok=True)
    for r in range(1, 7):
        rot = rotations[f"rotation_{r}"]
        doc = {"rotation": r, "withheld_fold": r,
               "derived_from_folds": rot["training_folds"],
               "roster": rot["roster"]["directional"],
               "masks": rot["masks"],
               "roster_detail": rot["roster"],
               "c2_demotions": rot["c2_demotions"],
               "synthetic": False,
               "provenance": "re-derived from 5 training folds via the "
                             "frozen rule texts (BUILD_SPEC §2.3, "
                             "TOURNAMENT §4.4.1/§4.4.6)"}
        (rot_dir / f"rotation_{r}.json").write_text(
            json.dumps(doc, indent=1, default=float))

    # ---- production roster (mechanical map, TOURNAMENT §4.4.1, post-C2) ---
    gates_doc = json.loads(
        (PROTO / "gates" / "acceptance_007.json").read_text())
    passers = [m for m in ["M1", "M2", "M4", "M5"]
               if gates_doc["gates"][m]["PASS"]]
    c2_path = PROTO / "designs" / "c2_gate_matrix.json"
    c2_doc = json.loads(c2_path.read_text())
    prod_demoted = []
    if c2_doc["GATE"] != "PASS":
        # binding breaches demote the junior of each breaching pair only
        # after the remediation ladder is exhausted; ladder history is the
        # authority (each rung ledgered).
        for entry in c2_doc.get("remediation_ladder_history", []):
            if entry.get("action") == "demote" and entry.get("organ"):
                prod_demoted.append(entry["organ"])
    roster = [m for m in passers if m not in prod_demoted]
    roster_doc = {
        "roster": roster,
        "acceptance_passers": passers,
        "c2_gate": c2_doc["GATE"],
        "c2_demotions": prod_demoted,
        "M3": ("gain" if gates_doc["gates"]["M3"]["PASS"] else "B-disp"),
        "M4_variant": gates_doc["gates"]["M4"]["ab_decision"]["ships"],
        "M5": ("enabled" if gates_doc["gates"]["M5"]["PASS"]
               else "disabled"),
        "masks": "frozen Phase-0 production masks (G15/A9)",
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "provenance": "mechanical map: verdict-arm tilt roster = "
                      "{M1,M2,M4,M5} ∩ acceptance-gate passers, post-C2 "
                      "(TOURNAMENT §4.4.1)"}
    (PROTO / "store" / "roster_007.json").write_text(
        json.dumps(roster_doc, indent=1, default=float))
    GA.ledger({"kind": "build_artifact", "component": "crossfit_masks",
               "decision": f"EA contract files written: 6 rotation files + "
                           f"roster_007.json (roster={roster}, "
                           f"demotions={prod_demoted})",
               "provenance": "TOURNAMENT §4.4.1 mechanical map + §4.4.6"})
    print(f"production roster: {roster} (passers {passers}, "
          f"c2 {c2_doc['GATE']}, demotions {prod_demoted})", flush=True)


def _pair_signal_folds(ma, mb, sm, folds):
    from scipy.stats import spearmanr
    vals = []
    inc_n, tot_n = 0, 0
    for f in folds:
        A, B = ma[f], mb[f]
        inc = A["expressive"] & B["expressive"]
        tot_n += len(inc)
        inc_n += int(inc.sum())
        for j in np.nonzero(inc)[0]:
            i = A["idx"][j]
            m = sm[i] & np.isfinite(A["mu_std"][j]) & np.isfinite(B["mu_std"][j])
            if m.sum() >= 8 and A["mu_std"][j][m].std() > 1e-9 \
                    and B["mu_std"][j][m].std() > 1e-9:
                v = spearmanr(A["mu_std"][j][m], B["mu_std"][j][m]).statistic
                if np.isfinite(v):
                    vals.append(v)
    pooled = float(np.mean(vals)) if vals else np.nan
    return pooled, len(vals), (inc_n / tot_n if tot_n else np.nan)


def _pair_book_folds(ba, bb, folds):
    aa, bbv = [], []
    for f in folds:
        A, B = ba[f], bb[f]
        inc = A["expressive"] & B["expressive"] & np.isfinite(A["ret"]) \
            & np.isfinite(B["ret"])
        aa.append(A["ret"][inc])
        bbv.append(B["ret"][inc])
    a = np.concatenate(aa)
    b = np.concatenate(bbv)
    if len(a) < 20 or a.std() == 0 or b.std() == 0:
        return np.nan, len(a)
    return float(np.corrcoef(a, b)[0, 1]), len(a)


if __name__ == "__main__":
    main()
