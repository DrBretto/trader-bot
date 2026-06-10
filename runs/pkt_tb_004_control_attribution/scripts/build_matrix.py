"""Build ATTRIBUTION_MATRIX.md from cells/*/stats.json (+ canon_confirm/*.json).

Column blocks per the Methodologist: [O] full-period, [O] holdout, [C] holdout
(gross-of-costs). Delta convention: cell - C00; a layer's contribution = -delta.
"""
from __future__ import annotations

import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
RUN_DIR = os.path.join(REPO, 'runs', 'pkt_tb_004_control_attribution')

LAYER_NAMES = {
    'T1-A': 'Vol-bucket size adjust (vol_adj)',
    'T1-B': 'Regime size adjust (regime_adj)',
    'T1-LLM': 'LLM layer (size adj + buy veto + sell veto)',
    'T1-D': 'Ensemble disagreement multiplier',
    'T1-E': 'Expert modifier psm (size channel)',
    'T1-F': 'Risk throttle (throttle_scale)',
    'T1-G': 'Regime compatibility score multiplier',
    'T1-H': 'Regime-conditional buy thresholds',
    'T1-I': 'Regime-conditional cash floors',
    'T1-J': 'Regime label steering (fusion label flips)',
    'T1-K': 'Trailing stop',
    'T1-L': 'Health-collapse sell',
    'T1-M': 'Panic force-sell',
    'T1-O': 'Leverage hold cap',
    'T1-P': 'Reduce-on-health-drop',
    'T1-Q': 'Reduce-on-regime-shift',
    'T1-S': 'Vol-bucket buy filter',
    'T1-T': 'Panic asset-class buy filter',
    'T2-ALLOFF': 'ENTIRE CONTROL STACK (all-off bracket)',
}
ORDER = ['T1-A', 'T1-B', 'T1-LLM', 'T1-D', 'T1-E', 'T1-F', 'T1-G', 'T1-H',
         'T1-I', 'T1-J', 'T1-K', 'T1-L', 'T1-M', 'T1-O', 'T1-P', 'T1-Q',
         'T1-S', 'T1-T', 'T2-ALLOFF']


def f(x, nd=2, pct=False, signed=True):
    if x is None:
        return '—'
    v = x * 100 if pct else x
    s = f'{v:+.{nd}f}' if signed else f'{v:.{nd}f}'
    return s


def main():
    stats = {}
    for cid in ORDER:
        p = os.path.join(RUN_DIR, 'cells', cid, 'stats.json')
        if os.path.exists(p):
            stats[cid] = json.load(open(p))

    canon = {}
    cdir = os.path.join(RUN_DIR, 'canon_confirm')
    if os.path.isdir(cdir):
        for fn in os.listdir(cdir):
            if fn.endswith('.json'):
                d = json.load(open(os.path.join(cdir, fn)))
                canon[d['cell_id']] = d
    canon_base = canon.get('C00')

    with open(os.path.join(RUN_DIR, 'trigger_evaluation.json')) as fo:
        trig = json.load(fo)

    L = []
    L.append('# PKT-TB-004 — Attribution Matrix\n')
    L.append('Delta convention: **Δ = ablated-cell − C00 baseline** (layer OFF minus layer ON).')
    L.append('A layer\'s contribution to the strategy = **−Δ**: negative Δreturn ⇒ the layer')
    L.append('was adding return; positive Δreturn ⇒ removing the layer would have helped.')
    L.append('ΔmaxDD positive ⇒ removal made drawdown shallower (the guard was not earning its DD claim).\n')
    L.append('Baseline C00 = champion-as-deployed (active params, ranking blend 0.35) with stored')
    L.append('nightly LLM risks wired (`use_stored_llm_risks=true`). Harness [O] = optimizer replay,')
    L.append('from $100k cash, costed, 5 seeds {11,17,23,29,31}, median-seed values shown,')
    L.append('194 trading days 2025-08-05→2026-06-09 (valuation dates). Harness [C] = three-line')
    L.append('canon continuation from the real 2026-03-11 book, champion overlays, gross-of-costs,')
    L.append('deterministic. LLM rows: [O] full-period reads restricted to the LLM era (2026-01-29→).\n')
    L.append('**Gate** (pre-registered): paired daily |t|≥2.0 AND NW(5) |t|≥1.8 AND sign-consistent')
    L.append('across 5 seeds AND |Δret| > B (analytic slippage-noise bound). `MEANINGFUL` only if all four.\n')

    hdr = ('| Cell | Layer | Δret | ΔmaxDD | ΔSharpe | Δexp | Δtrades | t | NW-t | B(bps) | sign5 | gate |')
    sep = ('|---|---|---|---|---|---|---|---|---|---|---|---|')

    for seg_key, title in (('full', '[O] Full period (2025-08-05 → 2026-06-09)'),
                           ('holdout', '[O] Holdout only, from cash (2026-03-11 →)')):
        L.append(f'\n## {title}\n')
        L.append(hdr)
        L.append(sep)
        for cid in ORDER:
            st = stats.get(cid)
            if not st:
                continue
            s = st[seg_key]
            mc, mb, p = s['metrics_cell'], s['metrics_base'], s['paired']
            dret = mc['total_return'] - mb['total_return']
            ddd = mc['max_drawdown'] - mb['max_drawdown']
            dsh = (None if None in (mc['sharpe'], mb['sharpe'])
                   else mc['sharpe'] - mb['sharpe'])
            dexp = mc['exposure'] - mb['exposure']
            dtr = sum(mc['trades'].values()) - sum(mb['trades'].values())
            gate = st['decision_gate']
            note = 'LLM-era' if st.get('llm_era_restricted') else ''
            L.append(
                f'| {cid} | {LAYER_NAMES[cid]}{" *(" + note + ")*" if note and seg_key == "full" else ""} '
                f'| {f(dret, 2, True)}pp | {f(ddd, 2, True)}pp | {f(dsh, 2)} '
                f'| {f(dexp, 3)} | {dtr:+d} | {f(p.get("t"), 2)} | {f(p.get("nw5_t"), 2)} '
                f'| {st["seed_noise_bound_bps"]:.1f} | {"Y" if s.get("sign_consistent") else "N"} '
                f'| {"**MEANINGFUL**" if (seg_key == "full" and gate["meaningful"]) else ("agrees" if seg_key == "holdout" and gate.get("holdout_sign_agrees") else "ns")} |'
            )

    L.append('\n## [C] Holdout, canon-line continuation (gross-of-costs, deterministic)\n')
    if canon_base:
        L.append('| Cell | Layer | Δ final value | Δret (pp of book) | direction vs [O] holdout |')
        L.append('|---|---|---|---|---|')
        v0 = list(canon_base['date_value_map'].values())[0]
        base_fv = canon_base['final_value']
        for cid in ORDER:
            c = canon.get(cid)
            st = stats.get(cid)
            if not c or cid == 'C00':
                continue
            dfv = c['final_value'] - base_fv
            dret_c = dfv / v0
            agree = '—'
            if st:
                do = st['delta_total_return_holdout']
                if do == 0 and abs(dret_c) < 0.001:
                    agree = 'both ~0'
                elif do * dret_c > 0:
                    agree = 'CONFIRMS'
                elif do * dret_c < 0:
                    agree = 'CONTRADICTS'
                else:
                    agree = 'one ~0'
            L.append(f'| {cid} | {LAYER_NAMES[cid]} | {f(dfv, 0)}$ | {f(dret_c, 2, True)}pp | {agree} |')
    else:
        L.append('_canon_confirm runs pending_')

    L.append('\n## Bracket & triggers\n')
    s1 = trig.get('S1') or {}
    L.append(f'- Interaction mass R = Σ-residual of the all-off bracket vs sum of LOO deltas: '
             f'**{s1.get("R_bps_day", "—")} bps/day** (pre-holdout), Σ|δᵢ| = {s1.get("sum_abs")} bps/day, '
             f'SE(R) = {s1.get("SE_R")} → S1 trigger {"FIRED" if s1.get("fired") else "not fired"}.')
    t1b = trig.get('tier1b') or {}
    L.append(f'- Tier-1b (LLM channel split): parent T1-LLM t = {t1b.get("parent_t_full_llm_era")} (LLM-era) / '
             f'{t1b.get("parent_t_holdout")} (holdout) → {"FIRED" if t1b.get("fired") else "NOT fired; T1b-C/N/R = SKIPPED-PARENT-NULL"}.')
    L.append(f'- S2 wrong-sign trigger: {len(trig.get("S2") or [])} cells. S3 instability: {len(trig.get("S3") or [])} cells.')
    L.append('- Tier-3 pairwise cells run: **0** (no trigger fired).')

    L.append('\n## Substrate caveats (carry into every read of this matrix)\n')
    L.append('1. **2025 inference artifacts are backfilled one-hot heuristics** (S3 timestamps 2026-03-18),')
    L.append('   not live model output; real GRU+Transformer output exists only from 2026-01-31.')
    L.append('   Full-period reads of regime/health-conditioned layers partially reflect that backfill.')
    L.append('2. Endpoint deltas are sign-stable across seeds and ≫ B for several layers while paired daily')
    L.append('   t-stats stay <2: the effects are path-shaped (a changed buy compounds forever), not')
    L.append('   daily-mean-shaped. The pre-registered gate treats these as NOT MEANINGFUL; endpoint')
    L.append('   magnitudes are reported for honesty, not as proof.')
    L.append('3. From-cash holdout runs are not the canon book; [C] block is the on-book read (gross of costs).')
    L.append('4. One market path; ~10 months; no cross-path resampling. See Skeptic section of the')
    L.append('   committee report for what these ablations cannot conclude.')

    out = os.path.join(RUN_DIR, 'ATTRIBUTION_MATRIX.md')
    with open(out, 'w') as fo:
        fo.write('\n'.join(L) + '\n')
    print('wrote', out)


if __name__ == '__main__':
    main()
