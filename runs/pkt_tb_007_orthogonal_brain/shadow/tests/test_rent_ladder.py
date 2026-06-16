"""PKT-TB-011 attribution-spine gate tests.

Proves the four Analyst conditions the rent ladder must discharge:
  C6  multi-factor exposure strip (SPY+duration+commodity) + realized
      gross-differential term recovers selection rent free of a planted
      exposure leak (a static beta would miss it).
  C7  per-rung divergence ledger + stripped-column caveats; non-strippable
      (regime) rung labeled; ladder order-dependence disclosed.
  C8  ladder additivity exact in log-return space; bp/day residual printed
      and rung-sums labeled approximate.
  C9  BH-FDR(10%) family INCLUDES the M4 sub-window; three-valued verdict
      with the +/-1.5 bp materiality band; "1 of N" badge.

Pure-math unit tests over synthetic series -- no parquet / network.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

SHADOW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHADOW))

import shadow_lib as SL                                       # noqa: E402


# ---------------------------------------------------------------- helpers
def navs_from_returns(start: float, rets):
    """Build a NAV series from daily simple returns."""
    out = [start]
    for r in rets:
        out.append(out[-1] * (1.0 + r))
    return out


def _det_factor(n, amp, mean, phase=0.0, freq=0.7):
    return [mean + amp * math.sin(freq * i + phase) for i in range(n)]


# ==================================================================== C6
def test_c6_multifactor_strip_recovers_alpha_removing_exposure_leak():
    n = 60                                           # distinct freqs => identifiable
    r_mkt = _det_factor(n, 0.010, 0.0020, 0.0, freq=0.70)   # nonzero mean => leak
    r_dur = _det_factor(n, 0.006, -0.0010, 1.1, freq=0.23)
    r_com = _det_factor(n, 0.008, 0.0015, 2.0, freq=1.30)
    alpha = 0.5                                      # true selection rent bp/day
    b_mkt, b_dur, b_com = 80.0, -30.0, 20.0          # planted realized betas
    gross = [alpha + b_mkt * r_mkt[i] + b_dur * r_dur[i] + b_com * r_com[i]
             for i in range(n)]
    factor_daily = {"mkt": r_mkt, "duration": r_dur, "commodity": r_com}

    strip = SL.multifactor_strip(gross, factor_daily)
    # all three factors used (multi-factor, not SPY-only)
    assert set(strip["factors_used"]) == {"mkt", "duration", "commodity"}
    # realized betas recovered
    assert abs(strip["betas"]["mkt"] - b_mkt) < 1e-2
    assert abs(strip["betas"]["duration"] - b_dur) < 1e-2
    assert abs(strip["betas"]["commodity"] - b_com) < 1e-2
    # gross is inflated by the one-signed exposure leak; stripped recovers alpha
    assert abs(strip["stripped_bp_day"] - alpha) < 1e-2
    assert abs(strip["gross_diff_bp_day"] - strip["stripped_bp_day"]) > 0.05
    # the exposure term explains the gap (gross = stripped + exposure)
    assert abs(strip["gross_diff_bp_day"]
               - (strip["stripped_bp_day"] + strip["exposure_bp_day"])) < 1e-6


def test_c6_static_single_factor_would_miss_the_sleeve():
    """A SPY-only strip leaves the duration/commodity sleeve in 'stripped'."""
    n = 60
    r_mkt = _det_factor(n, 0.010, 0.0020, 0.0, freq=0.70)
    r_dur = _det_factor(n, 0.006, 0.0040, 1.1, freq=0.23)   # big one-signed dur leak
    gross = [0.5 + 80.0 * r_mkt[i] - 60.0 * r_dur[i] for i in range(n)]
    spy_only = SL.multifactor_strip(gross, {"mkt": r_mkt})
    full = SL.multifactor_strip(gross, {"mkt": r_mkt, "duration": r_dur})
    # SPY-only strip still carries the duration sleeve -> wrong alpha;
    # the full multi-factor strip recovers the true 0.5 bp.
    assert abs(full["stripped_bp_day"] - 0.5) < 1e-2
    assert abs(spy_only["stripped_bp_day"] - 0.5) > 0.1


# ==================================================================== C8
def _ladder_series(n=48):
    # five nested books with genuinely divergent paths
    base = [0.001 * math.sin(0.5 * i) for i in range(n)]
    I = navs_from_returns(100000.0, base)
    R = navs_from_returns(100000.0, [b + 0.00002 * math.cos(i) for i, b in enumerate(base)])
    F = navs_from_returns(100000.0, [b + 0.00010 + 0.00003 * math.sin(i) for i, b in enumerate(base)])
    E = navs_from_returns(100000.0, [b + 0.00012 + 0.00002 * math.cos(0.3 * i) for i, b in enumerate(base)])
    U = navs_from_returns(100000.0, [b + 0.00015 + 0.00001 * math.sin(0.2 * i) for i, b in enumerate(base)])
    dates = [f"2026-{7 + i // 28:02d}-{1 + i % 28:02d}" for i in range(n + 1)]
    return dates, {"I": I, "R": R, "F": F, "E": E, "U": U}


def test_c8_log_additivity_exact_and_bp_residual_printed():
    dates, series = _ladder_series()
    st = SL.compute_ladder_stats([], dates, series)
    add = st["additivity"]
    # log-space rungs telescope EXACTLY
    assert add["log_total_U_minus_I_bp"] is not None
    assert abs(add["log_residual_bp"]) < 1e-6
    assert abs(add["sum_log_rungs_bp"] - add["log_total_U_minus_I_bp"]) < 1e-6
    # bp/day residual is PRINTED (not hidden) and rung-sums labeled approximate
    assert "bp_day_residual" in add and add["bp_day_residual"] is not None
    assert "APPROXIMATE" in add["bp_day_note"]
    assert "exact" in add["log_note"]


# ==================================================================== C7
def test_c7_divergence_ledger_caveats_and_order_disclosure():
    dates, series = _ladder_series()
    st = SL.compute_ladder_stats([], dates, series, selected_universe_active=False)
    dv = st["divergence"]
    assert dv["order"] == ["regime", "forecast", "event", "universe"]
    assert "order" in dv["order_dependence_note"].lower()
    for comp in ("regime", "forecast", "event", "universe"):
        assert comp in dv["per_rung_path_bp"]
    by_comp = {r["component"]: r for r in st["organ_ledger"]}
    # regime rung is an exposure move -> not fully strippable, labeled
    assert by_comp["regime"]["strippable"] is False
    assert "exposure-strip incomplete" in by_comp["regime"]["caveat"]
    # universe rung not yet wired to the live engine -> labeled
    assert "not yet wired" in by_comp["universe"]["caveat"]
    # every rung carries gross AND stripped columns
    for r in st["organ_ledger"]:
        assert "gross_bp_day" in r and "stripped_bp_day" in r


# ==================================================================== C9
def test_c9_three_valued_verdict_band():
    assert SL.three_valued_verdict(3.0, [1.0, 5.0], 3.5) == "positive"
    z = SL.three_valued_verdict(0.1, [-1.0, 1.2], 0.3)
    assert z == "zero (measured at materiality scale)"
    ind = SL.three_valued_verdict(2.0, [-1.0, 5.0], 1.2)
    assert ind == "indeterminate at available power"
    assert SL.three_valued_verdict(None, None, None) == \
        "indeterminate at available power"


def test_c9_bh_fdr_basic():
    # classic BH: p=.001 survives at q=.10, large p's do not
    pv = [0.001, 0.20, 0.50, 0.80, None]
    surv = SL.bh_fdr(pv, q=0.10)
    assert surv[0] is True
    assert surv[1] is False and surv[2] is False
    assert surv[4] is False                          # None never survives


def test_c9_m4_subwindow_in_fdr_family():
    dates, series = _ladder_series()
    st = SL.compute_ladder_stats([], dates, series)
    # M4 sub-window read exists, is flagged an FDR family member
    assert st["m4_subwindow"]["in_fdr_family"] is True
    assert "m4_subwindow" in st["fdr"]["family"]
    # the family = 4 stripped rungs + the M4 sub-window
    assert st["fdr"]["family"] == ["regime", "forecast", "event",
                                   "universe", "m4_subwindow"]
    # every organ row carries an FDR survivor flag + p
    for r in st["organ_ledger"]:
        assert "fdr_survivor" in r and "fdr_p" in r


def test_c9_m4_subwindow_validity_rule():
    # base tilt (F-I) strictly positive everywhere -> a valid sub-window exists
    n = 30
    base = [0.001] * n
    I = navs_from_returns(100000.0, [0.0] * n)
    F = navs_from_returns(100000.0, base)
    R = list(I)
    # E above F by a steady margin -> a measurable E-F marginal
    E = navs_from_returns(100000.0, [b + 0.0002 for b in base])
    U = list(E)
    dates = [f"2026-08-{1 + i:02d}" for i in range(n + 1)]
    sub = SL._m4_subwindow(dates, {"I": I, "R": R, "F": F, "E": E, "U": U})
    assert sub["valid"] is True
    assert sub["n"] >= 3

    # base tilt negative everywhere -> NO valid sub-window (confound excluded)
    Fneg = navs_from_returns(100000.0, [-0.001] * n)
    sub2 = SL._m4_subwindow(dates, {"I": I, "R": R, "F": Fneg,
                                    "E": E, "U": U})
    assert sub2["valid"] is False


def test_c9_one_of_n_badge_on_lone_positive():
    # craft a ladder where exactly one rung is strongly positive (forecast)
    n = 80
    flat = [0.0] * n
    I = navs_from_returns(100000.0, flat)
    R = navs_from_returns(100000.0, flat)
    F = navs_from_returns(100000.0, [0.0005] * n)     # steady +5bp/day rung
    E = navs_from_returns(100000.0, [0.0005] * n)     # E-F ~ 0
    U = navs_from_returns(100000.0, [0.0005] * n)     # U-E ~ 0
    dates = [f"2026-{7 + i // 28:02d}-{1 + i % 28:02d}" for i in range(n + 1)]
    st = SL.compute_ladder_stats([], dates, {"I": I, "R": R, "F": F,
                                             "E": E, "U": U})
    by = {r["component"]: r for r in st["organ_ledger"]}
    assert by["forecast"]["verdict"] == "positive"
    # lone positive that hasn't survived FDR carries the 1-of-N badge
    if not by["forecast"].get("fdr_survivor"):
        assert "1 of N" in by["forecast"].get("badge", "")
