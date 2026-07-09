"""PKT-TB-007 — expression-layer unit tests (BUILD_SPEC_007 §8 step 2 gate):
projection math (incl. infeasible-tilt clipping), neutral-recovery OBJECT
identity, lot-fix assertions (both sites), tier-cap enforcement, no-REDUCE,
holdout-guard refusal.
"""
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROTO = Path(__file__).resolve().parents[1]
REPO = PROTO.parents[2]
for p in (str(PROTO), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from genome_007 import Genome007                                     # noqa: E402
from lot_fix_007 import aggregate_book, _execute_intents_lotfix      # noqa: E402
import run_replay_007 as RR                                          # noqa: E402
import tilt_adapter as TA                                            # noqa: E402
from chassis.utils.three_line_replay.replay_engine import Portfolio, Position  # noqa: E402
from chassis.utils.three_line_replay.strategies import StrategyContext   # noqa: E402

D = "2026-02-10"          # a real pre-holdout date (RiskStats uses the ohlcv cache)


# ================================================================ projection math
def _rand_case(seed, n=14):
    rng = np.random.default_rng(seed)
    d = rng.normal(0, 0.01, n)
    beta = rng.uniform(-0.2, 1.4, n)
    sigma = rng.uniform(0.004, 0.03, n)
    lb = -rng.uniform(0.005, 0.03, n)
    ub = rng.uniform(0.005, 0.03, n)
    return d, beta, sigma, lb, ub


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 42])
def test_projection_satisfies_all_three_constraints(seed):
    d, beta, sigma, lb, ub = _rand_case(seed)
    x, info = TA.project_parity(d, beta, sigma, lb, ub)
    tol = 1e-9 + 1e-6 * np.abs(x).sum()
    assert abs(x.sum()) <= tol or info["infeasible"]
    assert abs(x @ beta) <= tol or info["infeasible"]
    eps = TA.EPS_SIGMA_FRAC * float(np.abs(x) @ sigma)
    assert abs(x @ sigma) <= eps + tol
    assert (x >= lb - 1e-12).all() and (x <= ub + 1e-12).all()


def test_projection_infeasible_tilt_is_clipped_to_caps():
    """A huge one-name desired tilt gets clipped to its cap while the
    constraints still hold on what remains."""
    n = 8
    d = np.zeros(n); d[0] = 0.20                       # 20% NAV desired on one name
    beta = np.array([1.0, 1.1, 0.9, 0.2, 0.1, 1.0, 0.5, 0.3])
    sigma = np.full(n, 0.01)
    lb = np.full(n, -0.0125); ub = np.full(n, 0.0125)
    x, info = TA.project_parity(d, beta, sigma, lb, ub)
    assert x[0] <= 0.0125 + 1e-12                      # cap binds
    assert 0 in info["clipped"]
    tol = 1e-9 + 1e-6 * np.abs(x).sum()
    assert abs(x.sum()) <= tol
    assert abs(x @ beta) <= tol


def test_projection_degenerate_beta_drops_row():
    n = 5
    d = np.array([0.01, -0.004, 0.002, -0.005, -0.003])
    beta = np.ones(n)                                  # collinear with ones
    sigma = np.full(n, 0.01)
    x, info = TA.project_parity(d, beta, sigma, np.full(n, -1.0), np.full(n, 1.0))
    assert info.get("beta_row_dropped")
    assert abs(x.sum()) < 1e-9


def test_projection_zero_input():
    x, info = TA.project_parity(np.zeros(4), np.ones(4) * 0.5, np.full(4, 0.01),
                                np.full(4, -1.0), np.full(4, 1.0))
    assert np.allclose(x, 0) and not info["infeasible"]


# ================================================================ fixtures
def _lot(sym, shares, entry, **kw):
    return Position(symbol=sym, shares=float(shares), entry_price=float(entry),
                    entry_date="2026-01-31", peak_price=float(entry),
                    asset_class=kw.get("asset_class", "equity"),
                    sector=kw.get("sector", "broad"),
                    leverage_flag=kw.get("leverage_flag", 0))


SUPPORT_MARKS = {"SCHD": 25.0, "XLB": 90.0, "VLUE": 110.0, "VEA": 55.0,
                 "ITA": 145.0, "SOXX": 230.0, "XRT": 80.0, "TLT": 91.0,
                 "AGG": 98.0, "MUB": 106.0, "FXE": 103.0, "USO": 80.0,
                 "FXI": 38.0, "RSP": 180.0, "IYR": 95.0, "SHY": 82.0,
                 "KRE": 60.0}


def _features_df():
    rows = [{"symbol": s, "date": "2026-02-09", "close": px}
            for s, px in SUPPORT_MARKS.items()]
    return pd.DataFrame(rows)


def _portfolio():
    return Portfolio(cash=100_000.0,
                     positions=[_lot("SCHD", 737, 24.6), _lot("XLB", 446, 88.0),
                                _lot("VLUE", 149, 108.0), _lot("VEA", 332, 54.0)])


def _ctx(intents_unused=None):
    return StrategyContext(
        inputs_date=D, portfolio=_portfolio(),
        variant_config={"decision_params": {"min_order_dollars": 250}},
        expert_signals={}, expert_metrics={}, decisions={}, panic_streak=0,
        last_regime=None, features_df=_features_df(), inference={}, llm_risks={})


def _organ_file(tmp, date=D, gain=0.6):
    mu1 = {"ITA": 1.4, "SOXX": 1.1, "USO": 0.8, "FXI": 0.5, "RSP": 0.2,
           "XRT": -0.2, "TLT": -0.9, "AGG": -0.6, "MUB": -0.4, "FXE": 0.1,
           "IYR": -0.3, "SHY": -0.7, "KRE": 0.3}
    mu2 = {k: -v * 0.5 + 0.1 for k, v in mu1.items()}
    payload = {"date": date, "schema_version": "organ_inputs_007.v1",
               "organs": {"M1": {"mu": mu1, "q": gain},
                          "M2": {"mu": mu2, "q": 0.5}},
               "disp_z": 0.8, "p_exceed": {}, "manifest": {"synthetic": True}}
    nd = Path(tmp) / "nightly"
    nd.mkdir(parents=True, exist_ok=True)
    (nd / f"{date}.json").write_text(json.dumps(payload))
    return nd


def _strategy(tmp, genome, log=True):
    return TA.make_tilt_strategy(genome, _organ_file(tmp),
                                 log_dir=(Path(tmp) / "out") if log else None)


ACTIVE_GENOME = dict(organ_trust={"M1": 0.5, "M2": -0.5}, tilt_gain=0.6,
                     conviction_temp=1.0, dead_zone=0.05, disp_gain=0.5,
                     cap_core=0.0125, cap_conditional=0.00625,
                     defensive_fraction=0.25, event_damp_strength=0.0)


# ================================================================ neutral recovery
def test_neutral_recovery_b0_returns_same_object(tmp_path):
    strat = _strategy(tmp_path, Genome007.b0(("M1", "M2")))
    intents = [{"symbol": "ITA", "action": "BUY", "shares": 10, "dollars": 1450.0,
                "price": 145.0, "reason": "SCORE_0.70"}]
    snapshot = deepcopy(intents)
    out = strat.post_decision(_ctx(), intents)
    assert out is intents                              # SAME list object
    assert out == snapshot                             # nothing mutated
    assert out[0] is intents[0]                        # same dicts
    log = json.loads((tmp_path / "out" / "expression_log" / f"{D}.json").read_text())
    assert log["neutral"] is True
    assert log["neutral_reason"] == "tilt_gain_zero"


def test_neutral_recovery_no_organ_file_returns_same_object(tmp_path):
    strat = TA.make_tilt_strategy(Genome007.from_dict(ACTIVE_GENOME),
                                  tmp_path / "empty_nightly",
                                  log_dir=tmp_path / "out")
    intents = []
    out = strat.post_decision(_ctx(), intents)
    assert out is intents


def test_neutral_recovery_dead_zone_quantized(tmp_path):
    """Conviction alive but the tilt too small to clear min_order on a small
    NAV => dead_zone_quantized, same object back."""
    g = dict(ACTIVE_GENOME, tilt_gain=0.01)            # ~0.04% NAV one-sided
    strat = _strategy(tmp_path, Genome007.from_dict(g))
    intents = [{"symbol": "FXI", "action": "BUY", "shares": 30, "dollars": 1140.0,
                "price": 38.0, "reason": "SCORE_0.66"}]
    out = strat.post_decision(_ctx(), intents)
    assert out is intents
    log = json.loads((tmp_path / "out" / "expression_log" / f"{D}.json").read_text())
    assert log["neutral_reason"] in ("dead_zone_quantized", "dead_zone")


# ================================================================ non-neutral path
def _run_active(tmp_path, genome_overrides=None, intents=None):
    g = dict(ACTIVE_GENOME, **(genome_overrides or {}))
    strat = _strategy(tmp_path, Genome007.from_dict(g))
    intents = intents if intents is not None else [
        {"symbol": "ITA", "action": "BUY", "shares": 20, "dollars": 2900.0,
         "price": 145.0, "reason": "SCORE_0.71"}]
    out = strat.post_decision(_ctx(), intents)
    log = json.loads((tmp_path / "out" / "expression_log" / f"{D}.json").read_text())
    return intents, out, log


def test_active_tilt_changes_intents_and_satisfies_projection(tmp_path):
    intents, out, log = _run_active(tmp_path)
    assert out is not intents
    assert log["neutral"] is False
    tilt = log["tilt"]
    assert abs(sum(tilt.values())) < 1e-6              # Sum dw ~ 0
    assert abs(log["projection"]["beta_resid"]) < 1e-6
    assert abs(log["projection"]["sigma_resid"]) <= log["projection"]["eps_sigma"] + 1e-9
    one_sided = sum(abs(v) for v in tilt.values()) / 2
    assert one_sided <= TA.T_MAX + 1e-9                # turnover bounded
    assert "VIXY" not in tilt
    # incumbent dicts not mutated
    assert intents[0]["dollars"] == 2900.0


def test_tier_caps_enforced(tmp_path):
    _, _, log = _run_active(tmp_path, {"tilt_gain": 1.0, "cap_core": 0.004,
                                       "cap_conditional": 0.002})
    held = {"SCHD", "XLB", "VLUE", "VEA"}
    for sym, v in log["tilt"].items():
        if sym in TA.TILT_CORE:
            cap = min(0.004, TA.FXE_HARD_CAP) if sym == "FXE" else 0.004
            assert abs(v) <= cap + 1e-9, sym
        elif sym in TA.TILT_COND:
            assert abs(v) <= 0.002 + 1e-9, sym
        else:
            assert sym in held, f"ballast tilt on non-held {sym}"


def test_defensive_fraction_clip(tmp_path):
    _, _, log = _run_active(tmp_path, {"defensive_fraction": 0.10})
    assert log["budget"]["defensive_share_post"] <= 0.10 + 0.02


def test_no_reduce_ever_and_passthrough_of_chassis_reduce(tmp_path):
    chassis = [
        {"symbol": "ITA", "action": "BUY", "shares": 20, "dollars": 2900.0,
         "price": 145.0, "reason": "SCORE_0.71"},
        {"symbol": "VEA", "action": "REDUCE", "shares": 332, "price": 55.0,
         "reason": "EXPOSURE_TRIM"}]
    reduce_obj = chassis[1]
    intents, out, log = _run_active(tmp_path, intents=chassis)
    added = [it for it in out if it not in chassis]
    assert all(it["action"] in ("BUY", "SELL") for it in added)   # never REDUCE
    assert reduce_obj in out                                      # untouched object
    assert reduce_obj == {"symbol": "VEA", "action": "REDUCE", "shares": 332,
                          "price": 55.0, "reason": "EXPOSURE_TRIM"}


def test_integer_shares_and_min_order(tmp_path):
    _, out, _ = _run_active(tmp_path)
    for it in out:
        if it.get("reason", "").startswith("ORB1_TILT"):
            assert float(it["shares"]) == int(float(it["shares"]))
            assert it["dollars"] >= 250.0 - 1e-9


def test_expression_log_attribution_present(tmp_path):
    _, _, log = _run_active(tmp_path)
    assert set(log["organ_attribution"]) <= {"M1", "M2"}
    assert log["organ_attribution"], "which-organ-moved-which-name log missing"
    assert (tmp_path / "out" / "expression_log.jsonl").exists()


def test_fully_sold_names_get_no_tilt(tmp_path):
    chassis = [{"symbol": "SCHD", "action": "SELL", "shares": 737,
                "price": 25.0, "reason": "HEALTH_COLLAPSE"}]
    _, out, log = _run_active(tmp_path, intents=chassis)
    assert log["tilt"].get("SCHD", 0.0) == 0.0


# ================================================================ adapter lot site
def test_adapter_lot_aggregation_site(tmp_path):
    """C5 site 1 inside the adapter path: a 2-lot SCHD book is seen whole."""
    pf = Portfolio(cash=10_000.0,
                   positions=[_lot("SCHD", 738, 24.0), _lot("SCHD", 738, 26.0)])
    held, w_prev, nav, _ = aggregate_book(pf.positions, {"SCHD": 25.0}, pf.cash)
    assert held["SCHD"] == 1476.0
    assert w_prev["SCHD"] == pytest.approx(1476 * 25 / nav)


def test_harness_lot_fix_site():
    """C5 site 2: multi-lot full exit through the patched executor."""
    pf = Portfolio(cash=0.0,
                   positions=[_lot("SCHD", 738, 24.0), _lot("SCHD", 738, 26.0)])
    executed = _execute_intents_lotfix(
        pf, [{"symbol": "SCHD", "action": "SELL", "shares": 1476, "price": 25.0,
              "reason": "X"}],
        {"SCHD": {"open": 25.0, "close": 25.0}},
        {"min_order_dollars": 250}, D)
    assert executed[0]["shares"] == 1476.0
    assert pf.positions == []


# ================================================================ holdout guard
def test_holdout_guard_refuses_live_and_holdout_without_env():
    for window in ("live", "holdout"):
        with pytest.raises(SystemExit):
            RR.check_holdout_authorization(window, env={})
    RR.check_holdout_authorization("preholdout", env={})          # never guarded
    RR.check_holdout_authorization("live", env={RR.ENV_FLAG: "1"})


def test_genome_b0_and_ranges():
    b0 = Genome007.b0(("M1", "M2", "M5"))
    assert b0.is_b0 and b0.tilt_gain == 0.0
    with pytest.raises(ValueError):
        Genome007.from_dict({"tilt_gain": 1.5})
    with pytest.raises(ValueError):
        Genome007.from_dict({"organ_trust": {"M9": 0.0}})
