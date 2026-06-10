"""PKT-TB-006 — attribution battery tests (battery/stats/e1_reads/report_evidence).

Usage:  .venv/bin/python -m pytest tests/test_battery.py -v

Everything here runs on SYNTHETIC/smoke series only — zero replays, zero
holdout looks (the budget/ledger tests use throwaway tmp ledgers, never
prototype/holdout_looks.jsonl).
"""
from __future__ import annotations

import csv
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

PROTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROTO))

import battery                # noqa: E402
import e1_reads as e1         # noqa: E402
import ea                     # noqa: E402
import executive as ex        # noqa: E402
import report_evidence as rev # noqa: E402
import stats as st            # noqa: E402
import synth_oof as so        # noqa: E402


# ============================ §4.3 organ-verdict truth table ===================
@pytest.mark.parametrize("t,e2,gate_off,tag", [
    # gate-honest dominates everything
    (3.5, 1.0, True, "0 (gate-honest)"),
    (None, None, True, "0 (gate-honest)"),
    # positive: t >= +2.0 AND E2 sign >= 0
    (2.0, 0.0, False, "positive"),            # boundary t = 2.0, E2 = 0 exactly
    (2.5, 0.3, False, "positive"),
    # t >= 2 with E2 disagreement -> indeterminate
    (2.5, -0.1, False, "indeterminate"),
    (2.0, None, False, "indeterminate"),      # E2 unavailable
    # negative: t <= -2.0 (no E2 condition)
    (-2.0, 0.5, False, "negative"),           # boundary t = -2.0
    (-3.0, -0.5, False, "negative"),
    # zero (measured): |t| < 1.0
    (0.0, 0.0, False, "0 (measured)"),
    (0.999, -1.0, False, "0 (measured)"),
    (-0.999, 1.0, False, "0 (measured)"),
    # indeterminate band: 1.0 <= |t| < 2.0
    (1.0, 1.0, False, "indeterminate"),       # boundary t = +1.0 exactly
    (-1.0, 1.0, False, "indeterminate"),      # boundary t = -1.0 exactly
    (1.5, 0.2, False, "indeterminate"),
    (1.999, 0.2, False, "indeterminate"),
    (-1.999, 0.2, False, "indeterminate"),
    # missing E1 -> indeterminate
    (None, 0.2, False, "indeterminate"),
    (float("nan"), 0.2, False, "indeterminate"),
])
def test_organ_verdict_truth_table(t, e2, gate_off, tag):
    assert st.organ_verdict(t, e2, gate_off)["tag"] == tag


def test_scorecard_attr_formats():
    assert st.scorecard_attr(0.21, "indeterminate") == "+0.21 (indeterminate)"
    assert st.scorecard_attr(0.0, "0 (measured)") == "0 (measured)"
    assert st.scorecard_attr(None, "0 (gate-honest)") == "0 (gate-honest)"
    assert st.scorecard_attr(None, "indeterminate") == "n/a (indeterminate)"


# ============================ §4.2 BEATS/TIES/LOSES boundaries =================
@pytest.mark.parametrize("t,ds,verdict,gap", [
    (1.0, 0.1, "BEATS", False),     # boundary: t = +1.0 exactly, dSharpe > 0
    (1.0, 0.0, "UNDEFINED (rule-gap)", True),   # the uncovered cell
    (1.0, -0.1, "UNDEFINED (rule-gap)", True),
    (2.5, -0.5, "UNDEFINED (rule-gap)", True),
    (-1.0, -0.5, "LOSES", False),   # boundary: t = -1.0 exactly
    (-1.0, 0.5, "LOSES", False),    # LOSES has no dSharpe condition
    (0.999, 5.0, "TIES", False),
    (-0.999, -5.0, "TIES", False),
    (0.0, 0.0, "TIES", False),
])
def test_beats_ties_loses_boundaries(t, ds, verdict, gap):
    r = st.beats_ties_loses(t, ds)
    assert r["verdict"] == verdict
    assert r["rule_gap"] == gap


# ============================ HAC vs naive t ===================================
def test_hac_t_discounts_positive_autocorrelation():
    rng = np.random.default_rng(42)
    n, rho = 1500, 0.6
    e = rng.normal(0, 1, n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    x = x + 0.08                                  # small positive mean
    t_naive = st.naive_t(x)
    t_hac = st.hac_tstat(x, lags=10)
    assert np.isfinite(t_naive) and np.isfinite(t_hac)
    assert abs(t_hac) < abs(t_naive)              # NW widens the se under +autocorr
    # iid series: HAC ~ naive
    y = rng.normal(0.05, 1, n)
    assert abs(st.hac_tstat(y) - st.naive_t(y)) < 0.35 * abs(st.naive_t(y))


def test_paired_stats_identical_dates_only():
    d1 = np.array(["2026-01-02", "2026-01-03", "2026-01-04"])
    d2 = np.array(["2026-01-03", "2026-01-04", "2026-01-05"])
    r = st.paired_daily_stats(d1, [0.01, 0.02, 0.03], d2, [0.0, 0.0, 0.0])
    assert r["n"] == 2                            # intersection only
    assert r["mean"] == pytest.approx(0.025)


# ============================ drop-arm trust renorm ============================
def test_renorm_trust_sums_to_one_and_floor_preserved():
    rng = np.random.default_rng(7)
    tau = rng.dirichlet([2, 2, 2], size=50)       # [50,3] valid trust rows
    out = battery.renorm_trust(tau, [1, 0, 1], eps=0.05)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.allclose(out[:, 1], 0.0)            # dropped member at exactly 0
    assert (out[:, [0, 2]] >= 0.05 / 2 - 1e-12).all()   # floor eps/n_surv
    # survivors keep relative order
    sgn = np.sign(tau[:, 0] - tau[:, 2])
    assert np.array_equal(np.sign(out[:, 0] - out[:, 2]), sgn)


def test_renorm_trust_all_off_raises():
    with pytest.raises(ValueError):
        battery.renorm_trust(np.array([0.3, 0.3, 0.4]), [0, 0, 0])


def test_drop_genome_transform():
    g = ea.Genome.b0()
    g2 = battery.drop_genome(g, "gbm_cond")
    assert g2.member_gate == [1, 0, 1]
    assert g.member_gate == [1, 1, 1]             # original untouched
    a = battery.ARMS["R05"]
    h1 = a.config_hash(g.to_dict())
    h2 = a.config_hash(g2.to_dict())
    assert h1 != h2                               # hash sees the transform


# ============================ ledger + budget guards ===========================
@pytest.fixture
def fake_replay(tmp_path):
    """A stand-in replay runner that writes the 4 contract files."""
    script = tmp_path / "fake_replay.py"
    script.write_text(textwrap.dedent("""
        import argparse, csv, json
        from pathlib import Path
        ap = argparse.ArgumentParser()
        ap.add_argument("--arm-config"); ap.add_argument("--out")
        a = ap.parse_args()
        out = Path(a.out)
        for name, drift in (("daily_raw.csv", 1.0006), ("daily_costadj.csv", 1.0005)):
            with open(out / name, "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(["date", "value"])
                v = 100000.0
                for i in range(40):
                    w.writerow([f"2026-03-{(i % 28) + 1:02d}", f"{v:.2f}"])
                    v *= drift
        (out / "result.json").write_text(json.dumps(
            {"timeline": [], "actions": [], "date_value_map": {}}))
        (out / "manifest.json").write_text(json.dumps(
            {"code_sha": "fake", "seeds": [4242], "command": "fake_replay"}))
    """))
    return script


@pytest.fixture
def battery_env(tmp_path):
    ledger = tmp_path / "holdout_looks.jsonl"
    retrains = tmp_path / "retrains.jsonl"
    frozen = tmp_path / "genome_frozen.json"
    ea.Genome.b0().to_json(frozen)
    out_root = tmp_path / "runs_battery"
    return {"ledger": ledger, "retrain_ledger": retrains,
            "frozen_path": frozen, "out_root": out_root}


def test_ledger_append_on_execute(battery_env, fake_replay):
    env = battery_env
    status = battery.execute_arm(
        "R04", replay_cmd=[sys.executable, str(fake_replay)], **env)
    rows = battery._read_jsonl(env["ledger"])
    assert len(rows) == 1
    rec = rows[0]
    assert rec["run_id"] == "R04"
    assert set(rec) == {"run_id", "date_executed", "config_hash", "purpose"}
    assert rec["config_hash"] == status["config_hash"]
    assert status["outputs_ok"]
    arm_dir = Path(status["arm_dir"])
    assert battery.missing_outputs(arm_dir) == []
    cfg = json.loads((arm_dir / "arm_config.json").read_text())
    assert cfg["run_id"] == "R04"
    gen = json.loads((arm_dir / "genome.json").read_text())
    assert gen["member_gate"] == [0, 1, 1]        # cast OFF materialized


def test_ledger_written_even_when_replay_fails(battery_env, tmp_path):
    env = battery_env
    bad = tmp_path / "bad_replay.py"
    bad.write_text("import sys; sys.exit(1)")
    with pytest.raises(RuntimeError, match="ledgered"):
        battery.execute_arm("R04", replay_cmd=[sys.executable, str(bad)], **env)
    assert len(battery._read_jsonl(env["ledger"])) == 1   # look consumed honestly


def test_budget_refusal_at_21st_replay(battery_env, fake_replay):
    env = battery_env
    with open(env["ledger"], "w") as fh:
        for i in range(20):
            fh.write(json.dumps({"run_id": f"X{i:02d}", "date_executed": "t",
                                 "config_hash": "h", "purpose": "fill"}) + "\n")
    with pytest.raises(battery.BudgetExceeded, match="hard cap of 20"):
        battery.execute_arm("R04", replay_cmd=[sys.executable, str(fake_replay)],
                            **env)
    assert len(battery._read_jsonl(env["ledger"])) == 20  # 21st never ledgered


def test_retrain_budget_refusal_at_10th(tmp_path):
    led = tmp_path / "retrains.jsonl"
    for i in range(9):
        battery.register_retrain(f"RT-{i + 1}", "fill", ledger=led)
    battery.register_retrain("RT-3", "re-registration of existing id ok", ledger=led)
    with pytest.raises(battery.BudgetExceeded, match="hard cap"):
        battery.register_retrain("RT-10", "tenth distinct cycle", ledger=led)


def test_retrain_precondition_blocks_unregistered_arm(battery_env, fake_replay):
    env = battery_env
    with pytest.raises(battery.PreconditionFailed, match="RT-3"):
        battery.execute_arm("R10", replay_cmd=[sys.executable, str(fake_replay)],
                            **env)
    assert len(battery._read_jsonl(env["ledger"])) == 0   # no look consumed


def test_unfrozen_genome_refused(battery_env, fake_replay):
    env = dict(battery_env)
    env["frozen_path"] = env["frozen_path"].with_name("missing.json")
    with pytest.raises(battery.PreconditionFailed, match="freeze"):
        battery.execute_arm("R04", replay_cmd=[sys.executable, str(fake_replay)],
                            **env)


def test_reserved_arm_refused(battery_env, fake_replay):
    with pytest.raises(battery.PreconditionFailed, match="reserved"):
        battery.execute_arm("R19", replay_cmd=[sys.executable, str(fake_replay)],
                            **battery_env)


def test_default_cmd_maps_arm_onto_landed_runner_cli(battery_env):
    arm = battery.ARMS["R04"]
    arm_dir, _ = battery.materialize_arm(arm, battery_env["out_root"],
                                         battery_env["frozen_path"])
    cmd = battery.default_replay_cmd(arm, arm_dir)
    s = " ".join(cmd)
    assert "--arm syn1" in s and "--window full" in s
    assert f"--genome {arm_dir / 'genome.json'}" in s
    assert "--exec-dir" in s and "--out" in s
    cmd_inc = battery.default_replay_cmd(battery.ARMS["R02"],
                                         battery_env["out_root"] / "R02")
    assert "--arm incumbent" in " ".join(cmd_inc)
    assert "--genome" not in " ".join(cmd_inc)


@pytest.mark.parametrize("run_id,fragment", [
    ("R08", ["--exec-mode", "equal_trust"]),     # executive bypass knob landed
    ("R13", ["--cost-seed", "4243"]),            # slippage-seed override landed
    ("R14", ["--cost-seed", "4244"]),
    ("R07", ["--sigma-source", "trailing21"]),   # replay-time sigma swap landed
    ("R01", ["--cost-seed", "4242"]),            # paired seed passed explicitly
])
def test_former_capability_gaps_now_expressed(battery_env, run_id, fragment):
    arm = battery.ARMS[run_id]
    arm_dir, _ = battery.materialize_arm(arm, battery_env["out_root"],
                                         battery_env["frozen_path"])
    cmd = battery.default_replay_cmd(arm, arm_dir)
    s = " ".join(cmd)
    assert " ".join(fragment) in s, s


def test_r07_sigma_swap_supersedes_variant_nightly_store(battery_env):
    arm = battery.ARMS["R07"]
    arm_dir, _ = battery.materialize_arm(arm, battery_env["out_root"],
                                         battery_env["frozen_path"])
    cmd = battery.default_replay_cmd(arm, arm_dir)
    assert "--sigma-source" in cmd
    assert "--nightly-dir" not in cmd            # never both mechanisms at once
    # arms whose variant store carries member opinions still pass it
    arm10 = battery.ARMS["R10"]
    d10, _ = battery.materialize_arm(arm10, battery_env["out_root"],
                                     battery_env["frozen_path"])
    assert "--nightly-dir" in battery.default_replay_cmd(arm10, d10)


def test_nonstandard_bypass_shape_still_refused(battery_env):
    """The landed knob is exactly equal_trust+f=0.7 — anything else refuses
    BEFORE a holdout look is consumed."""
    arm = battery.Arm("RX", "nonstandard bypass",
                      {"strategy": "syn1", "genome": "frozen",
                       "executive_bypass": {"equal_trust": True, "f_fixed": 0.5},
                       "slippage_seed": 4242}, None, e1=None, e2=None)
    with pytest.raises(battery.PreconditionFailed, match="exec-mode"):
        battery.default_replay_cmd(arm, battery_env["out_root"] / "RX")


def test_load_run_accepts_landed_runner_shape(tmp_path):
    """Shape B: daily_series.csv + result.json without timeline + timeline.json."""
    run_dir = tmp_path / "RB"
    run_dir.mkdir()
    rows = [("2026-03-11", 100000.0, 100000.0), ("2026-03-12", 100600.0, 100550.0),
            ("2026-03-13", 100900.0, 100800.0)]
    with open(run_dir / "daily_series.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "raw_value", "cost_adjusted_value"])
        w.writerows(rows)
    (run_dir / "result.json").write_text(json.dumps(
        {"actions": [{"date": "2026-03-12", "action": "SELL", "symbol": "SPY"}],
         "date_value_map": {}}))
    (run_dir / "timeline.json").write_text(json.dumps(
        [{"date": d, "ending_value": v, "ending_cash": 0.4 * v}
         for d, _, v in rows]))
    (run_dir / "manifest.json").write_text(json.dumps({"code_sha": "x"}))
    run = st.load_run(run_dir)
    assert list(run["dates"]) == [r[0] for r in rows]
    assert run["values"][1] == 100550.0 and run["values_raw"][1] == 100600.0
    assert len(run["result"]["timeline"]) == 3      # stitched from timeline.json
    m = st.run_metrics(run)
    assert m["realized_round_trips"] == 1
    assert np.isfinite(m["avg_gross_exposure"])
    assert m["cumulative_transaction_costs"] == pytest.approx(100.0)
    assert battery.missing_outputs(run_dir) == []   # shape B passes verification


def test_plan_covers_all_arms_no_ledger_writes(battery_env):
    doc = battery.plan(allow_unfrozen=True,
                       frozen_path=battery_env["frozen_path"])
    ids = [r["run_id"] for r in doc["arms"]]
    assert ids == [f"R{i:02d}" for i in range(1, 21)]
    assert doc["budgets"]["planned_replays"] == 14
    assert doc["budgets"]["planned_retrains"] == 5    # RT-1..RT-5
    assert all(len(r["config_hash"]) == 16 for r in doc["arms"])
    assert not battery_env["ledger"].exists()         # plan consumed zero looks


# ============================ E1 walk ==========================================
@pytest.fixture(scope="module")
def world():
    return so.make_synth_world(n_dates=300, seed=7)


@pytest.fixture(scope="module")
def oof_dir(world, tmp_path_factory):
    d = tmp_path_factory.mktemp("oof_e1")
    n = len(world["dates"])
    masks = {1: np.zeros(n, bool), 2: np.zeros(n, bool)}
    masks[1][20:140] = True
    masks[2][162:282] = True
    so.write_smoke_oofs(world, d, masks)
    return d


@pytest.fixture(scope="module")
def models():
    return {1: ex.Executive(seed=11).eval(), 2: ex.Executive(seed=13).eval()}


def _side(world, models, **kw):
    return e1.E1Side(genome=kw.pop("genome", ea.Genome.b0()), models=models,
                     sigma_source=kw.pop("sigma_source", "risknet"), **kw)


def test_e1_walk_matches_ea_fold_utility(world, oof_dir, models):
    """min over cost scenarios of the series-recomputed utility must equal
    ea.FitnessEngine.fold_utility (the walk is the same math)."""
    side = _side(world, models, label="base")
    eng = e1.build_side(side, world, oof_dir, [1, 2])
    rng = np.random.default_rng(3)
    for g in (ea.Genome.b0(), ea.Genome.random(rng)):
        if sum(g.member_gate) == 0:
            g.member_gate = [1, 1, 1]
        for f in (1, 2):
            u_series = eng.utility_from_series(g, f)
            u_ea = eng.fold_utility(g, f)             # inherited ea.py code
            assert u_series == pytest.approx(u_ea, abs=1e-9), (f, g.to_dict())


def test_e1_member_drop_read(world, oof_dir, models):
    side_w = _side(world, models, label="champion")
    side_wo = _side(world, models, label="cast-drop",
                    genome=battery.drop_genome(ea.Genome.b0(), "cast"))
    res = e1.run_e1(side_w, side_wo, world, oof_dir, [1, 2])
    assert res["n"] > 200
    assert np.isfinite(res["hac_t"])
    assert res["sd"] > 0                              # the drop changes the walk
    assert set(res["per_fold"]) == {"1", "2"}
    assert len(res["ci95"]) == 2 and res["ci95"][0] < res["ci95"][1]
    assert np.isfinite(res["mde"]) and res["mde"] > 0
    # synthetic oof dir has no risknet files -> fallback warning surfaced
    assert any("risknet" in w for w in res["warnings"])


def test_e1_bypass_and_genome_swap_reads(world, oof_dir, models):
    base = _side(world, models, label="base")
    byp = _side(world, models, label="equal-trust-f0.7",
                bypass={"equal_trust": True, "f_fixed": 0.7})
    res = e1.run_e1(base, byp, world, oof_dir, [1, 2])
    assert res["n"] > 200 and np.isfinite(res["hac_t"])
    rng = np.random.default_rng(11)
    champ = ea.Genome.random(rng)
    champ.member_gate = [1, 1, 1]
    res2 = e1.run_e1(_side(world, models, label="champ", genome=champ),
                     _side(world, models, label="B0"), world, oof_dir, [1, 2])
    assert res2["n"] > 200


def test_e1_member_swap_uses_member_map(world, oof_dir, models):
    """ridge-in-slot shape: map cast -> a different OOF file and the series moves."""
    base = _side(world, models, label="base")
    # reuse gbm_cond's OOF file as the stand-in 'twin' for cast
    swap = _side(world, models, label="ridge-in-slot",
                 oof_member_map={"cast": "gbm_cond"})
    res = e1.run_e1(base, swap, world, oof_dir, [1, 2])
    assert res["n"] > 200
    assert float(np.abs(res["mean"])) >= 0           # finite read
    assert res["sd"] > 0                             # the swap changes the walk


# ============================ report schema ====================================
def _write_fake_run(run_dir: Path, seed: int, drift: float):
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dates = [str(d)[:10] for d in
             np.array(np.datetime64("2026-01-31") + np.arange(180),
                      dtype="datetime64[D]") if np.is_busday(d)][:90]
    v_raw, v_adj = 100000.0, 100000.0
    rows_raw, rows_adj, timeline = [], [], []
    for d in dates:
        r = rng.normal(drift, 0.006)
        v_raw *= (1 + r)
        v_adj *= (1 + r - 0.0001)
        rows_raw.append((d, v_raw))
        rows_adj.append((d, v_adj))
        timeline.append({"date": d, "ending_value": v_adj,
                         "ending_cash": 0.3 * v_adj, "holdings_count": 10})
    for name, rows in (("daily_raw.csv", rows_raw), ("daily_costadj.csv", rows_adj)):
        with open(run_dir / name, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["date", "value"])
            w.writerows([(d, f"{v:.2f}") for d, v in rows])
    actions = [{"date": dates[5], "symbol": "SPY", "action": "BUY", "dollars": 1000},
               {"date": dates[50], "symbol": "SPY", "action": "SELL", "dollars": 1100},
               {"date": dates[70], "symbol": "QQQ", "action": "REDUCE", "dollars": 500}]
    (run_dir / "result.json").write_text(json.dumps(
        {"timeline": timeline, "actions": actions, "date_value_map": {}}))
    (run_dir / "manifest.json").write_text(json.dumps(
        {"code_sha": "synthetic", "params_hash": "ph", "seeds": [4242],
         "snapshot_range": [dates[0], dates[-1]], "command": "synthetic"}))


def test_comparison_json_md_schema(tmp_path):
    a, b = tmp_path / "R0X", tmp_path / "R0Y"
    _write_fake_run(a, 1, 0.0009)
    _write_fake_run(b, 2, 0.0004)
    e1_res = {"hac_t": 1.7, "n": 1480, "mean": 2e-4, "ci95": [-1e-5, 4e-4],
              "mde": 2.4e-4}
    cmp = rev.write_comparison("R0X", a, b, "SYN-1", "incumbent",
                               out_root=tmp_path / "evidence", e1_result=e1_res)
    for period in ("full_period", "holdout_only"):
        blk = cmp[period]
        for side in ("metrics_a", "metrics_b"):
            assert set(rev.METRIC_KEYS) <= set(blk[side])
            assert blk[side]["realized_round_trips"] >= 0
        assert set(rev.METRIC_KEYS) <= set(blk["delta"])
        p = blk["paired_daily"]
        for k in ("n", "mean", "sd", "t", "hac_t", "ci95", "mde",
                  "mean_bp_day", "sd_bp_day"):
            assert k in p
        assert p["n"] > 0
    # holdout slice really starts at the boundary
    assert cmp["holdout_only"]["metrics_a"]["window_start"] >= st.HOLDOUT_START
    assert cmp["full_period"]["metrics_a"]["n_days"] > \
        cmp["holdout_only"]["metrics_a"]["n_days"]
    jp = tmp_path / "evidence" / "R0X" / "comparison.json"
    mp = tmp_path / "evidence" / "R0X" / "comparison.md"
    assert jp.exists() and mp.exists()
    md = mp.read_text()
    for needle in ("Full period", "Holdout only", "Paired daily diff",
                   "Manifests", "E1 primary read"):
        assert needle in md, needle
    assert json.loads(jp.read_text())["e1_primary"]["hac_t"] == 1.7


def test_scorecard_assembly_schema(tmp_path):
    hold = tmp_path / "holdout_looks.jsonl"
    val = tmp_path / "validation_looks.jsonl"
    hold.write_text("".join(json.dumps({"run_id": f"R{i:02d}"}) + "\n"
                            for i in range(14)))
    val.write_text("".join(json.dumps({"c": i}) + "\n" for i in range(33)))
    bakeoff = {"paired": {"t": 0.62, "n": 62, "mean_bp_day": 1.8,
                          "sd_bp_day": 22.0},
               "holdout_dsharpe": 0.31, "holdout_dreturn": 0.012}
    organs = [
        {"organ": "transformer", "attr_key": "transformer", "run_id": "R03",
         "e1_hac_t": 0.4, "e1_n": 1480, "e2_holdout_dsharpe": 0.05,
         "e2_holdout_mean_delta": 1e-5},
        {"organ": "ensemble_cast", "attr_key": "ensemble", "member": "cast",
         "run_id": "R04", "e1_hac_t": 2.3, "e1_n": 1480,
         "e2_holdout_dsharpe": 0.2, "e2_holdout_mean_delta": 2e-5},
        {"organ": "ensemble_gbm", "attr_key": "ensemble", "member": "gbm",
         "run_id": "R05", "e1_hac_t": 1.2, "e1_n": 1480,
         "e2_holdout_dsharpe": -0.1, "e2_holdout_mean_delta": -1e-5},
        {"organ": "evolution", "attr_key": "evolution", "run_id": "R09",
         "e1_hac_t": 0.1, "e1_n": 1480, "e2_holdout_dsharpe": 0.0,
         "e2_holdout_mean_delta": 0.0},
        {"organ": "llm", "attr_key": "LLM", "run_id": "R10",
         "e1_hac_t": None, "e1_n": None, "e2_holdout_dsharpe": None,
         "e2_holdout_mean_delta": None, "gate_off": True},
        {"organ": "gdelt", "attr_key": "GDELT", "run_id": "R11",
         "e1_hac_t": -2.4, "e1_n": 1480, "e2_holdout_dsharpe": -0.3,
         "e2_holdout_mean_delta": -2e-5},
        {"organ": "meta_evaluator", "attr_key": "meta-evaluator", "run_id": "R08",
         "e1_hac_t": 1.7, "e1_n": 1480, "e2_holdout_dsharpe": 0.12,
         "e2_holdout_mean_delta": 5e-6},
        {"organ": "infotropy_b", "attr_key": "infotropy", "run_id": "R12",
         "e1_hac_t": 0.2, "e1_n": 1480, "e2_holdout_dsharpe": 0.01,
         "e2_holdout_mean_delta": 1e-6},
        {"organ": "infotropy_a", "attr_key": "infotropy", "run_id": "E1-only",
         "e1_hac_t": 0.5, "e1_n": 1480, "e2_holdout_dsharpe": None,
         "e2_holdout_mean_delta": None},
    ]
    gates = {"placebo": {"real_ic": 0.031, "percentile": 98.0, "pass_95": True,
                         "n_permutations": 50},
             "shift_gate": {"overall": "PASS"}}
    doc = rev.assemble_scorecard(
        bakeoff, organs, gates, cost_per_month=9.5,
        reductions=["CAST OOF seeds 3->2"], out_dir=tmp_path / "evidence",
        holdout_ledger=hold, validation_ledger=val)
    # ledger counts filled VERBATIM into the §4.5 line
    assert "consumed 14 holdout looks" in doc["multiplicity_line"]
    assert "33 validation-fold decisions" in doc["multiplicity_line"]
    # verdicts computed mechanically
    by_organ = {r["organ"]: r for r in doc["organ_table"]}
    assert by_organ["transformer"]["verdict"] == "0 (measured)"
    assert by_organ["ensemble_cast"]["verdict"] == "positive"
    assert by_organ["llm"]["verdict"] == "0 (gate-honest)"
    assert by_organ["gdelt"]["verdict"] == "negative"
    assert by_organ["meta_evaluator"]["verdict"] == "indeterminate"
    # infotropy: both zero -> no-transfer on the final line
    assert doc["organ_attrs"]["infotropy"] == "no-transfer"
    # bake-off verdict + final line shape
    assert doc["bakeoff"]["verdict"] == "TIES"
    fl = doc["final_line"]
    assert fl.startswith("BRAIN vs INCUMBENT: TIES by ")
    for organ in rev.FINAL_LINE_ORGANS:
        assert f"{organ}=" in fl
    assert "COST: $9.50/mo" in fl and "CAST OOF seeds 3->2" in fl
    # §4.6: all seven gate lines present; missing gates say NOT PROVIDED
    assert len(doc["gate_readouts"]) == 7
    assert sum("NOT PROVIDED" in g for g in doc["gate_readouts"]) == 5
    assert any("percentile 98.0" in g for g in doc["gate_readouts"])
    sj = tmp_path / "evidence" / "scorecard.json"
    sm = tmp_path / "evidence" / "SCORECARD.md"
    assert sj.exists() and sm.exists()
    md = sm.read_text()
    for needle in ("Organ verdicts", "Multiplicity", "Evidence gates",
                   "Final line", doc["multiplicity_line"][:60]):
        assert needle in md


def test_final_line_rule_gap_is_printed_not_coerced(tmp_path):
    bakeoff = {"paired": {"t": 1.4, "n": 62, "mean_bp_day": 3.0, "sd_bp_day": 20.0},
               "holdout_dsharpe": -0.1, "holdout_dreturn": 0.002}
    doc = rev.assemble_scorecard(
        bakeoff, [], {}, 9.5, [], out_dir=tmp_path / "evidence",
        holdout_ledger=tmp_path / "h.jsonl", validation_ledger=tmp_path / "v.jsonl")
    assert doc["bakeoff"]["rule_gap"] is True
    assert "UNDEFINED (rule-gap)" in doc["final_line"]
