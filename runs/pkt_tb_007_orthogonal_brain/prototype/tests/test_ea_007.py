"""PKT-TB-007 wave-2b tests — EA + surrogate machinery (BUILD_SPEC_007 §4/§5).

Covers: FITNESS(B0) ≡ 0, paired-walk vs adapter-projection consistency
(same genome + same inputs => same tilt decisions to 1e-9 through the shared
solve path), the s_min floor behavior, rotation withholding correctness,
gate arithmetic (adoption + rotation), genome range clamps and the
boundary-pin detector.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROTO = Path(__file__).resolve().parents[1]
REPO = PROTO.parents[2]
TB006 = REPO / "runs" / "pkt_tb_006_clean_sheet_brain" / "prototype"
for p in (str(PROTO), str(REPO), str(TB006)):
    if p not in sys.path:
        sys.path.insert(0, p)

import ea_007 as EA                                          # noqa: E402
import surrogate_007 as SG                                   # noqa: E402
import tilt_adapter as TA                                    # noqa: E402
from genome_007 import Genome007                             # noqa: E402

ROSTER = ("M1", "M2")
NAMES = list(TA.TILT_CORE) + list(TA.TILT_COND)


# ------------------------------------------------------------ tiny fixtures
def _organ_file(tmp: Path, date: str, seed: int = 7) -> None:
    rng = np.random.default_rng(seed + hash(date) % 1000)
    organs = {k: {"mu": {s: round(float(rng.standard_normal()), 6)
                         for s in NAMES}, "q": 0.6} for k in ROSTER}
    doc = {"date": date, "schema_version": "organ_inputs_007.v1",
           "organs": organs, "disp_z": 0.4, "p_exceed": {},
           "manifest": {"synthetic": True}}
    (tmp / f"{date}.json").write_text(json.dumps(doc))


def _mini_pack(n_days: int = 6, seed: int = 3, start_idx: int = 1) -> dict:
    """A tiny day-pack with full capacity (some held ballast to fund)."""
    rng = np.random.default_rng(seed)
    days = []
    support = sorted(set(NAMES) | {"SCHD", "XLB"})
    for t in range(n_days):
        D = f"2025-03-{start_idx + t:02d}"
        held = {"SCHD": 0.25, "XLB": 0.20, "TLT": 0.10}
        days.append({
            "date": D, "regime": "choppy", "nav": 100_000.0,
            "support": support,
            "beta": rng.uniform(-0.5, 1.5, len(support)),
            "sigma": rng.uniform(0.005, 0.02, len(support)),
            "has_stats": np.ones(len(support), dtype=bool),
            "w_prev": np.array([held.get(s, 0.0) for s in support]),
            "buys_w": np.zeros(len(support)),
            "sell_w": np.zeros(len(support)),
            "fully_sold": np.zeros(len(support), dtype=bool),
            "partially_sold": np.zeros(len(support), dtype=bool),
            "ovn": rng.normal(0, 0.004, len(support)),
            "intra": rng.normal(0, 0.006, len(support)),
            "hs_bps": np.full(len(support), 4.0),
            "marks": np.full(len(support), 10.0),   # cheap => rounding ~exact
            "n_holdings": 3, "n_buys": 0, "n_sells": 0,
            "book_empty_at_close": False,
        })
    return {"days": days, "params": SG.DEPLOYED_PARAMS,
            "asset_class": {s: "bond" for s in support}}   # no panic closure


@pytest.fixture()
def organs_dir(tmp_path):
    d = tmp_path / "organs"
    d.mkdir()
    for f in (1, 2):
        for t in range(6):
            _organ_file(d, f"2025-0{f + 2}-{1 + t:02d}")
    return d


@pytest.fixture()
def packs():
    return {1: _mini_pack(6, seed=3, start_idx=1),
            2: {**_mini_pack(6, seed=4, start_idx=1)}}


def _fix_pack_dates(packs):
    # fold 2 pack dates -> 2025-04-* so organ files can be distinct per fold
    for t, rec in enumerate(packs[2]["days"]):
        rec["date"] = f"2025-04-{1 + t:02d}"
    return packs


@pytest.fixture()
def engine(packs, organs_dir, tmp_path):
    packs = _fix_pack_dates(packs)
    for f, month in ((1, "03"), (2, "04")):
        for t in range(6):
            _organ_file(organs_dir, f"2025-{month}-{1 + t:02d}")
    return EA.FitnessEngine007(packs, organs_dir, TA.PRODUCTION_MASKS,
                               ROSTER, fold_list=[1, 2])


# ============================================== 1. FITNESS(B0) ≡ 0
def test_fitness_b0_is_exactly_zero(engine):
    b0 = Genome007.b0(ROSTER)
    res = engine.fitness(b0, [1, 2])
    assert res["fitness"] == 0.0
    assert all(v == 0.0 for v in res["fold_u"].values())
    assert res["shrinkage"] == 0.0
    # and the walk itself is exactly flat
    w = SG.walk_fold_paired(engine.packs[1], b0, engine.organs_dir,
                            engine.masks)
    assert float(np.abs(w["dr_gross"]).max()) == 0.0
    assert float(np.abs(w["cost1"]).max()) == 0.0
    assert w["n_active"] == 0


def test_nonneutral_genome_moves_fitness(engine):
    g = Genome007(organ_trust={"M1": 0.5, "M2": -0.3}, tilt_gain=0.6,
                  dead_zone=0.05)
    res = engine.fitness(g, [1, 2])
    assert res["fitness"] != 0.0
    w = SG.walk_fold_paired(engine.packs[1], g, engine.organs_dir,
                            engine.masks)
    assert w["n_active"] > 0
    assert float(np.abs(w["dr_gross"]).max()) > 0


# ============================================== 2. walk vs adapter consistency
def test_paired_walk_matches_adapter_projection(tmp_path):
    """Same genome + same inputs => the SAME tilt decisions through the
    replay adapter (post_decision, expression log) and through the walk's
    input derivation + shared solve, to 1e-9 (log precision 1e-8)."""
    from src.utils.three_line_replay.replay_engine import Portfolio, Position
    from src.utils.three_line_replay.strategies import StrategyContext

    D = "2025-03-03"
    genome = Genome007(organ_trust={"M1": 0.5, "M2": -0.3}, tilt_gain=0.6,
                       dead_zone=0.05)
    organs_dir = tmp_path / "organs"
    organs_dir.mkdir()
    _organ_file(organs_dir, D, seed=11)

    # one held book + marks (features rows strictly < D)
    held = {"SCHD": (1000, 25.0), "XLB": (400, 50.0), "TLT": (120, 90.0)}
    nav_syms = sorted(set(NAMES) | set(held))
    rng = np.random.default_rng(5)
    marks = {s: float(rng.uniform(20, 120)) for s in nav_syms}
    for s, (sh, px) in held.items():
        marks[s] = px
    feats = pd.DataFrame({"date": pd.Timestamp("2025-03-02"),
                          "symbol": nav_syms,
                          "close": [marks[s] for s in nav_syms]})
    positions = [Position(symbol=s, shares=float(sh), entry_price=px,
                          entry_date="2025-01-02", peak_price=px,
                          asset_class="equity", sector="broad",
                          leverage_flag=0)
                 for s, (sh, px) in held.items()]
    cash = 100_000.0 - sum(sh * px for _, (sh, px) in held.items())
    portfolio = Portfolio(cash=cash, positions=positions)
    nav = 100_000.0

    stats_tbl = {s: {"beta": float(rng.uniform(-0.2, 1.4)),
                     "sigma": float(rng.uniform(0.005, 0.02))}
                 for s in nav_syms}

    class _Risk:
        def table(self, symbols, date):
            return {s: stats_tbl[s] for s in symbols if s in stats_tbl}

    strat = TA.make_tilt_strategy(genome, organs_dir, log_dir=tmp_path,
                                  risk=_Risk())
    ctx = StrategyContext(inputs_date=D, portfolio=portfolio,
                          variant_config={"decision_params": {}},
                          expert_signals={}, expert_metrics={}, decisions={},
                          panic_streak=0, last_regime="choppy",
                          features_df=feats, inference={}, llm_risks={})
    out = strat.post_decision(ctx, [])
    log = json.loads((tmp_path / "expression_log" / f"{D}.json").read_text())
    assert log["neutral"] is False
    adapter_dw = {k: float(v) for k, v in log["tilt"].items()}

    # the walk side: derive the same day inputs and call the SHARED solve
    organs = TA.load_organ_outputs(organs_dir, D)
    conv = TA.compute_conviction(organs, genome)
    support = sorted((set(held) | set(TA.TILT_CORE) | set(TA.TILT_COND))
                     - {TA.VIXY})
    w_prev = {s: held[s][0] * held[s][1] / nav for s in held}
    lb, ub = [], []
    for s in support:
        wh = w_prev.get(s, 0.0)
        lo, hi = TA.tilt_bounds(s, genome, wh, 0.0, wh, False, False,
                                s in stats_tbl)
        lb.append(lo)
        ub.append(hi)
    beta = np.array([stats_tbl[s]["beta"] for s in support])
    sigma = np.array([stats_tbl[s]["sigma"] for s in support])
    dw, pdiag, _, _ = TA.solve_tilt(organs, genome, TA.PRODUCTION_MASKS,
                                    support, conv["active"], conv["tau"],
                                    conv["T_t"], beta, sigma,
                                    np.array(lb), np.array(ub))
    assert not pdiag["projection"]["infeasible"]
    walk_dw = {s: v for s, v in dw.items() if abs(v) > 1e-8}
    assert set(walk_dw) == set(adapter_dw)
    for s in walk_dw:
        # adapter log rounds at 1e-8; identical solve path => agreement there
        assert abs(walk_dw[s] - adapter_dw[s]) < 1e-8, (s, walk_dw[s],
                                                        adapter_dw[s])
    # determinism of the shared path at 1e-15 (bit-for-bit re-solve)
    dw2, _, _, _ = TA.solve_tilt(organs, genome, TA.PRODUCTION_MASKS,
                                 support, conv["active"], conv["tau"],
                                 conv["T_t"], beta, sigma,
                                 np.array(lb), np.array(ub))
    for s in dw:
        assert dw[s] == dw2[s]
    assert isinstance(out, list)


# ============================================== 3. floor behavior (s_min)
def test_fold_u_floor_behavior():
    # microscopic but consistent positive series: sd << s_min => U degrades
    # to sqrt(252) * mean / s_min (pure-mean reward, C3 spend pressure)
    dr = np.full(100, 1e-5)
    dr[::2] = 1.2e-5                              # tiny sd, well under 2bp
    u = SG.fold_U(dr, np.zeros(100), scenarios=(1.0,))
    expect = np.sqrt(252) * dr.mean() / SG.S_MIN_DAILY
    assert abs(u - expect) < 1e-12
    # high-sd series: the floor must NOT bind
    rng = np.random.default_rng(0)
    dr2 = rng.normal(5e-5, 5e-3, 200)
    u2 = SG.fold_U(dr2, np.zeros(200), scenarios=(1.0,))
    expect2 = np.sqrt(252) * dr2.mean() / dr2.std(ddof=1)
    assert abs(u2 - expect2) < 1e-12
    # min over cost scenarios: higher cost multiple can only lower U
    cost = np.full(200, 2e-5)
    assert (SG.fold_U(dr2, cost) <=
            SG.fold_U(dr2, cost, scenarios=(1.0,)) + 1e-15)
    # zero series => exactly 0 (the B0 anchor of the fitness scale)
    assert SG.fold_U(np.zeros(50), np.zeros(50)) == 0.0


def test_worst_fold_floor_and_shrinkage(engine):
    # FITNESS = mean_f U_f - max(0, -min_f U_f) - lambda * dist^2
    g = Genome007(organ_trust={"M1": 1.0, "M2": -1.0}, tilt_gain=0.8,
                  dead_zone=0.0)
    res = engine.fitness(g, [1, 2])
    u = np.array(list(res["fold_u"].values()))
    expected = u.mean() - max(0.0, -u.min()) - res["shrinkage"]
    assert abs(res["fitness"] - expected) < 1e-5      # reported shrinkage rounds at 1e-6
    d = EA.to_unit(g, ROSTER) - EA.b0_unit(ROSTER)
    assert abs(res["shrinkage"] - EA.LAMBDA_REG * float(d @ d)) < 1e-5


# ============================================== 4. rotation withholding
def test_rotation_withholds_fold_everywhere(packs, organs_dir, monkeypatch,
                                            tmp_path):
    packs = _fix_pack_dates(packs)
    walked = []
    orig = SG.walk_fold_paired

    def spy(pack, genome, organs_dir_, masks, **kw):
        walked.append(pack["days"][0]["date"][:7])
        return orig(pack, genome, organs_dir_, masks, **kw)

    monkeypatch.setattr(SG, "walk_fold_paired", spy)
    eng = EA.FitnessEngine007(packs, organs_dir, TA.PRODUCTION_MASKS,
                              ROSTER, fold_list=[1])     # fold 2 withheld
    res = EA.run_ga(eng, tmp_path / "rot", "test_rot", P=4, G=1,
                    fold_subsample=1, verbose=False)
    assert all(m == "2025-03" for m in walked), walked   # fold 2 never walked
    assert res["champion"]["champion"] is not None
    # unseen-fold evaluation walks ONLY fold 2
    walked.clear()
    unseen = EA.FitnessEngine007(packs, organs_dir, TA.PRODUCTION_MASKS,
                                 ROSTER, fold_list=[2])
    unseen.fold_utility(res["champion"]["champion"], 2)
    assert set(walked) == {"2025-04"}


def test_load_rotation_spec_contract(tmp_path):
    with pytest.raises(FileNotFoundError):
        EA.load_rotation_spec(tmp_path, 3)
    doc = {"rotation": 3, "withheld_fold": 3, "derived_from_folds": [1, 2],
           "roster": ["M1"], "masks": {"M1": {"ITA": 1.0}}, "synthetic": True}
    (tmp_path / "rotation_3.json").write_text(json.dumps(doc))
    spec = EA.load_rotation_spec(tmp_path, 3)
    assert spec["roster"] == ["M1"]


# ============================================== 5. gate arithmetic
def test_adoption_gate_arithmetic():
    res = {"fitness": 1.0, "fold_u": {str(f): v for f, v in
           enumerate([1.2, 0.9, 1.1, 0.8, 1.0, 1.0], 1)}}
    g = EA.adoption_gate(res, k_eff=2)            # sqrt(2 ln 2)=1.18 -> 1.4
    assert g["bar_multiplier"] == 1.4
    sd = np.std([1.2, 0.9, 1.1, 0.8, 1.0, 1.0], ddof=1)
    assert abs(g["bar"] - 1.4 * sd) < 1e-6
    assert g["adopted"] is True
    g2 = EA.adoption_gate(res, k_eff=400)         # sqrt(2 ln 400) ~ 3.46
    assert abs(g2["bar_multiplier"] - np.sqrt(2 * np.log(400))) < 1e-3
    assert g2["adopted"] == (1.0 > g2["bar"])
    # high-variance champion fails at the same margin
    res3 = {"fitness": 0.3, "fold_u": {str(f): v for f, v in
            enumerate([2.0, -1.5, 1.0, -0.8, 0.9, 0.2], 1)}}
    assert EA.adoption_gate(res3, 2)["adopted"] is False


def test_rotation_gate_arithmetic():
    du = {1: 0.2, 2: 0.1, 3: 0.15, 4: 0.05, 5: 0.12, 6: 0.08}
    g = EA.rotation_gate(du)
    v = np.array(list(du.values()))
    assert g["n_nonneg"] == 6
    assert abs(g["pooled_se"] - v.std(ddof=1) / np.sqrt(6)) < 1e-5
    assert g["passed"] is True
    # 3 negative folds => <4/6 nonneg => fail regardless of mean
    g2 = EA.rotation_gate({1: 0.5, 2: -0.01, 3: -0.01, 4: -0.01, 5: 0.4, 6: 0.4})
    assert g2["n_nonneg"] == 3 and g2["passed"] is False
    # nonneg but pooled mean below 1 x se => fail
    g3 = EA.rotation_gate({1: 0.3, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0})
    assert g3["n_nonneg"] == 6
    assert g3["passed"] == (g3["pooled_mean"] > g3["pooled_se"])


# ============================================== 6. clamps + boundary pins
def test_genome_range_clamps():
    with pytest.raises(ValueError):
        Genome007(organ_trust={"M1": 0.0}, tilt_gain=1.5)
    with pytest.raises(ValueError):
        Genome007(organ_trust={"M1": 3.0})
    with pytest.raises(ValueError):
        Genome007(organ_trust={"M9": 0.0})
    # from_unit clips out-of-box vectors back into range
    v = np.array([2.0, -1.0] + [5.0] * 8)
    g = EA.from_unit(v, ("M1", "M2"))
    assert g.organ_trust["M1"] == 2.0 and g.organ_trust["M2"] == -2.0
    assert g.tilt_gain == 1.0
    # round trip in unit space
    g2 = Genome007(organ_trust={"M1": 0.7, "M2": -1.1}, tilt_gain=0.4,
                   conviction_temp=1.3, dead_zone=0.2)
    u = EA.to_unit(g2, ("M1", "M2"))
    g3 = EA.from_unit(u, ("M1", "M2"))
    for nm in EA.gene_names(("M1", "M2")):
        lo, hi, _ = EA.gene_bounds(nm)
        a = (g2.organ_trust[nm.split(".")[1]] if "." in nm
             else getattr(g2, nm))
        b = (g3.organ_trust[nm.split(".")[1]] if "." in nm
             else getattr(g3, nm))
        assert abs(a - b) < 1e-12


def test_boundary_pin_detector():
    b0 = Genome007.b0(ROSTER)
    pin0 = EA.boundary_pin_fraction(b0, ROSTER)
    # B0 pins exactly tilt_gain (0.0 = lower edge) and event_damp (0.0)
    assert set(pin0["pinned_genes"]) == {"tilt_gain", "event_damp_strength"}
    assert pin0["noise_flag"] is False
    g = Genome007(organ_trust={"M1": 2.0, "M2": -2.0}, tilt_gain=1.0,
                  conviction_temp=2.0, dead_zone=0.0, disp_gain=2.0)
    pin = EA.boundary_pin_fraction(g, ROSTER)
    assert pin["fraction_pinned"] > 1 / 3
    assert pin["noise_flag"] is True


def test_dedupe_count():
    a = np.zeros(10)
    near = a + 0.001
    far = a + 1.0
    assert EA.dedupe_count([a, near, far]) == 2


# ============================================== firewall rails
def test_fold_dates_respect_firewall_and_fitness_end():
    dates = [f"2025-{m:02d}-15" for m in range(1, 13)] + \
            ["2026-01-15", "2026-02-05", "2026-02-15", "2026-03-12"]
    fd = SG.fold_dates(sorted(dates), 6)
    assert all(d <= SG.FITNESS_END for d in fd)
    assert all(d < SG.HOLDOUT_START for d in fd)
    assert "2026-02-15" not in fd and "2026-03-12" not in fd
