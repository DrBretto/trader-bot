"""PKT-TB-006 — brain tests: solo-book rule, trust ledger, executive, EA.

Usage:  .venv/bin/python -m pytest tests/test_brain.py -v

Everything runs on SYNTHETIC seeded data (synth_oof.make_synth_world) —
hermetic, no panel/network dependency.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROTO))

import books as bk            # noqa: E402
import ea                     # noqa: E402
import executive as ex        # noqa: E402
import folds as fd            # noqa: E402
import synth_oof as so        # noqa: E402


# ============================ shared fixtures ==================================
@pytest.fixture(scope="session")
def world():
    return so.make_synth_world(n_dates=300, seed=7)


@pytest.fixture(scope="session")
def oof_dir(world, tmp_path_factory):
    d = tmp_path_factory.mktemp("oof_smoke")
    n = len(world["dates"])
    masks = {1: np.zeros(n, bool), 2: np.zeros(n, bool)}
    masks[1][20:140] = True
    masks[2][162:282] = True          # 22-td embargo gap between folds
    so.write_smoke_oofs(world, d, masks)
    return d


@pytest.fixture(scope="session")
def exec_data(world, oof_dir):
    return ex.assemble_exec_data(world, oof_dir, [1, 2])


@pytest.fixture(scope="session")
def trained_exec(exec_data):
    log = []
    m = ex.train_executive(exec_data, seed=11, max_epochs=25, patience=25,
                           loss_log=log)
    return m, log


@pytest.fixture(scope="session")
def lofo_models(exec_data):
    return ex.train_lofo_executives(exec_data, [1, 2], max_epochs=3, patience=3)


# ============================ solo-book rule ===================================
class TestSoloBook:
    def test_unit_gross_cap_long_only(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            mu = rng.normal(0, 1, 64)
            sigma = 0.3 + rng.random(64)
            w = bk.solo_book(mu, sigma)
            assert (w >= 0).all()                                  # long-only
            assert w.max() <= bk.W_CAP + 1e-9                      # cap
            n_pos = (np.maximum(mu, 0) > 0).sum()
            if n_pos >= 10:
                assert abs(w.sum() - 1.0) < 1e-9                   # unit gross

    def test_few_names_infeasible_gross(self):
        mu = np.zeros(64)
        mu[:4] = 1.0
        w = bk.solo_book(mu, np.ones(64))
        assert (w[:4] == pytest.approx(bk.W_CAP)) and abs(w.sum() - 4 * bk.W_CAP) < 1e-9

    def test_all_negative_abstains(self):
        w = bk.solo_book(-np.ones(64), np.ones(64))
        assert (w == 0).all()

    def test_determinism(self):
        rng = np.random.default_rng(3)
        mu, sigma = rng.normal(0, 1, 64), 0.3 + rng.random(64)
        assert np.array_equal(bk.solo_book(mu, sigma), bk.solo_book(mu.copy(), sigma.copy()))

    def test_sigma_floor_and_q95_clip(self):
        mu = np.ones(64) * 0.1
        mu[0] = 100.0                                              # huge outlier
        w = bk.solo_book(mu, np.full(64, 1e-9))                    # sigma below floor
        # outlier clipped at q95 -> cannot dominate beyond the cap
        assert w[0] <= bk.W_CAP + 1e-9
        assert np.isfinite(w).all()


# ============================== trust ledger ===================================
class TestLedger:
    def test_lag_honesty(self):
        """Stats at decision date D must not move when u changes at D' > D-h-1."""
        rng = np.random.default_rng(1)
        n = 120
        dates = np.array([f"d{i:03d}" for i in range(n)])
        u = rng.normal(0, 1e-3, n)
        k = 80
        u2 = u.copy()
        u2[k:] += 1.0                                              # future shock at index k
        fold_ids = np.zeros(n, dtype=int)
        l1 = bk.build_trust_ledger(dates, {"m": u}, fold_ids)["m"]
        l2 = bk.build_trust_ledger(dates, {"m": u2}, fold_ids)["m"]
        lag = bk.H + 1
        for col in ("ewma21_raw", "ewma63_raw", "hit_rate", "cf_drawdown"):
            a, b = l1[col][: k + lag], l2[col][: k + lag]
            same = np.isnan(a) & np.isnan(b) | np.isclose(a, b, equal_nan=True)
            assert same.all(), f"{col} leaked before D-h-1"
            assert not np.allclose(np.nan_to_num(l1[col][k + lag:]),
                                   np.nan_to_num(l2[col][k + lag:])), f"{col} ignored u"

    def test_realized_u_uses_costs(self, world):
        books = np.zeros((40, 64))
        books[:, :10] = 0.1
        rets = np.zeros((40, 64))
        u = bk.realized_u_series(books, rets, world["sigma_hat"][:40],
                                 np.full(64, 3.0))
        assert u[0] < 0                                            # pure cost on day 0
        assert np.isnan(u[-(bk.H - 1):]).all()                     # unrealized tail

    def test_walk_forward_refusal(self, tmp_path):
        np.savez(tmp_path / "fold_1_cast.npz", dates=np.array(["2020-01-02"]),
                 mu=np.zeros((1, 64)), sigma=np.ones((1, 64)), c=np.array([0.5]),
                 manifest=json.dumps({"walk_forward": False}))
        with pytest.raises(RuntimeError, match="walk_forward"):
            bk.load_oof(tmp_path, 1, "cast")


# =============================== executive =====================================
class TestExecutive:
    def test_param_counts(self):
        n = ex.Executive().n_params()
        assert 400 <= n <= 700, n                                  # ~560
        assert n <= ex.PARAM_CEILING
        assert ex.LinearGate().n_params() <= 150                   # ~100 twin

    def test_shapes_and_entropy_floor(self, exec_data):
        t = ex.ExecTensors(exec_data, np.arange(min(50, len(exec_data.dates))))
        m = ex.Executive(seed=11)
        out = ex.forward_batch(m, t, torch.zeros(t.n, 64), torch.zeros(t.n),
                               sigma_cap=0.10, no_trade_band=0.01, train=False)
        assert out["tau"].shape == (t.n, 3)
        assert torch.allclose(out["tau"].sum(-1), torch.ones(t.n), atol=1e-5)
        assert float(out["tau"].min()) >= ex.EPS_TAU / 3 - 1e-6    # entropy floor
        assert out["w_tgt"].shape == (t.n, 64)
        assert (out["f"] >= 0).all() and (out["f"] <= ex.F_MAX).all()

    def test_vol_cap_binds(self, exec_data):
        t = ex.ExecTensors(exec_data, np.arange(30))
        m = ex.Executive(seed=11)
        cap = 0.01                                                 # absurdly tight cap
        out = ex.forward_batch(m, t, torch.zeros(t.n, 64), torch.zeros(t.n),
                               sigma_cap=cap, no_trade_band=0.0, train=False)
        vol = (out["w_tgt"] * t.sigma_hat).sum(-1)
        assert float(vol.max()) <= cap + 1e-6
        out2 = ex.forward_batch(m, t, torch.zeros(t.n, 64), torch.zeros(t.n),
                                sigma_cap=10.0, no_trade_band=0.0, train=False)
        assert float((out2["w_tgt"].sum(-1) - out["w_tgt"].sum(-1)).max()) > 0  # binding

    def test_loss_decreases(self, trained_exec):
        _, log = trained_exec
        first = np.mean([r["train_loss"] for r in log[:3]])
        last = np.min([r["train_loss"] for r in log[-5:]])
        assert last < first, f"loss did not decrease: {first} -> {last}"

    def test_lofo_indexing(self, exec_data):
        for f in (1, 2):
            tr, va = ex.lofo_train_indices(exec_data, exclude_fold=f)
            used = np.concatenate([tr, va])
            assert (exec_data.fold_of[used] != f).all(), f"LOFO {f} saw fold {f}"
            other = 2 if f == 1 else 1
            assert (exec_data.fold_of[used] == other).any()

    def test_fine_tune_touches_exactly_named_params(self, exec_data, trained_exec):
        model, _ = trained_exec
        import copy
        m = copy.deepcopy(model)
        pre = {k: v.detach().clone() for k, v in m.state_dict().items()}
        res = ex.fine_tune(m, exec_data, start=str(exec_data.dates[10]),
                           cutoff=str(exec_data.dates[-1]), max_epochs=5, patience=5)
        post = m.state_dict()
        for k in pre:
            if k in ex.FINE_TUNE_PARAMS:
                continue
            assert torch.equal(pre[k], post[k]), f"frozen param {k} changed"
        assert set(res["changed_params"]) <= set(ex.FINE_TUNE_PARAMS)
        assert len(ex.FINE_TUNE_PARAMS) == 5                       # 7 effective scalars
        assert sum(post[k].numel() for k in ex.FINE_TUNE_PARAMS) == 7

    def test_meta_decision_schema(self, exec_data, trained_exec, tmp_path):
        model, _ = trained_exec
        rec = ex.write_meta_decision(model, exec_data, 20, tmp_path)
        for key in ("date", "model_version", "code_sha", "input_hash", "experts",
                    "trust_entropy", "deployment_fraction", "sizing_attrib", "book",
                    "counterfactuals", "expected", "realized"):
            assert key in rec, key
        assert len(rec["experts"]) == 3
        e0 = rec["experts"][0]
        for key in ("name", "confidence", "rolling", "trust", "trust_logit",
                    "logit_attrib", "solo_book_corr"):
            assert key in e0, key
        # IG completeness: attributions + bias reproduce the logit
        tot = sum(e0["logit_attrib"].values())
        assert tot == pytest.approx(e0["trust_logit"], abs=1e-3)
        assert (tmp_path / "meta_decision.json").exists()
        assert rec["realized"] is None


# ================================ genome =======================================
class TestGenome:
    def test_b0_matches_spec_table(self):
        b0 = ea.Genome.b0()
        assert b0.trust_prior == [0.0, 0.0, 0.0]
        assert b0.trust_halflife_days == 21.0
        assert b0.member_gate == [1, 1, 1] and b0.feature_gate == [1] * 8
        assert (b0.gross_target, b0.vol_target_ann, b0.max_symbol_weight) == (0.60, 0.10, 0.08)
        assert (b0.dd_brake_threshold, b0.dd_brake_strength) == (0.10, 0.50)
        assert (b0.no_trade_band, b0.conviction_temp, b0.cash_floor) == (0.010, 1.0, 0.10)
        assert (b0.risk_aversion_lambda, b0.abstain_threshold) == (2.0, 0.10)
        assert (b0.record_weight_eps, b0.event_weight_cap) == (0.25, 1.0)
        assert ea.N_GENES == 27

    def test_roundtrip(self, tmp_path):
        rng = np.random.default_rng(5)
        for _ in range(20):
            g = ea.Genome.random(rng)
            fv, bv = g.to_vector()
            g2 = ea.Genome.from_vector(fv, bv)
            for k, v in g.to_dict().items():
                v2 = getattr(g2, k)
                np.testing.assert_allclose(v, v2, rtol=1e-9, err_msg=k)
            g.to_json(tmp_path / "g.json")
            assert ea.Genome.from_json(tmp_path / "g.json").to_dict() == g.to_dict()

    def test_clamps(self):
        nf = len(ea.Genome.b0().to_vector()[0])
        nb = len(ea.Genome.b0().to_vector()[1])
        hi = ea.Genome.from_vector(np.full(nf, 99.0), np.ones(nb, dtype=int))
        lo = ea.Genome.from_vector(np.full(nf, -99.0), np.zeros(nb, dtype=int))
        for name, ln, kind, lo_b, hi_b, _ in ea.GENE_SPECS:
            if kind == "bit":
                continue
            vh, vl = getattr(hi, name), getattr(lo, name)
            vh = vh if isinstance(vh, list) else [vh]
            vl = vl if isinstance(vl, list) else [vl]
            assert all(abs(x - hi_b) < 1e-6 for x in vh), name
            assert all(abs(x - lo_b) < 1e-6 for x in vl), name
        assert lo.n_active_gates() == 11                           # 3 members + 8 families

    def test_ea_seed_formula(self):
        import hashlib
        we = "2026-02-06"
        assert ea.ea_seed(we) == int(hashlib.sha256(
            f"PKT-TB-006-EA{we}".encode()).hexdigest()[:8], 16)


# ================================== EA =========================================
class TestEA:
    @pytest.fixture(scope="class")
    def engine(self, exec_data, lofo_models):
        return ea.FitnessEngine(exec_data, lofo_models, [1, 2])

    def test_genes_do_something(self, engine):
        b0 = ea.Genome.b0()
        u0 = engine.fold_utility(b0, 1)
        for change in ({"member_gate": [0, 1, 1]}, {"feature_gate": [1, 1, 1, 1, 0, 1, 1, 1]},
                       {"record_weight_eps": 0.50}, {"event_weight_cap": 2.0},
                       {"gross_target": 0.30}, {"conviction_temp": 4.0},
                       {"trust_prior": [2.0, -2.0, 0.0]}, {"no_trade_band": 0.030},
                       {"vol_target_ann": 0.06}):
            g = ea.Genome.from_dict({**b0.to_dict(), **change})
            u = engine.fold_utility(g, 1)
            assert u != pytest.approx(u0, abs=1e-12), f"gene dead in walk: {change}"

    def test_ea_smoke(self, engine, tmp_path):
        res = ea.run_ea(engine, tmp_path, "2026-02-06", P=8, G=2, verbose=False)
        gen_files = sorted(tmp_path.glob("generation_*.jsonl"))
        assert len(gen_files) >= 1
        for gf in gen_files:
            rows = [json.loads(l) for l in open(gf)]
            assert len(rows) == 8
            assert any(r["is_b0"] for r in rows), "B0 missing from a generation"
        assert "adopted_champion" in res
        assert res["shipped"] in ("champion", "B0")
        assert (tmp_path / "genome_2026-02-06.json").exists()
        assert (tmp_path / "ea_manifest_2026-02-06.json").exists()
        # determinism of the seed
        assert res["seed"] == ea.ea_seed("2026-02-06")

    def test_adoption_gate_arithmetic(self):
        champ = {"fitness": 1.0, "fold_u": {"1": 1.0, "2": 1.2}}
        b0 = {"fitness": 0.5, "fold_u": {"1": 0.6, "2": 0.7}}
        g = ea.adoption_gate(champ, b0, [1, 2])
        diffs = np.array([0.4, 0.5])
        assert g["cross_fold_sd"] == pytest.approx(diffs.std())
        assert g["margin"] == pytest.approx(0.5)
        assert g["adopted"] is True                                # 0.5 > 0.05
        # not adopted when margin under 1 sd
        champ2 = {"fitness": 0.52, "fold_u": {"1": 1.0, "2": 0.0}}
        b02 = {"fitness": 0.5, "fold_u": {"1": 0.0, "2": 1.0}}
        g2 = ea.adoption_gate(champ2, b02, [1, 2])
        assert g2["adopted"] is False

    def test_fitness_min_over_cost_scenarios(self, engine):
        """Higher-cost scenario can only hurt: U(min over {1.0,1.5}) <= U at 1.0."""
        b0 = ea.Genome.b0()
        import ea as ea_mod
        orig = ea_mod.COST_SCENARIOS
        try:
            ea_mod.COST_SCENARIOS = (1.0,)
            engine._gate_cache.clear()
            u_low = engine.fold_utility(b0, 1)
            ea_mod.COST_SCENARIOS = (1.0, 1.5)
            engine._gate_cache.clear()
            u_min = engine.fold_utility(b0, 1)
        finally:
            ea_mod.COST_SCENARIOS = orig
            engine._gate_cache.clear()
        assert u_min <= u_low + 1e-12


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
