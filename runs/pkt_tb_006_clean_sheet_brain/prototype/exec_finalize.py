"""PKT-TB-006 — executive FINAL diagnostics + §7.6 fine-tune (BUILD_SPEC §15 step 8).

Run AFTER executive.py main() has written exec_out/. Does, in order:
  1. val-utility ladder read: 5 MLP deploy seeds vs the linear twin on the same
     embargoed global validation split (challenger-parity rung, §7.5#5);
  2. per-fold std(tau) static-trust diagnostic (§7.5#2, read per-fold);
  3. calibration corr (§7.5#4): corr(U_h_hat = (tau·ewma21)_sum, realized
     (tau·u_real)_sum) on the fine-tune window (proposal §6 estimator);
  4. §7.6 fine-tune of each deploy seed (exact 7-param set), pre/post val delta;
     pre-fine-tune weights stashed in exec_out/pre_finetune/ (no glob clash);
  5. writes exec_out/final_diagnostics.json with everything + manifest fields.

The fine-tune window is clipped at FITNESS_END by data construction (OOF member
outputs end with F6); the registered cutoff 2026-03-03 stays the upper bound
(vacuously). Forced choice logged to validation_looks.jsonl.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROTO = Path(__file__).resolve().parent
sys.path.insert(0, str(PROTO))

import executive as ex   # noqa: E402
import folds as fd       # noqa: E402
import members as M      # noqa: E402
import synth_oof as so   # noqa: E402


def val_utility(model, data, va_idx, sigma_cap=ex.SIGMA_CAP_B0,
                lam_dn=ex.LAMBDA_DN_B0, no_trade_band=0.010) -> float:
    t_va = ex.ExecTensors(data, va_idx)
    T_u = float(max(np.std(data.u_real[va_idx]), 1e-4))
    model.eval()
    with torch.no_grad():
        wp, dd = ex.sequential_state(model, t_va, sigma_cap, no_trade_band)
        out = ex.forward_batch(model, t_va, wp, dd, sigma_cap, no_trade_band, False)
        _, u = ex.compute_loss(model, out, t_va, wp, lam_dn, T_u, None, False)
    return float(u)


def tau_series(model, data, idx) -> np.ndarray:
    t = ex.ExecTensors(data, idx)
    with torch.no_grad():
        zz = t.z.unsqueeze(1).expand(-1, ex.N_MEMBERS, -1)
        x_tr = torch.cat([t.r, t.c.unsqueeze(-1), t.agree.unsqueeze(-1), zz], dim=-1)
        tau, _ = model.trust(x_tr)
    return tau.numpy()


def main():
    t0 = time.time()
    out_dir = PROTO / "exec_out"
    world = so.world_from_panel(PROTO / "store" / "panel.npz")
    fold_list = [1, 2, 3, 4, 5, 6]
    data = ex.assemble_exec_data(world, PROTO / "oof", fold_list,
                                 end_date=fd.FITNESS_END)
    tr_idx, va_idx = ex.lofo_train_indices(data, None)

    seeds, models = list(ex.DEPLOY_SEEDS), {}
    for s in seeds:
        m = ex.Executive(seed=s)
        m.load_state_dict(torch.load(out_dir / f"executive_seed{s}.pt"))
        models[s] = m
    twin = ex.LinearGate(seed=11)
    twin.load_state_dict(torch.load(out_dir / "linear_twin.pt"))

    # 1. ladder read --------------------------------------------------------
    mlp_val = {s: val_utility(models[s], data, va_idx) for s in seeds}
    twin_val = val_utility(twin, data, va_idx)
    mlp_mean = float(np.mean(list(mlp_val.values())))
    ladder = "mlp" if mlp_mean > twin_val else "linear_twin"

    # 2. per-fold std(tau) (mean ensemble tau, per member) -------------------
    std_tau = {}
    for f in fold_list:
        idx = np.where(data.fold_of == f)[0]
        taus = np.mean([tau_series(models[s], data, idx) for s in seeds], axis=0)
        std_tau[f"F{f}"] = [float(x) for x in taus.std(axis=0)]
    static_trust_dead = {k: bool(max(v) < 0.02) for k, v in std_tau.items()}

    # 3. calibration corr on the fine-tune window ----------------------------
    sel = (data.dates >= fd.FINE_TUNE_START) & (data.dates <= fd.FINE_TUNE_CUTOFF)
    idx_ft = np.where(sel)[0]
    taus_ft = np.mean([tau_series(models[s], data, idx_ft) for s in seeds], axis=0)
    u_hat = (taus_ft * data.r[idx_ft, :, 0]).sum(axis=1)       # proposal §6 estimator
    u_realized = (taus_ft * data.u_real[idx_ft]).sum(axis=1)
    calib_corr = float(np.corrcoef(u_hat, u_realized)[0, 1])

    # 4. fine-tune (§7.6) each deploy seed -----------------------------------
    pre_dir = out_dir / "pre_finetune"
    pre_dir.mkdir(exist_ok=True)
    ft_results = {}
    for s in seeds:
        torch.save(models[s].state_dict(), pre_dir / f"executive_seed{s}.pt")
        ft = ex.fine_tune(models[s], data)
        ft_results[str(s)] = ft
        torch.save(models[s].state_dict(), out_dir / f"executive_seed{s}.pt")
    deltas = [ft_results[str(s)]["delta"] for s in seeds]
    post_val = {s: val_utility(models[s], data, va_idx) for s in seeds}

    res = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "walk_forward": True, "seeds": seeds, "code_sha": M.code_sha(),
        "command": "python " + " ".join(sys.argv),
        "n_days": int(len(data.dates)), "val_days": int(len(va_idx)),
        "mlp_val_util": {str(k): v for k, v in mlp_val.items()},
        "mlp_val_util_mean": mlp_mean,
        "linear_twin_val_util": twin_val,
        "ladder_ships": ladder,
        "std_tau_per_fold": std_tau,
        "static_trust_dead_per_fold": static_trust_dead,
        "calibration_corr": calib_corr,
        "calibration_window": [fd.FINE_TUNE_START, str(sorted(data.dates[idx_ft])[-1])],
        "fine_tune": ft_results,
        "fine_tune_mean_delta": float(np.mean(deltas)),
        "post_ft_val_util": {str(k): v for k, v in post_val.items()},
        "wall_clock_s": round(time.time() - t0, 1),
    }
    (out_dir / "final_diagnostics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in
                      ["mlp_val_util_mean", "linear_twin_val_util", "ladder_ships",
                       "calibration_corr", "fine_tune_mean_delta"]}, indent=1))
    for f in fold_list:
        print(f"F{f} std(tau) per member: "
              + " ".join(f"{x:.4f}" for x in std_tau[f"F{f}"])
              + ("  [STATIC-TRUST FLAG]" if static_trust_dead[f"F{f}"] else ""))


if __name__ == "__main__":
    main()
