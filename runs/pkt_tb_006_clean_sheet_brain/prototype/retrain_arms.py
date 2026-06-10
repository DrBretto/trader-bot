"""PKT-TB-006 — retrain-arm builder (TOURNAMENT §4.4 RT-2..RT-5; FREEZE_SYN1.md).

Produces the four retrain-arm artifact sets the battery's declared contracts
name (battery.py RETRAIN-DIR NAME CONTRACT), honoring the SYN-1 freeze:

  RT-2 ridge-in-slot     OOF member map cast -> ridge_twin (files exist);
                         executive re-fit -> exec_out_rt2/
  RT-3 LLM-neutral       GBM-Cond + EventHead retrained with ALL llm_* columns
                         at neutral constants (masks 0) -> oof/*_llm_neutral +
                         models_out/<member>_llm_neutral; executive re-fit
                         (z LLM_block columns zeroed) -> exec_out_rt3/
  RT-4 GDELT-ablated     same with the G1-G5 blocks zeroed everywhere ->
                         *_gdelt_ablated + exec_out_rt4/ (z G1/G4G5 zeroed)
  RT-5 Infotropy-B       cast -> cast_uniform OOF (exists) + uniform CAST
                         deploy seeds; executive re-fit with w_rec UNIFORM in
                         the loss AND the walk -> exec_out_rt5/. The shipping
                         GBM is ALREADY uniform (freeze §9.2), EventHead /
                         RiskNet carry no Transfer-B weighting — so R12's
                         contrast is EXACTLY {CAST record-weighting, executive
                         w_rec utility weighting, trust-tilt record weighting}.

Every executive re-fit applies the §7.3 LADDER RULE mechanically: 5 MLP deploy
seeds + the linear twin train on the SAME data; the twin ships unless the MLP
mean val-util beats it; the shipped rung is recorded in <dir>/ladder.json and
LOFO executives OF THE SHIPPED CLASS are trained for the E1 walks. §7.6
fine-tune is NOT applied (de-claimed at freeze: delta exactly 0, §4.6#9).

RiskNet is NOT retrained for RT-3/RT-4: under the FROZEN configuration the
deployed sigma instrument is the trailing-21 proxy (FREEZE sigma-source row);
RiskNet's llm/gdelt columns enter the deployed path nowhere (E4 heads appear
only in the R07 contrast). Forced choice, logged.

Usage (each subcommand idempotent):
  .venv/bin/python retrain_arms.py members --which rt3|rt4|rt5cast|noscreen
  .venv/bin/python retrain_arms.py exec --rt rt2|rt3|rt4|rt5
  .venv/bin/python retrain_arms.py lofo-twins-base
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

PROTO = Path(__file__).resolve().parent
sys.path.insert(0, str(PROTO))

import executive as ex      # noqa: E402
import folds as fd          # noqa: E402
import members as M         # noqa: E402
import synth_oof as so      # noqa: E402
import train_members as TM  # noqa: E402
from e1_reads import member_map_ctx  # noqa: E402
from ea import FEATURE_GATE_Z_MAP    # noqa: E402

import torch                # noqa: E402

OOF = PROTO / "oof"
MODELS = PROTO / "models_out"
FOLDS = [1, 2, 3, 4, 5, 6]

# ---- neutralization column lists (panel T/B; masks set to 0 = "never there")
LLM_T_COLS = ["llm_sent", "llm_conf", "llm_sal", "llm_sent_d1", "m_llm_available"]
LLM_B_COLS = (["llm_sent_x", "llm_conf_x", "llm_sal_x", "llm_sent_d1_x",
               "llm_available"] + [f"llm_event_{i:02d}_x" for i in range(12)])
GDELT_T_COLS = ["gd_g1_share", "gd_g1_z", "gd_g3_tone", "gd_g4_burst_z",
                "gd_novelty", "gd_novelty_21", "gd_hhi_loc", "gd_hhi_org",
                "m_gdelt_available"]
GDELT_B_COLS = ["g1_z", "g3_tone", "g3_tone_dispersion", "g2_goldstein",
                "g2_conflict", "g4_burst_z", "g4_novelty", "g5_hhi_loc",
                "g5_hhi_org"]

# executive z columns zeroed at TRAIN time per arm (mirrors the arm's gates)
EXEC_Z_OFF = {
    "rt2": [],
    "rt3": FEATURE_GATE_Z_MAP["LLM_block"],
    "rt4": FEATURE_GATE_Z_MAP["G1_themes"] + FEATURE_GATE_Z_MAP["G4G5_novelty_conc"],
    "rt5": [],
}
EXEC_OOF_MAP = {
    "rt2": {"cast": "ridge_twin"},
    "rt3": {"gbm_cond": "gbm_cond_llm_neutral",
            "event_head": "event_head_llm_neutral"},
    "rt4": {"gbm_cond": "gbm_cond_gdelt_ablated",
            "event_head": "event_head_gdelt_ablated"},
    "rt5": {"cast": "cast_uniform"},
}
EXEC_UNIFORM_WREC = {"rt2": False, "rt3": False, "rt4": False, "rt5": True}
EXEC_DIR = {t: PROTO / f"exec_out_{t}" for t in EXEC_OOF_MAP}


def log_look(component: str, decision: str, provenance: str) -> None:
    entry = {"ts": dt.datetime.now().isoformat(timespec="seconds"),
             "component": component, "decision": decision,
             "provenance": provenance, "phase": "D-battery-build"}
    with (PROTO / "validation_looks.jsonl").open("a") as f:
        f.write(json.dumps(entry) + "\n")


def neutralize_panel(panel: dict, mode: str) -> dict:
    """Return a panel copy with the mode's T/B columns at neutral constants
    (0 = the 'feed never available' value: every llm/gdelt column is
    mask-interacted or zero-when-unavailable in feature_store)."""
    t_cols = [str(c) for c in panel["T_cols"]]
    b_cols = [str(c) for c in panel["B_cols"]]
    tc, bc = (LLM_T_COLS, LLM_B_COLS) if mode == "llm" else (GDELT_T_COLS, GDELT_B_COLS)
    out = dict(panel)
    T = panel["T"].copy()
    B = panel["B"].copy()
    for c in tc:
        T[:, :, t_cols.index(c)] = 0.0
    for c in bc:
        B[:, :, b_cols.index(c)] = 0.0
    out["T"], out["B"] = T, B
    return out


# ============================ member retrains ==================================
def members_rt(which: str) -> None:
    command = "python " + " ".join(sys.argv)
    panel = TM.load_panel()
    dates = panel["dates"]
    screen = json.loads((PROTO / "store" / "infotropy_a_screen.json").read_text())

    if which in ("rt3", "rt4"):
        mode = "llm" if which == "rt3" else "gdelt"
        suffix = "_llm_neutral" if which == "rt3" else "_gdelt_ablated"
        pn = neutralize_panel(panel, mode)
        # shipping config mirrored exactly: GBM uniform weights + r3_only
        # routing (FREEZE), EventHead r3_only — only the columns differ.
        TM.run_gbm(pn, dates, FOLDS, screen, uniform=True, deploy=True,
                   command=command, smoke=False, variant="r3_only",
                   member=f"gbm_cond{suffix}")
        TM.run_event(pn, dates, FOLDS, screen, deploy=True, command=command,
                     smoke=False, variant="r3_only", member=f"event_head{suffix}")
        log_look(f"retrain_{which}_members",
                 f"{which.upper()}: GBM-Cond + EventHead retrained with "
                 f"{mode} columns at neutral constants (masks 0): T={ {'llm': LLM_T_COLS, 'gdelt': GDELT_T_COLS}[mode] }, "
                 f"B={ {'llm': LLM_B_COLS, 'gdelt': GDELT_B_COLS}[mode] }; shipping config mirrored "
                 f"(uniform GBM weights, r3_only routing). CAST is text/GDELT-free "
                 f"(X = price/volume only) — not retrained. RiskNet NOT retrained: "
                 f"frozen deployed sigma = trailing-21 proxy; E4 heads appear only "
                 f"in the R07 contrast (forced choice).",
                 f"retrain_arms.py members --which {which}")
    elif which == "rt5cast":
        # uniform CAST deploy seeds (the uniform OOF twin exists from wave 3)
        sec_ids, cls_ids = M.sector_class_ids(PROTO.parents[2] / "config" / "universe.csv")
        dm = fd.deploy_train_mask(np.asarray(dates).astype(str))
        train_idx = np.nonzero(dm)[0]
        eps = []
        for f in FOLDS:
            p = OOF / f"fold_{f}_cast_uniform.npz"
            man = json.loads(str(np.load(p, allow_pickle=True)["manifest"]))
            eps += man.get("best_epochs", [])
        ep = max(5, int(np.median(eps))) if eps else 20
        out = MODELS / "cast_uniform"
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        deploy_seeds = fd.DEPLOY_SEEDS[:3]          # mirrors shipping reduction
        for seed in deploy_seeds:
            sd, *_ = M.train_cast_one(panel, train_idx, np.array([], dtype=int),
                                      seed, sec_ids, cls_ids, uniform_w=True,
                                      max_epochs=ep, patience=ep, days_per_step=8)
            torch.save(sd, out / f"seed_{seed}.pt")
            print(f"[cast_uniform deploy seed {seed}] done "
                  f"({time.time()-t0:.0f}s)", flush=True)
        (out / "manifest.json").write_text(TM.manifest(
            "cast_uniform", fd.FINE_TUNE_CUTOFF, deploy_seeds,
            dict(epochs=ep, days_per_step=8, lr=1e-3, wd=1e-3, uniform_w=True),
            time.time() - t0, command, False,
            extra={"deploy": True,
                   "fixed_epochs_provenance": "median best_epoch across the "
                                              "cast_uniform OOF folds"}))
        log_look("retrain_rt5_cast_uniform_deploy",
                 f"RT-5 deploy-side: uniform-weight CAST twin deploy ensemble "
                 f"(3 seeds {deploy_seeds}, fixed epochs {ep} = median uniform-OOF "
                 f"best_epoch) trained for the variant nightly store; identical "
                 f"architecture/recipe to the shipping CAST except w_rec==1 in "
                 f"the Huber loss term",
                 "retrain_arms.py members --which rt5cast")
    elif which == "noscreen":
        # Infotropy-A fold-level read twin: EventHead with NO Transfer-A screen
        TM.run_event(panel, dates, FOLDS, screen, deploy=False, command=command,
                     smoke=False, variant="no_screen", member="event_head_noscreen")
        log_look("infotropy_a_noscreen_twin",
                 "Infotropy-A fold-level read (D5: E1-only, no replay): the "
                 "shipping EventHead is R3-only screened (freeze: conjunctive "
                 "gate DEAD), so the read's without-side is a NO-SCREEN twin "
                 "(every family passes; same elastic-net recipe) -> "
                 "oof/fold_*_event_head_noscreen.npz. Member-twin training for "
                 "an E1-only read; consumes NO replay look and NO RT cycle "
                 "(forced choice)",
                 "retrain_arms.py members --which noscreen")
    else:
        raise SystemExit(f"unknown --which {which}")


# ============================ executive re-fits ================================
def refit_exec(rt: str, max_epochs: int = 200) -> dict:
    out_dir = EXEC_DIR[rt]
    out_dir.mkdir(exist_ok=True)
    world = so.world_from_panel(PROTO / "store" / "panel.npz")
    with member_map_ctx(EXEC_OOF_MAP[rt]):
        data = ex.assemble_exec_data(world, OOF, FOLDS, end_date=fd.FITNESS_END)
    z_off = EXEC_Z_OFF[rt]
    if z_off:
        data.z[:, z_off] = 0.0                       # arm's z gating at train time
    if EXEC_UNIFORM_WREC[rt]:
        data.w_rec = np.ones_like(data.w_rec)        # uniform loss weighting
        data.w_rec_raw = np.ones_like(data.w_rec_raw)  # uniform walk/tilt weighting
    # ---- 5 MLP deploy seeds + the linear twin (same recipe as exec_out) -------
    t0 = time.time()
    for s in ex.DEPLOY_SEEDS:
        log: list = []
        m = ex.train_executive(data, seed=s, max_epochs=max_epochs, loss_log=log)
        torch.save(m.state_dict(), out_dir / f"executive_seed{s}.pt")
        (out_dir / f"loss_seed{s}.json").write_text(json.dumps(log))
        print(f"[{rt} exec seed {s}] done ({time.time()-t0:.0f}s)", flush=True)
    twin = ex.train_executive(data, seed=11, model_cls=ex.LinearGate,
                              max_epochs=max_epochs)
    torch.save(twin.state_dict(), out_dir / "linear_twin.pt")
    print(f"[{rt} twin] done ({time.time()-t0:.0f}s)", flush=True)
    # ---- §7.3 ladder rule, mechanical -----------------------------------------
    from exec_finalize import val_utility
    _, va_idx = ex.lofo_train_indices(data, None)
    mlp_val = {}
    for s in ex.DEPLOY_SEEDS:
        m = ex.Executive(seed=s)
        m.load_state_dict(torch.load(out_dir / f"executive_seed{s}.pt"))
        mlp_val[str(s)] = val_utility(m, data, va_idx)
    twin_val = val_utility(twin, data, va_idx)
    mlp_mean = float(np.mean(list(mlp_val.values())))
    ships = "mlp" if mlp_mean > twin_val else "linear_twin"
    # ---- LOFO executives of the SHIPPED class (E1 walk inputs) ----------------
    cls, pat = ((ex.LinearGate, "lofo_twin_fold{f}.pt") if ships == "linear_twin"
                else (ex.Executive, "lofo_fold{f}.pt"))
    for f in FOLDS:
        m = ex.train_executive(data, seed=11, model_cls=cls, exclude_fold=f,
                               max_epochs=max_epochs)
        torch.save(m.state_dict(), out_dir / pat.format(f=f))
        print(f"[{rt} lofo {ships} F{f}] done ({time.time()-t0:.0f}s)", flush=True)
    ladder = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "rt": rt, "oof_member_map": EXEC_OOF_MAP[rt],
        "z_cols_zeroed_at_train": z_off,
        "uniform_w_rec": EXEC_UNIFORM_WREC[rt],
        "mlp_val_util": mlp_val, "mlp_val_util_mean": mlp_mean,
        "linear_twin_val_util": twin_val,
        "ships": ships,
        "replay_exec_mode": "learned" if ships == "mlp" else "linear_twin",
        "fine_tune": "not applied (de-claimed at freeze: delta exactly 0, §4.6#9)",
        "lofo_files": pat.replace("{f}", "<f>"),
        "n_days": int(len(data.dates)), "val_days": int(len(va_idx)),
        "wall_clock_s": round(time.time() - t0, 1),
    }
    (out_dir / "ladder.json").write_text(json.dumps(ladder, indent=1))
    log_look(f"retrain_{rt}_executive",
             f"{rt.upper()} executive re-fit (same seeds/recipe as exec_out; "
             f"OOF map {EXEC_OOF_MAP[rt]}; z zeroed {z_off}; uniform w_rec "
             f"{EXEC_UNIFORM_WREC[rt]}): ladder rule applied mechanically — "
             f"MLP mean val util {mlp_mean:.3e} vs twin {twin_val:.3e} -> "
             f"SHIPS {ships}; LOFO executives of the shipped class trained for "
             f"the E1 walks; fine-tune not applied (de-claimed at freeze)",
             f"{out_dir.name}/ladder.json")
    print(json.dumps(ladder, indent=1))
    return ladder


def lofo_twins_base(max_epochs: int = 200) -> None:
    """LOFO LinearGate twins for the BASE exec_out (the frozen gate's E1 walk)
    + ladder.json transcribing the already-recorded freeze decision."""
    out_dir = PROTO / "exec_out"
    world = so.world_from_panel(PROTO / "store" / "panel.npz")
    data = ex.assemble_exec_data(world, OOF, FOLDS, end_date=fd.FITNESS_END)
    t0 = time.time()
    for f in FOLDS:
        m = ex.train_executive(data, seed=11, model_cls=ex.LinearGate,
                               exclude_fold=f, max_epochs=max_epochs)
        torch.save(m.state_dict(), out_dir / f"lofo_twin_fold{f}.pt")
        print(f"[base lofo twin F{f}] done ({time.time()-t0:.0f}s)", flush=True)
    diag = json.loads((out_dir / "final_diagnostics.json").read_text())
    (out_dir / "ladder.json").write_text(json.dumps({
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "rt": "rt1_base",
        "mlp_val_util_mean": diag["mlp_val_util_mean"],
        "linear_twin_val_util": diag["linear_twin_val_util"],
        "ships": diag["ladder_ships"],                  # 'linear_twin' (FROZEN)
        "replay_exec_mode": ("learned" if diag["ladder_ships"] == "mlp"
                             else "linear_twin"),
        "fine_tune": "executed, delta exactly 0 on all seeds -> de-claimed (§4.6#9)",
        "lofo_files": "lofo_twin_fold<f>.pt (frozen gate class; MLP "
                      "lofo_fold<f>.pt retained for EA fitness walks per TR S2)",
        "provenance": "exec_out/final_diagnostics.json (freeze record)",
        "wall_clock_s": round(time.time() - t0, 1),
    }, indent=1))
    log_look("base_lofo_twins",
             "FROZEN-GATE MIRRORING: 6 LOFO LinearGate twins trained for "
             "exec_out (same recipe as the LOFO MLPs, model_cls=LinearGate) so "
             "the E1 base side walks the SAME gate class the replay deploys "
             "(linear twin per FREEZE). MLP LOFO files retained unchanged for "
             "the EA fitness walks (TR S2). exec_out/ladder.json transcribes "
             "the freeze ladder decision",
             "exec_out/ladder.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    mp = sub.add_parser("members")
    mp.add_argument("--which", required=True,
                    choices=["rt3", "rt4", "rt5cast", "noscreen"])
    ep_ = sub.add_parser("exec")
    ep_.add_argument("--rt", required=True, choices=list(EXEC_OOF_MAP))
    ep_.add_argument("--max-epochs", type=int, default=200)
    lt = sub.add_parser("lofo-twins-base")
    lt.add_argument("--max-epochs", type=int, default=200)
    args = ap.parse_args()
    if args.cmd == "members":
        members_rt(args.which)
    elif args.cmd == "exec":
        refit_exec(args.rt, args.max_epochs)
    elif args.cmd == "lofo-twins-base":
        lofo_twins_base(args.max_epochs)


if __name__ == "__main__":
    main()
