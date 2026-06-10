"""PKT-TB-006 — end-to-end brain smoke on REAL panel dates + SYNTHETIC member OOFs.

    .venv/bin/python smoke_brain.py [--folds 5,6] [--epochs 30]

Pipeline: synthetic OOFs (walk_forward-flagged) -> solo books -> trust ledger ->
executive 30-epoch train (loss curve printed) -> linear twin -> LOFO executives ->
meta_decision.json sample -> fine-tune (7 params) -> EA smoke (P=8, G=2) ->
champion-vs-B0 adoption-gate readout. Everything seeded. Writes ONLY under
smoke_out/ (synthetic artifacts never land in oof/ or ea/).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import books as bk
import ea
import executive as ex
import folds as fd
import synth_oof as so

PROTO = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="5,6")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--ea-pop", type=int, default=8)
    ap.add_argument("--ea-gens", type=int, default=2)
    ap.add_argument("--b1-k", type=int, default=20)
    args = ap.parse_args()
    t0 = time.time()
    out = PROTO / "smoke_out"
    out.mkdir(exist_ok=True)
    fold_list = [int(x) for x in args.folds.split(",")]

    print("== [1] world from REAL panel (synthetic member opinions) ==")
    world = so.world_from_panel(PROTO / "store" / "panel.npz")
    print(f"   dates {world['dates'][0]}..{world['dates'][-1]} n={len(world['dates'])}")

    print("== [2] synthetic OOFs (seeded, walk_forward-flagged) ==")
    masks = {f: fd.fold_date_mask(world["dates"], f) for f in fold_list}
    for f in fold_list:
        fd.assert_no_holdout(world["dates"], masks[f])
        print(f"   fold {f}: {masks[f].sum()} decision dates (<= {fd.FITNESS_END})")
    paths = so.write_smoke_oofs(world, out / "oof_smoke", masks)
    print(f"   wrote {len(paths)} OOF files -> {out / 'oof_smoke'}")

    print("== [3] solo books + trust ledger ==")
    data = ex.assemble_exec_data(world, out / "oof_smoke", fold_list)
    gross = data.books.sum(axis=2)
    print(f"   rows {len(data.dates)}; solo-book gross mean by member "
          f"{dict(zip(bk.MEMBERS, np.round(gross.mean(axis=0), 3)))}")
    print(f"   ledger stats nonzero frac: {np.mean(np.any(data.r != 0, axis=(1, 2))):.2f}")
    u_means = {m: float(np.nanmean(data.u_real[:, i])) for i, m in enumerate(bk.MEMBERS)}
    print(f"   realized counterfactual u means: { {k: round(v, 6) for k, v in u_means.items()} }")

    print(f"== [4] executive train ({args.epochs} epochs, seed 11) ==")
    print(f"   executive params: {ex.Executive().n_params()} "
          f"(ceiling {ex.PARAM_CEILING}); linear twin: {ex.LinearGate().n_params()}")
    log: list = []
    model = ex.train_executive(data, seed=11, max_epochs=args.epochs,
                               patience=args.epochs, verbose=True, loss_log=log)
    with open(out / "exec_loss_curve.json", "w") as fh:
        json.dump(log, fh, indent=1)
    torch.save(model.state_dict(), out / "executive_seed11.pt")
    print(f"   loss {log[0]['train_loss']:+.6f} -> {log[-1]['train_loss']:+.6f}; "
          f"val_U {log[0]['val_util']:+.6f} -> best {max(r['val_util'] for r in log):+.6f}")

    print("== [5] linear-gate twin ==")
    twin_log: list = []
    twin = ex.train_executive(data, seed=11, model_cls=ex.LinearGate,
                              max_epochs=args.epochs, patience=args.epochs,
                              loss_log=twin_log)
    print(f"   twin params {twin.n_params()}; "
          f"best val_U {max(r['val_util'] for r in twin_log):+.6f} "
          f"(MLP best {max(r['val_util'] for r in log):+.6f})")

    print("== [6] LOFO executives ==")
    lofo = ex.train_lofo_executives(data, fold_list, max_epochs=max(5, args.epochs // 3),
                                    patience=5)
    for f, m in lofo.items():
        torch.save(m.state_dict(), out / f"lofo_fold{f}.pt")
    print(f"   trained {len(lofo)} LOFO executives (folds {fold_list})")

    print("== [7] audit record (meta_decision.json, 16-step IG) ==")
    rec = ex.write_meta_decision(model, data, len(data.dates) // 2, out)
    print(f"   date {rec['date']} tau "
          f"{[round(e['trust'], 3) for e in rec['experts']]} "
          f"f {rec['deployment_fraction']:.3f} gross {rec['book']['gross']:.3f} "
          f"entropy {rec['trust_entropy']:.3f}")

    print("== [8] fine-tune (the 7 named params only) ==")
    ft = ex.fine_tune(model, data)  # real window 2025-08-04..2026-03-03 on fold-6 dates
    print(f"   window {ft['window']} pre {ft['pre_val_util']:+.6f} "
          f"post {ft['post_val_util']:+.6f} delta {ft['delta']:+.6f}; "
          f"changed {ft['changed_params']}")

    print(f"== [9] EA smoke (P={args.ea_pop}, G={args.ea_gens}) ==")
    engine = ea.FitnessEngine(data, lofo, fold_list)
    res = ea.run_ea(engine, out / "ea_smoke", fd.FITNESS_END,
                    P=args.ea_pop, G=args.ea_gens)
    print("== [10] champion vs B0 adoption-gate readout ==")
    print(f"   champion FITNESS {res['champion_fitness']:+.4f} | "
          f"B0 {res['b0_fitness']:+.4f} | margin {res['margin']:+.4f} | "
          f"cross-fold sd {res['cross_fold_sd']:.4f} -> SHIPPED: {res['shipped']}")
    print(f"   champion fold U {res['champion_fold_u']} | B0 fold U {res['b0_fold_u']}")
    b1 = ea.run_random_baseline(engine, fd.FITNESS_END, K=args.b1_k,
                                out_path=out / "ea_smoke" / "b1_smoke.json")
    print(f"   B1 random-search (K={args.b1_k} smoke) best {b1['best_fitness']:+.4f}")
    print(f"== DONE in {time.time() - t0:.1f}s; artifacts in {out} ==")


if __name__ == "__main__":
    main()
