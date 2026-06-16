"""Extract the live brain's *runtime subset* out of the research prototype trees
into a small, deployable staging dir the Lambda image bakes (PKT-TB-012,
DESIGN_DOSSIER reports/03 §0/§2).

The frozen brain's forward inference (`forward_inference.run_inference`) imports a
handful of modules that today live inside the 4.2 GB TB-006/TB-007 prototype
trees (laptop-only). "Productionizing" is NOT a launchd->EventBridge swap; it is
carving the runtime CODE + the frozen seed CACHES out of those trees into a
deployable artifact. This script is that carve, scripted and idempotent so a
fresh checkout reproduces the exact baked subset.

    python -m src.brain.bake_runtime_subset            # full bake (code + data)
    python -m src.brain.bake_runtime_subset --code-only # CI: resolve modules only

Output (default ``build/brain_bake/``, git-ignored — assembled at build time):
    code/            the runtime .py modules + dicts/ (sys.path entry)
    models_out_007/  the 4 frozen model files (320 KB)  [also copied to brain/]
    seed_caches/     ohlcv / cboe / fred / cot / gdelt frozen seed caches (~108 MB)
    reference/nightly_007/  parity reference JSONs
    FREEZE_ORB1.json, universe.csv

Dockerfile.lambda then ``COPY build/brain_bake/`` into the image and sets
``BRAIN_RUNTIME_SUBSET=${LAMBDA_TASK_ROOT}/brain/code``. The research trees stay
read-only and are NEVER deployed.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

_REPO = Path(__file__).resolve().parents[2]
_TB7 = _REPO / "runs" / "pkt_tb_007_orthogonal_brain"
_TB7_PROTO = _TB7 / "prototype"
_TB7_SHADOW = _TB7 / "shadow"
_TB6_PROTO = _REPO / "runs" / "pkt_tb_006_clean_sheet_brain" / "prototype"

# The runtime CODE subset (forward_inference + everything it imports at the
# inference path). tilt_adapter / lot_fix_007 / run_replay_007 are pulled in by
# shadow_lib's lazy imports; included so the import graph resolves, but the LIVE
# brain is the two-stage engine and never calls the tilt.
CODE_MODULES: List[Tuple[Path, str]] = [
    (_TB7_SHADOW / "forward_inference.py", "forward_inference.py"),
    (_TB7_SHADOW / "shadow_lib.py", "shadow_lib.py"),
    (_TB7_PROTO / "organs_007.py", "organs_007.py"),
    (_TB7_PROTO / "folds.py", "folds.py"),
    (_TB7_PROTO / "make_targets_007.py", "make_targets_007.py"),
    (_TB7_PROTO / "risk_stats_007.py", "risk_stats_007.py"),
    (_TB7_PROTO / "genome_007.py", "genome_007.py"),
    (_TB7_PROTO / "lot_fix_007.py", "lot_fix_007.py"),
    (_TB7_PROTO / "run_replay_007.py", "run_replay_007.py"),
    (_TB7_PROTO / "tilt_adapter.py", "tilt_adapter.py"),
    (_TB7_PROTO / "precompute_nightly_007.py", "precompute_nightly_007.py"),
    (_TB7_PROTO / "gates_007.py", "gates_007.py"),
    (_TB7_PROTO / "stats.py", "stats.py"),
    (_TB6_PROTO / "feature_store.py", "feature_store.py"),
    (_TB6_PROTO / "features_gdelt.py", "features_gdelt.py"),
    (_TB6_PROTO / "data_layer.py", "data_layer.py"),
    (_TB6_PROTO / "gdelt_backfill.py", "gdelt_backfill.py"),
]

MODELS_ROOT = _TB7_PROTO / "models_out_007"
DICTS_DIR = _TB7_PROTO / "dicts"
NIGHTLY_REF = _TB7_PROTO / "store" / "nightly_007"
SEED_CACHES: List[Tuple[Path, str]] = [
    (_TB6_PROTO / "cache" / "ohlcv", "seed_caches/ohlcv"),
    (_TB6_PROTO / "cache" / "cboe", "seed_caches/cboe"),
    (_TB6_PROTO / "cache" / "fred", "seed_caches/fred"),
    (_TB6_PROTO / "cache" / "cot", "seed_caches/cot"),
    (_TB6_PROTO / "gdelt_cache" / "daily", "seed_caches/gdelt/daily"),
]


def _copy_dir(src: Path, dst: Path) -> int:
    if not src.exists():
        return -1
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return sum(1 for _ in dst.rglob("*") if _.is_file())


def bake(stage: Path, code_only: bool = False) -> int:
    stage.mkdir(parents=True, exist_ok=True)
    missing: List[str] = []

    # 1. code modules
    code = stage / "code"
    code.mkdir(parents=True, exist_ok=True)
    for src, name in CODE_MODULES:
        if src.exists():
            shutil.copy2(src, code / name)
        else:
            missing.append(str(src))
    n_dicts = _copy_dir(DICTS_DIR, code / "dicts")
    if n_dicts < 0:
        missing.append(str(DICTS_DIR))

    # 2. frozen models (also placed under brain/models_out_007 for the cold-start
    #    assertion path that resolve_model_root() prefers).
    n_models = _copy_dir(MODELS_ROOT, stage / "models_out_007")
    if n_models < 0:
        missing.append(str(MODELS_ROOT))
    else:
        _copy_dir(MODELS_ROOT, _REPO / "brain" / "models_out_007")

    # 3. parity reference
    n_ref = _copy_dir(NIGHTLY_REF, stage / "reference" / "nightly_007")
    if n_ref < 0:
        missing.append(str(NIGHTLY_REF))

    # 4. pointers
    for rel in ("brain/FREEZE_ORB1.json", "config/universe.csv"):
        src = _REPO / rel
        if src.exists():
            shutil.copy2(src, stage / Path(rel).name)
        else:
            missing.append(str(src))

    # 5. seed caches (heavy; skip under --code-only)
    seed_files = 0
    if not code_only:
        for src, rel in SEED_CACHES:
            n = _copy_dir(src, stage / rel)
            if n < 0:
                missing.append(str(src))
            else:
                seed_files += n

    print(f"baked code modules: {len(CODE_MODULES)} (dicts files={n_dicts})")
    print(f"baked models: {n_models} | reference: {n_ref} | seed files: {seed_files}"
          + (" (skipped: --code-only)" if code_only else ""))
    if missing:
        print("MISSING SOURCES (bake incomplete):")
        for m in missing:
            print("  - " + m)
        return 1
    print(f"bake OK -> {stage}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default=str(_REPO / "build" / "brain_bake"))
    ap.add_argument("--code-only", action="store_true",
                    help="resolve/copy code modules only (skip the heavy seed caches)")
    args = ap.parse_args(argv)
    return bake(Path(args.stage), code_only=args.code_only)


if __name__ == "__main__":
    sys.exit(main())
