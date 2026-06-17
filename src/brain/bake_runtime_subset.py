"""Extract the live brain's *runtime subset* out of the research prototype trees
into a deployable staging dir the Lambda image bakes (PKT-TB-012,
DESIGN_DOSSIER reports/03 §0/§2).

The frozen brain's forward inference (`forward_inference.run_inference`) imports a
handful of modules + frozen seed caches that live inside the TB-006/TB-007
prototype trees (laptop-only). "Productionizing" is carving the runtime CODE +
the frozen seed CACHES out of those trees into a deployable artifact.

We bake the subset **mirroring its original repo-relative layout** under
``build/brain_bake/``, because every module computes its paths from its own file
location (e.g. ``REPO = PROTO7.parents[2]``, ``TB6_PROTO = REPO/runs/...``). Baking
at the original paths means those constants resolve byte-for-byte inside the image
— no monkeypatching. The only thing redirected at runtime is the WRITABLE state
tree (``BRAIN_STATE_DIR`` → ``/tmp`` in Lambda; the read-only seeds are copied in
at cold start by ``ensure_seed_caches``).

    python -m src.brain.bake_runtime_subset            # full bake (code + seeds)
    python -m src.brain.bake_runtime_subset --code-only # CI: resolve modules only

Dockerfile.lambda then ``COPY build/brain_bake/ ${LAMBDA_TASK_ROOT}/`` so the
mirrored ``runs/...`` tree lands at ``/var/task/runs/...``. The research trees stay
read-only and are NEVER deployed.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List

_REPO = Path(__file__).resolve().parents[2]
_TB7 = "runs/pkt_tb_007_orthogonal_brain"
_TB6 = "runs/pkt_tb_006_clean_sheet_brain"

# Repo-relative FILES copied verbatim (preserving layout).
CODE_FILES: List[str] = [
    f"{_TB7}/shadow/forward_inference.py",
    f"{_TB7}/shadow/shadow_lib.py",
    f"{_TB7}/prototype/organs_007.py",
    f"{_TB7}/prototype/folds.py",
    f"{_TB7}/prototype/make_targets_007.py",
    f"{_TB7}/prototype/risk_stats_007.py",
    f"{_TB7}/prototype/genome_007.py",
    f"{_TB7}/prototype/lot_fix_007.py",
    f"{_TB7}/prototype/run_replay_007.py",
    f"{_TB7}/prototype/tilt_adapter.py",
    f"{_TB7}/prototype/precompute_nightly_007.py",
    f"{_TB7}/prototype/gates_007.py",
    f"{_TB7}/prototype/stats.py",
    f"{_TB6}/prototype/feature_store.py",
    f"{_TB6}/prototype/features_gdelt.py",
    f"{_TB6}/prototype/data_layer.py",
    f"{_TB6}/prototype/gdelt_backfill.py",
    f"{_TB6}/prototype/store/llm_features.parquet",
]

# Repo-relative DIRS copied verbatim (dicts, reference, models).
CODE_DIRS: List[str] = [
    f"{_TB7}/prototype/dicts",
    f"{_TB7}/prototype/store/nightly_007",     # parity reference
    f"{_TB7}/prototype/models_out_007",        # the frozen weights
    f"{_TB6}/prototype/dicts",
]

# Repo-relative SEED CACHE dirs (read-only; copied to BRAIN_STATE_DIR at cold start).
SEED_DIRS: List[str] = [
    f"{_TB6}/prototype/cache/ohlcv",
    f"{_TB6}/prototype/cache/cboe",
    f"{_TB6}/prototype/cache/fred",
    f"{_TB6}/prototype/cache/cot",
    f"{_TB6}/prototype/gdelt_cache/daily",
]


def _copy_file(rel: str, stage: Path, missing: List[str]) -> None:
    src = _REPO / rel
    if not src.exists():
        missing.append(rel)
        return
    dst = stage / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_dir(rel: str, stage: Path, missing: List[str]) -> int:
    src = _REPO / rel
    if not src.exists():
        missing.append(rel)
        return -1
    dst = stage / rel
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return sum(1 for _ in dst.rglob("*") if _.is_file())


def bake(stage: Path, code_only: bool = False) -> int:
    stage.mkdir(parents=True, exist_ok=True)
    missing: List[str] = []

    for rel in CODE_FILES:
        _copy_file(rel, stage, missing)
    code_dir_files = 0
    for rel in CODE_DIRS:
        n = _copy_dir(rel, stage, missing)
        if n > 0:
            code_dir_files += n

    # Mirror the frozen weights to brain/models_out_007 too, so
    # freeze.resolve_model_root() finds them via either path.
    _models_src = _REPO / _TB7 / "prototype" / "models_out_007"
    if _models_src.exists():
        shutil.copytree(_models_src, _REPO / "brain" / "models_out_007",
                        dirs_exist_ok=True)

    seed_files = 0
    if not code_only:
        for rel in SEED_DIRS:
            n = _copy_dir(rel, stage, missing)
            if n > 0:
                seed_files += n

    print(f"baked code files: {len(CODE_FILES)} | code-dir files: {code_dir_files} "
          f"| seed files: {seed_files}" + (" (skipped: --code-only)" if code_only else ""))
    if missing:
        print("MISSING SOURCES (bake incomplete):")
        for m in missing:
            print("  - " + m)
        return 1
    print(f"bake OK -> {stage} (mirrored layout under runs/)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default=str(_REPO / "build" / "brain_bake"))
    ap.add_argument("--code-only", action="store_true",
                    help="copy code/dicts/models/reference only (skip heavy seed caches)")
    args = ap.parse_args(argv)
    return bake(Path(args.stage), code_only=args.code_only)


if __name__ == "__main__":
    sys.exit(main())
