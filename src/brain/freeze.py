"""FREEZE_ORB1 cold-start assertion (PKT-TB-012, the cutover gate).

The live brain is the native two-stage engine (PKT-TB-008), frozen by
``brain/FREEZE_ORB1.json`` (PKT-TB-010). Before any forward inference or any
live ``trade_intents.json`` write, the night Lambda asserts that the engine code
and the model artifacts are byte-identical to the frozen contract:

  - ``engine_sha`` over ``src/brain/engine/*.py`` (basename + bytes, sorted,
    ``__pycache__`` excluded) == FREEZE_ORB1.engine.engine_sha
  - ``model_sha`` over the four ``models_out_007`` files (concatenated bytes,
    fixed order, first 12 hex) == FREEZE_ORB1.model_artifacts.model_sha

A mismatch is a parity failure that **ABORTS the night** (abort-never-degrade):
``assert_cold_start`` raises ``BrainFreezeError`` and the caller falls back to
the incumbent intents + SNS CRITICAL. This converts "trust the deploy" into
"verify the deploy" (DESIGN_DOSSIER reports/03 §1).

No artifact below may move without terminating the LIVE_PREREG reads and opening
a fresh pre-registration (LIVE_PREREG.md §3).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# This module lives at src/brain/freeze.py.
_THIS = Path(__file__).resolve()
_ENGINE_DIR = _THIS.parent / "engine"
_REPO_ROOT = _THIS.parents[2]

# The four model files, in the exact order the model_sha is computed over
# (forward_inference.model_sha convention; pinned in FREEZE_ORB1).
MODEL_FILES_ORDERED = (
    "m1_cast/seed_4242.pt",
    "m1_cast/seed_4243.pt",
    "m4_evt_a/model.pkl",
    "m3_disp/coefs.npz",
)

# Candidate roots for the baked model artifacts. In the Lambda image the four
# files are baked under ${LAMBDA_TASK_ROOT}/brain/models_out_007/ (320 KB,
# DESIGN_DOSSIER reports/03 §1); in the repo they live in the PKT-TB-007
# prototype tree. The first existing root wins.
_MODEL_ROOT_CANDIDATES = (
    "brain/models_out_007",
    "runs/pkt_tb_007_orthogonal_brain/prototype/models_out_007",
)


class BrainFreezeError(RuntimeError):
    """Raised when the cold-start freeze assertion fails. ABORTS the night."""


@dataclass(frozen=True)
class FreezeAssertion:
    ok: bool
    engine_sha: str
    model_sha: str
    expected_engine_sha: str
    expected_model_sha: str
    model_root: str
    detail: str


def _freeze_root() -> Path:
    """Directory holding brain/FREEZE_ORB1.json. ${LAMBDA_TASK_ROOT} in the
    Lambda image, the repo root in a checkout."""
    task_root = os.environ.get("LAMBDA_TASK_ROOT")
    if task_root and (Path(task_root) / "brain" / "FREEZE_ORB1.json").exists():
        return Path(task_root)
    return _REPO_ROOT


def freeze_path() -> Path:
    return _freeze_root() / "brain" / "FREEZE_ORB1.json"


def load_freeze() -> dict:
    p = freeze_path()
    if not p.exists():
        raise BrainFreezeError(f"FREEZE_ORB1.json not found at {p} — refusing "
                               "to run the live brain without its freeze contract")
    return json.loads(p.read_text())


def compute_engine_sha(engine_dir: Optional[Path] = None) -> str:
    """sha256 over each src/brain/engine/*.py (basename + bytes), sorted,
    __pycache__ excluded — the FREEZE_ORB1.engine.engine_sha_method."""
    engine_dir = Path(engine_dir) if engine_dir is not None else _ENGINE_DIR
    files: List[Path] = sorted(
        p for p in engine_dir.glob("*.py") if "__pycache__" not in p.parts
    )
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()


def resolve_model_root() -> Optional[Path]:
    root = _freeze_root()
    for cand in _MODEL_ROOT_CANDIDATES:
        p = root / cand
        if (p / MODEL_FILES_ORDERED[0]).exists():
            return p
    # Fall back to the repo root for checkouts where LAMBDA_TASK_ROOT pointed
    # at a baked image dir that lacks the prototype tree.
    for cand in _MODEL_ROOT_CANDIDATES:
        p = _REPO_ROOT / cand
        if (p / MODEL_FILES_ORDERED[0]).exists():
            return p
    return None


def compute_model_sha(model_root: Optional[Path] = None) -> str:
    """sha256 over the concatenated bytes of the four model files, fixed order,
    first 12 hex chars — the forward_inference.model_sha() convention."""
    if model_root is None:
        model_root = resolve_model_root()
    if model_root is None:
        raise BrainFreezeError(
            "model artifacts not found in any baked/prototype root "
            f"({_MODEL_ROOT_CANDIDATES}) — cannot assert model_sha")
    h = hashlib.sha256()
    for rel in MODEL_FILES_ORDERED:
        fp = Path(model_root) / rel
        if not fp.exists():
            raise BrainFreezeError(f"frozen model file missing: {fp}")
        h.update(fp.read_bytes())
    return h.hexdigest()[:12]


def assert_cold_start(model_root: Optional[Path] = None) -> FreezeAssertion:
    """Assert engine_sha + model_sha against FREEZE_ORB1 BEFORE any inference.

    Returns a FreezeAssertion on success; raises BrainFreezeError on any
    mismatch or missing artifact (abort-never-degrade). The live brain is the
    native two-stage engine, not tilt_adapter — this assertion is what proves
    the deployed code IS that frozen engine.
    """
    freeze = load_freeze()
    exp_engine = freeze.get("engine", {}).get("engine_sha", "")
    exp_model = freeze.get("model_artifacts", {}).get("model_sha", "")

    root = Path(model_root) if model_root is not None else resolve_model_root()
    got_engine = compute_engine_sha()
    got_model = compute_model_sha(root)

    problems: List[str] = []
    if not exp_engine or got_engine != exp_engine:
        problems.append(
            f"engine_sha mismatch: got {got_engine[:16]}… expected "
            f"{exp_engine[:16]}… (deployed engine is NOT the frozen "
            "two-stage engine)")
    if not exp_model or got_model != exp_model:
        problems.append(
            f"model_sha mismatch: got {got_model} expected {exp_model} "
            "(model artifacts moved — a retrained brain is a NEW prereg)")

    assertion = FreezeAssertion(
        ok=not problems,
        engine_sha=got_engine,
        model_sha=got_model,
        expected_engine_sha=exp_engine,
        expected_model_sha=exp_model,
        model_root=str(root) if root else "",
        detail=("freeze OK" if not problems else "; ".join(problems)),
    )
    if problems:
        raise BrainFreezeError(
            "FREEZE_ORB1 cold-start assertion FAILED — night ABORTS, "
            "morning falls back to incumbent intents: " + assertion.detail)
    return assertion
