"""Frozen ORB-1 brain isolation guard (PKT-TB-014 / D-AUTO-20260616).

The monthly training + evolutionary search retrain the OLD regime/health models
(-> ``s3://<bucket>/models/``). They must NEVER read or overwrite the frozen
ORB-1 New Brain weights — ``brain/FREEZE_ORB1.json`` plus the four model
artifacts under ``runs/pkt_tb_007_orthogonal_brain/prototype/models_out_007``,
which carry ``no_retraining: true`` and are sha-asserted at the Lambda cold
start. Retraining or clobbering them would silently break the live brain.

Isolation holds at two layers:

  1. **Structural** — the AWS Batch training image (``Dockerfile.training``) does
     not ``COPY runs/`` or ``COPY brain/``, so the frozen artifacts are not
     present in the training container at all; reads are impossible there.
  2. **Enforced** (this guard) — every training entrypoint calls
     ``assert_frozen_brain_isolated(<output dirs>)`` before doing any work. It
     raises ``FrozenBrainIsolationError`` if any training write target resolves
     under the frozen tree, so even a local (laptop) run cannot clobber the
     frozen weights.
"""
from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

# Repo-relative paths that hold the frozen ORB-1 New Brain weights / contract.
FROZEN_BRAIN_PATHS = (
    "brain/FREEZE_ORB1.json",
    "runs/pkt_tb_007_orthogonal_brain/prototype/models_out_007",
    "runs/pkt_tb_006_clean_sheet_brain/prototype/models_out",
)


class FrozenBrainIsolationError(RuntimeError):
    """Raised when a training job would read/write the frozen ORB-1 weights."""


def _frozen_abspaths():
    return [(_REPO / p).resolve() for p in FROZEN_BRAIN_PATHS]


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def assert_frozen_brain_isolated(*output_dirs: str) -> None:
    """Fail fast unless every training output dir is disjoint from the frozen
    ORB-1 brain tree. Call this at the top of any training/evolution entrypoint.
    """
    frozen = _frozen_abspaths()
    for od in output_dirs:
        if not od:
            continue
        odp = Path(od).resolve()
        for fp in frozen:
            if _is_within(odp, fp) or _is_within(fp, odp):
                raise FrozenBrainIsolationError(
                    f"Training output '{od}' overlaps the frozen ORB-1 brain path "
                    f"'{fp}'. The frozen New Brain weights are no_retraining "
                    f"(PKT-TB-014 / D-AUTO-20260616); training must write only to "
                    f"its own model prefix (e.g. /tmp/models, /tmp/evolution, "
                    f"s3://<bucket>/models/)."
                )
    surfaced = ", ".join(d for d in output_dirs if d) or "(none)"
    print(f"[frozen-brain-guard] ORB-1 New Brain weights isolated from training "
          f"(output dirs: {surfaced}).")
