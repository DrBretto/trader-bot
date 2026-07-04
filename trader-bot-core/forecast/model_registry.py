"""Model registry for the relocated forecast spine (PKT-TRADER-BOT-FORECAST-
SPINE-RELOCATE, P3).

First-class surface over the frozen model artifacts that were relocated verbatim
into ``forecast/ref/p7proto/models_out_007`` (the four files pinned in
FREEZE_ORB1). This is the single place the spine names the model root and the
model_sha convention — no ``sys.path.append(runs/pkt_tb_00X/...)`` and no
``__file__``-relative walks out into the prototype trees.

The ``model_sha`` here is byte-identical to the original
``forward_inference.model_sha()`` and to ``freeze.compute_model_sha`` (same four
files, same order, same first-12-hex convention), so the cold-start freeze
assertion is unchanged by the relocation.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

# forecast/ref/p7proto/models_out_007 — the relocated verbatim model artifacts.
MODEL_ROOT = Path(__file__).resolve().parent / "ref" / "p7proto" / "models_out_007"

# The four frozen model files, in the exact order the model_sha is computed over
# (forward_inference.model_sha convention; pinned in FREEZE_ORB1).
MODEL_FILES_ORDERED = (
    "m1_cast/seed_4242.pt",
    "m1_cast/seed_4243.pt",
    "m4_evt_a/model.pkl",
    "m3_disp/coefs.npz",
)


def model_path(rel: str) -> Path:
    """Absolute path to a model artifact under the first-class model root."""
    return MODEL_ROOT / rel


def model_sha() -> str:
    """sha256 over the concatenated bytes of the four frozen model files, fixed
    order, first 12 hex — the ``forward_inference.model_sha()`` convention,
    unchanged by the relocation (byte-identical artifacts)."""
    h = hashlib.sha256()
    for rel in MODEL_FILES_ORDERED:
        fp = MODEL_ROOT / rel
        if not fp.exists():
            raise FileNotFoundError(f"frozen model file missing: {fp}")
        h.update(fp.read_bytes())
    return h.hexdigest()[:12]
