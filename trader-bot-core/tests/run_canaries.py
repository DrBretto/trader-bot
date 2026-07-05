#!/usr/bin/env python3
"""Reality-suite runner — the ONE entry the nightly + post-pipeline wiring calls.

Tiers (P7 / dossier output 4):
  * commit    — Tier-1 unit-fixture (deterministic, on real recorded data). Fast,
                every commit / CI.
  * live      — Tier-2 live-canary (assert against TODAY's real S3/CloudWatch).
                Nightly + POST-PIPELINE — catches a freeze/degrade the day it
                happens (the check missing for two weeks).
  * external  — Tier-3 external-cross-check (call Yahoo/GDELT independently).
                Nightly.
  * nightly   — live + external (the full nightly gate).
  * all       — every tier.

Exit code is pytest's: non-zero == a canary went RED == a real failure mode is
live. The post-pipeline caller (monitors.canary_gate) turns a non-zero into an
SNS CRITICAL.

Usage:  python run_canaries.py <commit|live|external|nightly|all> [pytest args…]
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

_TIERS = {
    "commit": ["-m", "unit", "tier1_unit"],
    "live": ["-m", "live_canary", "tier2_live_canary"],
    "external": ["-m", "external_check", "tier3_external_check"],
    "nightly": ["-m", "live_canary or external_check", "tier2_live_canary", "tier3_external_check"],
    "all": ["."],
}


def main(argv) -> int:
    tier = argv[0] if argv else "nightly"
    if tier not in _TIERS:
        print(f"unknown tier {tier!r}; choose one of {list(_TIERS)}", file=sys.stderr)
        return 2
    cmd = [sys.executable, "-m", "pytest", *_TIERS[tier], *argv[1:]]
    print(f"[canaries] {tier} -> {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(_HERE))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
