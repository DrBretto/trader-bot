"""Post-pipeline reality-canary gate (P7 wiring).

The two-week silent freeze happened because NOTHING ran a substrate-currency
check the day it broke. This is the POST-PIPELINE hook: the nightly pipeline
calls ``run_post_pipeline_canaries()`` immediately after publish; it runs the
live-canary tier against TODAY's real S3 output and, on ANY red, fires an SNS
CRITICAL so a freeze/degrade is caught the day it happens.

Wiring (P8/P9 own cutting prod over — this module only provides the callable):
    from monitors.canary_gate import run_post_pipeline_canaries
    run_post_pipeline_canaries()      # after the night's publish_line(...)

Scheduling: the nightly gate (live + external) also runs on cron via
``.github/workflows/reality-canaries-nightly.yml`` and can be invoked directly
with ``python trader-bot-core/tests/run_canaries.py nightly``.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

_TESTS = Path(__file__).resolve().parents[1] / "tests"
_RUNNER = _TESTS / "run_canaries.py"


def run_post_pipeline_canaries(tier: str = "live", *, alert: bool = True) -> dict:
    """Run the live-canary tier post-pipeline. Returns
    {ok, tier, returncode, output_tail}. On red (returncode != 0), fires an SNS
    CRITICAL when ``alert`` (alerting failures never crash the caller)."""
    proc = subprocess.run(
        [sys.executable, str(_RUNNER), tier, "-q"],
        capture_output=True, text=True, cwd=str(_TESTS))
    out = (proc.stdout or "") + (proc.stderr or "")
    ok = proc.returncode == 0
    result = {"ok": ok, "tier": tier, "returncode": proc.returncode,
              "output_tail": "\n".join(out.strip().splitlines()[-25:])}
    if not ok and alert:
        _alert_canary_red(result)
    return result


def _alert_canary_red(result: dict) -> None:
    body = (
        "POST-PIPELINE REALITY CANARY FAILED — a real failure mode is LIVE (a "
        "frozen store / frozen mu / degraded publish / stored-value drift). This "
        "is the check that was missing for the two-week silent freeze.\n\n"
        f"tier         = {result['tier']}\n"
        f"returncode   = {result['returncode']}\n\n"
        f"{result['output_tail']}\n")
    try:
        from src.utils.sns_alerts import send_alert
        send_alert(subject="[TraderBot] CRITICAL: post-pipeline reality canary RED",
                   body=body)
    except Exception as e:  # noqa: BLE001 — alerting must never crash the pipeline
        print(f"[canary_gate] SNS alert failed (non-fatal): {e}")


if __name__ == "__main__":
    res = run_post_pipeline_canaries(sys.argv[1] if len(sys.argv) > 1 else "live",
                                     alert=False)
    print(res["output_tail"])
    raise SystemExit(res["returncode"])
