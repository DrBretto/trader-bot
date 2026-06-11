"""PKT-TB-007 — B0-EXPR exactness check (BUILD_SPEC_007 §8 step 3 — the gate
nothing downstream is trusted before).

Runs the incumbent arm and the ORB-1 arm under the B0 genome over the
PRE-HOLDOUT live window (decision dates 2026-02-02..2026-03-06; the holdout
guard env is FORCIBLY UNSET in both child processes) — one OS process per arm
(§2.1 discipline) — and asserts:

  - sha256(timeline.json)      identical   (raw daily series, full state)
  - sha256(daily_series.csv)   identical   (raw AND cost-adjusted series)
  - executed actions lists     identical

Writes b0_expr_result.json next to the run dirs.

    .venv/bin/python b0_expr_check.py [--out runs_battery_007/B0_EXPR]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

PROTO = Path(__file__).resolve().parent
REPO = PROTO.parents[2]
PY = REPO / ".venv" / "bin" / "python"

sys.path.insert(0, str(PROTO))
from run_replay_007 import ENV_FLAG, sha256_file  # noqa: E402


def run_arm(arm: str, out: Path, extra=()) -> None:
    env = dict(os.environ)
    env.pop(ENV_FLAG, None)                      # guard UNSET by construction
    cmd = [str(PY), str(PROTO / "run_replay_007.py"), "--arm", arm,
           "--window", "preholdout", "--out", str(out), *extra]
    print("->", " ".join(cmd))
    r = subprocess.run(cmd, env=env, cwd=str(PROTO))
    if r.returncode != 0:
        raise SystemExit(f"{arm} arm failed (rc={r.returncode})")


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROTO / "runs_battery_007" / "B0_EXPR"))
    args = ap.parse_args()
    base = Path(args.out)
    inc, orb = base / "incumbent", base / "orb1_b0"

    run_arm("incumbent", inc)
    run_arm("orb1", orb)                         # default genome = B0

    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
           "window": "preholdout", "holdout_guard_env_unset": True,
           "arms": {}}
    for name, d in (("incumbent", inc), ("orb1_b0", orb)):
        res["arms"][name] = {
            "sha256_timeline": sha256_file(d / "timeline.json"),
            "sha256_daily_series": sha256_file(d / "daily_series.csv"),
            "sha256_cost_overlay": sha256_file(d / "cost_overlay.json"),
        }
        summary = json.loads((d / "manifest.json").read_text())["summary"]
        res["arms"][name]["n_decision_dates"] = summary["n_decision_dates"]
        res["arms"][name]["final_value_raw"] = summary["final_value_raw"]
        res["arms"][name]["n_executed_actions"] = summary["n_executed_actions"]

    a, b = res["arms"]["incumbent"], res["arms"]["orb1_b0"]
    actions_a = json.loads((inc / "result.json").read_text())["actions"]
    actions_b = json.loads((orb / "result.json").read_text())["actions"]
    # actions carry the variant name (same 'hybrid' config object both arms) —
    # compare the full records as-is
    res["checks"] = {
        "timeline_hash_equal": a["sha256_timeline"] == b["sha256_timeline"],
        "daily_series_hash_equal": (a["sha256_daily_series"]
                                    == b["sha256_daily_series"]),
        "cost_overlay_hash_equal": (a["sha256_cost_overlay"]
                                    == b["sha256_cost_overlay"]),
        "actions_identical": actions_a == actions_b,
        "n_actions": len(actions_a),
    }
    # every orb1 expression-log day must be neutral under B0
    neutral_ok, n_logs = True, 0
    logdir = orb / "expression_log"
    if logdir.exists():
        for f in sorted(logdir.glob("*.json")):
            n_logs += 1
            if not json.loads(f.read_text()).get("neutral", False):
                neutral_ok = False
    res["checks"]["all_expression_logs_neutral"] = neutral_ok
    res["checks"]["n_expression_logs"] = n_logs
    res["pass"] = all(v for k, v in res["checks"].items()
                      if isinstance(v, bool))
    (base / "b0_expr_result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["checks"], indent=1))
    print("B0-EXPR:", "HASH MATCH — PASS" if res["pass"] else "FAIL")
    if not res["pass"]:
        raise SystemExit(1)
    return res


if __name__ == "__main__":
    main()
