"""PKT-TB-007 — non-neutral tilt smoke (BUILD_SPEC_007 §8 step 2/3 sanity;
wave-1 organ stub).

Generates SYNTHETIC organ-input files (deterministic per date; clearly tagged
synthetic=true — these are plumbing tests, NEVER evidence), runs the ORB-1 arm
twice over the pre-holdout window with a small non-zero tilt_gain, and checks:

  - intents/series CHANGE vs the incumbent (the channel is alive)
  - projection constraints hold on every non-neutral day
    (Sum dw ~ 0, beta-neutral, sigma inside the budget, one-sided <= T_max)
  - turnover bounded
  - determinism across reruns (separate processes, identical hashes)
  - per-decision expression log writes (which organ moved which name)

    .venv/bin/python smoke_tilt_007.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

PROTO = Path(__file__).resolve().parent
REPO = PROTO.parents[2]
PY = REPO / ".venv" / "bin" / "python"
sys.path.insert(0, str(PROTO))

from run_replay_007 import ENV_FLAG, sha256_file  # noqa: E402
from tilt_adapter import TILT_CORE, TILT_COND, T_MAX  # noqa: E402

SMOKE_NIGHTLY = PROTO / "store" / "nightly_007_smoke"
SMOKE_GENOME = {
    "organ_trust": {"M1": 0.5, "M2": -0.3},
    "tilt_gain": 0.5, "conviction_temp": 1.0, "dead_zone": 0.05,
    "disp_gain": 0.5, "cap_core": 0.0125, "cap_conditional": 0.00625,
    "defensive_fraction": 0.25, "event_damp_strength": 0.0,
}


def make_synthetic_organs(dates) -> int:
    SMOKE_NIGHTLY.mkdir(parents=True, exist_ok=True)
    syms = list(TILT_CORE) + list(TILT_COND)
    n = 0
    for d in dates:
        seed = int(hashlib.sha256(f"orb1-smoke-{d}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        mu1 = {s: round(float(v), 6) for s, v in zip(syms, rng.normal(0, 1, len(syms)))}
        mu2 = {s: round(float(v), 6) for s, v in zip(syms, rng.normal(0, 1, len(syms)))}
        payload = {"date": d, "schema_version": "organ_inputs_007.v1",
                   "organs": {"M1": {"mu": mu1, "q": 0.65},
                              "M2": {"mu": mu2, "q": 0.45}},
                   "disp_z": round(float(rng.normal(0, 1)), 6),
                   "p_exceed": {},
                   "manifest": {"synthetic": True,
                                "purpose": "wave-1 expression smoke — never evidence",
                                "seed": seed}}
        (SMOKE_NIGHTLY / f"{d}.json").write_text(json.dumps(payload, indent=1))
        n += 1
    return n


def run_orb1(out: Path, genome_path: Path) -> None:
    env = dict(os.environ)
    env.pop(ENV_FLAG, None)
    cmd = [str(PY), str(PROTO / "run_replay_007.py"), "--arm", "orb1",
           "--window", "preholdout", "--out", str(out),
           "--genome", str(genome_path), "--nightly-dir", str(SMOKE_NIGHTLY)]
    print("->", " ".join(cmd))
    r = subprocess.run(cmd, env=env, cwd=str(PROTO))
    if r.returncode != 0:
        raise SystemExit(f"smoke arm failed (rc={r.returncode})")


def main() -> dict:
    base = PROTO / "runs_battery_007" / "SMOKE_TILT"
    base.mkdir(parents=True, exist_ok=True)
    gpath = base / "genome_smoke.json"
    gpath.write_text(json.dumps(SMOKE_GENOME, indent=1))

    # synthetic organ files for every cached dir in the preholdout window
    sys.path.append(str(REPO / "runs" / "pkt_tb_006_clean_sheet_brain" / "prototype"))
    from data_layer import DiskCachedS3Cache
    from run_replay_007 import build_trading_dates
    cache = DiskCachedS3Cache(s3_client=None)
    dates = build_trading_dates(cache, "preholdout")
    n_files = make_synthetic_organs(dates)

    run1, run2 = base / "orb1_run1", base / "orb1_run2"
    run_orb1(run1, gpath)
    run_orb1(run2, gpath)

    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
           "window": "preholdout", "n_synthetic_organ_files": n_files,
           "genome": SMOKE_GENOME, "checks": {}, "stats": {}}
    c = res["checks"]

    # determinism across reruns (separate processes)
    c["rerun_timeline_hash_equal"] = (sha256_file(run1 / "timeline.json")
                                      == sha256_file(run2 / "timeline.json"))
    c["rerun_daily_series_hash_equal"] = (sha256_file(run1 / "daily_series.csv")
                                          == sha256_file(run2 / "daily_series.csv"))
    el1 = sorted((run1 / "expression_log").glob("*.json"))
    el2 = sorted((run2 / "expression_log").glob("*.json"))
    c["rerun_expression_logs_equal"] = (
        [f.name for f in el1] == [f.name for f in el2]
        and all(a.read_bytes() == b.read_bytes() for a, b in zip(el1, el2)))

    # the channel is alive: differs from the incumbent B0_EXPR baseline
    inc = PROTO / "runs_battery_007" / "B0_EXPR" / "incumbent"
    if (inc / "timeline.json").exists():
        c["differs_from_incumbent"] = (sha256_file(run1 / "timeline.json")
                                       not in (sha256_file(inc / "timeline.json"),))
        a_inc = json.loads((inc / "result.json").read_text())["actions"]
        a_orb = json.loads((run1 / "result.json").read_text())["actions"]
        tilt_actions = [a for a in a_orb
                        if str(a.get("reason", "")).startswith("ORB1_TILT")]
        res["stats"]["incumbent_n_actions"] = len(a_inc)
        res["stats"]["orb1_n_actions"] = len(a_orb)
        res["stats"]["n_tilt_actions_executed"] = len(tilt_actions)
        c["intents_changed"] = len(a_orb) != len(a_inc) or a_orb != a_inc

    # per-day projection constraints + turnover from the expression logs
    n_active = n_neutral = 0
    bad = []
    turnovers, parity = [], {"cash": [], "beta": [], "sigma_frac": []}
    for f in el1:
        log = json.loads(f.read_text())
        if log.get("neutral"):
            n_neutral += 1
            continue
        n_active += 1
        tilt = log["tilt"]
        one_sided = sum(abs(v) for v in tilt.values()) / 2
        turnovers.append(one_sided)
        pr = log["projection"]
        parity["cash"].append(abs(sum(tilt.values())))
        parity["beta"].append(abs(pr["beta_resid"]))
        parity["sigma_frac"].append(abs(pr["sigma_resid"])
                                    / max(pr["eps_sigma"], 1e-12))
        ok = (abs(sum(tilt.values())) < 1e-6
              and abs(pr["beta_resid"]) < 1e-6
              and abs(pr["sigma_resid"]) <= pr["eps_sigma"] + 1e-9
              and one_sided <= T_MAX + 1e-9
              and "VIXY" not in tilt
              and not pr["infeasible"]
              and bool(log.get("organ_attribution")))
        if not ok:
            bad.append(f.name)
    c["all_active_days_satisfy_projection"] = bad == []
    c["expression_logs_written"] = (n_active + n_neutral) > 0
    res["stats"].update({
        "n_active_days": n_active, "n_neutral_days": n_neutral,
        "bad_days": bad,
        "turnover_one_sided_mean": round(float(np.mean(turnovers)), 6) if turnovers else None,
        "turnover_one_sided_max": round(float(np.max(turnovers)), 6) if turnovers else None,
        "t_max": T_MAX,
        "parity_resid_max_cash": float(np.max(parity["cash"])) if parity["cash"] else None,
        "parity_resid_max_beta": float(np.max(parity["beta"])) if parity["beta"] else None,
        "parity_sigma_frac_of_eps_max": (round(float(np.max(parity["sigma_frac"])), 4)
                                         if parity["sigma_frac"] else None),
    })
    res["pass"] = all(v for v in c.values() if isinstance(v, bool))
    (base / "smoke_result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({**c, **res["stats"]}, indent=1))
    print("SMOKE_TILT:", "PASS" if res["pass"] else "FAIL")
    if not res["pass"]:
        raise SystemExit(1)
    return res


if __name__ == "__main__":
    main()
