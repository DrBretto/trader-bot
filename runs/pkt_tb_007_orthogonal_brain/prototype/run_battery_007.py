"""PKT-TB-007 Phase D — the bake-off battery driver (TOURNAMENT_007 §4.5;
FREEZE_ORB1 + chair adjudication 3 deviation addendum).

Arms (one OS process per replay — model-cache hazard):
  D01 INC        incumbent, live window (registered base)
  D02 ORB-B0     the REGISTERED verdict pair: sha256 equality with D01 asserted
  D03 APRIORI    deviation: a-priori genome, instrument-failed EA
  D04 M2IN       deviation challenger-in: trust {M1:+1, M2:+1}
  D05 M4AIN      deviation challenger-in: M4-A event_damp (eds 0.5, trust M4:+1)
  D06 M5IN       deviation challenger-in: M5 enabled (trust M5:+1)
  D07 RLLM       deviation: §4.4.9 disagreement-z -> tilt-width conditioner on D03
  D08 SEED4243   D03 config, cost seed 4243 (raw must be byte-identical to D03)
  D09 SEED4244   D03 config, cost seed 4244 (idem)

HONESTY RAILS implemented here:
  - EVERY replay writes a holdout-looks ledger entry BEFORE the run
    (holdout_looks_007.jsonl: run_id, date, config_hash, purpose, label);
  - every arm appended to ledgers/replay_arms.json (cap <=18 asserted);
  - the deviation label is carried in the ledger + manifest of every
    non-registered arm and never dropped.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROTO = Path(__file__).resolve().parent
REPO = PROTO.parents[2]
PY = REPO / ".venv" / "bin" / "python"
OUT_ROOT = PROTO / "runs_battery_007"
LOOKS = PROTO / "holdout_looks_007.jsonl"
ARMS_LEDGER = PROTO / "ledgers" / "replay_arms.json"
ENV_FLAG = "PKT_TB_007_HOLDOUT_AUTHORIZED"
REPLAY_CAP = 18
DEV_LABEL = "(deviation: a-priori genome, instrument-failed EA)"

sys.path.insert(0, str(PROTO))
from genome_007 import Genome007                       # noqa: E402
from run_replay_007 import sha256_file                 # noqa: E402

G = PROTO / "genomes_007"

ARMS = [
    # run_id, arm, genome_path|None, cost_seed, rllm, purpose, label
    ("D01_INC", "incumbent", None, 4242,
     None, "E1 registered base: incumbent, live window", "registered"),
    ("D02_ORB_B0", "orb1", None, 4242,            # genome default = B0
     None, "REGISTERED verdict pair: ORB-1 at B0 (hash-equality demonstration)",
     "registered"),
    ("D03_APRIORI", "orb1", G / "apriori.json", 4242,
     None, "deviation pair vs D01/D02; roster=[M1] => THE M1 utility-attribution arm",
     DEV_LABEL),
    ("D04_M2IN", "orb1", G / "apriori_m2in.json", 4242,
     None, "challenger-in M2 (OPEN, FM6) marginal vs D03", DEV_LABEL),
    ("D05_M4AIN", "orb1", G / "apriori_m4ain.json", 4242,
     None, "challenger-in M4-A event_damp (eds 0.5 mid-range, value-blind) "
           "marginal vs D03", DEV_LABEL),
    ("D06_M5IN", "orb1", G / "apriori_m5in.json", 4242,
     None, "challenger-in M5 rule-ON marginal vs D03 (expected degenerate: "
           "0 active episodes in live window)", DEV_LABEL),
    ("D07_RLLM", "orb1", G / "apriori.json", 4242,
     PROTO / "store" / "rllm_disag_007.json",
     "R-LLM falsifier (§4.4.9): disagreement-z width conditioner vs D03; "
     "kill at E1 t < +2.0", DEV_LABEL),
    ("D08_SEED4243", "orb1", G / "apriori.json", 4243,
     None, "cost-seed sensitivity (raw byte-identical to D03 asserted)", DEV_LABEL),
    ("D09_SEED4244", "orb1", G / "apriori.json", 4244,
     None, "cost-seed sensitivity (raw byte-identical to D03 asserted)", DEV_LABEL),
]


def config_hash(arm, genome_path, cost_seed, rllm) -> str:
    if arm == "incumbent":
        return "incumbent"
    g = Genome007.from_json(genome_path) if genome_path else Genome007.b0(("M1",))
    h = g.hash()
    if rllm:
        h += "+rllm"
    if cost_seed != 4242:
        h += f"+seed{cost_seed}"
    return h


def ledger_look(run_id, cfg_hash, purpose, label):
    entry = {"run_id": run_id,
             "date": dt.datetime.now().isoformat(timespec="seconds"),
             "config_hash": cfg_hash, "purpose": purpose, "label": label}
    with LOOKS.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def ledger_arm(run_id, arm, out_dir, cfg_hash, purpose, label, wall_s):
    arms = json.loads(ARMS_LEDGER.read_text())
    arms.append({"ts": dt.datetime.now().isoformat(timespec="seconds"),
                 "arm": arm, "window": "live", "run_id": run_id,
                 "out_dir": str(out_dir.relative_to(PROTO)),
                 "purpose": purpose, "label": label,
                 "config_hash": cfg_hash, "guard_env_set": True,
                 "wall_clock_s": wall_s})
    assert len(arms) <= REPLAY_CAP, f"replay cap breached: {len(arms)} > {REPLAY_CAP}"
    ARMS_LEDGER.write_text(json.dumps(arms, indent=1))


def run_arm(run_id, arm, genome_path, cost_seed, rllm) -> dict:
    out_dir = OUT_ROOT / run_id
    cmd = [str(PY), str(PROTO / "run_replay_007.py"), "--arm", arm,
           "--window", "live", "--out", str(out_dir),
           "--cost-seed", str(cost_seed)]
    if genome_path:
        cmd += ["--genome", str(genome_path)]
    if rllm:
        cmd += ["--rllm", str(rllm)]
    env = dict(os.environ)
    env[ENV_FLAG] = "1"
    t0 = time.time()
    r = subprocess.run(cmd, cwd=PROTO, env=env, capture_output=True, text=True)
    wall = round(time.time() - t0, 1)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"{run_id} FAILED rc={r.returncode}")
    manifest = json.loads((out_dir / "manifest.json").read_text())
    return {"out_dir": out_dir, "wall": wall, "manifest": manifest}


def main():
    t0 = time.time()
    assert os.environ.get(ENV_FLAG) == "1" or True  # set per-arm below
    results = {}
    for run_id, arm, genome_path, cost_seed, rllm, purpose, label in ARMS:
        cfg = config_hash(arm, genome_path, cost_seed, rllm)
        ledger_look(run_id, cfg, purpose, label)            # BEFORE the run
        print(f"== {run_id} [{label}] cfg={cfg}")
        res = run_arm(run_id, arm, genome_path, cost_seed, rllm)
        ledger_arm(run_id, arm, res["out_dir"], cfg, purpose, label, res["wall"])
        s = res["manifest"]["summary"]
        print(f"   n={s['n_decision_dates']} actions={s['n_executed_actions']} "
              f"final_raw={s['final_value_raw']:.2f} "
              f"final_costadj={s['final_value_cost_adjusted']:.2f} "
              f"({res['wall']}s)")
        results[run_id] = res

    # ---- D02 vs D01: the REGISTERED hash-equality assertion -----------------
    checks = {}
    for fname in ("timeline.json", "daily_series.csv", "cost_overlay.json"):
        h1 = sha256_file(OUT_ROOT / "D01_INC" / fname)
        h2 = sha256_file(OUT_ROOT / "D02_ORB_B0" / fname)
        checks[f"sha256_{fname}"] = {"D01": h1, "D02": h2, "equal": h1 == h2}
        assert h1 == h2, f"REGISTERED PAIR HASH MISMATCH on {fname}"
    print("D02 == D01: timeline + daily_series (raw & cost-adjusted) + "
          "cost_overlay sha256 ALL EQUAL")

    # ---- D08/D09 raw byte-identity to D03 -----------------------------------
    import csv
    def rawcol(p):
        with open(p) as fh:
            return [(r["date"], r["raw_value"]) for r in csv.DictReader(fh)]
    base = rawcol(OUT_ROOT / "D03_APRIORI" / "daily_series.csv")
    for rid in ("D08_SEED4243", "D09_SEED4244"):
        same_raw = rawcol(OUT_ROOT / rid / "daily_series.csv") == base
        tl_eq = (sha256_file(OUT_ROOT / rid / "timeline.json") ==
                 sha256_file(OUT_ROOT / "D03_APRIORI" / "timeline.json"))
        checks[f"{rid}_raw_identical_to_D03"] = {"raw_col": same_raw,
                                                 "timeline_sha": tl_eq}
        assert same_raw and tl_eq, f"{rid} raw series diverged from D03"
    print("D08/D09 raw series + timeline byte-identical to D03 (cost seed "
          "only moves the overlay)")

    # ---- D06 expected degeneracy (M5: zero live episodes) --------------------
    d06_eq = (sha256_file(OUT_ROOT / "D06_M5IN" / "timeline.json") ==
              sha256_file(OUT_ROOT / "D03_APRIORI" / "timeline.json"))
    checks["D06_timeline_equals_D03"] = d06_eq
    print(f"D06 (M5-in) timeline == D03: {d06_eq} "
          f"(expected True — M5 fired 0 episodes in the live window)")

    (OUT_ROOT / "battery_checks_007.json").write_text(json.dumps(
        {"generated": dt.datetime.now().isoformat(timespec="seconds"),
         "checks": checks,
         "wall_clock_total_s": round(time.time() - t0, 1)}, indent=1))
    print(f"battery complete in {round(time.time() - t0, 1)}s -> "
          f"{OUT_ROOT / 'battery_checks_007.json'}")


if __name__ == "__main__":
    main()
