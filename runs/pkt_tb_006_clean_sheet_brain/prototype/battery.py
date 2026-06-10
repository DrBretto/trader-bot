"""PKT-TB-006 — attribution battery runner (TOURNAMENT §4.4; BUILD_SPEC §13).

The battery table below is the §4.4 plan VERBATIM: R01–R14 planned (14 replays,
5 retrains) + R15–R20 contingency (≤6 replays, ≤4 contingency retrains).
Hard caps (Feasibility, binding): ≤20 replay runs, ≤9 retrain cycles — enforced
here, refusal is a hard error.

LEDGER DISCIPLINE (TOURNAMENT §5): every executed replay appends
    {run_id, date_executed, config_hash, purpose}
to ``prototype/holdout_looks.jsonl`` AT EXECUTION TIME, *before* the replay
subprocess is launched — the write is in the runner, impossible to forget.
``--plan`` consumes zero looks (no ledger writes, no replays).

REPLAY CONTRACT (parallel lane owns run_replay.py / strategy_adapter.py /
precompute_nightly.py). The LANDED runner CLI is
    run_replay.py --arm {syn1|incumbent} --window full --out <arm_dir>
                  [--genome <arm_dir>/genome.json] [--exec-dir D] [--nightly-dir D]
                  [--exec-mode {learned|equal_trust}] [--cost-seed N]
                  [--sigma-source {trailing21|risknet}]
and leaves ``daily_series.csv`` (date,raw_value,cost_adjusted_value),
``result.json`` (timeline split to ``timeline.json``), ``manifest.json`` in
<arm_dir> — a DIVERGENCE from the packet-documented shape (daily_raw.csv +
daily_costadj.csv, timeline inline); stats.load_run accepts both. Arm configs
alter ONLY: genome (member/feature gates), executive weights dir, member
opinion sources via a variant --nightly-dir (ridge-in-slot / retrained-model
opinions are PRECOMPUTED into store/nightly_<tag>/ by precompute_nightly), or
retrained model dirs — materialized by this module into <arm_dir>.

CAPABILITY GAPS — CLOSED: the runner now expresses (a) the R08 executive
bypass via ``--exec-mode equal_trust`` (tau=1/M over active members, f fixed
0.7, same vol-cap rails), (b) the R13/R14 slippage-seed overrides via
``--cost-seed`` (default 4242), (c) the R07 replay-time sigma swap via
``--sigma-source trailing21|risknet`` ('trailing21' names the landed/R01
default behavior; the swap supersedes the variant-nightly-store mechanism, so
R07's variant --nightly-dir is NOT passed). default_replay_cmd still REFUSES
an executive_bypass shape other than {equal_trust, f_fixed: 0.7} — that knob
does not exist.

RETRAIN-DIR NAME CONTRACT (for the retrain owner): RT-2 → exec_out_rt2 + OOF
member names ``ridge_twin`` (exists); RT-3 → exec_out_rt3 + ``<member>_llm_neutral``;
RT-4 → exec_out_rt4 + ``<member>_gdelt_ablated``; RT-5 → exec_out_rt5 +
``<member>_uniform`` (member OOFs exist); contingency RT-6..RT-9 per arm spec.

EXECUTION PRECONDITIONS (TOURNAMENT §5, enforced): no run executes before the
frozen SYN-1 genome exists at ``ea/genome_2026-02-06.json`` (the single
pre-registered configuration), except via --allow-unfrozen for harness smoke
OUTSIDE the real run tree.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

import ea
import folds as fd

PROTO = Path(__file__).resolve().parent
HOLDOUT_LEDGER = PROTO / "holdout_looks.jsonl"
RETRAIN_LEDGER = PROTO / "retrains.jsonl"
FROZEN_GENOME = PROTO / "ea" / f"genome_{fd.FITNESS_END}.json"
PLAN_PATH = PROTO / "battery_plan.json"
RUNS_ROOT = PROTO / "runs_battery"

MAX_REPLAYS = 20      # §4.4 hard cap
MAX_RETRAINS = 9      # §4.4 hard cap
SEED_PAIRED = 4242
SEED_SENS = (4243, 4244)

MEMBER_SLOTS = ("cast", "gbm_cond", "event_head")


class BudgetExceeded(RuntimeError):
    pass


class PreconditionFailed(RuntimeError):
    pass


# ============================ trust renorm (drop arms) =========================
def renorm_trust(tau: np.ndarray, member_gate, eps: float = 0.05) -> np.ndarray:
    """BUILD_SPEC §13 drop-arm state transform: zero the dropped member's trust,
    renormalize over survivors, redistribute the entropy floor over survivors
    (floor eps/n_survivors each). Pure function; sums to 1; NO retrain.

    tau [..., M]; member_gate length-M 0/1.
    """
    g = np.asarray(member_gate, dtype=np.float64)
    n_surv = g.sum()
    if n_surv <= 0:
        raise ValueError("drop arm with all members gated off")
    t = np.asarray(tau, dtype=np.float64) * g
    norm = t.sum(axis=-1, keepdims=True)
    t = np.divide(t, norm, out=np.full_like(t, 1.0 / n_surv) * g, where=norm > 1e-12)
    return (1.0 - eps) * t + eps * g / n_surv


def drop_genome(genome: "ea.Genome", member: str) -> "ea.Genome":
    """Member OFF via the genome member_gate (the walk/adapter renormalizes)."""
    g = ea.Genome.from_dict(genome.to_dict())
    mg = list(g.member_gate)
    mg[MEMBER_SLOTS.index(member)] = 0
    g.member_gate = mg
    return g


# ============================ arm table (§4.4 verbatim) ========================
@dataclass
class Arm:
    run_id: str
    description: str
    arm_config: dict                 # replay-time config (materialized to arm dir)
    retrain_required: str | None     # RT-id or None
    e1: dict | None                  # E1 read spec (None: no E1 for this run)
    e2: dict | None                  # E2 comparison spec
    contingency: bool = False
    purpose: str = ""

    def config_hash(self, genome_doc: dict | str) -> str:
        doc = {"run_id": self.run_id, "arm_config": self.arm_config,
               "retrain": self.retrain_required, "genome": genome_doc}
        return hashlib.sha256(
            json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]


def _e2(vs: str = "R01") -> dict:
    return {"pair_vs": vs, "slippage_seed": SEED_PAIRED,
            "read": "holdout-only paired daily stats (sign confirmation)"}


def build_arms() -> dict[str, Arm]:
    arms: list[Arm] = []
    base = {"strategy": "syn1", "genome": "frozen", "exec_dir": "exec_out",
            "slippage_seed": SEED_PAIRED}
    arms.append(Arm("R01", "SYN-1, the ONE pre-registered config (bake-off + "
                    "base of every paired arm)", dict(base), "RT-1",
                    e1=None, e2={"pair_vs": "R02", "slippage_seed": SEED_PAIRED,
                                 "read": "bake-off primary (§4.2)"},
                    purpose="bake-off base / paired-arm base"))
    arms.append(Arm("R02", "Incumbent (bake-off opponent)",
                    {"strategy": "incumbent", "slippage_seed": SEED_PAIRED},
                    None, e1=None, e2={"pair_vs": "R01",
                                       "slippage_seed": SEED_PAIRED,
                                       "read": "bake-off primary (§4.2)"},
                    purpose="bake-off opponent"))
    arms.append(Arm("R03", "Ridge-in-slot (transformer qua transformer)",
                    {**base, "oof_member_map": {"cast": "ridge_twin"},
                     "exec_dir": "exec_out_rt2",
                     "nightly_dir": "store/nightly_rt2_ridge_slot"}, "RT-2",
                    e1={"kind": "member_swap", "slot": "cast",
                        "swap_to": "ridge_twin", "exec_dir_without": "exec_out_rt2"},
                    e2=_e2(), purpose="organ: transformer (primary)"))
    for rid, m in (("R04", "cast"), ("R05", "gbm_cond"), ("R06", "event_head")):
        arms.append(Arm(rid, f"{m} OFF, trust renorm (no retrain, K17)",
                        {**base, "member_gate_off": m}, None,
                        e1={"kind": "member_drop", "member": m},
                        e2=_e2(), purpose=f"organ: ensemble ({m} drop)"))
    arms.append(Arm("R07", "RiskNet+ -> trailing-21d realized replacement",
                    {**base, "sigma_source": "trailing21",
                     "nightly_dir": "store/nightly_trailing_sigma"}, None,
                    e1={"kind": "sigma_swap", "without_source": "trailing21"},
                    e2=_e2(), purpose="organ: ensemble (RiskNet+ instrument)"))
    arms.append(Arm("R08", "Executive -> equal-trust + fixed f=0.7 (same vol cap)",
                    {**base, "executive_bypass": {"equal_trust": True,
                                                  "f_fixed": 0.7}}, None,
                    e1={"kind": "executive_bypass", "f_fixed": 0.7},
                    e2=_e2(), purpose="organ: meta-evaluator"))
    arms.append(Arm("R09", "B0 DEFAULT_GENOME brain (evolution B2 read)",
                    {**base, "genome": "B0"}, None,
                    e1={"kind": "genome_swap", "without_genome": "B0"},
                    e2=_e2(), purpose="organ: evolution (champion vs B0)"))
    arms.append(Arm("R10", "LLM neutral-constants retrain",
                    {**base, "oof_suffix": "_llm_neutral",
                     "feature_gate_off": ["LLM_block"],
                     "exec_dir": "exec_out_rt3",
                     "nightly_dir": "store/nightly_rt3_llm_neutral"}, "RT-3",
                    e1={"kind": "retrain_ablation", "oof_suffix": "_llm_neutral",
                        "feature_gate_off": ["LLM_block"],
                        "exec_dir_without": "exec_out_rt3"},
                    e2=_e2(), purpose="organ: LLM (primary)"))
    arms.append(Arm("R11", "GDELT G1-G5 ablated retrain",
                    {**base, "oof_suffix": "_gdelt_ablated",
                     "feature_gate_off": ["G1_themes", "G4G5_novelty_conc"],
                     "exec_dir": "exec_out_rt4",
                     "nightly_dir": "store/nightly_rt4_gdelt_ablated"}, "RT-4",
                    e1={"kind": "retrain_ablation", "oof_suffix": "_gdelt_ablated",
                        "feature_gate_off": ["G1_themes", "G4G5_novelty_conc"],
                        "exec_dir_without": "exec_out_rt4"},
                    e2=_e2(), purpose="organ: GDELT (block arm)"))
    arms.append(Arm("R12", "Infotropy-B uniform-weights retrain (scorecard read)",
                    {**base, "oof_suffix": "_uniform", "uniform_w_rec": True,
                     "exec_dir": "exec_out_rt5",
                     "nightly_dir": "store/nightly_rt5_uniform"}, "RT-5",
                    e1={"kind": "retrain_ablation", "oof_suffix": "_uniform",
                        "uniform_w_rec": True, "exec_dir_without": "exec_out_rt5"},
                    e2=_e2(), purpose="organ: infotropy Transfer B"))
    for rid, seed in zip(("R13", "R14"), SEED_SENS):
        arms.append(Arm(rid, f"Seed-sensitivity: full brain at master seed {seed}",
                        {**base, "slippage_seed": seed}, None, e1=None,
                        e2={"pair_vs": "R01", "slippage_seed": seed,
                            "read": "seed sensitivity (context only)"},
                        purpose=f"seed sensitivity ({seed})"))
    # ---- contingency (≤6 replays, ≤4 contingency retrains) -------------------
    arms.append(Arm("R15", "CONTINGENCY: LLM feature-neutralized, NO retrain",
                    {**base, "feature_gate_off": ["LLM_block"]}, None,
                    e1={"kind": "gate_only", "feature_gate_off": ["LLM_block"]},
                    e2=_e2(), contingency=True,
                    purpose="organ: LLM (secondary, no-retrain)"))
    arms.append(Arm("R16", "CONTINGENCY: G1 theme-dictionary alone ablated",
                    {**base, "oof_suffix": "_g1_ablated",
                     "feature_gate_off": ["G1_themes"],
                     "exec_dir": "exec_out_rt6",
                     "nightly_dir": "store/nightly_rt6_g1_ablated"}, "RT-6",
                    e1={"kind": "retrain_ablation", "oof_suffix": "_g1_ablated",
                        "feature_gate_off": ["G1_themes"],
                        "exec_dir_without": "exec_out_rt6"},
                    e2=_e2(), contingency=True, purpose="organ: GDELT (G1 sub-arm)"))
    arms.append(Arm("R17", "CONTINGENCY: S1/S3/S2 data-edge block ablated",
                    {**base, "oof_suffix": "_dataedge_ablated",
                     "feature_gate_off": ["CBOE_S1", "FRED_S3", "COT_S2"],
                     "exec_dir": "exec_out_rt7",
                     "nightly_dir": "store/nightly_rt7_dataedge"}, "RT-7",
                    e1={"kind": "retrain_ablation", "oof_suffix": "_dataedge_ablated",
                        "feature_gate_off": ["CBOE_S1", "FRED_S3", "COT_S2"],
                        "exec_dir_without": "exec_out_rt7"},
                    e2=_e2(), contingency=True, purpose="data-edge block arm"))
    arms.append(Arm("R18", "CONTINGENCY: retrained-executive member-drop arm (D6)",
                    {**base, "member_gate_off": "<member-TBD-from-R04-R06>",
                     "exec_dir": "exec_out_rt8"}, "RT-8",
                    e1={"kind": "member_drop_retrained",
                        "member": "<member-TBD-from-R04-R06>",
                        "exec_dir_without": "exec_out_rt8"},
                    e2=_e2(), contingency=True,
                    purpose="dissent D6: drop arm with executive re-fit"))
    arms.append(Arm("R19", "CONTINGENCY: reserved", {**base, "reserved": True},
                    None, e1=None, e2=None, contingency=True, purpose="reserved"))
    arms.append(Arm("R20", "CONTINGENCY: reserved (RT-9 reserve)",
                    {**base, "reserved": True}, "RT-9", e1=None, e2=None,
                    contingency=True, purpose="reserved"))
    return {a.run_id: a for a in arms}


ARMS = build_arms()

# E1-only reads (no replay, no holdout look) — TOURNAMENT §4.3 infotropy row:
# Transfer A gated-vs-R3-only-twin on training folds, printed in parentheses.
E1_ONLY_READS = {
    "infotropy_a": {
        "kind": "member_swap", "slot": "event_head",
        "swap_to": "event_head_r3only",
        "note": "Transfer A: gated EventHead vs R3-only twin, E1 only, no replay",
    },
}


# ============================ ledgers & budgets ================================
def _read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def replay_count(ledger: Path = HOLDOUT_LEDGER) -> int:
    return len(_read_jsonl(ledger))


def retrain_ids(ledger: Path = RETRAIN_LEDGER) -> list[str]:
    return sorted({r["rt_id"] for r in _read_jsonl(ledger)})


def append_holdout_look(run_id: str, config_hash: str, purpose: str,
                        ledger: Path = HOLDOUT_LEDGER) -> dict:
    """The §5 ledger write — called by execute_arm BEFORE the replay launches."""
    rec = {"run_id": run_id,
           "date_executed": dt.datetime.now().isoformat(timespec="seconds"),
           "config_hash": config_hash, "purpose": purpose}
    with open(ledger, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def register_retrain(rt_id: str, what: str, ledger: Path = RETRAIN_LEDGER) -> dict:
    """Retrain-cycle budget: hard refuse beyond 9 distinct RT cycles."""
    ids = retrain_ids(ledger)
    if rt_id not in ids and len(ids) >= MAX_RETRAINS:
        raise BudgetExceeded(
            f"REFUSED: retrain {rt_id} would be cycle #{len(ids) + 1} > "
            f"hard cap {MAX_RETRAINS} (TOURNAMENT §4.4). Executed: {ids}")
    rec = {"rt_id": rt_id, "what": what,
           "date": dt.datetime.now().isoformat(timespec="seconds")}
    with open(ledger, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


# ============================ genome resolution ================================
def resolve_genome(spec: str, frozen_path: Path = FROZEN_GENOME,
                   allow_unfrozen: bool = False):
    """'frozen' -> the committed SYN-1 genome; 'B0' -> DEFAULT_GENOME."""
    if spec == "B0":
        return ea.Genome.b0(), {"genome": "B0_DEFAULT"}
    if Path(frozen_path).exists():
        g = ea.Genome.from_json(Path(frozen_path))
        return g, g.to_dict()
    if allow_unfrozen:
        return ea.Genome.b0(), "GENOME:UNFROZEN(plan-placeholder)"
    raise PreconditionFailed(
        f"REFUSED: frozen SYN-1 genome not found at {frozen_path} — no replay "
        f"executes before the single-configuration freeze commit (TOURNAMENT §5).")


def materialize_arm(arm: Arm, out_root: Path, frozen_path: Path = FROZEN_GENOME,
                    allow_unfrozen: bool = False) -> tuple[Path, str]:
    """Write <arm_dir>/arm_config.json (+ genome.json transform) for the runner."""
    arm_dir = Path(out_root) / arm.run_id
    arm_dir.mkdir(parents=True, exist_ok=True)
    genome_doc = "N/A(incumbent)"
    if arm.arm_config.get("strategy") == "syn1":
        g, genome_doc = resolve_genome(arm.arm_config.get("genome", "frozen"),
                                       frozen_path, allow_unfrozen)
        if arm.arm_config.get("member_gate_off"):
            g = drop_genome(g, arm.arm_config["member_gate_off"])
        if arm.arm_config.get("feature_gate_off"):
            fg = list(g.feature_gate)
            for fam in arm.arm_config["feature_gate_off"]:
                fg[ea.FEATURE_GATE_NAMES.index(fam)] = 0
            g.feature_gate = fg
        g.to_json(arm_dir / "genome.json")
        if not isinstance(genome_doc, str):
            genome_doc = g.to_dict()           # hash the TRANSFORMED genome
    ch = arm.config_hash(genome_doc)
    cfg = {**asdict(arm), "config_hash": ch,
           "files": {"genome": "genome.json"} if arm.arm_config.get("strategy") == "syn1" else {}}
    with open(arm_dir / "arm_config.json", "w") as fh:
        json.dump(cfg, fh, indent=1)
    return arm_dir, ch


# ============================ execution ========================================
REQUIRED_ALWAYS = ("result.json", "manifest.json")
REQUIRED_SERIES_A = ("daily_raw.csv", "daily_costadj.csv")   # documented shape
REQUIRED_SERIES_B = ("daily_series.csv",)                    # landed runner shape


def missing_outputs(arm_dir: Path) -> list[str]:
    arm_dir = Path(arm_dir)
    missing = [f for f in REQUIRED_ALWAYS if not (arm_dir / f).exists()]
    if not all((arm_dir / f).exists() for f in REQUIRED_SERIES_A) and \
            not all((arm_dir / f).exists() for f in REQUIRED_SERIES_B):
        missing.append("daily series (daily_raw.csv+daily_costadj.csv OR daily_series.csv)")
    return missing


def shipped_exec_mode(exec_dir: str | Path) -> str:
    """The replay exec-mode for an exec dir: <dir>/ladder.json
    'replay_exec_mode' (the §7.3 ladder rung that ships, recorded by the
    retrain driver; 'linear_twin' is the FROZEN base rung per FREEZE_SYN1.md).
    Missing ladder.json defaults to the frozen rung."""
    p = PROTO / exec_dir / "ladder.json"
    if p.exists():
        return json.loads(p.read_text()).get("replay_exec_mode", "linear_twin")
    return "linear_twin"


def default_replay_cmd(arm: Arm, arm_dir: Path) -> list[str]:
    """Map an arm onto the LANDED run_replay.py CLI. Refuses (PreconditionFailed,
    naming the missing knob) when the runner cannot express the arm — the
    orchestrator must then supply --replay-cmd explicitly."""
    cfg = arm.arm_config
    bp = cfg.get("executive_bypass")
    if bp and (not bp.get("equal_trust") or bp.get("f_fixed") != 0.7):
        raise PreconditionFailed(
            f"REFUSED: {arm.run_id} executive_bypass {bp} cannot be expressed "
            f"by run_replay.py — the only landed bypass is --exec-mode "
            f"equal_trust (tau=1/M, f=0.7). Supply --replay-cmd only once the "
            f"capability exists.")
    cmd = [sys.executable, str(PROTO / "run_replay.py"),
           "--arm", cfg.get("strategy", "syn1"), "--window", "full",
           "--out", str(arm_dir),
           "--cost-seed", str(cfg.get("slippage_seed", SEED_PAIRED))]
    if cfg.get("strategy") == "syn1":
        cmd += ["--genome", str(Path(arm_dir) / "genome.json")]
        if cfg.get("exec_dir"):
            cmd += ["--exec-dir", str(PROTO / cfg["exec_dir"])]
        # R07's replay-time sigma swap supersedes the variant-nightly-store
        # expression of the same transform — never pass both.
        if cfg.get("nightly_dir") and not cfg.get("sigma_source"):
            cmd += ["--nightly-dir", str(PROTO / cfg["nightly_dir"])]
        if cfg.get("sigma_source"):
            cmd += ["--sigma-source", cfg["sigma_source"]]
        # executive gate: R08 bypass, else the dir's SHIPPED ladder rung
        # (linear_twin for the frozen base; per-arm rung from ladder.json)
        cmd += ["--exec-mode", "equal_trust" if bp
                else shipped_exec_mode(cfg.get("exec_dir", "exec_out"))]
    return cmd


def execute_arm(run_id: str, replay_cmd: list[str] | None = None,
                out_root: Path = RUNS_ROOT, ledger: Path = HOLDOUT_LEDGER,
                retrain_ledger: Path = RETRAIN_LEDGER,
                frozen_path: Path = FROZEN_GENOME,
                allow_unfrozen: bool = False) -> dict:
    """Execute one battery replay: budget guard -> materialize -> command build
    (capability-gap refusals happen here, BEFORE a look is consumed) -> LEDGER
    WRITE -> replay subprocess (with PKT_TB_006_HOLDOUT_AUTHORIZED=1: the
    ledger write IS the authorization record) -> output verification. Raises,
    never warns, on any budget/precondition breach."""
    if run_id not in ARMS:
        raise KeyError(f"unknown run_id {run_id}")
    arm = ARMS[run_id]
    if arm.arm_config.get("reserved"):
        raise PreconditionFailed(f"{run_id} is a reserved contingency slot — "
                                 f"define its arm_config before executing")
    n = replay_count(ledger)
    if n >= MAX_REPLAYS:
        raise BudgetExceeded(
            f"REFUSED: replay #{n + 1} exceeds the hard cap of {MAX_REPLAYS} "
            f"holdout looks (TOURNAMENT §4.4/§5). Ledger: {ledger}")
    if arm.retrain_required:
        done = retrain_ids(retrain_ledger)
        if arm.retrain_required not in done:
            raise PreconditionFailed(
                f"REFUSED: {run_id} requires retrain {arm.retrain_required} "
                f"which is not registered in {retrain_ledger} (executed: {done}). "
                f"Run `battery.py register-retrain --rt {arm.retrain_required}` "
                f"after the retrain cycle completes.")
    arm_dir, ch = materialize_arm(arm, out_root, frozen_path, allow_unfrozen)
    if replay_cmd:
        cmd = list(replay_cmd) + ["--arm-config", str(arm_dir / "arm_config.json"),
                                  "--out", str(arm_dir)]
    else:
        cmd = default_replay_cmd(arm, arm_dir)   # may refuse BEFORE the look
    look = append_holdout_look(run_id, ch, arm.purpose or arm.description, ledger)
    env = dict(os.environ, PKT_TB_006_HOLDOUT_AUTHORIZED="1")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    status = {"run_id": run_id, "config_hash": ch, "arm_dir": str(arm_dir),
              "ledger_entry": look, "returncode": proc.returncode,
              "cmd": cmd}
    if proc.returncode != 0:
        status["error"] = (proc.stderr or proc.stdout)[-2000:]
        with open(arm_dir / "execute_status.json", "w") as fh:
            json.dump(status, fh, indent=1)
        raise RuntimeError(f"replay for {run_id} failed (look already consumed "
                           f"and ledgered — that is the honest record): "
                           f"{status['error'][:400]}")
    missing = missing_outputs(arm_dir)
    status["outputs_ok"] = not missing
    status["missing_outputs"] = missing
    with open(arm_dir / "execute_status.json", "w") as fh:
        json.dump(status, fh, indent=1)
    if missing:
        raise RuntimeError(f"replay for {run_id} completed but outputs missing: "
                           f"{missing} in {arm_dir}")
    return status


# ============================ plan mode ========================================
def arm_blockers(arm: Arm, frozen_path: Path = FROZEN_GENOME,
                 retrain_ledger: Path = RETRAIN_LEDGER) -> list[str]:
    """Execution-readiness check for one arm (NO ledger writes, NO replays):
    everything execute_arm would refuse on, plus artifact existence for the
    knobs default_replay_cmd would pass. Empty list = runnable now."""
    blockers: list[str] = []
    cfg = arm.arm_config
    if cfg.get("reserved"):
        return ["reserved contingency slot (arm_config undefined)"]
    if arm.retrain_required and arm.retrain_required not in retrain_ids(retrain_ledger):
        blockers.append(f"retrain {arm.retrain_required} not registered")
    if cfg.get("strategy") != "syn1":
        return blockers
    if cfg.get("genome", "frozen") == "frozen" and not Path(frozen_path).exists():
        blockers.append(f"frozen genome missing at {frozen_path}")
    if cfg.get("member_gate_off") and "<" in str(cfg["member_gate_off"]):
        blockers.append(f"member_gate_off placeholder {cfg['member_gate_off']} "
                        f"(decided from R04-R06)")
    exec_dir = cfg.get("exec_dir", "exec_out")
    bp = cfg.get("executive_bypass")
    if not bp:
        mode = shipped_exec_mode(exec_dir)
        need = ("linear_twin.pt" if mode == "linear_twin" else "executive_seed11.pt")
        if not (PROTO / exec_dir / need).exists():
            blockers.append(f"{exec_dir}/{need} missing (exec-mode {mode})")
    elif not bp.get("equal_trust") or bp.get("f_fixed") != 0.7:
        blockers.append(f"inexpressible executive_bypass {bp}")
    nd = cfg.get("nightly_dir")
    if nd and not cfg.get("sigma_source"):
        if not (PROTO / nd / "manifest.json").exists():
            blockers.append(f"variant nightly store {nd} missing")
    return blockers


def plan(allow_unfrozen: bool = True, frozen_path: Path = FROZEN_GENOME) -> dict:
    """Print + persist the full battery with config hashes BEFORE anything runs
    (the orchestrator commits this plan). Zero ledger writes."""
    rows = []
    for run_id, arm in ARMS.items():
        if arm.arm_config.get("strategy") == "syn1":
            try:
                g, gdoc = resolve_genome(arm.arm_config.get("genome", "frozen"),
                                         frozen_path, allow_unfrozen)
                if arm.arm_config.get("member_gate_off") and \
                        "<" not in arm.arm_config["member_gate_off"]:
                    g = drop_genome(g, arm.arm_config["member_gate_off"])
                    gdoc = g.to_dict() if not isinstance(gdoc, str) else gdoc
                if arm.arm_config.get("feature_gate_off") and not isinstance(gdoc, str):
                    fg = list(g.feature_gate)
                    for fam in arm.arm_config["feature_gate_off"]:
                        fg[ea.FEATURE_GATE_NAMES.index(fam)] = 0
                    g.feature_gate = fg
                    gdoc = g.to_dict()
            except PreconditionFailed:
                gdoc = "GENOME:UNFROZEN(plan-placeholder)"
        else:
            gdoc = "N/A(incumbent)"
        blockers = arm_blockers(arm, frozen_path)
        rows.append({
            "run_id": run_id, "description": arm.description,
            "retrain": arm.retrain_required or "—",
            "contingency": arm.contingency,
            "config_hash": arm.config_hash(gdoc),
            "e1": (arm.e1 or {}).get("kind", "—"),
            "e2": (arm.e2 or {}).get("pair_vs", "—"),
            "slippage_seed": arm.arm_config.get("slippage_seed"),
            "exec_mode": ("equal_trust" if arm.arm_config.get("executive_bypass")
                          else shipped_exec_mode(arm.arm_config.get("exec_dir", "exec_out"))
                          if arm.arm_config.get("strategy") == "syn1" else None),
            "runnable": not blockers,
            "blockers": blockers,
            "arm_config": arm.arm_config,
        })
    genome_frozen = Path(frozen_path).exists()
    doc = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "genome_frozen": genome_frozen,
        "frozen_genome_path": str(frozen_path),
        "note": (None if genome_frozen else
                 "PLAN PRINTED BEFORE GENOME FREEZE — config hashes for syn1 arms "
                 "use a placeholder genome doc; the plan MUST be re-printed and "
                 "re-committed after the freeze commit, before any execution."),
        "budgets": {"max_replays": MAX_REPLAYS, "max_retrains": MAX_RETRAINS,
                    "planned_replays": sum(1 for a in ARMS.values() if not a.contingency),
                    "planned_retrains": sum(1 for a in ARMS.values()
                                            if a.retrain_required and not a.contingency),
                    "replays_consumed": replay_count(),
                    "retrains_consumed": len(retrain_ids())},
        "seeds": {"paired": SEED_PAIRED, "sensitivity": list(SEED_SENS)},
        "e1_only_reads": E1_ONLY_READS,
        "planned_arms_runnable": all(r["runnable"] for r in rows
                                     if not r["contingency"]),
        "not_runnable": {r["run_id"]: r["blockers"] for r in rows
                         if not r["runnable"]},
        "arms": rows,
    }
    with open(PLAN_PATH, "w") as fh:
        json.dump(doc, fh, indent=1)
    return doc


def print_plan(doc: dict) -> None:
    print(f"PKT-TB-006 attribution battery plan ({doc['generated']}) — "
          f"genome_frozen={doc['genome_frozen']}")
    if doc.get("note"):
        print(f"  NOTE: {doc['note']}")
    b = doc["budgets"]
    print(f"  budgets: {b['planned_replays']} planned replays (cap {b['max_replays']}, "
          f"{b['replays_consumed']} consumed); {b['planned_retrains']} planned retrains "
          f"(cap {b['max_retrains']}, {b['retrains_consumed']} consumed)")
    hdr = (f"{'run':<5}{'retrain':<8}{'cont':<6}{'hash':<18}{'E1 read':<22}"
           f"{'E2 vs':<7}{'seed':<6}{'exec':<13}{'run?':<6}description")
    print(hdr)
    print("-" * len(hdr))
    for r in doc["arms"]:
        print(f"{r['run_id']:<5}{r['retrain']:<8}{str(r['contingency'])[0]:<6}"
              f"{r['config_hash']:<18}{r['e1']:<22}{r['e2']:<7}"
              f"{str(r['slippage_seed'] or '—'):<6}"
              f"{str(r.get('exec_mode') or '—'):<13}"
              f"{('YES' if r['runnable'] else 'NO'):<6}{r['description']}")
    if doc.get("not_runnable"):
        print("  NOT RUNNABLE:")
        for rid, bl in doc["not_runnable"].items():
            print(f"    {rid}: {'; '.join(bl)}")
    print(f"  planned (non-contingency) arms all runnable: "
          f"{doc['planned_arms_runnable']}")
    print(f"  plan written: {PLAN_PATH}")


# ============================ CLI ==============================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan", help="print + persist the battery plan (no runs)")
    rp = sub.add_parser("run", help="execute one battery replay (consumes a look)")
    rp.add_argument("--run-id", required=True)
    rp.add_argument("--replay-cmd", default=None,
                    help="override replay command (space-split)")
    rp.add_argument("--out-root", default=str(RUNS_ROOT))
    rp.add_argument("--allow-unfrozen", action="store_true",
                    help="harness smoke only — never for the real battery")
    rr = sub.add_parser("register-retrain", help="record one retrain cycle (cap 9)")
    rr.add_argument("--rt", required=True)
    rr.add_argument("--what", default="")
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        print_plan(plan())
    elif args.cmd == "run":
        cmd = args.replay_cmd.split() if args.replay_cmd else None
        status = execute_arm(args.run_id, replay_cmd=cmd,
                             out_root=Path(args.out_root),
                             allow_unfrozen=args.allow_unfrozen)
        print(json.dumps(status, indent=1))
    elif args.cmd == "register-retrain":
        print(json.dumps(register_retrain(args.rt, args.what), indent=1))


if __name__ == "__main__":
    main()
