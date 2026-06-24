"""Tilt cutover — the ``tilt_adapter`` book (the COMPARISON line, not the live engine).

NOTE (PKT-TB-BV-01 docstring correction): this module's old docstring was inverted.
The brain changed several times on 2026-06-23 (git): ``a35ca30`` briefly made the
tilt the live engine and retired the two-stage model, but ``d3495b1`` restored the
regime chassis and **``ec9561d`` "Make the regime-restored two-stage the live engine
+ comparison vs the tilt"** is the current live state. So:

  LIVE ENGINE  = the regime-restored two-stage engine (``runtime.run_cutover`` ->
                 ``src/brain/engine/``), which writes the live ``trade_intents.json``
                 (``engine == "native_two_stage"``); the restored regime model lives
                 inside it as a per-name ``regime_score_mult`` in Stage-1 ranking.
  THIS MODULE  = the ``tilt_adapter`` book (deterministic chassis intents + a SMALL
                 capped M1 tilt — the shadow's yellow line, book F = regime throttle
                 + M1 tilt). It is the COMPARISON line, invoked only when explicitly
                 configured (``engine == "tilt_adapter"``), and is NOT the default
                 live path.

``run_tilt_cutover`` mirrors the shadow's ``process_book_date`` for the F book but
writes REAL intents on that configured-comparison path. Hard fail-safe: ANY error
returns ok=False and the caller leaves the deterministic incumbent intents untouched
(abort-never-degrade).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[2]

# The frozen a-priori M1 genome = the yellow book F (shadow_A). event_damp=0 → the
# pure M1 forecast tilt; the deterministic rules carry everything else.
GENOME_REL = "runs/pkt_tb_007_orthogonal_brain/shadow/genomes/shadow_A.json"


class _Pos:
    """Minimal position the tilt's aggregate_book reads (symbol/shares/marks)."""
    __slots__ = ("symbol", "shares", "entry_price", "last_close")

    def __init__(self, symbol, shares, mark):
        self.symbol = str(symbol)
        self.shares = float(shares)
        self.entry_price = float(mark) if mark else 1.0
        self.last_close = float(mark) if mark else None


class _Book:
    __slots__ = ("positions", "cash")

    def __init__(self, positions, cash):
        self.positions = positions
        self.cash = float(cash)


def _config_root() -> Path:
    task_root = os.environ.get("LAMBDA_TASK_ROOT")
    if task_root and (Path(task_root) / "config").exists():
        return Path(task_root)
    return _REPO_ROOT


def run_tilt_cutover(
    run_date: str,
    features_df,
    regime_label: str,
    portfolio_state: dict,
    incumbent_intents: List[dict],
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    """Apply the ML tilt to the deterministic incumbent intents. Returns
    ``{ok, trade_intents, reason, n_edits}``. On ANY failure ok=False and the
    caller keeps the incumbent intents (fail-safe to the deterministic rules)."""
    config = config or {}
    try:
        # (1) forward inference — writes organ_inputs/<D>.json (same models the
        #     retired two-stage path used) and sets up sys.path for the tilt libs.
        from .runtime import production_forecaster
        production_forecaster([run_date])

        import shadow_lib as SL                       # noqa: E402 (path set above)
        from tilt_adapter import make_tilt_strategy   # noqa: E402
        from src.utils.three_line_replay.strategies import StrategyContext

        organ_file = SL.ORGAN_DIR / f"{run_date}.json"
        if not organ_file.exists():
            return {"ok": False, "reason": f"no organ outputs for {run_date}"}

        # (2) regime throttle (R organ): scales risk-ADD legs in panic/defensive;
        #     identity in normal regimes, so it only ever de-risks.
        throttled = SL._regime_throttle([dict(a) for a in incumbent_intents], regime_label)

        # (3) the M1 tilt strategy (book F = the yellow line).
        genome = _config_root() / GENOME_REL if (_config_root() / GENOME_REL).exists() \
            else _REPO_ROOT / GENOME_REL
        uni_csv = _config_root() / "config" / "universe.csv"
        strat = make_tilt_strategy(genome, nightly_dir=SL.ORGAN_DIR, universe_csv=uni_csv)

        # (4) book state for the tilt's parity/lot solve.
        marks = {}
        for h in portfolio_state.get("holdings", []) or []:
            mk = h.get("current_price") or h.get("close_price") or h.get("entry_price")
            if mk:
                marks[str(h.get("symbol"))] = float(mk)
        positions = [_Pos(h.get("symbol"), h.get("shares", 0), marks.get(str(h.get("symbol"))))
                     for h in portfolio_state.get("holdings", []) or [] if h.get("symbol")]
        book = _Book(positions, portfolio_state.get("cash", 0.0) or 0.0)

        ctx = StrategyContext(
            inputs_date=run_date, portfolio=book,
            variant_config={"decision_params": dict(config.get("decision_params", {}))},
            expert_signals={}, expert_metrics={}, decisions={},
            panic_streak=0, last_regime=None,
            features_df=features_df, inference={}, llm_risks={})

        tilted = strat.post_decision(ctx, throttled)
        if tilted is None:
            tilted = throttled  # tilt went neutral -> the deterministic rules stand

        n_edits = sum(1 for a in tilted if str(a.get("reason", "")).startswith("ORB1_TILT"))
        intents = {
            "generated_date": run_date,
            "generated_timestamp": "",
            "regime": regime_label,
            "actions": tilted,
            "buy_candidates": [],
            "expert_metrics": {"engine": "tilt_adapter", "genome": "shadow_A",
                               "n_tilt_edits": n_edits},
            "expires_after_days": 3,
        }
        return {"ok": True, "trade_intents": intents,
                "reason": f"tilt_adapter live ({n_edits} ML edits on the deterministic rules)",
                "n_edits": n_edits}
    except Exception as e:  # noqa: BLE001 — abort-never-degrade: keep incumbent
        return {"ok": False, "reason": f"tilt cutover failed (ABORT -> incumbent): "
                                       f"{type(e).__name__}: {e}"}
