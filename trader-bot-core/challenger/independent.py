"""The INDEPENDENT challenger reconstruction (PKT-TRADER-BOT-SEED-CANON-BY-REPLAY-V2).

challenger = the LEGACY incumbent brain (``incumbent`` = ported
``decision_engine``: ``final_score = base_score(health) x regime_multiplier``,
top-``max_positions`` by score) run per settled day + the ORB-1 ``M1`` conviction
tilt (``tilt`` = ported ``tilt_adapter``, ``T_MAX`` <= 8% NAV) overlaid — run
INDEPENDENTLY of the two-stage canon. It NEVER reads the two-stage's intents / the
coupled ``publish/challenger`` (that coupled M1-tilt-of-two-stage is superseded).

Per settled day D (the SAME as-of-D substrate the canon two-stage consumes):
  1. build the incumbent's inputs from the corrected substrate — the FULL 64
     universe, the RECORDED regime label (``regime_multiplier`` from the same
     ``regime_compatibility`` table), decision-time marks (D-1 settled close);
     health is UNKNOWN on this substrate (no reproducible health signal — the same
     benefit-of-doubt the canon adapter applies), so the incumbent ranks on
     ``regime_multiplier`` (its regime picker) — the deliberate contrast to the
     two-stage's ``mu_M1``-PRIMARY ranking;
  2. run the ported incumbent ``run()`` (legacy path, no expert-signal fusion) ->
     incumbent intents;
  3. overlay the ported ``M1`` tilt from the FRESH organ inputs
     ``forecast/inference.py`` wrote as-of-D (``organs.M1.mu`` = fresh mu, ``q``,
     ``disp_z``, ``p_exceed``; GDELT-as-was in ``p_exceed``) -> tilted intents;
  4. execute the tilted intents at the settled OPEN via the ported live-fidelity
     executor (``lot_fix._execute_intents_lotfix``), then mark peak/last-close;
  5. the day's line value = the ONE settled-close mark (``settled_close_value`` =
     cash + sum(shares x settled close)) — identical valuation to canon/SPY.

The book carries per-lot entry metadata across days so the incumbent's multi-day
sell/trim gates (HEALTH_COLLAPSE, HEALTH_DROP, REGIME_SHIFT, trailing stops) fire.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from challenger import incumbent as INC
from challenger.book import Portfolio, settled_close_value
from challenger.lot_fix import _execute_intents_lotfix
from challenger.risk_stats import RiskStats
from challenger.strategy import StrategyContext
from challenger.tilt import make_tilt_strategy

_THIS = Path(__file__).resolve()
CORE_ROOT = _THIS.parents[1]
GENOME_PATH = _THIS.parent / "shadow_A.json"           # the frozen shadow-A genome
UNIVERSE_CSV = CORE_ROOT / "config" / "universe.csv"


def _vol_bucket(v: Optional[float]) -> str:
    """Same bucketing the canon adapter uses (adapter.forecast_adapter._vol_bucket)."""
    if v is None:
        return "med"
    if v < 0.12:
        return "low"
    if v > 0.30:
        return "high"
    return "med"


def _asset_health(universe_df, features_df) -> List[Dict[str, Any]]:
    """Health-blind asset_health for the incumbent, on the reconstruction
    substrate: no reproducible per-symbol health signal exists (the live LLM/expert
    health was a point-in-time artifact, not part of the deterministic seed — the
    same reason the canon adapter defaults UNKNOWN health to a benefit-of-doubt
    pass). Each row carries the health-neutral 1.0 score + the symbol's vol_bucket
    (from the as-of-D realized vol), so the incumbent's ``final_score = 1.0 x
    regime_multiplier`` ranks on its regime picker — the intended incumbent-ranker
    contrast to the two-stage's mu-primary ranking."""
    vol_by_sym: Dict[str, float] = {}
    if features_df is not None and len(features_df) > 0 and "vol_21d" in features_df.columns:
        for _, r in features_df.iterrows():
            vol_by_sym[str(r["symbol"])] = r.get("vol_21d")
    rows: List[Dict[str, Any]] = []
    for sym in universe_df["symbol"].astype(str):
        rows.append({"symbol": sym, "health_score": 1.0,
                     "vol_bucket": _vol_bucket(vol_by_sym.get(sym))})
    return rows


class IndependentChallenger:
    """Stateful per-day incumbent+tilt book, marked by the ONE settled-close
    machinery. Constructed once per reconstruction; ``step`` advances it one settled
    day; ``value`` returns the day's marked line value."""

    def __init__(self, *, cash_anchor: float, organ_dir: Path, ohlcv_dir: Path,
                 decision_params: Dict[str, Any],
                 regime_compat: Mapping[str, Mapping[str, float]],
                 log_dir: Optional[Path] = None):
        self.book = Portfolio(cash=float(cash_anchor))
        self.decision_params = dict(decision_params)
        self.regime_compat = dict(regime_compat or {})
        # the ported ORB-1 tilt as a Strategy; RiskStats bound to the EXTENDED core
        # OHLCV store (seed + spliced post-split bars) for as-of-D beta/sigma.
        self.strategy = make_tilt_strategy(
            genome=GENOME_PATH, nightly_dir=Path(organ_dir),
            risk=RiskStats(ohlcv_dir=Path(ohlcv_dir)),
            universe_csv=UNIVERSE_CSV, log_dir=log_dir)

    def step(self, D: str, *, features_df, universe_df, regime_label: str,
             ohlc: Mapping[str, Dict[str, float]]) -> Dict[str, Any]:
        """Run incumbent+tilt for settled day D and execute at D's open.

        ``features_df``: one row/symbol carrying the decision-time mark (D-1 settled
        close) + vol_21d (the driver's ``_features_df_asof``).
        ``ohlc``: {symbol: {"open": .., "close": ..}} for D (settled fills + close).
        """
        marks = {str(r["symbol"]): float(r["close"])
                 for _, r in features_df.iterrows()
                 if r.get("close") is not None and float(r["close"]) > 0}
        portfolio_state, _missing = self.book.to_state_dict_with_marks(marks)

        asset_health = _asset_health(universe_df, features_df)
        inference_output = {
            "date": D,
            "asset_health": asset_health,
            "regime": {"label": regime_label, "position_size_multiplier": 1.0,
                       "disagreement": 0.0, "probs": {}, "confidence": 1.0},
        }
        config = {
            "decision_params": self.decision_params,
            "regime_compatibility": self.regime_compat,
            "decision_engine_overrides": {},
            "universe": universe_df,
            "portfolio_state": portfolio_state,
        }
        validation = {"price_coverage": 1.0, "degraded_mode": False}

        # incumbent (ported decision_engine) — legacy path (expert_signals=None: the
        # recorded regime label is used directly, no v3 fusion). VERBATIM run().
        buf = io.StringIO()
        with redirect_stdout(buf):
            decisions = INC.run(inference_output, {}, features_df, config,
                                validation, expert_signals=None)
        incumbent_intents = list(decisions.get("actions", []))

        # overlay the ported M1 tilt (edits the incumbent intents at the margin;
        # neutral day => incumbent intents unchanged)
        ctx = StrategyContext(
            inputs_date=D, portfolio=self.book, variant_config=config,
            expert_signals={}, expert_metrics={}, decisions=decisions,
            panic_streak=0, last_regime=None, features_df=features_df,
            inference=inference_output, llm_risks={})
        if self.strategy.post_decision is not None:
            final_intents = self.strategy.post_decision(ctx, incumbent_intents)
        else:
            final_intents = incumbent_intents

        # execute at the settled OPEN (ported live-fidelity executor), then mark
        # peak/last-close to the settled close (the incumbent's stop/peak state).
        executed = _execute_intents_lotfix(
            self.book, final_intents, dict(ohlc), self.decision_params, D,
            entry_regime=regime_label)
        closes = {s: q.get("close") for s, q in ohlc.items() if q.get("close")}
        _mark_close(self.book, closes)
        return {"n_incumbent_intents": len(incumbent_intents),
                "n_final_intents": len(final_intents),
                "n_executed": len(executed),
                "tilt_active": len(final_intents) != len(incumbent_intents)
                or any(it.get("reason", "").startswith("ORB1") for it in final_intents)}

    def value(self, closes: Mapping[str, float]) -> float:
        """The ONE settled-close mark: cash + sum(shares x settled close)."""
        return settled_close_value(self.book, closes)


def _mark_close(book: "Portfolio", closes: Mapping[str, float]) -> None:
    """Update per-lot last_close + peak_price at the settled close (mirrors the
    legacy ``replay_engine._mark_to_close``) so the incumbent's trailing-stop /
    peak-health gates track correctly across days. Valuation is unchanged."""
    for p in book.positions:
        c = closes.get(p.symbol)
        if c is not None:
            p.last_close = float(c)
            if float(c) > p.peak_price:
                p.peak_price = float(c)
