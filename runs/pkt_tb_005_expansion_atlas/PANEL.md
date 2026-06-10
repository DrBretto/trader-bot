# PKT-TB-005 Expansion Atlas — Panel Declaration

Declared 2026-06-09, before deliberation, per packet requirement.
Each panelist runs as a real subagent (Agent tool), distinct charter prompt,
return captured verbatim under `panel_returns/`.

| # | Role | Charter (owned output) |
|---|------|------------------------|
| 1 | **Analyst** (mandatory) | Atlas structure + triage ledger: every candidate gets {hypothesis, mechanism, data availability+cost, Lambda/training fit, evidence path to E2, triage verdict}. Synthesizes the converged ATLAS.md. |
| 2 | **Skeptic** (mandatory) | Graveyard prior: attacks each candidate's mechanism ("why would this edge exist AND be free AND survive daily cadence?"); enforces multiple-candidates-vs-holdout honesty; designs the mandatory dumb-baseline pilot. |
| 3 | **Information Scout** (ad hoc) | Enumerates DATA angles with a concrete FREE source within sandbox/allowlist reach for each, or parks with named cost. |
| 4 | **Model Architect** (ad hoc) | Enumerates MODEL angles (regime alternatives/baselines, supervised heads, sizing, ensemble weighting, expert-signal stacking) within monthly-local-training compute. |
| 5 | **Portfolio Strategist** (ad hoc) | Enumerates STRATEGY-SHAPE angles (universe composition, regime-conditional universes, hedging budget, cash-as-position yield, rebalance cadence). |
| 6 | **Feasibility Auditor** (ad hoc) | Grades every shortlisted candidate against the operating envelope (cost/cadence/compute/allowlist) BEFORE pilot effort; kills/parks with the breach named. |

## Deliberation order

1. Scout / Architect / Strategist enumerate in parallel (breadth pass).
2. Skeptic + Feasibility Auditor attack/grade the combined candidate list in parallel.
3. Analyst synthesizes the triage ledger; orchestrator (this thread) carries
   `pilot-now` winners to replay evidence; dissent recorded in COMMITTEE_REPORT.md.

## Shared substrate brief given to all panelists

- 65-ETF daily system; GRU+Transformer 5-regime ensemble (equal-weight avg, disagreement
  throttle); AE/VAE health scores; 4 expert signals (macro/credit, vol complexity
  VIX/VVIX/SKEW, fragility PCA, entropy shift); priority-ordered regime fusion;
  layered sizing; LLM risk veto; sim-only fills (no broker).
- Regime models are trained on pseudo-labels from a rule-based threshold labeler
  (`training/utils/metrics.py:compute_regime_labels_from_baseline`).
- Stored per-day artifacts on S3 (`daily/<date>/`): features, signals, full inference
  (incl. separate gru_prediction / transformer_prediction probs), prices.
  Coverage 2025-08-04 → 2026-06-10 (235 dates). Holdout boundary 2026-03-11.
- Local parquets: context+GDELT 2014-12 → 2026-02 (training/data/).
- Replay harnesses: `optimizer/replay.py` (pipeline-faithful walk-forward substrate)
  and `src/utils/three_line_replay/` (production-path, strategy hooks).
- Envelope: ~$9/mo, free data only, daily-or-slower signal, thin Lambda handlers,
  monthly local training window, network allowlist (Stooq, FRED, yfinance hosts,
  GDELT, Alpha Vantage free tier, major crypto public APIs).
