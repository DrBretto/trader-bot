# PKT-TB-004 — Committee Panel Declaration

Declared 2026-06-09 before deliberation, per packet `## Committee panel`.

| # | Role | Source | Charter |
|---|------|--------|---------|
| 1 | Analyst | mandatory (STABLE-001) | Owns the attribution matrix: mechanism × (return Δ, Sharpe Δ, max-DD Δ, exposure Δ, trade-count Δ), full period and holdout. |
| 2 | Skeptic | mandatory (STABLE-002) | Attacks attribution validity: layer interaction effects, replay path-dependence, veto-scoring survivorship. Forces the panel to state what the ablations CANNOT conclude. |
| 3 | Causal Attribution Methodologist | ad hoc (forced relevant lens) | Designs the ablation battery: leave-one-out per layer, all-off/all-on bracket, pairwise interactions where LOO flags surprises; prescribes seeds/replays per cell per EVIDENCE_PROTOCOL §2; sets the holdout-look budget. |
| 4 | Risk Architect | ad hoc | Owns the exposure-asymmetry question: designs the target-gross-exposure pro-rata trim CANDIDATE (triggers, hysteresis, cost drag) and the VIXY policy candidate; both submit to the same replay proof. |
| 5 | Model Calibration Specialist | ad hoc | Owns the regime-confidence question: full confidence/disagreement history from stored artifacts; typical-vs-episodic diagnosis; temperature scaling and 5→3 collapse candidates; downstream counterfactual. |
| 6 | LLM Veto Auditor | ad hoc | Scores every historical Haiku veto/downsize against subsequent asset outcomes: hit rate vs base rate, dollar impact of obeying vs ignoring; the Bedrock step must earn its place. |

Each role runs as a real subagent (Agent tool); invocation digests recorded in the
committee report receipts. Synthesis is performed as a separate step by the committee
chair after reading each return.
