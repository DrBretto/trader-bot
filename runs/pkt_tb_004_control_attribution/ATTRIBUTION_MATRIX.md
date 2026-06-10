# PKT-TB-004 — Attribution Matrix

Delta convention: **Δ = ablated-cell − C00 baseline** (layer OFF minus layer ON).
A layer's contribution to the strategy = **−Δ**: negative Δreturn ⇒ the layer
was adding return; positive Δreturn ⇒ removing the layer would have helped.
ΔmaxDD positive ⇒ removal made drawdown shallower (the guard was not earning its DD claim).

Baseline C00 = champion-as-deployed (active params, ranking blend 0.35) with stored
nightly LLM risks wired (`use_stored_llm_risks=true`). Harness [O] = optimizer replay,
from $100k cash, costed, 5 seeds {11,17,23,29,31}, median-seed values shown,
194 trading days 2025-08-05→2026-06-09 (valuation dates). Harness [C] = three-line
canon continuation from the real 2026-03-11 book, champion overlays, gross-of-costs,
deterministic. LLM rows: [O] full-period reads restricted to the LLM era (2026-01-29→).

**Gate** (pre-registered): paired daily |t|≥2.0 AND NW(5) |t|≥1.8 AND sign-consistent
across 5 seeds AND |Δret| > B (analytic slippage-noise bound). `MEANINGFUL` only if all four.


## [O] Full period (2025-08-05 → 2026-06-09)

| Cell | Layer | Δret | ΔmaxDD | ΔSharpe | Δexp | Δtrades | t | NW-t | B(bps) | sign5 | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T1-A | Vol-bucket size adjust (vol_adj) | +0.01pp | +0.01pp | +0.00 | -0.002 | +2 | -0.02 | -0.04 | 6.7 | N | ns |
| T1-B | Regime size adjust (regime_adj) | +2.60pp | -1.31pp | +0.18 | +0.020 | +2 | +0.88 | +1.10 | 7.0 | Y | ns |
| T1-LLM | LLM layer (size adj + buy veto + sell veto) *(LLM-era)* | +1.10pp | +0.14pp | +0.15 | -0.021 | -2 | +0.41 | +0.63 | 6.8 | Y | ns |
| T1-D | Ensemble disagreement multiplier | -6.51pp | -2.97pp | -0.46 | +0.018 | -27 | -1.36 | -1.40 | 7.3 | Y | ns |
| T1-E | Expert modifier psm (size channel) | -7.17pp | -7.16pp | -0.45 | +0.018 | -4 | -0.91 | -0.97 | 6.9 | Y | ns |
| T1-F | Risk throttle (throttle_scale) | -1.19pp | -1.53pp | -0.10 | +0.006 | +2 | -0.61 | -0.61 | 6.7 | Y | ns |
| T1-G | Regime compatibility score multiplier | +3.59pp | -1.54pp | +0.28 | +0.037 | +1 | +0.65 | +0.67 | 6.7 | Y | ns |
| T1-H | Regime-conditional buy thresholds | +0.26pp | +0.02pp | +0.02 | +0.002 | +2 | +0.53 | +0.66 | 6.8 | Y | ns |
| T1-I | Regime-conditional cash floors | +3.32pp | +0.28pp | +0.25 | +0.099 | +13 | +1.63 | +1.58 | 7.0 | Y | ns |
| T1-J | Regime label steering (fusion label flips) | -0.27pp | +0.29pp | -0.01 | +0.087 | +24 | -0.09 | -0.11 | 7.3 | Y | ns |
| T1-K | Trailing stop | -1.13pp | +0.90pp | -0.08 | -0.032 | -11 | -0.39 | -0.40 | 6.3 | Y | ns |
| T1-L | Health-collapse sell | -0.16pp | +0.98pp | +0.00 | +0.020 | -19 | -0.10 | -0.15 | 6.6 | Y | ns |
| T1-M | Panic force-sell | +6.01pp | +8.68pp | +0.94 | -0.020 | -32 | +0.44 | +0.48 | 6.0 | Y | ns |
| T1-O | Leverage hold cap | +0.00pp | +0.00pp | +0.00 | +0.000 | +0 | — | — | 6.7 | Y | ns |
| T1-P | Reduce-on-health-drop | -3.31pp | -0.65pp | -0.26 | +0.058 | -66 | -0.61 | -0.81 | 6.2 | Y | ns |
| T1-Q | Reduce-on-regime-shift | -0.90pp | +0.00pp | -0.07 | +0.005 | -5 | -1.38 | -1.10 | 6.7 | Y | ns |
| T1-S | Vol-bucket buy filter | +37.15pp | +0.94pp | +1.87 | +0.067 | +30 | +1.58 | +1.93 | 6.9 | Y | ns |
| T1-T | Panic asset-class buy filter | -0.35pp | +0.63pp | -0.02 | -0.003 | +6 | -0.36 | -0.38 | 6.7 | Y | ns |
| T2-ALLOFF | ENTIRE CONTROL STACK (all-off bracket) | +34.92pp | -14.57pp | +1.01 | +0.382 | -144 | +1.16 | +1.21 | 5.1 | Y | ns |

## [O] Holdout only, from cash (2026-03-11 →)

| Cell | Layer | Δret | ΔmaxDD | ΔSharpe | Δexp | Δtrades | t | NW-t | B(bps) | sign5 | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T1-A | Vol-bucket size adjust (vol_adj) | -0.03pp | +0.03pp | -0.01 | -0.004 | +0 | -0.43 | -0.51 | 6.7 | Y | ns |
| T1-B | Regime size adjust (regime_adj) | +0.01pp | -1.33pp | +0.12 | +0.007 | +1 | +0.11 | +0.12 | 7.0 | N | agrees |
| T1-LLM | LLM layer (size adj + buy veto + sell veto) | -1.00pp | -0.03pp | -0.21 | -0.036 | -2 | -1.26 | -1.17 | 6.8 | Y | ns |
| T1-D | Ensemble disagreement multiplier | -2.14pp | -2.99pp | -0.15 | +0.046 | -5 | -0.62 | -0.77 | 7.3 | Y | agrees |
| T1-E | Expert modifier psm (size channel) | -6.74pp | -7.38pp | -0.49 | +0.080 | -4 | -0.90 | -1.02 | 6.9 | Y | agrees |
| T1-F | Risk throttle (throttle_scale) | -1.13pp | -1.63pp | -0.12 | +0.026 | +2 | -0.60 | -0.62 | 6.7 | Y | agrees |
| T1-G | Regime compatibility score multiplier | -1.21pp | -1.63pp | -0.19 | -0.020 | +3 | -0.57 | -0.72 | 6.7 | Y | ns |
| T1-H | Regime-conditional buy thresholds | +0.93pp | +0.18pp | +0.18 | -0.020 | +3 | +1.00 | +1.05 | 6.8 | Y | agrees |
| T1-I | Regime-conditional cash floors | +0.00pp | +0.00pp | +0.00 | +0.000 | +0 | — | — | 7.0 | Y | ns |
| T1-J | Regime label steering (fusion label flips) | +0.00pp | +0.00pp | +0.00 | +0.000 | +0 | — | — | 7.3 | Y | ns |
| T1-K | Trailing stop | -2.75pp | +0.56pp | -0.69 | -0.110 | -7 | -1.61 | -1.62 | 6.3 | Y | agrees |
| T1-L | Health-collapse sell | +0.53pp | +0.05pp | +0.11 | +0.001 | -11 | +0.71 | +0.82 | 6.6 | Y | ns |
| T1-M | Panic force-sell | -1.51pp | +0.67pp | -0.34 | +0.025 | -1 | -1.04 | -0.99 | 6.0 | Y | ns |
| T1-O | Leverage hold cap | +0.00pp | +0.00pp | +0.00 | +0.000 | +0 | — | — | 6.7 | Y | agrees |
| T1-P | Reduce-on-health-drop | -1.18pp | -0.80pp | -0.23 | +0.104 | -16 | -0.55 | -0.64 | 6.2 | Y | agrees |
| T1-Q | Reduce-on-regime-shift | -0.82pp | +0.00pp | -0.17 | +0.021 | -5 | -1.39 | -1.14 | 6.7 | Y | agrees |
| T1-S | Vol-bucket buy filter | +1.83pp | +7.49pp | -1.02 | -0.055 | +9 | +0.11 | +0.12 | 6.9 | Y | agrees |
| T1-T | Panic asset-class buy filter | -0.29pp | +0.65pp | -0.10 | -0.014 | +6 | -0.34 | -0.36 | 6.7 | Y | agrees |
| T2-ALLOFF | ENTIRE CONTROL STACK (all-off bracket) | +5.43pp | +9.49pp | +1.20 | +0.499 | -54 | +0.42 | +0.51 | 5.1 | Y | agrees |

## [C] Holdout, canon-line continuation (gross-of-costs, deterministic)

| Cell | Layer | Δ final value | Δret (pp of book) | direction vs [O] holdout |
|---|---|---|---|---|
| T1-LLM | LLM layer (size adj + buy veto + sell veto) | -2152$ | -2.10pp | CONFIRMS |
| T1-K | Trailing stop | -676$ | -0.66pp | CONFIRMS |
| T1-L | Health-collapse sell | +768$ | +0.75pp | CONFIRMS |
| T1-M | Panic force-sell | -6371$ | -6.22pp | CONFIRMS |
| T1-O | Leverage hold cap | +0$ | +0.00pp | both ~0 |
| T1-P | Reduce-on-health-drop | -579$ | -0.56pp | CONFIRMS |
| T1-Q | Reduce-on-regime-shift | -303$ | -0.30pp | CONFIRMS |
| T2-ALLOFF | ENTIRE CONTROL STACK (all-off bracket) | -10728$ | -10.47pp | CONTRADICTS |

## Bracket & triggers

- Interaction mass R = Σ-residual of the all-off bracket vs sum of LOO deltas: **-7.5506 bps/day** (pre-holdout), Σ|δᵢ| = 34.7418 bps/day, SE(R) = 16.032 → S1 trigger not fired.
- Tier-1b (LLM channel split): parent T1-LLM t = 0.413 (LLM-era) / -1.26 (holdout) → NOT fired; T1b-C/N/R = SKIPPED-PARENT-NULL.
- S2 wrong-sign trigger: 0 cells. S3 instability: 0 cells.
- Tier-3 pairwise cells run: **0** (no trigger fired).

## Substrate caveats (carry into every read of this matrix)

1. **2025 inference artifacts are backfilled one-hot heuristics** (S3 timestamps 2026-03-18),
   not live model output; real GRU+Transformer output exists only from 2026-01-31.
   Full-period reads of regime/health-conditioned layers partially reflect that backfill.
2. Endpoint deltas are sign-stable across seeds and ≫ B for several layers while paired daily
   t-stats stay <2: the effects are path-shaped (a changed buy compounds forever), not
   daily-mean-shaped. The pre-registered gate treats these as NOT MEANINGFUL; endpoint
   magnitudes are reported for honesty, not as proof.
3. From-cash holdout runs are not the canon book; [C] block is the on-book read (gross of costs).
4. One market path; ~10 months; no cross-path resampling. See Skeptic section of the
   committee report for what these ablations cannot conclude.
