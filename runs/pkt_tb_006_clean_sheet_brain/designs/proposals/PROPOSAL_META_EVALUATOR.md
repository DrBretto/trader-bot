# PROPOSAL — Meta-Evaluator (the brain's executive)

**Panel role 7 (Meta-Evaluator Designer), PKT-TB-006, 2026-06-10.**
**Designed blind to the incumbent, from the packet + ASSIGNMENT_BRIEF.md only.**
**Audience: Architects Alpha/Beta/Gamma (graft this organ into your system) and the build engineer (implement from this document alone).**

This proposal is architect-agnostic. Wherever it says "expert," substitute whatever upstream
organs your design has (forecast heads, allocation policies, GDELT/event models, the LLM
sentiment organ). The executive's contract is deliberately narrow so it grafts cleanly.

---

## 0. Design thesis (one paragraph)

With ~2,959 daily decision points of public history and ~210 days of own record (brief §7),
the executive cannot be a large learned policy — but it must be genuinely learned, on
decision-grade targets. The resolution: a **tiny gating network (≤ ~3k effective parameters)
with hard structural priors**, trained by **differentiable decision replay** — the loss IS the
realized forward utility of the actions the network would have taken, computed through the
same fill/cost mechanics the bake-off harness uses. No regime labels anywhere in the loss
path. Per-model trust is a separate, observable softmax head whose main *input* is each
expert's rolling counterfactual P&L — so most of the "which model do I trust today"
intelligence comes from honest measurement, and the learned part only has to map
(measurement, context, agreement) → (trust, deployment), a low-dimensional problem the
sample budget can actually pay for.

---

## 1. Contract: what the executive sees and what it emits

### 1.1 Inputs (decision date D; everything derived from `daily/D/*`, i.e. data through D-1 close)

Let M = number of experts (expect 3–6) and S = universe size (64).

1. **Per-expert opinion matrix** `mu[M, S]` — each expert's standardized per-symbol edge
   (expected h-day excess return, z-scored cross-sectionally per expert per day). Experts that
   natively emit something else (probabilities, scores, event intensities) are adapted to this
   unit by their own organ; the executive requires this one common currency.
2. **Per-expert uncertainty** `sigma[M, S]` (expert-reported, or its rolling residual sd if
   the expert reports none) and **self-confidence scalar** `c[M]` per expert per day.
3. **Per-expert rolling counterfactual performance** `r[M, K]` — K≈4 stats per expert from
   the trust ledger (§4): EWMA counterfactual utility at 21d/63d half-lives, hit-rate, and
   counterfactual max-drawdown, all computed only from windows fully realized by D-1 (§5).
4. **Context vector** `z[~16]` — market state from `daily/D/context.parquet` (vol level,
   term slope, yield slope, credit proxy, risk-off proxy) **plus** the sentiment organs'
   day-level aggregates (LLM sentiment score/dispersion, GDELT tone/intensity features as
   defined by roles 9–10). Sentiment enters the executive twice: as experts (their `mu` rows)
   *and* as context (their day-level aggregates condition trust and deployment).
5. **Agreement features** `g[~6]` — cross-expert sign-agreement rate, rank correlation of
   opinion vectors, dispersion of per-expert portfolio implications. (Disagreement is the
   single most decision-relevant statistic an executive can see; it is cheap and unlearned.)
6. **Current portfolio weights** `w_prev[S]` (from the replay/portfolio state) — the
   executive prices its own turnover.

### 1.2 Outputs (the decision-grade calls)

1. **Trust vector** `tau[M]` — simplex (sums to 1): how much to listen to which model today.
2. **Capital deployment fraction** `f ∈ [0, f_max]` — how much to spend (gross exposure as a
   fraction of portfolio value; `f_max = 1.0`, no leverage flags in the universe).
3. **Target portfolio weights** `w_tgt[S]` — derived deterministically from (tau, f, mu,
   sigma) by the blending rule below, then diffed against `w_prev` into BUY/SELL/REDUCE
   intents for the harness (fills at next open, harness cluster cap applies downstream).

The executive emits (1)–(3) plus the audit record (§6) as `daily/D/meta_decision.json`
(run-dir equivalent during the prototype).

---

## 2. Architecture and functional form

### 2.1 The chosen form: structured two-head gating MLP with per-symbol weight sharing

**Trust head** (per-expert, weight-shared across experts — this is the key parameter economy):

```
e_m   = phi([r_m, c_m, agree_m, z])          # phi: 1 hidden layer, width 8, tanh; SAME phi for every expert
s_m   = v . e_m + b                          # scalar logit per expert
tau   = softmax((s_1..s_M) / T) with entropy floor: tau <- (1-eps)*tau + eps/M,  eps = 0.05
```

`phi` shared across experts means trust generalizes across experts ("an expert whose recent
counterfactual P&L is good and who agrees with consensus in a calm tape gets weight") rather
than memorizing expert identities. An optional per-expert bias `b_m` (M params) is the only
identity-specific capacity, and it is the first thing pruned if validation says overfit.

**Blend** (deterministic, no parameters):

```
mu_blend[s]  = sum_m tau[m] * mu[m, s]
prec[s]      = sum_m tau[m] / sigma[m, s]^2          # trust-weighted precision
w_raw[s]     = clip(mu_blend[s], -q, +q) * prec[s]   # q = cross-sectional 95th pct, fixed
w_dir[s]     = max(w_raw[s], 0)                      # long-only unless the host design says otherwise
w_unit       = w_dir / sum(w_dir)                    # unit-gross candidate book
```

**Sizing head** (how much to spend):

```
f = f_max * sigmoid(psi([z, g, |mu_blend| stats, tau entropy, realized vol of w_unit book]))
    # psi: 1 hidden layer, width 8
w_tgt = f * w_unit, then vol-target overlay: scale so trailing-21d predicted book vol <= sigma_cap (fixed, pre-registered)
```

**Intents:** `delta = w_tgt - w_prev`; emit BUY/SELL/REDUCE intents only where
`|delta[s]| > min_trade` (fixed band, e.g. 25 bps of portfolio value) — the no-trade band is
the executive's built-in turnover brake and is priced in training (§3).

**Parameter count:** with M=5, K=4, |z|=16, |g|=6: phi ≈ (4+1+1+16+1)·8+8 ≈ 192; v,b ≈ 9;
b_m ≈ 5; psi ≈ (16+6+4+1+1+1)·8 + 8 + 9 ≈ 249. **Total ≈ 450–600 parameters.** Even tripling
feature widths stays under ~3k. Against ~2.4k usable pretraining decisions (and ~600
effectively independent h-day windows, §5), this is a defensible ratio; a 50k-parameter
gate would not be.

### 2.2 Alternatives considered and rejected (engineer: do not "upgrade" to these)

- **Attention-over-experts / set transformer.** Permutation invariance over a *fixed set of
  3–6 experts* buys nothing the shared `phi` doesn't already give, and spends parameters on
  machinery (keys/queries/heads) rather than signal. The packet's transformer requirement is
  the ensemble's job (roles 3–5), not the executive's. Rejected: capacity without cause.
- **Gradient-boosted decision head (LightGBM-style).** Strong on tabular, but (a) the loss
  here is realized utility *through* a portfolio construction — non-differentiable for trees,
  so you'd have to discretize the action space or regress a value function and argmax it,
  each an extra approximation layer; (b) a simplex-valued trust output is awkward as a tree
  target; (c) trees on 2.4k rows with ~40 features memorize unless strangled to the point of
  being a lookup table. **Kept as the challenger ablation** (§7): a GBM trained to regress
  realized utility of a small discrete action grid. If the challenger matches the gate on
  validation, report it — that is a finding about how much learnable structure exists.
- **RL (PPO / Q-learning) over the daily episode.** Brief §7 names this dead on arrival:
  long-horizon credit assignment from 2.9k steps from scratch. The differentiable h-day
  utility (§3) is the degenerate-but-honest version: a myopic policy gradient with exact
  credit assignment and no bootstrapping.
- **Pure heuristic (inverse-variance trust, fixed f).** This is the leave-one-out *baseline*,
  not the organ (§7). If the learned executive cannot beat it, the honest verdict is that it
  cannot, and the dossier says so.

---

## 3. Decision-grade training targets and loss (the load-bearing section)

### 3.1 Differentiable decision replay — the loss is the realized utility of the choice

There is no label. For each training decision date D, the network maps inputs (§1.1) to
`w_tgt(theta)`; we then compute, from *recorded* prices, the utility that choice would have
realized over horizon h, including costs, and ascend it. Targets-as-labels (e.g. "the best
expert in hindsight") are explicitly avoided: they re-introduce a classification problem and
discard sizing information.

**Forward utility of the choice, horizon h = 5 trading days (pre-registered, not tuned):**

```
# All prices from the deep-history parquet / Stooq OHLCV, aligned to the harness convention:
# decision at D uses data through D-1 close; fill at D's open; mark at closes D..D+h-1.

r_book(t)   = sum_s w_path[s, t] * ret[s, t]                  # daily book returns, t = D..D+h-1
cost(D)     = sum_s |w_tgt[s] - w_prev[s]| * (half_spread_bps[s] + E_slip) / 1e4
              # half_spread_bps from the SAME per-sector table as src/utils/transaction_costs.py
              # E_slip = 0 (slippage is mean-zero ±2bps uniform); spread is the systematic cost
U_h(D)      = (1/h) * sum_t log(1 + r_book(t))                # growth term
              - lam_dn * (1/h) * sum_t min(0, r_book(t))^2    # downside penalty (one-sided variance)
              - cost(D) / h                                   # amortized entry cost
Loss        = - mean_D [ U_h(D) ]
              + beta_to * mean_D [ sum_s |w_tgt(D) - w_tgt(D-1)| ]     # turnover smoothness
              + beta_H  * mean_D [ -H(tau(D)) only when H below floor ] # anti-collapse
              + weight decay (lam_wd) on all parameters
```

`lam_dn` is fixed at a pre-registered value (declared in TOURNAMENT.md before any holdout
read; candidate default 5.0 on daily log-return units) — it is the brain's risk-aversion
constant and is an *evolution-visible knob* (§3.4), never holdout-tuned by hand.

The gradient flows: prices are constants; `w_tgt` is differentiable in theta through tau and
f (softmax/sigmoid/clip — use soft clip); `|.|` terms use smooth-abs (sqrt(x²+1e-8)). The
no-trade band is applied with a straight-through estimator at train time, hard at inference.
This is a standard differentiable-backtest setup; it trains in seconds at this scale.

### 3.2 Why w_prev during training

Utility of a *choice* includes what it costs to move there. During pretraining, `w_prev` is
generated by running the executive sequentially over the training window (teacher-forced in
the first epoch with `w_prev = w_tgt(D-1)` detached; sequential self-consistent thereafter).
This makes the executive learn turnover discipline rather than having it bolted on.

### 3.3 Counterfactual per-expert utility — the trust ledger's raw material

For every expert m and date D, compute `u_m(D)` = the utility (same `U_h` formula, same cost
table) of the **solo-expert book**: the portfolio the blend rule would produce with
`tau = onehot(m)`, f from a fixed reference sizing (f=0.7, same vol cap). This is computed
once per training pass and nightly in production; it is pure bookkeeping, no learning.

`u_m` serves twice:
1. **As input** — the rolling stats `r[M, K]` in §1.1 (this is what makes trust observable
   and honest: most of its variance comes from measured counterfactual P&L, not from weights).
2. **As auxiliary supervision** — a decoupled trust-alignment term, weight `beta_tr` small
   (≈0.1 of main loss):

```
Loss_trust = mean_D [ KL( softmax(u_*(D..D+h realized) / T_u)  ||  tau(D) ) ]
```

   i.e. the trust head is nudged toward the experts that *turned out* to deserve trust. This
   is auxiliary only — the main loss already rewards good trust through realized utility —
   but it densifies the gradient signal for the trust head specifically (M targets per day
   instead of 1), which matters at this sample size.

### 3.4 Division of labor with evolution (role 8 interface)

Gradients train theta (the gate/sizing weights) against fixed scalarization constants.
**Evolution owns the constants gradient cannot honestly set:** `lam_dn` (risk aversion),
`f_max`/`sigma_cap`, `eps` (trust entropy floor), `h` if role 8 wants it on the genome, and
the no-trade band. These are few (≤8 genes), bounded, and evaluated by full-window replay
fitness — exactly the slow/outer vs fast/inner loop split that keeps the EA meaningful
rather than ceremonial. The executive exposes them in one config block for the EA to bite on.

---

## 4. "How much to listen to which model today" — the observable mechanism

The trust answer must be inspectable, time-varying, and supervised. Three layers:

1. **Measured (no learning):** the trust ledger — a per-day, per-expert table
   (`trust_ledger.parquet`): `date, expert, u_m_realized, ewma21, ewma63, hit_rate,
   cf_drawdown, tau_assigned`. The counterfactual columns are the *fact record* of which
   expert has been earning trust; anyone can read it without trusting the network.
2. **Learned mapping (tiny):** `tau = softmax over phi(r_m, c_m, agree_m, z)` — the network
   only learns *how to convert* the fact record + context into today's listening weights
   (e.g. "in high-vol regimes, fade the momentum-shaped expert faster"). Supervised by the
   main utility loss + the decoupled counterfactual-KL term (§3.3).
3. **Floor (structural):** entropy floor `eps = 0.05` guarantees no expert is ever fully
   muted — every expert keeps generating counterfactual track record, so trust can recover
   when an expert's regime returns. (A muted expert with no live attribution would otherwise
   be unrecoverable and would also sabotage the leave-one-out attribution in Phase D.)

Time variation is then *guaranteed observable*: plot `tau` by date. If `tau` is flat
(std(tau_m) over time < 0.02 for all m on the holdout), the trust head is doing nothing and
§7's falsifier fires.

---

## 5. No-look-ahead and sample-budget discipline

### 5.1 Look-ahead audit of every input and target

- Inputs at D: `daily/D/*` contains data through D-1 close (brief §3.1) — clean by
  construction in production; in pretraining, the feature builder must reproduce the same
  convention from the deep-history parquet (features at D use closes ≤ D-1).
- Fill at D's open: in pretraining, D's open comes from refetched OHLCV (Stooq); in replay,
  from `daily/D+1/prices.parquet` — same price, two storages. One alignment test is
  mandatory: for 20 random (symbol, date) pairs in the overlap window, assert pretrain fill
  price == replay fill price (post VUG-split handling; never double-apply the 6:1).
- Rolling trust stats at D may include only counterfactual windows **fully realized by D-1
  close** → latest usable decision date in the EWMA is `D - h - 1`. The ledger builder
  enforces this lag; it is the easiest leak to make and the unit test for it is required.
- Targets for D consume prices through D+h-1 close → **purged walk-forward splits**: any
  train/validation boundary discards h+1 days on each side (embargo), so no target window
  crosses a split.
- Holdout (2026-03-11 →) is touched exactly once per pre-registered configuration, per
  EVIDENCE_PROTOCOL; the executive contributes **one** configuration to the bake-off, chosen
  on pre-holdout validation only. Looks are counted in the run journal.

### 5.2 Sample budget statement (Training Realist's audit line)

- Decisions are daily and book-level: the executive's sample is **~2,959 days**, not 189k
  symbol-days (symbols share one tau and one f). With h=5 overlapping windows, effectively
  ~600 quasi-independent utility observations spanning ~6–8 distinct market regimes.
- Parameters ≈ 450–600 (hard ceiling 3k). Ratio ≈ 1:4 to 1:1 against effective samples —
  workable only because of the structural priors (shared phi, deterministic blend,
  measurement-driven trust inputs, bounded outputs).
- Anti-memorization battery (all mandatory, all cheap): weight decay; dropout 0.1 on z;
  input noise on `r` (counterfactual stats are themselves noisy — train like it); 5-seed
  ensemble with averaged outputs (seeds recorded); early stop on purged validation utility;
  and the **simplification ladder**: if validation utility of the MLP ≤ that of a linear
  gate (`tau = softmax(A·[r,c,g] + B·z)`, ~100 params), ship the linear gate and say so in
  the dossier. The organ's claim is "learned executive," not "deep executive."

### 5.3 Pretrain / fine-tune split

- **Pretrain (public history, ~2014/15 → 2026-02):** requires per-expert opinions over
  history. **Hard requirement on every upstream organ: historical opinions must be generated
  walk-forward** (expert trained only on data before the dates it opines on). If experts
  hand the executive in-sample-fitted history, the executive will learn to over-trust the
  most overfit expert — the single most likely silent failure of this whole design. The
  executive's spec therefore demands a `walk_forward: true` attestation field in each
  expert's historical-opinion manifest, and refuses (build error) without it.
- **Fine-tune (own record, ~210 days):** freeze `phi` and the blend; fine-tune only the
  trust logit layer (v, b, b_m) and the sizing head psi at LR×0.1, 50 epochs max, on the
  live-record window *before* holdout start (~130 days). Purpose: calibration to the real
  morning-fill/cost regime, not new structure — 130 days cannot support structure and we do
  not ask it to. The last ~62 days (holdout) are never trained on.

---

## 6. Auditability — the per-decision attribution record

Every night the executive writes `meta_decision.json` (schema below; prototype writes to the
run dir, production-shaped path `daily/<D>/meta_decision.json`):

```json
{
  "date": "2026-06-10", "model_version": "meta_v3_seed-ens5", "code_sha": "…", "input_hash": "…",
  "experts": [
    {"name": "xsec_transformer", "confidence": 0.61,
     "rolling": {"ewma21": 0.0012, "ewma63": 0.0007, "hit": 0.54, "cf_dd": -0.041},
     "trust": 0.38, "trust_logit": 1.21,
     "logit_attrib": {"ewma21": 0.55, "ewma63": 0.18, "confidence": 0.10, "agree": 0.21, "context": 0.17, "bias": -0.02},
     "solo_book_corr": 0.74}
    /* … one entry per expert … */
  ],
  "trust_entropy": 0.89, "deployment_fraction": 0.63,
  "sizing_attrib": {"vol_level": -0.31, "disagreement": -0.22, "sentiment_agg": 0.08, "edge_magnitude": 0.41, "bias": 0.05},
  "book": {"gross": 0.63, "n_positions": 14,
           "top_contributions": [{"symbol": "XLE", "w": 0.06,
             "expert_share": {"xsec_transformer": 0.031, "gdelt_event": 0.018, "llm_sent": 0.011}}]},
  "counterfactuals": {"equal_weight_baseline_book_gross": 0.70, "divergence_from_baseline_L1": 0.21},
  "expected": {"U_h_hat": 0.0009}, "realized": null
}
```

- `logit_attrib` / `sizing_attrib`: exact first-order decomposition — with one hidden layer
  this is computed by integrated gradients (16-step, deterministic) per input group;
  groups, not raw features, keep it readable. For the linear-gate fallback it is exact.
- `expert_share` per holding: `tau[m]*mu[m,s]*prec_share` renormalized — "who put this
  position on."
- `realized` is back-filled h days later by the nightly job (expected-vs-realized utility is
  the executive's own running calibration curve, plotted in the dossier).
- The trust ledger (§4) plus these records answer, for any date: *what did the gate see, who
  did it listen to, why, how big did it bet, and was it right* — without re-running anything.

---

## 7. Failure modes and falsifiers (what kills this organ)

**Leave-one-out attribution (Phase D, pre-registered):** replace the executive with the
**naive baseline**: `tau = 1/M` (equal trust), `f` = fixed 0.7 with the same vol-cap overlay,
same blend rule, same no-trade band — everything else in the brain identical. Run both
through the identical harness on the holdout.

- **The organ counts iff** the full brain beats the baseline-brain on holdout with positive
  mean paired daily return difference and the paired t respecting EVIDENCE_PROTOCOL form,
  AND ΔSharpe > 0. Report effect size with its (wide, ~62-day) error bars honestly.
- **Kill criteria (any one):**
  1. Paired daily mean Δ ≤ 0 on holdout (the learned executive adds nothing over equal
     weight + fixed sizing) → report meta-evaluator attribution ≈ 0; the assignment item
     fails honestly rather than cosmetically.
  2. **Static trust:** std over holdout days of every tau_m < 0.02 → the "today" in "which
     model today" is dead; the organ is a relabeled fixed blend.
  3. **Trust collapse:** any tau_m pinned at the entropy floor for >80% of holdout days
     while that expert's counterfactual EWMA is positive → the gate is ignoring its own
     measurement layer; inspect `logit_attrib`, retrain or fall back to linear gate.
  4. **Calibration failure:** corr(expected U_h_hat, realized U_h) ≤ 0 over the
     fine-tune+holdout window → the sizing head's confidence is noise; revert f to fixed.
  5. **Challenger parity:** the GBM challenger (§2.2) or the linear gate matches the MLP on
     pre-holdout validation → ship the simpler form (this is a *graceful degradation*, not a
     kill — but the dossier must say which rung of the ladder shipped and why).

**Named residual risks:** (a) upstream walk-forward attestation is the soft spot — if any
expert's history is subtly in-sample, trust learning is poisoned; mitigation is the manifest
gate + a spot-check (retrain one expert's last fold, compare opinions). (b) h=5 overlap
inflates effective sample if anyone computes naive standard errors — all reported stats use
non-overlapping or HAC-adjusted forms. (c) ~62-day holdout cannot certify small effects;
the dossier states the minimum detectable effect up front.

---

## 8. Cost lines (Feasibility Auditor worksheet entries)

- **Training (local Mac, monthly):** counterfactual ledger build over full history ≈ minutes
  (vectorized pandas); gate training: ~600 params × 2.9k samples × 200 epochs × 5 seeds —
  well under 5 minutes CPU on torch 2.1.2. Fits the existing monthly launchd window with
  >95% headroom. **$0.**
- **Inference (nightly Lambda):** one forward pass (<1 ms) + ledger update + JSON write,
  inside the existing 3008 MB / ~110 s night run; adds <1 s. Incremental Lambda GB-s ≈ $0.00.
- **Storage:** model artifact <1 MB (state_dict + scaler + manifest) under `models/` with
  `latest.json` pointer per existing convention; `meta_decision.json` ~4 KB/day; trust
  ledger parquet <1 MB/yr. S3 incremental ≈ **$0.01/mo**.
- **No Bedrock usage by this organ** (it consumes the LLM organ's structured output only).
- **Total executive line: ≈ $0.01/month.** The executive spends its budget in design, not
  in compute — by construction.

---

## 9. Build order (engineer's checklist)

1. Blend rule + intent diff + no-trade band (pure functions, unit-tested against the cost
   table and the VUG split case).
2. Counterfactual ledger builder (per-expert solo-book U_h, lag-enforced rolling stats) +
   the look-ahead unit tests of §5.1.
3. Differentiable replay loss (teacher-forced then sequential) + purged walk-forward
   splitter.
4. Gate/sizing heads + 5-seed training loop + simplification ladder (linear-gate twin
   trained in the same run, always).
5. Fine-tune pass on own record (pre-holdout window only).
6. `meta_decision.json` writer + integrated-gradients attribution + expected-vs-realized
   backfill job.
7. LOO baseline runner (equal trust, fixed f) wired to the identical harness — built
   *before* the holdout is ever read, per pre-registration.
