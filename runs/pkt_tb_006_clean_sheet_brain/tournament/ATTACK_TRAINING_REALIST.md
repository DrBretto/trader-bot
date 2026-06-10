# ATTACK — TRAINING REALIST (panel role 11)

**Packet:** PKT-TB-006-CLEAN-SHEET-TRADERS-BRAIN — Phase B tournament
**Author:** Training Realist, 2026-06-10
**Scope:** sample-budget audit of every learnable component in DESIGN_ALPHA / DESIGN_BETA /
DESIGN_GAMMA; transformer-specific gradient arithmetic; target learnability; the
pretrain/fine-tune split; walk-forward/embargo adequacy; binding forced simplifications.
This is arithmetic, not opinion. Where a design's own number disagrees with mine, both
numbers appear and the derivation is shown.

---

## 0. The three numbers everything hangs on

Every effective-sample claim below derives from these. They are computed once, here, and
reused; any design wanting a different number must show different arithmetic.

**N1 — Cross-sectional effective breadth.** 64 ETFs are not 64 draws. With average pairwise
daily-return correlation ρ̄ ≈ 0.45–0.55 (broad equity sleeve internally 0.85–0.97 — SPY/VOO/
IVV/VTI are quadruplicates; sectors 0.6–0.8 to SPY; bonds in 2022–25 often positively
correlated to equities), the equal-correlation effective count is
`N_eff = S / (1 + (S−1)·ρ̄)` ≈ 64 / (1 + 63·0.5) ≈ **2–4 effective names for LEVEL targets**.
After per-day cross-sectional de-meaning (market removed), residual block correlations
(~0.15–0.3 within sector/duration/commodity clusters) give
**≈ 10–15 effective names for MARKET-NEUTRAL targets** (block count: ~1 broad-equity
residual, ~4–5 country/intl, ~6–8 sector/industry, ~1–2 factor tilts, ~2–3 duration/credit,
~3 commodity, ~1.5 fx, 1 vol). I use **12** as the working figure. At 13 sleeves (Gamma)
the same logic gives **≈ 8** effective sleeves.

**N2 — Independent time samples.** 2,959 trading days (2014-08-29 → 2026-06-05). h=5
overlapping targets share 4/5 of their return → **≈ 580–590 quasi-independent 5-day
windows** full-history; **≈ 558** on the GDELT-rich panel (2015-02-18 →, 2,790 d);
**≈ 104** on the LLM Tier-2 panel (~520 d); **≈ 25–29** on the pre-holdout live record.
h=21 targets: ≈ 140 windows full-history.

**N3 — Regime count.** ~6–8 distinct regimes in 11.7y (2015-16 chop, 2018 vol events, 2020
crash/rebound, 2021 bull, 2022 bear, 2023–25 bull, 2025-26 current). Any component whose
parameters encode *regime-conditional disposition* (drawdown brakes, trust half-lives,
gross targets, dd genes) has an effective sample of **~6–8**, full stop.

**Master conversion** for pooled symbol-day rows with market-neutral targets:
`effective ≈ rows × (12/64) ÷ 5` → 189k symbol-days → **≈ 7,100**; 150k rows/fold →
**≈ 5,600**; 180k → **≈ 6,800**. For LEVEL targets divide by a further ~4 (N1 level case).
Designs claiming 20–35k effective (Alpha §10) are using cluster counts of ~8/day but not
discounting the 5-day overlap consistently — **their numbers are 2–3× optimistic**; the
honest band for the densest target in the program is **~7–12k**.

**N4 — Holdout power (context for every verdict).** ~62 holdout days; paired daily diff sd
between two similar long-only books ≈ 0.15–0.25%/day → minimum detectable daily mean at
t=2 ≈ 2·0.2%/√62 ≈ **5 bp/day ≈ 13%/yr**. Almost no organ-level true effect is that big.
Holdout attribution will be sign-and-magnitude reporting, not certification — all three
designs concede this; I confirm the arithmetic and bind them to printing the MDE.

---

## 1. Component-by-component audit tables

Verdict scale: **SUPPORTED** (params ≲ ⅓ of effective sample, or convex/strongly shrunk),
**MARGINAL** (ratio 1:3 to ~1.5:1, defensible only with the full stated battery),
**UNSUPPORTED** (params exceed honest effective sample with no dense-signal escape).
Defense grade: **real / partial / hand-waving**.

### 1.1 DESIGN_ALPHA

| Component | Claimed params | Target | Design's eff. sample | Honest eff. sample | Verdict | Defense grade |
|---|---|---|---|---|---|---|
| CAST (E1) 2-layer xsec transformer | 45k (fallback 22k) | 5d xsec rank z | 20–35k | **7–12k** (189k×12/64÷5) | **UNSUPPORTED at 45k; MARGINAL at 22k** | **real** (no symbol ID, shared nets, rank loss, ridge ladder, 5 seeds) but wd 1e-4 is weak and the 20–35k claim double-counts breadth vs overlap |
| XGB-Cond (E2) | ≤9.3k leaf values | P(5d excess>0) per symbol-day | ~25k | **~5.6k/fold** | **MARGINAL→UNSUPPORTED at 300 iters** — 9.3k local leaves vs 5.6k effective rows is ≳1.5:1 | **partial** (depth/L2/monotone real; iteration cap too generous; early-stop on purged folds must be binding, not advisory) |
| EventHead (E3) | 675 coefs (27×25 elastic-net) | bucket 5d abnormal return | 8–15k | ~558 windows × ~10 eff. buckets ≈ **5.5k**; per-bucket 25 coefs vs ~550 | **SUPPORTED** (elastic net, abstains) | **real** |
| RiskNet (E4) | 90 (MLP 370) | 5d realized vol + beta | "ample" | vol target SNR is high (vol autocorr ≈ 0.7+); honest | **SUPPORTED** | **real** |
| MetaTrigger (E5) | ≤1.5k leaves | binary clears-2×spread | 12–20k | ~80k events ×12/64÷5 ≈ **3k** | **MARGINAL** | **partial** — the label is mostly \|move\| vs spread, i.e. vol; risk of duplicating RiskNet (see §3.5); trains on equal-trust blend, serves τ-tilted (stated, unquantified) |
| Executive | 450–600 (cap 3k) | h=5 book utility + KL aux | ~600 | **~600 windows / 6–8 regimes** | **MARGINAL** (≈1:1) | **real** — the full §5.2 battery + linear ladder + KL densification (M=3 targets/day) is the best executive defense on the table |
| EA genome | 32 genes, K≤400 | 6 fold-scores | ~6 | **6–8 regime obs** | **MARGINAL by design** | **real** — adoption gate >1 cross-fold sd + B1 control + max-of-K arithmetic done honestly in the proposal |

System total ≈ 57k trained params vs ~7–12k effective on the densest target. The design's
shape (dense supervised perception, tiny utility-trained judgment) is right; the flagship's
size is not.

### 1.2 DESIGN_BETA

| Component | Claimed params | Target | Design's eff. sample | Honest eff. sample | Verdict | Defense grade |
|---|---|---|---|---|---|---|
| P1 transformer policy | 26k | −U_h(book), h=5, end-to-end | "~600 indep windows; 189k credit terms" | **~580 book-windows; gradient breadth ≤ ~12/day → ≤7k constraint-equivalents** (§2.2) | **UNSUPPORTED as written** | **partial** — symbol dropout / no-ID / twin are the best defenses in any design, but "credit densification" is the single most hand-waving sentence in the three documents (§2.2) |
| P2 ×2 linear policies | ~30 each | U_h h=21 / h=5 | same | same | **SUPPORTED** | **real** (convex floor — excellent) |
| P3 GBM action-value | ~2.4k leaves (300×d3) | per-symbol 5d **absolute** utility | ~600 cross-sections | LEVEL target → 189k×(3/64)÷5 ≈ **1.8k** | **MARGINAL, and the target is wrong** (§3.3) — absolute utility is ~80% market factor → a 580-sample market-timer in disguise | **partial** |
| P4 event MLP | ~1.1k | U_h h=5 | ~540 / ~100 (LLM) | GDELT path ~540; **LLM loadings ~100 windows** | **MARGINAL**; LLM-input slice UNSUPPORTED unless dims capped (§6) | **partial** |
| Executive (book blend) | 450–600 | h=5 blended-book utility | ~600 | ~600; convex blend of unit-gross books is *simpler* than Alpha's μ/σ machinery and makes the ledger exact | **MARGINAL→SUPPORTED** | **real** |
| EA genome + slot selectors | 36 genes | 6 fold-scores | ~6 | 6–8 | **MARGINAL by design** | **real** (library trick keeps eval cached; adoption gate honest) |

### 1.3 DESIGN_GAMMA

| Component | Claimed params | Target | Design's eff. sample | Honest eff. sample | Verdict | Defense grade |
|---|---|---|---|---|---|---|
| ETT event-theme transformer | 28k | per-sleeve 5d excess (13 sleeves) | "~600 indep windows" | 558 windows × ~8 eff. sleeves ≈ **4.5k**, and event-conditioned (quiet nights ≈ no signal) → practical **~2–3k** | **UNSUPPORTED at 28k** — worst params:signal ratio in the program | **partial** — dropout/seeds/MLP-rung real; but 120×16 theme embedding table = memorization slots for rare codes (no min-frequency rule); LLM token annotations exist only on ~610 backfill days, undefended |
| ISRE GRU regime encoder | 12–14k | 5d per-sleeve vol + market corr | "dense & high-SNR, defensible" | vol IS the high-SNR target — honest; but ~2,800 sequences, overlap → ~560 windows; **HAR-RV gets most of this with ~3 params/symbol** | **MARGINAL** | **partial** — *no simpler twin is pre-registered for ISRE*; the only learnable-component in any design without a fallback rung |
| GBM ranker | ≤6.4k leaves (400×d4) | lambdarank 5d excess | ~600 cross-sections | **~7k** (same master conversion) | **MARGINAL** — ~1:1 leaves:effective | **partial** (min_child 200 real; 400 trees too generous) |
| LLM organ | 0 trained | — | — | — | n/a | **real** (zero-shot, schema-clamped) |
| Member-5 contrarian | ~30 | COT-conditioned tilt | 1,040 weekly obs | ~1,040, but **~13 holdout obs → no holdout verdict possible** | **SUPPORTED (params), unevaluable (holdout)** | **real** (evolution-gated, declared) |
| Executive (adopted) | 450–600 | h=5 book utility | ~600 | ~600 | **MARGINAL** | **real** |
| EA genome (funnel gates) | ≤40 genes | 6 fold-scores | ~6 | 6–8 | **MARGINAL by design** | **real** |

Gamma-specific aggravator: the theme→sector dictionary (~60–100 hand-curated codes) is an
un-counted degree-of-freedom reservoir. Each curation choice is a parameter fit by human
eye to the same history. The freeze-before-holdout rule (Gamma §13.3) is necessary but the
dictionary must ALSO be frozen before any *validation-fold* reads used for model selection,
not just before the holdout — otherwise it is tuned on the same folds that pick the models.

---

## 2. The transformers — gradient-information arithmetic

All three designs stake assignment item 1 partly on a transformer. The question per
transformer: what does one day of its loss actually deliver in independent gradient
information, and can it beat its pre-registered simpler twin on that diet?

### 2.1 Alpha CAST (45k) — densest signal of the three

**What one day delivers.** Huber + soft-Spearman over the 64-token cross-section. 64
residuals at ~12 effective → **~12 independent scalar constraints/day**, plus the rank
surrogate adds no new information beyond reweighting them. Over 2,900 days ÷ 5 overlap:
**~7,000 independent constraint-equivalents** for 45k weights → 6.4 params per constraint.

**Signal ceiling.** If true daily IC (Spearman) is 0.03 — optimistic for liquid-ETF 5-day
relative value — sd of daily IC at breadth 12 ≈ 1/√11 ≈ 0.30, so the full-history t-stat
of the *target itself* is ≈ 0.03/(0.30/√580) ≈ 2.4. The signal is barely detectable over
the entire pretraining history; a 45k-param net is being asked to carve fine structure out
of a phenomenon whose existence is a 2-sigma event.

**Verdict: cannot honestly beat the ridge twin at 45k.** The most learnable honest version
(this becomes binding in §6): **CAST-Small ≈ 15–22k** — 1 encoder layer (not 2), d_model
32, FFN 64, heads 4; weight decay 1e-3 (not 1e-4); dropout 0.2–0.3; **add a dense
auxiliary head** (next-day per-symbol return z, weight 0.3) to feed the GRU encoder a
64-targets/day unsmoothed signal; early stop on purged-fold Spearman; 5-seed averaging
(keep); per-day Gaussian-rank target (see §3.1). At 15–22k vs 7–12k effective with the
aux head, CAST has a genuine fighting chance of positive attribution vs ridge — the
fallback ladder stays and CAST-45k is demoted to a logged challenger, never the default.

### 2.2 Beta P1 (26k) — the weakest gradient supply

**What one day delivers.** The loss is ONE scalar per day: U_h of the book. The design's
"~189k symbol-day credit terms" defense: the gradient decomposes into Σ_s (∂w_s/∂θ)·r_s,
so 64 per-symbol terms exist. **The arithmetic against it:** 64 gradient *terms* are not 64
*constraints*. The information content of a gradient step is bounded by the loss surface,
and one scalar loss per day pins one direction in parameter space per day; the
cross-sectional structure of r_s does color that direction (parameters routing weight
toward symbols that subsequently rose get pushed), which is at best equivalent to a
weighted regression of returns on features through the policy — i.e. exactly CAST's
information diet, **≤ ~12 effective constraints/day, ≤ ~7k total**, and in practice less:
the softplus-renormalization couples all symbols (a gradient on XLE is partially a gradient
on everything), and the downside-penalty and cost terms add variance without adding
constraints. 26k params end-to-end on this diet is **UNSUPPORTED**. The "trained by the
panel" sentence is the program's one load-bearing hand-wave.

**Fighting-chance settings (binding in §6):** the encoder must NOT be trained from scratch
on utility. Either (a) **auxiliary dense pretraining**: pretrain GRU+token machinery with a
per-symbol h=5 return-z head (the dense supervised signal — explicitly internal machinery,
allowed under Beta's own thesis since the showcased output remains the action), then train
attention + score head + gross head on utility with encoder LR ×0.1 or frozen; or
(b) shrink to 1 encoder layer (≈ 17k → 8.5k; total ≈ 17k) AND keep the full battery
(48-of-64 symbol dropout, feature noise, wd 1e-3, 5 seeds, purged early stop). Route (a)
preferred — it is the only route by which 26k params receive enough gradient. The
no-attention twin stays as the attribution referee either way.

### 2.3 Gamma ETT (28k) — best story, thinnest signal

**What one day delivers.** 13 sleeve targets at ~8 effective sleeves → **~8 constraints/
day**, ×558 GDELT-panel windows ≈ **4.5k** — and the signal is event-conditioned: on quiet
nights mu ought to be ≈0, so informative days are maybe half the panel → practical **2–3k**
constraint-equivalents for 28k params. Additionally: (i) the 120-code theme embedding
table at 16-dim gives rare codes private parameters — a rare theme that co-occurred once
with a big sleeve move is a memorized lookup; (ii) the LLM annotation features on tokens
(llm_sent×conf, direction/magnitude tuples from §6.3) exist only over the ~610-day
backfill (~120 windows) — the transformer's "richest token features" train on the
program's thinnest slice.

**Verdict: UNSUPPORTED at 28k; the bag-of-events MLP (~3k) is the right size for this
signal.** Fighting-chance minimal ETT (binding in §6): theme vocab capped at codes with
≥200 training-panel occurrences (≈ 30–50 codes; the rest pooled into per-bucket OTHER),
embeddings 8-dim, d_model 24, 1 layer, 2 heads, FFN 48 → **≈ 8–10k params**; annotation
features behind an `ann_available` mask with dims ≤4; an auxiliary dense target
(next-day per-sleeve abnormal volume or realized vol — event→volume transmission is far
higher-SNR than event→return) to feed the embeddings; dropout/seeds as designed. The MLP
twin is trained as **co-primary**, and ETT ships only if it beats the MLP on purged
validation — Gamma already commits to this; I make it binding with the size cap.

### 2.4 Cross-transformer verdict

Ranked by honest probability of genuine positive attribution: **CAST-Small (Alpha) >
ETT-Small (Gamma) > P1 (Beta, even with aux pretraining)** — because CAST's target supplies
the most independent constraints per day, ETT's supplies a differentiated but sparse
signal, and P1's supplies one scalar. The assignment needs a transformer to carry weight;
the synthesis should put its transformer chips on the CAST slot and keep the event
transformer at MLP-challenger scale unless it earns more.

---

## 3. Targets audit — is the label itself learnable?

**3.1 h=5 cross-sectional rank z (Alpha CAST, Gamma GBM).** 5-day idiosyncratic excess
return on liquid ETFs: target sd ~1.5–2.5%; plausible predictable fraction (IC 0.02–0.05)
→ signal is ~0.1–0.25% of target variance. Learnable ONLY through rank-flavored losses +
heavy averaging; level regression would be noise-chasing. Required changes: per-day
**Gaussian-rank transform** of the target (replaces z-score-and-clip; tames the fat tails
that dominate Huber gradients); report IC with HAC errors or non-overlapping windows only.
Horizon: h=5 is right (h=1 is microstructure noise at these costs; h=21 leaves ~140
windows). Optional label smoothing: average the h=5 target over entry days D−1,D,D+1
(reduces open-print noise ~30% at no leakage cost since all inputs are ≤D−1 close — the
D−1 entry variant must shift the feature window back one day accordingly; if that
complexity is unwanted, skip it — Gaussian-rank is the binding change).

**3.2 h=5 book utility (all executives; Beta members).** Book daily vol at 0.6–0.7 gross
diversified ≈ 0.6–0.9%; a genuinely good allocator adds maybe 2–5 bp/week of utility →
per-window SNR ≈ 0.02–0.03 → full-history t ≈ 0.5–0.8. **The utility target cannot
support rich structure learning, period.** It supports: (i) tiny gates with structural
priors (the 600-param executive — and even there expect ties with equal-trust; the kill
criteria handle this honestly), (ii) the EA's ≤40 dispositions under a 1-sd adoption gate,
(iii) NOTHING with four-digit parameter counts (hence §2.2). The one high-SNR component
inside U_h is the vol/sizing channel (vol predictability is real), which is why
sizing-head learning and vol-cap overlays will produce most of whatever executive
attribution appears.

**3.3 Beta P3's absolute utility target — must change.** `y = mean log(1+r_s) − cost` is
~80% market factor; its effective sample is the LEVEL-target count (~1.8k, ~6 regimes),
and the GBM will spend its leaves learning 2020/2022 macro story. **Forced: cross-
sectionally de-mean y per day** (excess action-value). Beta's own thesis (the book is
cross-sectional) endorses this; the design text as written contradicts it.

**3.4 Counterfactual trust targets (all designs, KL aux).** Solo-member long-only books on
the same tape will correlate 0.8–0.95; the trust-discriminating signal is the *difference*
of correlated series — thin but honestly measured, and the KL term is auxiliary at 0.1
weight, which is the right size. Binding addition: the dossier must report the mean
pairwise correlation of member solo-book daily returns; **if it exceeds ~0.9, trust
learning is arithmetic cosmetics** and the expected (and acceptable) result is the
equal-trust tie. This single statistic is the cheapest predictor of whether assignment
item 5 can show attribution at all.

**3.5 MetaTrigger label (Alpha E5).** "Clears 2× half-spread over 5d" at 1–8 bp spreads
vs ~2% 5-day moves: the spread threshold is ~3–16 bp against a ~200 bp-sd move, so the
label is ≈ sign(move) — i.e. ~50/50 and its predictable part is mostly \|move\|-scale =
vol, which RiskNet already supplies. Risk: E5 re-learns E4. Binding: add a vol-only
logistic rung to the MetaTrigger ladder (features: sigma_hat, cost_bps, \|mu_blend\|); if
the GBM cannot beat it on purged AUC, ship the logistic.

**3.6 ISRE targets (Gamma).** 5d realized vol per sleeve + market correlation: genuinely
high-SNR (the one easy problem). But the learnability question is the *increment over
HAR*: HAR-RV (3 coefficients) captures most of daily-frequency vol forecastability;
EWMA/DCC-lite captures most of correlation dynamics. Binding: pre-register HAR-RV and
EWMA-corr as ISRE's simpler twin (currently absent — the only learnable component in any
design without a rung); ship the GRU only on a purged QLIKE win.

---

## 4. Fine-tuning split audit

**The honest split, stated once.** Live record: 2025-08-04 → present, ~210 trading days of
decision-grade artifacts. Holdout starts **2026-03-11** INSIDE it (~62 d and growing).
Fine-tune data must satisfy: decision date D's h=5 label consumes opens through D+5, so
the last admissible fine-tune decision date is **holdout_start − (h+1) ≈ 2026-03-03**
(6 trading days clear), and ledger/EWMA inputs must be lag-enforced (D−h−1) so no rolling
stat ingests post-2026-03-10 outcomes. That leaves **≈ 125–145 usable pre-holdout live
days ≈ 25–29 independent h=5 windows.** (The "~130 days" all three designs quote is
approximately right but none of the three subtracts the h+1 label embargo explicitly —
binding fix: the fine-tune cutoff is `holdout_start − h − 1` by formula, not "~130".)

**Is 25–29 windows enough for what each design fine-tunes?**

- **All three** fine-tune only the executive (trust logits + sizing head ψ, LR×0.1, ≤50
  epochs; φ and blend frozen; members never touched). Direction correct. But ψ alone is
  ~250 params against 25–29 windows — "calibration not structure" is the right *claim*
  and the wrong *parameter set*. **Binding for all three: restrict the fine-tune to
  ≤~20 effective params** — per-expert bias b_m (M), softmax temperature T (1), ψ output
  gain+bias (2), optionally the f_max-adjacent sigmoid bias — via freezing ψ's hidden
  layer. With LR×0.1 + early stop the proposals' version probably lands near this anyway;
  make it structural, not incidental.
- **Marginal value note (honesty, not a change):** the live record is itself simulated
  fills through the same cost model the pretraining uses (no broker; Alpaca removed
  2026-06-08), so "calibrate to the real morning-fill/cost regime" buys distribution-shift
  adaptation of features/opinions only, not a different fill physics. Expect the fine-tune
  delta to be ≈0; it is cheap and low-risk, keep it, but the dossier must not narrate it
  as adapting to "real" market frictions.

**Violations found: none in the split itself.** Alpha §9, Beta §11.3/stage-4, Gamma §9
phase-3 all bound fine-tuning to pre-holdout live days and never train members on it.
Two flags short of violation: (1) the missing h+1 embargo arithmetic above (all three);
(2) Beta §15.4's "cost-regime baked in" discussion and Gamma's calibration language both
imply the live record carries fill information that pretraining lacks — it doesn't (same
simulator); wording must change so the dossier doesn't overclaim what fine-tuning did.

---

## 5. Walk-forward / embargo verdicts

**Fold geometry (shared F1–F6, ~250 d each, 2020-02 → 2026-02, expanding head-train from
2014-08, 21-trading-day embargo).** Verdict: **adequate**. Purge need is h+1 = 6 d for
h=5 and 22 d for h=21; the 21-d embargo covers h=5 with 3.5× margin and h=21 with zero
margin — **binding micro-fix: embargo = 22 trading days** (or h_max+1 computed) so the
h=21 targets (Beta P2-slow, record-formation scores at h=21) cannot touch a fold by one
day. Single-sided embargo is correct for expanding-window-before-fold layouts; the meta
proposal's both-sides rule covers the executive's internal splits. Trailing-90d
standardization uses only days < d — clean.

**Regime coverage.** Each fold ≈ one regime chunk; cross-fold sd is a real robustness
read; 2014–2019 regimes inform perception only (declared). Acceptable — but note the EA's
dispositions are then fit to 6 fold-scores of which 2020 and 2022 dominate the variance;
the cost-scenario min and mean−0.5·std partially hedge. Nothing to force beyond what the
proposal already does; the adoption gate is the load-bearing control and it is honest.

**OOF stacking — one real circularity found.** Members are trained per-fold and emit OOF
opinions; the executive trains on the pooled OOF panel (F1–F6); the EA then scores
genomes — including executive-shaping genes (λ_dn, abstain, conviction_temp) and trust
priors — on the SAME folds the executive trained on. The champion genome is therefore
selected partially in-sample with respect to the executive's weights. This is mild (the
executive is ~600 params and heavily regularized) but it is the one leak-shaped hole in an
otherwise clean stack. **Binding fix (cheap):** train six leave-one-fold-out executives
(600 params × 6 ≈ minutes) and have the EA's fitness walk use, on fold f, the executive
trained without fold f. The deployed executive trains on all folds as designed. This
makes genome selection fully out-of-fold. Applies to all three designs identically (all
adopted the same machinery).

**Other checks, clean:** GDELT `visible_from` joins (all three); COT publication-date
keying; FRED weekly publication lags; LLM artifacts stamped ≥10.5 h pre-open; ledger
D−h−1 lag with required unit test; the 20-pair pretrain-vs-replay fill-price alignment
test; VUG 6:1 single application; holdout touched once per pre-registered configuration
with looks counted. The walk-forward attestation (`walk_forward: true` manifest, build
error without it) is the single best leakage control in the program — Alpha satisfies it
natively, Beta by owning both sides, Gamma adopts it; keep it mandatory in the synthesis.

**Record-formation weights (Infotropy B) — a sample-budget side effect nobody priced.**
Down-weighting round-trip moves toward eps shrinks the effective sample further
(eps=0.05 in Alpha/Gamma can cut effective N by ~30–50% depending on the retracement
distribution). Beta's eps=0.25 floor is the right instinct. **Binding for all: eps ≥ 0.2
at training time** (the EA may explore Alpha's [0.05, 0.5] gene range, but the gradient
trainers' default is ≥0.2), and the A/B falsifier (uniform vs weighted) stays mandatory —
if record-weighting helps, it must help *after* paying its own effective-sample cost.

---

## 6. Forced simplifications (binding list)

Each entry: the change, and the arithmetic that forces it.

**ALPHA**
- **A1. CAST registers at ≤22k (d_model 32 / FFN 64), preferably CAST-Small ≈15–22k with
  1 encoder layer + dense next-day auxiliary head; the 45k variant is a logged challenger
  only.** Arithmetic: 45k params vs 7–12k effective (189k×12/64÷5); 6.4 params per
  independent constraint cannot beat ridge honestly.
- **A2. CAST weight decay 1e-4 → 1e-3; target = per-day Gaussian-rank.** Arithmetic: §3.1
  (signal is ≤0.25% of target variance; tails dominate Huber at z-clip ±3).
- **A3. XGB-Cond capped at ≤150 iterations / 15 leaves (~2.3k leaf values) with binding
  early stop.** Arithmetic: 9.3k local leaves vs ~5.6k effective rows/fold ≳ 1.5:1.
- **A4. MetaTrigger gains a vol-only logistic rung; ship the logistic on AUC parity.**
  Arithmetic: §3.5 (the label ≈ sign(move); predictable part ≈ vol = RiskNet's output).
- **A5. Executive fine-tune restricted to ≤~20 params (b_m, T, ψ gain/bias).** Arithmetic:
  25–29 independent live windows.

**BETA**
- **B1. P1 may not train 26k end-to-end on utility. Either auxiliary dense pretraining of
  the encoder (per-symbol h=5 return-z head) with encoder LR×0.1/frozen during utility
  training, or 1 encoder layer (total ≈17k) — preferably both.** Arithmetic: one scalar
  loss/day ≤ 12 constraint-equivalents/day ≤ ~7k total vs 26k params (§2.2).
- **B2. P3's target is cross-sectionally de-meaned (excess action-value).** Arithmetic:
  absolute utility is a LEVEL target → ~1.8k effective / ~6 regimes; the GBM becomes a
  market-timer with 6 observations.
- **B3. P4's LLM-derived input dims capped at ≤8 (pooled bucket sent×conf + masks).**
  Arithmetic: LLM loadings train on ~100 independent windows.
- **B4. Same A5 fine-tune restriction; same §5 LOFO-executive fix for the EA.**

**GAMMA**
- **G1. ETT shrinks to ≈8–10k: theme vocab ≥200-occurrence floor (rare codes pooled),
  8-dim embeddings, d_model 24, 1 layer, FFN 48; LLM annotation features ≤4 dims behind
  an `ann_available` mask; add a dense auxiliary target (next-day sleeve abnormal
  volume/vol). Bag-of-events MLP (~3k) is co-primary; ETT ships only on a purged-
  validation win.** Arithmetic: 28k params vs ~4.5k effective (≤8 sleeve constraints/day
  × 558 windows), halved again by event-conditioning; rare-code embeddings are
  memorization slots; annotations cover ~120 windows.
- **G2. ISRE gets a pre-registered HAR-RV + EWMA-correlation twin; GRU ships only on a
  purged QLIKE win.** Arithmetic: HAR captures most vol forecastability with ~3 coefs;
  14k params must beat 3, on the record.
- **G3. GBM ranker ≤200 trees with binding early stop.** Arithmetic: 6.4k leaves vs ~7k
  effective ≈ 1:1.
- **G4. Theme→sector dictionary frozen and committed BEFORE any validation-fold model
  selection, not merely before the holdout.** Arithmetic: ~60–100 curated codes are
  uncounted fitted parameters; freezing them after fold-driven iteration launders them
  into the models.
- **G5. Same A5 fine-tune restriction; same §5 LOFO-executive fix.**

**ALL THREE (program-wide)**
- **S1. Embargo 21 → 22 trading days (h_max+1).**
- **S2. EA fitness walk uses leave-one-fold-out executives (§5).**
- **S3. Record-formation weight floor eps ≥ 0.2 for gradient trainers; A/B falsifier
  mandatory.**
- **S4. Fine-tune cutoff specified as `holdout_start − h − 1` by formula; live-record
  fine-tuning may not be narrated as adapting to "real" fills (same simulator).**
- **S5. Dossier must print: member solo-book pairwise correlation (the item-5 viability
  statistic, §3.4), the holdout MDE (~5 bp/day, §0-N4), and every effective-sample claim
  recomputed with the §0 master conversion — no more 20–35k-style double counting.**

---

## 7. Cross-design verdict

**Most honest training plan as written: ALPHA.** Its perception layer trains on the only
target in the program that genuinely delivers thousands of independent constraints
(per-symbol supervised, market-neutral, 64 readings/day), its executive/EA adoption is
verbatim-careful, and it pre-registers the ridge ladder against its own flagship. Its one
real sin is quantitative (effective-sample claims 2–3× high; CAST sized to the
optimistic number), fixed by A1–A3 — surgery, not amputation.

**Most surgery needed: GAMMA.** The thesis (information funnel, sizing/attention edge) is
the most differentiated and its §1 concession on direction-level efficiency is the most
intellectually honest paragraph in the three designs — but its flagship trains the most
parameters (28k) on the thinnest honest signal (~2–4.5k), its second net lacks any simple
twin, its central artifact (theme dictionary) is an unpriced parameter reservoir, and its
LLM-annotation centerpiece trains on ~120 windows. G1–G4 are load-bearing, not cosmetic.

**BETA in between.** Most honest about the thin spine (~600 windows, stated plainly; the
P2 convex floor and the dumb-twin guard are the best humility devices in the program;
the book-blend executive is a genuine simplification the synthesis should steal) — but
its central training claim ("credit densification" feeding 26k params from one scalar
loss/day) is the largest single overstatement in the tournament, and P3's absolute target
is a design error.

**What the SYNTHESIS should take (training-plan view):**
- From **Alpha**: the cross-sectional supervised target as the transformer's diet —
  CAST-Small is the program's best shot at "transformer demonstrably carries weight";
  the M=3 trust dimensionality; EventHead's elastic-net scale as the event-direction
  organ; the RiskNet-style cheap vol organ (or Gamma's ISRE-with-HAR-twin if covariance
  forecasting is wanted richer).
- From **Beta**: the executive-over-books convex blend (exact counterfactual ledger); the
  no-attention twin + symbol-subset dropout + feature-noise battery (apply to whichever
  transformer ships); the P2 linear floor and the dumb-twin guard; eps=0.25 instinct on
  record weights; the policy LIBRARY trick for giving the EA loss-shaping genes at zero
  marginal eval cost.
- From **Gamma**: the information-health block in the executive's context (the cheapest
  high-value six features in any design); the record-grade R1∧R2∧R3 gate scoped to event
  features with the R3-only twin; the funnel-gate genes (evolution deciding which streams
  pay); the bag-of-events MLP as the event-interaction challenger — and the ETT only at
  G1 size if the synthesis wants a second transformer, which on this arithmetic it should
  not: one well-fed transformer beats two starving ones.
- Killed by arithmetic regardless of synthesis: any 26k+ component trained end-to-end on
  daily book utility (B1); absolute-level GBM targets (B2); 120-code embedding tables
  without frequency floors (G1); fine-tuning >~20 params on the live record (A5/B4/G5).

— end —
