# LLM Veto Audit — PKT-TB-004 (LLM Veto Auditor return)

**Date:** 2026-06-10
**Charter:** Score every historical Haiku veto/downsize against what subsequently
happened to the vetoed asset: hit rate vs base rate, dollar impact of obeying vs
ignoring. The Bedrock step costs money and latency every night — it must earn its
place like every other layer.

**Evidence class:** E1-style historical scoring of the stored decision record.
This is NOT a replay counterfactual: no ablated replay was run for this audit; the
events are scored off-policy against subsequent prices. Per EVIDENCE_PROTOCOL, any
keep/cut decision still requires the E2 ablation cell (`llm_confidence` layer off)
from the Methodologist's battery. This audit tells the panel what the layer
*actually did* in production; the ablation tells what removing it *would do*.

All numbers below are backed by `veto_audit_data.json` and
`veto_audit_supplement.json` in this directory.

---

## 1. Data manifest

| Item | Value |
|---|---|
| Record window | `daily/2026-01-29/` … `daily/2026-06-10/` (S3 `investment-system-data`) |
| Nights with `llm_risk.json` + `decisions.json` | **88** (of 112 daily folders in window) |
| Price snapshot | `daily/2026-06-10/prices.parquet` (closes 2025-06-11 → 2026-06-09), pinned local copy `cache/prices_20260610.parquet` |
| Holdout boundary | 2026-03-11 (`config/optimizer.committee_20260606.json`) |
| Code rev at audit | `af91c88` |
| Scripts | `scripts/veto_audit.py`, `scripts/veto_audit_supplement.py` (deterministic, no seeds, re-runnable) |
| Consumption points audited | `src/steps/decision_engine.py` — buy filter line 246–252, forced-sell line 437–447 (`LLM_VETO`), sizing line 582 (`llm_adj = 1 − confidence_adjustment`) |
| LLM step | `src/steps/llm_risk_check.py` — `anthropic.claude-3-haiku-20240307-v1:0` via Bedrock, holdings + top-5-by-health, max 16 calls/night |

**Substrate contradiction #1 (record completeness).** The packet brief states
`llm_risk.json` + `decisions.json` exist **daily** 2026-01-29 → 2026-06-10. They do
not: 24 of the 112 daily folders carry no night-run artifacts at all (only
morning-execution/portfolio files). Most are Mondays/holidays, but
**2026-05-11 → 2026-05-20 is a 7-trading-day stretch with no night runs**
(2026-05-11, 12, 13, 14, 15, 18, 19, 20). The scoreable record is 88 nights, not ~90+
— and the May gap means the layer was entirely absent for a week and a half of
production without that absence being part of anyone's mental model.

## 2. Inventory of the record

- **804** (night, symbol) assessments over 88 nights — mean **9.14 calls/night**.
- **3 structural vetoes** ever fired — **0.37%** of assessments. All three on
  **VIXY**, all in April (2026-04-22, 04-25, 04-29). No other symbol was ever vetoed.
- **737 of 804 assessments (91.7%) carry a downsize** (`confidence_adjustment > 0`).
  Value distribution: 0.1 ×509, 0.2 ×204, 0.05 ×15, 0.15 ×5, ≥0.25 ×4. The channel
  is effectively a near-constant ~0.87–0.9× multiplier on everything it touches, not
  selective risk detection.
- **Severity** is 2 ("caution") in **686/804 (85%)** of assessments; severity 3 only
  twice (both VIXY veto days).
- **Boilerplate:** the exact sentence *"Elevated regulatory scrutiny but no immediate
  threat"* is the rationale on **392/804 (48.8%)** of assessments, across unrelated
  assets (GLD, SLV, SCHD, XLU, ITA, VIXY, …). The model is pattern-completing its
  own prompt example (that string is the literal example in `RISK_CHECK_PROMPT`),
  not analyzing the asset.
- **Silent failure mode:** 18/804 assessments (2.2%) are the hardcoded default
  *"LLM check unavailable - defaulting to no risk"* — on error the layer silently
  becomes a no-op. Only 49 of the 67 adj=0 reads are genuine model output.
- **Instability:** for the same symbol on consecutive assessed nights, the
  adjustment value changes **43%** of the time (e.g. VIXY 0.1 → 0.2 across one day
  with no new information) — consistent with sampling noise, not signal.

Counts by month (downsizes): 2026-01: 8, 02: 161, 03: 197, 04: 234, 05: 92, 06: 45.
Vetoes by month: 2026-04: 3. By symbol: top downsize targets SLV 66, USO 47, GLD 41,
XLF 37, SOXX 37 (full table in JSON).

Regime mix of the 88 scored nights (`decisions.json` final label): risk_on_trend 41,
choppy 20, high_vol_panic 19, risk_off_trend 7, calm_uptrend 1. The window is
risk-on-tilted but does include a Feb–Mar stress episode; it is still one ~4.5-month
regime cycle.

## 3. Veto channel — classification and outcome

Cross-referenced against `decisions.json` actions and `portfolio_state.json` on each
veto date, plus a full scan of all 88 nights for any `LLM_VETO` sell action:

| Date | Symbol | Forced sell? | Blocked a passing buy? | Classification |
|---|---|---|---|---|
| 2026-04-22 | VIXY | No | No (not on the scored watchlist; not held) | **no-op** |
| 2026-04-25 | VIXY | No | No | **no-op** |
| 2026-04-29 | VIXY | No | No | **no-op** |

- **Zero `LLM_VETO` sell actions exist anywhere in the 88-night record.**
- No veto ever removed a symbol that had passed the score/health/vol filters.
- **Dollar impact of obeying the veto channel: $0.00 full period, $0.00 holdout.**
  There is nothing to score against a base rate — the channel never engaged the book.

VIXY was assessed on those nights only because it ranked in the top-5 raw *health*
scores; the score threshold + regime compatibility had already excluded it from the
buy path. The vetoes were *substantively correct in content* (VIXY does decay
structurally) but fired exclusively when they could not matter.

### 3a. The one live test — and the layer failed it

On **2026-06-04** the engine emitted **BUY VIXY $5,139.93**
(`daily/2026-06-04/decisions.json`, reason `SCORE_0.68_HEALTH_0.95`) — exactly the
structurally-decaying asset the LLM had vetoed three times in April. That night's
`llm_risk.json` for VIXY: **`structural_risk_veto: false`, `confidence_adjustment:
0.1`, rationale "Elevated regulatory scrutiny but no immediate threat"** — the
boilerplate string. The order was stopped only by an accident of plumbing:
`daily/2026-06-04/morning_execution.json` records
`FAIL BUY VIXY: broker error: Order notional $5139.93 exceeds cap $5000.00`.

The only time in 88 nights the veto layer faced a consequential decision on the one
asset class it had ever vetoed, it waved the trade through. (VIXY's forward window
is truncated at the 2026-06-09 panel end: +3.2% over the 4 observable days — too
short to score, and irrelevant to the structural point.)

**Substrate contradiction #2 (for the chair):** the packet states "VIXY is in the
canon book now." `daily/2026-06-10/portfolio_state.json` holdings are XBI, XLF,
ARKK, EEM, VWO — no VIXY, because the 06-04 buy intent failed on the broker cap.
Flagging rather than resolving; whichever book is canon, the LLM-layer finding above
is unchanged (the *decision* layer approved VIXY).

## 4. Downsize channel — the only channel that ever touched money

The `confidence_adjustment` only changes the book when a BUY is actually sized
(`llm_adj = 1 − adj` at `decision_engine.py:582`). Consequential events:

- 82 BUY intents in the record; **40 of them (49%) — $276,935 of $543,395 intent
  dollars (51%) — were never assessed by the LLM at all** (see §5).
- **39 consequential downsize events** (BUY with adj>0 that night; 38 scoreable at
  +21d). Holdout: 18 events.
- Total dollars withheld by haircuts: **$38,277** full period ($23,880 holdout).

### Hit rates vs base rates (+21d forward return < 0 = "hit")

| Cohort | n | % negative at +21d | mean fwd 21d |
|---|---|---|---|
| Downsized buys — **full** | 38 | **71.1%** | −2.47% |
| All executed buys (base rate, full) | 80 | 68.8% | −0.49% |
| Downsized buys — **holdout** | 17 | **52.9%** | −0.30% |
| All executed buys (base rate, holdout) | 41 | 53.7% | +0.88% |
| SPY on holdout buy dates | 16 | 31.2% | +3.71% |
| Watchlist candidates fwd 21d (full) | 678 | 54.9% | +0.43% |

Full-period the downsized cohort looks 2.3 pts better than base rate; **in the
holdout it is exactly base rate (52.9% vs 53.7%)**. A coin flip with API latency.

### Did bigger haircuts land on worse assets? No — mildly the opposite.

| Correlation (downsize events) | Full (n=38) | Holdout (n=17) |
|---|---|---|
| adj vs forward 21d **return** (Spearman) | **+0.38** | **+0.32** |
| adj vs forward 21d realized **vol** (Spearman) | +0.41 | +0.07 |
| adj vs forward 21d max adverse excursion (Spearman) | +0.14 | −0.07 |

Positive adj↔return correlation means **larger haircuts fell on buys that
subsequently did better**, full period and holdout. The full-period vol correlation
(+0.41) collapses to +0.07 in the holdout — consistent with the LLM echoing the
*trailing* vol_21d it is handed in its own prompt, which carries no stable forward
information. At n≈38 none of these clear conventional significance; the honest read
is "no detectable targeting skill, point estimates lean the wrong way."

### Dollar impact, obey vs ignore (+21d)

| Counterfactual for withheld dollars | Full period | Holdout |
|---|---|---|
| Obey vs ignore, withheld dollars sit in cash | **+$557.86** | **+$81.50** |
| Obey vs ignore, withheld dollars in SPY instead | +$126.42 | +$167.19 |

On a ~$100k book over 4.5 months, the entire downsize channel moved the needle by
**+$558 (≈0.5% of book, ≈1.5% of the $38k it withheld) full period and +$82
holdout** — indistinguishable from zero, and produced almost entirely by the
full-period Feb–Mar drawdown coinciding with a falling buy cohort (everything
withheld from a falling market "saves" money; that is the base rate, not skill).

### Severity calibration

severity → mean adj: 0 → 0.000 (n=18), 1 → 0.059 (n=98), 2 → 0.128 (n=686),
3 → 0.400 (n=2). Monotone, but with 85% of mass at severity 2 the scale carries
almost no discrimination.

## 5. The coverage hole

`llm_risk_check.run()` assesses **holdings + top-5 by raw health score** (max 16).
Buys, however, come from the *final_score* ranking after regime compatibility — a
different list. Result: **40 of 82 buys (49%), $276,935 of $543,395 in buy intents
(51%), were sized with no LLM assessment at all** (adj defaults to 0, no veto
possible). The layer the operator believes is "a veto/downsize check on trades" in
fact never sees half the trades. Whatever protective story is told about this layer,
it is structurally only half-attached to the decision flow.

## 6. Cost and latency

- Model: `anthropic.claude-3-haiku-20240307-v1:0` (Bedrock, on-demand), 9.14
  calls/night avg (cap 16), ~300 input + ~100 output tokens/call (prompt template +
  one-sentence JSON; `max_tokens=256`).
- At Claude 3 Haiku's long-standing Bedrock on-demand rate ($0.25/MTok in,
  $1.25/MTok out): **≈$0.0018/night ≈ $0.46/year**. Pricing caveat: the model is
  legacy and no longer listed on current pricing pages (checked
  platform.claude.com/docs and aws.amazon.com/bedrock/pricing on 2026-06-10); even
  at Claude Haiku 3.5 rates ($0.80/$4.00) the bound is **$1.47/year**. No cost
  telemetry exists in `run_report.json`.
- Latency: calls are sequential in `run()`; at ~1–2 s/call, **~9–18 s per night** in
  a batch pipeline. Negligible.

The honest cost framing: **the dollar cost is trivially small and is NOT the
argument against this layer.** The costs that matter are (a) a false sense of
coverage (§5), (b) a silent no-op failure mode (§2), (c) one more nondeterministic
input making replays/attribution harder, and (d) operational dependency on a
deprecated model id.

## 7. What this scoring CANNOT see (honesty section)

1. **Off-policy / survivorship:** stored vetoes and adjustments were conditioned on
   the realized book and realized candidate lists. We observe the vetoed/downsized
   assets' subsequent prices, not the counterfactual book: a blocked buy frees a
   slot and cash for the *next* candidate, sells change subsequent nights' holdings
   list (and therefore which symbols the LLM is even asked about). Only the
   Methodologist's replay ablation closes this.
2. **The withheld-dollars counterfactual is simplified:** haircut dollars are scored
   as if parked in cash (and separately vs SPY). In the real engine they raise the
   cash buffer and can be deployed into other buys later — path effects not modeled.
3. **One regime cycle, 88 nights:** risk_on 41 / choppy 20 / panic 19 / risk_off 7 /
   calm 1. A structural-risk veto is a tail-event layer; 88 nights containing zero
   true structural blowups (no ETF closure, halt, or fraud event in the universe)
   cannot prove the layer worthless in the tail — it can only show the tail never
   arrived and what the layer did meanwhile. The 06-04 VIXY pass-through (§3a) is
   the closest thing to a tail test in the record, and it failed.
4. **Small n:** 3 vetoes and 39 downsize events. Every distributional statement
   above carries wide intervals; the verdict leans on the *structure* of the record
   (no-ops, coverage hole, boilerplate, near-constant adjustment), which does not
   depend on n.
5. **Horizon truncation:** forward returns capped at 2026-06-09; +63d is fully
   observable only for events before ~2026-03-10; June events are 4-day-truncated
   (flagged per event in the JSON via `fwd_*_actual`).
6. **Intent vs execution:** dollar impacts use decision-layer intent sizes
   (`decisions.json`); live execution occasionally diverges (alpaca caps, gaps).
   The charter is the decision layer, so intent is the right basis, but book-level
   dollar effects in production differ.

## 8. Verdict input to the panel

**Does not earn its place — at E1-style historical-scoring evidence (not a replay
counterfactual; the E2 ablation cell remains the removal bar per EVIDENCE_PROTOCOL).**
In 88 production nights and 804 paid assessments, the veto channel engaged the book
zero times ($0.00 impact, full and holdout) and its three firings were no-ops on a
non-candidate; when the engine finally emitted a real order for the very asset those
vetoes named (VIXY, 2026-06-04), the layer approved it and a broker cap did the
vetoing. The downsize channel is a near-constant ~0.9× haircut (91.7% of assessments,
mass at 0.1/0.2) whose targeting is base-rate in the holdout (52.9% vs 53.7% hit
rate), whose magnitude correlates *positively* with subsequent returns, and whose
total obey-vs-ignore impact is +$558 full / +$82 holdout on a $100k book — while 51%
of buy dollars are never assessed at all. The ~$0.50–1.50/year Bedrock cost is
irrelevant; the layer's real liabilities are unearned trust, a silent-failure
default, and added nondeterminism. Recommendation to the Analyst/Methodologist: in
the ablation battery, model this layer's sizing effect as the constant 0.88×
multiplier it empirically is — if the ablation confirms the constant reproduces its
P&L contribution, the "LLM" part of the layer is dead weight at E2 and belongs on
the cut list (operator disposition, not removed in this packet).

---
*Files: `veto_audit_data.json` (per-event records, base rates, costs),
`veto_audit_supplement.json` (holdout cuts, severity-3 detail),
`scripts/veto_audit.py`, `scripts/veto_audit_supplement.py` (re-runnable),
`cache/` (pinned S3 pulls).*
