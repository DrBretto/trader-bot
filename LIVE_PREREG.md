# LIVE_PREREG — the forward contract for the native two-stage brain (the LIVE arm)

**Registered:** 2026-06-16, BEFORE the first forward LIVE outcome exists.
**Authority:** `D-AUTO-20260616` (operator, 2026-06-16; conscious pre-registration override) +
DESIGN_DOSSIER.md §6 (the contents spec), committee
`20260616_trader-bot-new-brain-overhaul-committee-v1`, governed on the Infotropy spine.
**Packet:** `PKT-TB-010-LIVE-PREREG-V1-20260616`.
**Frozen artifacts:** `brain/FREEZE_ORB1.json` (content hashes asserted at cold start).
**This file is the read contract for the LIVE arm. The reads below are the ONLY
pre-registered LIVE looks. Anything else is exploratory and must be labeled as such.**

> **⚠ RE-PRE-REGISTRATION — 2026-06-23 (start date 2026-06-24).** Under the §3
> no-mid-stream bound, the LIVE arm has been re-frozen and the prior forward reads
> (registered 2026-06-16, start 2026-06-12) are **terminated**; a fresh forward
> read begins 2026-06-24. **Reason:** the cutover had dropped the regime chassis —
> `config/regime_compatibility.json`, the regime picker's multiplier series the
> orthogonal members were specced to add value ON TOP OF (PROPOSAL_ORTHOGONALITY
> §0) — leaving the live book regime-blind. It loaded ~51% into one correlated
> high-beta complex and lost −3.37% vs SPY −1.27% on 2026-06-23 (biggest single-day
> drop on record). The re-freeze restores the chassis: regime enters Stage-1 ranking
> as a soft per-name multiplier (`ForecastBundle.regime_score_mult`, the faithful
> port of `decision_engine.score_candidates`) and Stage-2 as a book-level gross cut
> (`θ_size.regime_exposure_multiplier`); a correlation-group cluster cap now binds on
> the tech/metals complexes. `θ_sel` is unchanged. New hashes are in §0. See
> `docs/plans/2026-06-23-restore-regime-chassis-into-new-brain.md`.

> **The standing fact, stated first (DESIGN_DOSSIER §1).** The brain's dollar conversion
> is currently measured at exactly zero: the exposure-stripped selection residual is
> **−0.29 bp/day, t≈−0.22**. The forecast signal (M1, pooled weekly rank-IC 0.1076,
> p=1.6e-8) is real *as a forecast*; its conversion into dollars is real only as a
> question this pre-registration measures forward. **This document freezes the
> instrument that measures conversion. It does not assert conversion. No surface
> produced under it may imply the live brain has a dollar edge.**

> **Relationship to SHADOW_PREREG.** The paper shadow (`runs/pkt_tb_007_orthogonal_brain/shadow/SHADOW_PREREG.md`)
> is **left untouched** and continues in parallel as the independent, un-overridden
> out-of-sample verdict clock (its READ F1 ≈ 2027-01-27, READ U2 ≈ 2027-08-10 stand
> exactly as registered). This LIVE_PREREG governs a *different* arm: the native
> two-stage engine deployed forward per the dated override (§4). The two clocks run
> side by side; neither is re-cut to match the other.

---

## §0. FROZEN ARTIFACTS (content-hashed, asserted at cold start)

All hashes live in `brain/FREEZE_ORB1.json` and are asserted at the night Lambda's
cold start (PKT-TB-012). A hash mismatch is a parity failure that ABORTS the night.

| Artifact | Identity (frozen) |
|---|---|
| **engine_sha** | `4ae7b79d9d62fa5255ba485e287d4493333225296ad9f82da6200f1413a6c51b` — content hash of `src/brain/engine/*.py` (re-frozen 2026-06-23: `contracts.py` + `selection.py` carry the restored `regime_score_mult` Stage-1 tilt; prior `0e1d22ec…` was the regime-blind cutover). **The live brain is THIS engine, not `tilt_adapter`.** |
| **model_sha** | `5b2428c66c14` — `forward_inference.model_sha()` over the four `models_out_007/` files (`m1_cast/seed_4242.pt`, `m1_cast/seed_4243.pt`, `m4_evt_a/model.pkl`, `m3_disp/coefs.npz`), per-file sha256 pinned in FREEZE_ORB1. **No retraining, ever, inside this LIVE arm** — a retrained brain is a NEW prereg with a new start date. |
| **θ_sel** | content hash `f154f3ce…cbd24b5e` (`SelectionParams.content_hash()`). Frozen by hand: `N=10`, `h_min=0.60`, `core_fraction=0.55` (tier edge), `regime_admissibility={}` (admits-all; regime enters only as a Stage-2 book-level multiplier), `tiebreak=(mu_M1_desc, idio_vol_asc, symbol_asc)`, `lot_policy{min_order=$250, reference_nav=$100,000}`. **Each parameter carries a provenance note in FREEZE_ORB1 ("prior to falsify, not fit to any forward read")** — this discharges **C4** (closes defect-B-via-the-human). |
| **θ_size + parity gain** | content hash `9561c1d0…f204b8cf` (re-frozen 2026-06-23; prior `4c4844ae…`). `gross_target=1.0`, `max_position_weight=0.20`, `max_cluster_weight=0.35`, `cash_reserve_pct=0.10`, **`regime_exposure_multiplier={calm 1.0, risk_on 1.0, choppy 0.90, risk_off 0.75, panic 0.50}`** (restored — cuts book-level gross in risk-off/panic), **`parity_gain=0.5`** (closed-loop ex-post parity controller, the F8 fix). θ_size never reaches Stage-1 Select (enforced by `assert_no_dollar_surface`). |
| **go-live universe** | `config/universe.csv` sha `438abff0…` (64 names, all eligible; the membership/eligibility contract) + `universe_exploitability.json` sha `c3cf5abe…` (the forecast-skill/tradability derivation). **Every tilt name is `forward_confirmed:false`** — a strong in-sample prior to falsify, tilted live before any forward fold confirms it (DESIGN_DOSSIER Attack 2, operator-accepted residual). The `min_fresh_fold_end ≥ 2026-06-11` floor (C2) governs the first forward re-derivation (PKT-TB-013), not this frozen go-live set; the forward `universe_manifest_{T}.json` does not yet exist (see BUILD_RECEIPT executor_concern). |

**C4 [GATE] discharge.** The full `θ_sel` vector is content-hashed in `brain/FREEZE_ORB1.json`
with a per-parameter provenance note and is bound under the no-mid-stream clause (§3).
Without this, "frozen by hand" is unaudited in-sample selection (defect B executed by a
human). With it, the hand-choice is pre-committed before the first forward date and the
Analyst sign-off does not revert to WITHHOLD.

## §1. START BOUNDARY

- The first processable LIVE decision date is the first trading date **strictly after the
  freeze commit AND after the two-stage engine writes `daily/<D>/trade_intents.json` with a
  green invariant self-check** (`held_symbols_invariant()` True across the θ_size grid).
  No backfit line: the engine structurally refuses any decision date ≤ the go-live boundary
  (`assert_forward_only`, extended to the publish path).
- The displayed champion line is **frozen-as-artifact through 2026-06-11** (the in-sample
  replay, kept for reference). It is **never spliced for computation and never justifies the
  forward LIVE line**. The New Brain forward line is re-anchored C0-continuous to the 06-11
  terminal (~$114.9k) by PKT-TB-012 — display continuity only, not a computational splice.
- The invariant self-check runs nightly as a hard gate. A False result **ABORTS the night**
  (abort-never-degrade); the morning executor falls back to the incumbent intents.

## §2. PRE-REGISTERED FORWARD READS (ledger-count events, not calendar dates)

The read nights are **ledger-count events**: the nightly job accrues the ledgers and the
read lands automatically when the count is reached. Interim dashboard numbers are **display,
not verdict**.

**(a) Forecast-IC leg — the certified skill receipt (never multiplied by notional).**
Every LIVE decision date D, the engine's M1 `mu` over the valid universe is appended to the
forecast ledger with a UTC `recorded_at` and a sha of the exact feature tensor (records
stamped on/after their own maturity excluded; late records flagged). **Statistic
(`gates_007.windowed_ics` convention, verbatim):** weekly sample dates = every 5th trading
day from the LIVE start boundary; per sample date, cross-sectional Spearman rank-IC between
recorded `mu` and realized open(D)→open(D+5) demeaned return, min 8 symbols; mean IC and
se=sd/√n over the weekly series. **READ — IC existence:** at **n=31** scored non-overlapping
weeks, CERTIFY "forecast signal real forward" iff mean weekly IC > 0 with one-sided t ≥ 2.0;
otherwise the line reads "forward IC not distinguishable from zero at the pre-registered
power" — no extension, no re-cut. *This leg is forecast-altitude ONLY; it is never converted
to a dollar claim.*

**(b) Utility leg — the conversion measurement (the UNPROVEN thing).** The
**exposure-stripped, MULTI-FACTOR forecast-rung rent** F−R (the forecast organ's marginal,
PKT-TB-011's 5-rung ladder), bp/day, with **Newey-West HAC(5)** CI and a **three-valued
verdict**. The strip is multi-factor (SPY + duration + broad-commodity) and adds a realized
gross-differential term (C6) — a static-beta strip cannot see the F8 gross leak. **Verdict
rule (the materiality band, SHADOW_PREREG §2 / TB-006 §7, pre-registered):**
- `positive` iff one-sided HAC t ≥ +2.0 at the pre-registered read count;
- `zero (measured at materiality scale)` iff the 95% CI is **fully contained in
  (−1.5, +1.5) bp/day** — conversion, if any, is too small to matter at this book size;
- `indeterminate at available power` otherwise — **with the point estimate and sign printed**.
Never "paying rent" on a point estimate alone. The conversion verdict is pre-committed here,
before the first forward dollar is read.

**(c) Realized ex-post parity series — the F8 disclosure form.** Nightly, the realized
gross / β / σ versus target plus the lot residual are appended to the parity ledger
(the closed-loop controller's record). This is the standing evidence that the realized book
tracks the intended book; a persistent one-signed residual is the F8 leak made visible.

**(d) M4-A event-damping sub-window read — BH-FDR family member (C9).** Carried **verbatim
from SHADOW_PREREG §3** so it is not a garden-of-forking-paths leak:

> - **Statistic:** paired daily return difference (book B minus book A), bp/day, HAC(5),
>   computed ONLY over the pre-registered sub-windows.
> - **Sub-window validity rule (the Realist's discriminating condition):** the M4-A read is
>   VALID ONLY on sub-windows where the base tilt's point estimate (book A minus book I) is
>   **positive**. The mechanical confound — "damping a losing tilt scores positive by
>   construction" — is thereby excluded by design. At each utility read date, the M4-A line is
>   evaluated on the largest contiguous sub-window(s) with A−I > 0, reported with its n.
> - **Bar:** B−A > 0 with HAC t ≥ 2.0 on a valid sub-window of n ≥ 66 days (power ≈ 75% if the
>   battery's +0.08 bp/day is real; honest prior 15–25%).
> - **Pre-commitment (verbatim Realist):** "One arm, pre-commit to dropping the thread on a
>   miss. Running it twice would be noise-chasing." A miss closes the M4 damping thread
>   permanently.

**FDR family membership (C9).** All conversion-class reads are members of one
**Benjamini–Hochberg FDR(10%)** family: the five ladder rungs (regime / forecast / event /
universe, PKT-TB-011) **and the M4-A sub-window read (d)**. The sub-window A−I>0 boundaries
above are the pre-registered ledger-event boundaries. A single green organ renders with a
"1 of N — not narratable as a discovery alone" caveat until it survives FDR. The IC leg (a)
is a *skill* read, reported separately, never collapsed into the conversion family.

## §3. NO-MID-STREAM-CHANGES

- The engine (`engine_sha`), model artifacts (`model_sha`), `θ_sel`, `θ_size`, parity gain,
  go-live universe, fill convention, cost seed, statistic definitions, and read bars above
  are **frozen as of this registration**. No knob moves, no statistic is re-cut, no read
  count is extended or shortened.
- **Permitted maintenance:** pure-engineering fixes (crash handling, S3 retries, log wording)
  that provably do not change any emitted number; each lands with the **nightly invariant
  self-check green** (held_symbols invariance across the θ_size grid + the FREEZE_ORB1 cold-
  start hash assertions). **A self-check failure ABORTS the night** rather than writing
  divergent forecasts.
- **Any change that WOULD alter an emitted number** (data-source swap, split handling, a model
  artifact change, any `θ_sel`/`θ_size` edit that moves its content hash) **terminates this
  LIVE arm's reads → fresh prereg + fresh start date** for the affected leg. A retrained brain
  is a new prereg.

## §4. DATED OVERRIDE

Per **`D-AUTO-20260616`** (2026-06-16) the operator **consciously overrides SHADOW_PREREG's
"no deploy before ~2027" verdict-wait for the LIVE arm only**. The paper shadow is **left
untouched** and continues in parallel as the independent, un-overridden out-of-sample verdict
clock. **The override authorizes forward measurement of conversion; it does NOT assert
conversion.** No surface produced under this pre-registration may imply the live brain has a
dollar edge.

The override is legitimate *because* the instrument that measures it is built into the same
design (this LIVE_PREREG + PKT-TB-011's attribution spine), its conversion verdict is
pre-committed before the first forward dollar is read (§2b), the paper shadow clock runs
untouched, and the frozen champion line is byte-immutable through 2026-06-11. Rationale of the
override (D-AUTO-20260616): the retired models merely reinforced the regime picker (redundant);
the new brain adds a genuinely orthogonal forecast (M1 pooled rank-IC 0.1076, p=1.6e-8);
fake money; no reason to keep running the old models. **None of that asserts a dollar edge —
it justifies measuring forward.**

— registered before first LIVE outcome; the ledgers do the rest —
