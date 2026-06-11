# SHADOW_PREREG — PKT-TB-007 dual forward shadow (the verdict contract)

**Registered:** 2026-06-11, BEFORE the first shadow outcome exists.
**Authority:** Training Realist Phase D review
(`tournament/REVIEW_TRAINING_REALIST_007_PHASE_D.md` §5 items 1–2) +
`COMMITTEE_REPORT_007.md` §Cross-packet synthesis item 4, operator-approved.
**Plan:** `docs/plans/2026-06-11-dual-forward-shadow.md`.
**This file is the read contract. The reads below are the ONLY pre-registered
looks. Anything else is exploratory and must be labeled as such.**

---

## 0. Frozen configuration

- **Start boundary:** the shadow is forward-only. The first processable
  decision date is the first trading date AFTER **2026-06-10**
  (= **2026-06-11**, confirmed at first run). The engine structurally
  refuses anything ≤ 2026-06-10 (tested).
- **Forecast model:** the frozen TB-007 deploy artifacts —
  M1 CAST-XS 2-seed mean (`models_out_007/m1_cast/seed_{4242,4243}.pt`),
  M3 disp coefs, M4-A event net (`models_out_007/m4_evt_a/model.pkl`);
  combined model sha printed in every forecast record (`model_sha`).
  No retraining, ever, inside this shadow. A retrained brain is a NEW
  shadow with a new prereg.
- **Shadow book A — a-priori genome** (`genomes/shadow_A.json`,
  hash `952a2f5a565e`): tilt_gain 0.5, organ_trust {M1:+1.0}, disp_gain 1.0,
  dead_zone 0.05, conviction_temp 1.0, caps at B0 mids (cap_core 0.0125,
  cap_conditional 0.00625, defensive_fraction 0.25), event_damp 0.0 —
  exactly the FREEZE_ORB1 ADDENDUM value-blind a-priori genome (the D03
  configuration).
- **Shadow book B — a-priori + M4-A damping** (`genomes/shadow_B.json`,
  hash `c9b8ee96c859`): book A + organ_trust {M4:+1.0},
  event_damp_strength 0.5 — exactly the D05 challenger-in arm.
- **Book I — paper incumbent:** the chassis's actual intents
  (`daily/<D>/trade_intents.json`), no tilt, same fill convention. All paired
  utility statistics are computed against book I (paper-vs-paper; the live
  broker line is mirrored for display only, never used in a verdict).
- **Expression channel:** `prototype/tilt_adapter.py` frozen production
  masks, T_max 0.08, the full §2 projection — the ONE implementation, no
  fork. Fills at D's open from `daily/<D+1>/prices.parquet`, marks at D's
  close, post-hoc cost overlay seed 4242 (TB-006 wiring adjudication #2).
  Cluster cap + min_order from the chassis's live
  `config/decision_params.active.json` (recorded per night).

## 1. Forecast-IC leg (the §2.3(a) shared-inflation test)

- **Record:** every decision date D, M1 mu over the full valid universe is
  appended to `forecast_ledger.jsonl` with a UTC `recorded_at` timestamp and
  a sha of the exact feature tensor. Records stamped on/after their own
  maturity date are excluded from scoring (enforced + tested); records
  stamped after D's 09:30 ET open carry `late_record: true` (catch-up
  transparency; their inputs are still the frozen point-in-time S3
  artifacts) — the count is reported at the read.
- **Statistic (the gates_007.windowed_ics convention, verbatim):** weekly
  sample dates = every 5th trading day from 2026-06-11; per sample date,
  cross-sectional Spearman rank-IC between recorded mu and the realized
  open(D)→open(D+5) return minus its cross-sectional mean, over valid
  symbols (minimum 8). Mean IC, se = sd/√n over the weekly series.

**READ F1 — IC existence (the Realist's ~31-week read).**
- **When:** the night the IC ledger reaches **n = 31** scored
  non-overlapping weeks. Projected calendar date ≈ **2027-01-27**
  (155 trading days of records + 5 of maturity from 2026-06-11; the ledger
  row count governs, not the calendar projection).
- **Bar (quoting the Realist):** "Weekly IC sd ≈ 0.30 at this breadth ⇒
  distinguishing IC 0.107 from 0 at 2σ needs ~31 independent weeks ≈ 7
  months."
  **CERTIFY "forecast signal real forward" iff mean weekly IC > 0 with
  one-sided t ≥ 2.0 at n = 31.** Otherwise the verdict line reads "forward
  IC not distinguishable from zero at the pre-registered power" — no
  extension, no re-cut.
- **READ F2 — magnitude (secondary, pre-registered now):** at **n = 110**
  weeks (≈ 2.1 years, projected ≈ 2028-08), test mean IC against the shrunk
  prior: "distinguishing 0.107 from the shrunk 0.05 needs ~110 weeks." Bar:
  one-sided t ≥ 2.0 against H0: IC = 0.05 certifies the unshrunk panel IC;
  mean below 0.05 with the CI excluding 0.107 certifies the shared-inflation
  hypothesis (§2.3a) at panel scale.

## 2. Utility leg (conversion at materiality)

- **Statistic:** paired daily return difference (cost-adjusted book A minus
  cost-adjusted book I), bp/day; mean with Newey-West HAC(5) se; 95% CI =
  mean ± 1.96·HAC se, over all settled days since 2026-06-11.

**READ U1 — 8 months.** At **164 settled trading days** (projected ≈
**2027-02-08**): the window certifies |effect| ≥ 2 bp/day (the Realist's
"(2·12.8/2)² ≈ 164 td"). Sign language is permitted ONLY if |mean| ≥ 2
bp/day with the CI excluding zero; otherwise the printed sentence is
"no conversion effect at the ≥2 bp/day scale this window can see."

**READ U2 — 14 months (the verdict read).** At **291 settled trading
days** (projected ≈ **2027-08-10**): certifies |effect| ≥ 1.5 bp/day
("≥ 1.5 in 291 td ≈ 14 months").
- **Equivalence band (the TB-006 §7 rule, pre-registered):** if the 95% CI
  is **fully contained in (−1.5, +1.5) bp/day**, the verdict is
  **"measured zero at materiality scale"** — conversion, if any, is too
  small to matter at this book size, and the thread closes.
- If the CI excludes zero AND |mean| ≥ 1.5 bp/day: a signed conversion
  verdict (either sign) is certified.
- Anything else prints "indeterminate at pre-registered power" — and the
  Realist's arithmetic stands: if the truth is 0.5 bp/day, nothing feasible
  certifies it and the equivalence read correctly returns "too small to
  matter."

## 3. M4-A damping confirmation arm (one arm, riding inside the shadow)

- **Statistic:** paired daily return difference (book B minus book A),
  bp/day, HAC(5), computed ONLY over the pre-registered sub-windows.
- **Sub-window validity rule (the Realist's discriminating condition):**
  the M4-A read is VALID ONLY on sub-windows where the base tilt's point
  estimate (book A minus book I) is **positive**. The mechanical confound —
  "damping a losing tilt scores positive by construction" — is thereby
  excluded by design. At each utility read date (U1, U2), the M4-A line is
  evaluated on the largest contiguous sub-window(s) with A−I > 0,
  reported with its n.
- **Bar:** B−A > 0 with HAC t ≥ 2.0 on a valid sub-window of n ≥ 66 days
  (power ≈ 75% if the battery's +0.08 bp/day is real; honest prior 15–25%).
- **Pre-commitment (verbatim Realist):** "One arm, pre-commit to dropping
  the thread on a miss. Running it twice would be noise-chasing." A miss at
  U2 closes the M4 damping thread permanently.

## 4. No-mid-stream-changes clause

- The genomes, masks, model weights, support tiers, fill convention, cost
  seed, statistic definitions, and read bars above are **frozen as of this
  commit**. No knob moves, no statistic is re-cut, no window is extended or
  shortened after this registration.
- Permitted maintenance: pure-engineering fixes (crash handling, S3
  retries, log wording) that provably do not change any emitted number;
  each lands with the nightly parity self-check green (M1/disp_z/M4
  recomputation vs the frozen prototype `store/nightly_007` files — a
  parity failure ABORTS the night rather than writing divergent forecasts).
- Any change that WOULD alter an emitted number (data-source swap, split
  handling, model artifact change) terminates this shadow's reads and
  requires a fresh prereg with a fresh start date for the affected leg.
- The reads land automatically: the nightly job accrues the ledgers; the
  read nights are ledger-count events. Interim peeks at the dashboard line
  are display, not verdicts; **no trading decision is keyed to this shadow
  before its read dates** (cross-packet synthesis: "No deploy decision is
  forced by anything in this dossier").

## 5. Recorded instrumentation notes (honest limitations, not changes)

1. Forward OHLCV rows splice S3 `daily/<D>/prices.parquet` bars with
   adj_close := close — ex-dividend days appear as real price drops in
   forward features (the training history was dividend-adjusted). Small,
   symmetric across all three books and both legs.
2. Splits are not auto-adjusted; any >35% overnight move raises an ALERT
   log line for operator action (the VUG 6:1 precedent).
3. The FOMC-week calendar dummy ends 2026-06-17 (the frozen builder list);
   later FOMC weeks read 0 in the M3 disp design. Affects only the disp_z
   conviction input, identically across books A and B.
4. GDELT gaps degrade through the trained `gdelt_available` masking; CBOE
   staleness > 5 days nulls disp_z to the B-disp constant (the adapter's
   registered fallback). Both are logged.
5. Alerting = ALERT lines in `shadow/logs/` only (operator SNS out of
   scope for this packet — noted per plan).
6. The M3 turn-of-month dummy is computed on the true exchange calendar
   (month-end trading days projected from the rule-based NYSE holiday
   calendar). The frozen reference panel, truncated at 2026-06-10, carried
   a month-end labeling artifact on its final two rows; the forward
   convention matches the training distribution for every complete month
   (parity-verified on May 2026). Affects only disp_z, identically across
   books A and B.

## 6. Where everything lives

- Engine + state: `runs/pkt_tb_007_orthogonal_brain/shadow/`
  (`shadow_nightly.py`, `state/`, `logs/`).
- S3 mirror: `s3://investment-system-data/shadow/pkt_tb_007/`
  (state.json, ledgers, organ inputs, this file). Overwrite-in-place,
  no versioning.
- Dashboard surface: `s3://investment-system-data/dashboard/shadow_timeseries.json`
  ({as_of, live_line (mirrored), shadow_A, shadow_B, ic_series, stats,
  prereg_pointer}).
- Schedule: `~/Library/LaunchAgents/com.traderbot.shadow.plist`,
  23:30 ET Mon–Fri (after the 22:00 ET night pipeline; catch-up makes
  missed nights harmless).

— registered before first outcome; the ledgers do the rest —
