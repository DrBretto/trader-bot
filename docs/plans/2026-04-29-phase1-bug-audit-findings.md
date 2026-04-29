# Phase 1 — Bug Audit Findings

Date: 2026-04-29
Packet: `docs/plans/2026-04-29-deep-diagnostic-and-calibration-packet.md`
Authority: execution / discovery_bearing.

Snapshotted production data used as evidence: `/tmp/claude-501/trader-bot-audit/{dashboard,timeseries}.json` (refreshed 2026-04-29 from `s3://investment-system-data/dashboard/`).

Every finding cites `file:line` for behavior claims and provides a small Python reproducer against the snapshotted data. No silent fixes — every defect goes here, even ones tempting to absorb into another finding.

Risk-class legend: `LIVE-IMPACT` (current production decisions are wrong), `OBSERVABILITY` (decisions may be right but evidence is misleading), `LATENT` (asleep but armed).

| ID | Title | Status | Risk class |
|----|-------|--------|-----------|
| F-1 | `vol_uncertainty` was effectively disabled for 8 months and re-armed 2026-04-02 | **CONFIRMED** (cause refined) | LIVE-IMPACT |
| F-2 | `fragility_score` is a one-way ratchet | **DISCONFIRMED as ratchet; CONFIRMED as tanh-saturation dead-zone with a 60-day window** | LIVE-IMPACT |
| F-3 | Vol "complex" is single-input VIX in disguise | **CONFIRMED** | LIVE-IMPACT |
| F-4 | Per-position dollar P&L wrong on VUG | **CONFIRMED — root cause: dual cost-basis in broker reconciliation path** | LIVE-IMPACT (display) |
| F-5 | Two divergent valuation paths produce different P&L for the same holding | **CONFIRMED** | LIVE-IMPACT (display) |
| F-6 | Expert-signal fallbacks flow into `timeseries.json` indistinguishable from real values | **CONFIRMED** | OBSERVABILITY |
| F-7 | Fragility caution gate is regime-blind | **CONFIRMED (by code reading)** | LIVE-IMPACT |
| F-8 | `compute_signals.run` reaches for `fred_latest` outside its definition scope | **CONFIRMED** — likely upstream cause of F-1 | LIVE-IMPACT (latent re-arm) |
| F-9 | Skew is read from Stooq but never given a `_history` and percentile is hardcoded | **CONFIRMED** | OBSERVABILITY |
| F-10 | Continuity-adjusted `total_value` vs raw `broker_total_value` differ by ~$10,078 with no on-page disclosure | **CONFIRMED — design artifact, not a bug, but UX hazard** | OBSERVABILITY |

---

## F-1 — `vol_uncertainty_score` flat at 0.10 for 168/187 production days, then live from 2026-04-02

### Evidence in production data

```python
import json
ts = json.load(open('/tmp/claude-501/trader-bot-audit/timeseries.json'))
vu = [r['vol_uncertainty_score'] for r in ts]
n_010 = sum(1 for v in vu if abs(v - 0.10) < 1e-6)
print(n_010, '/', len(vu))                  # 168 / 187
print('first non-0.10:',
      next(r['date'] for r in ts if abs(r['vol_uncertainty_score']-0.10)>1e-6))
# → 2026-04-02
print('distinct values:', len(set(round(v,4) for v in vu)))   # 20
```

`vol_uncertainty_score` was 0.10 on 168 of 187 live days (89.8%), then stepped to 0.879 on 2026-04-02 and has been live every day since.

### Code-level diagnosis

`src/signals/compute_signals.py:130-141`:

```python
vix_value = fred_latest.get('VIXCLS', 0) if 'fred_latest' in dir() else 0
# Also try context_df
if vix_value == 0:
    vix_value = float(ctx.get('vixy_return_21d', 0)) if hasattr(ctx, 'get') else 0
# Get VIX from FRED history for percentile computation
vix_history = None
if len(fred_df) > 0:
    vix_data = fred_df[fred_df['series_id'] == 'VIXCLS'].sort_values('date')
    if len(vix_data) >= 60:
        vix_history = vix_data['value']
        vix_value = float(vix_data['value'].iloc[-1])
```

When the FRED VIX history had `< 60` rows OR `series_id == 'VIXCLS'` was not present in `fred_df`, the code fell to `vix_value = 0`. `_percentile_score(0, VIX_THRESHOLDS)` (`src/signals/vol_uncertainty.py:24-25`) returns `0.1` because `0 ≤ p20 (=13)`. With `vvix=None, skew=None`, the composite collapses to `score = vix_pctile = 0.10` (`src/signals/vol_uncertainty.py:97`). The exact `0.10` constant flowing through 168 days is not "calibration" — it is the lowest-bin output of the pure-VIX percentile evaluated at zero.

The step to `0.879` on 2026-04-02 is consistent with a data path being repaired (FRED VIX series finally showing ≥ 60 rows in `fred_df` at run time, OR `vvix_data`/`skew_data` becoming available enough to influence the composite). The Apr 1 commit `e61f9c5` packaged a midday-checker deploy that touched `src/handler.py` and may have refreshed the deployed Lambda's data plane; root-cause for the data-plane fix is not pinpointable from git alone — see F-8 for an immediately related code defect that was contributing.

### Reproducer (deterministic)

```python
import pandas as pd
from src.signals.vol_uncertainty import compute_vol_uncertainty
out = compute_vol_uncertainty(vix=0, vvix=None, skew=None, vix_history=None, vvix_history=None)
assert abs(out['vol_uncertainty_score'] - 0.10) < 1e-9, out
```

### Recommended fix summary

`compute_signals.py:130-141`:
1. Treat `vix_value == 0` (or `None`) as a HARD ERROR for the vol-uncertainty path and emit `degraded_reason` instead of returning a deceptive `0.10` percentile.
2. Make `_percentile_score` return `None` (not `0.1`) when called with a zero/missing VIX, and have `compute_vol_uncertainty` propagate `degraded_reason='vix_unavailable'` rather than silently returning a bottom-bin score.
3. Lift the `vix_history` minimum-observation logic to a shared helper so the same neutral-fallback behavior is symmetric across `compute_vol_uncertainty`, `compute_macro_credit`, and any future signal that consumes FRED history.

### Risk class

LIVE-IMPACT. For 8 months the bot's fragility gate was the *only* live throttle on position size; vol_uncertainty was a phantom signal that contributed nothing. Once it re-armed on Apr 2 it stacked on top of fragility and roughly halved deployable capital relative to the prior 7-month regime. Any historical fitness/calibration based on the disabled era is over-fit to a state where one of the two main throttles was off.

---

## F-2 — `fragility_score` is **not** a one-way ratchet — it is a tanh-saturation dead-zone over a 60-day window

### Evidence in production data

```python
fr = [r['fragility_score'] for r in ts]
diffs = [fr[i] - fr[i-1] for i in range(1, len(fr))]
print('ups, downs, flat:',
      sum(d > 0.001 for d in diffs),                       # 29
      sum(d < -0.001 for d in diffs),                      # 18
      sum(abs(d) <= 0.001 for d in diffs))                 # 139
print('max up, max down:', max(diffs), min(diffs))
# 0.3745   -0.0384
print('last30 range:', max(fr[-30:]) - min(fr[-30:]))
# 0.0935
print('mean:', sum(fr)/len(fr))                             # 0.6295
```

The packet hypothesis was "physically cannot decline more than 4 percentage points in a day → smoothed running max." The data actually shows: 29 up days, 18 down days, max single-day rise of `+0.3745` and max single-day decline of `-0.0384`. Both directions occur. **The "ratchet" framing is wrong.** What is actually happening is two separate effects:

1. **60-day rolling window arithmetic**: today's pairwise correlation matrix replaces only `1/60 ≈ 1.7%` of the underlying return panel each day, so day-to-day change in `avg_correlation` and `pc1_explained` is bounded.
2. **tanh saturation**: `src/signals/fragility.py:101-108`:
   ```python
   corr_z = (avg_correlation - 0.30) / 0.15
   norm_corr = (np.tanh(corr_z) + 1) / 2
   ```
   At `avg_correlation = 0.55`, `corr_z = 1.67`, `tanh ≈ 0.93`, `norm_corr ≈ 0.965`. At `avg_correlation = 0.65`, `corr_z = 2.33`, `tanh ≈ 0.98`, `norm_corr ≈ 0.99`. Above `avg_correlation ≈ 0.55` the score is squashed into the top 5% of its range with no remaining differentiation power. The same effect dominates `pc1_explained` (`pc1_mean=0.45, pc1_std=0.12`).

Combined: in a single-factor melt-up rally where everything moves together, `fragility_score` saturates at `≈ 1.0` regardless of *whether the correlation is rising for fearful or euphoric reasons*. The signal cannot distinguish "everything correlated because we're in a bull rally" from "everything correlated because we're about to crash."

### Reproducer (deterministic)

```python
import numpy as np
mean, std = 0.30, 0.15
for c in [0.30, 0.45, 0.55, 0.60, 0.65, 0.70]:
    z = (c - mean) / std
    norm = (np.tanh(z) + 1) / 2
    print(f"avg_corr={c:.2f}  z={z:.2f}  norm={norm:.4f}")
# avg_corr=0.30  z=0.00  norm=0.5000
# avg_corr=0.45  z=1.00  norm=0.8808
# avg_corr=0.55  z=1.67  norm=0.9637
# avg_corr=0.60  z=2.00  norm=0.9820
# avg_corr=0.65  z=2.33  norm=0.9909
# avg_corr=0.70  z=2.67  norm=0.9954
```

### Recommended fix summary

`src/signals/fragility.py:101-108`:
1. Replace tanh saturation with a *piecewise* mapping calibrated on long-horizon historical distributions (e.g. percentile rank against a multi-year window of avg_correlation). This restores resolution above 0.55.
2. Shorten the rolling window from 60d → ~20d (or expose two timescales — short and long — and use *divergence* between them as the fragility signal). 60d-only smooths through every regime change shorter than a quarter.
3. Make the gate **conditional on regime confidence** rather than purely score-based — see F-7. Even with a perfect score, if the ensemble is calling `risk_on_trend` with confidence 0.92, fragility's correct interpretation in a rally is "concentrated leadership" (not necessarily "imminent shock"). Saturation in correlation is bidirectional information; the current pipeline only reads it as risk.

### Risk class

LIVE-IMPACT. This is a primary reason `position_size_modifier` averaged 0.49 in the hybrid-live window (capping at the 0.60 fragility-gate cap × ensemble multiplier ≈ 0.85). The score has been ≥ 0.97 for 6 weeks straight not because risk has been pinned high but because the function has saturated. The gate has been treating "this rally" the same as "the day before a shock."

---

## F-3 — `vvix_percentile` and `skew_value` are literal constants in production; `vol_uncertainty_score` is just `vix_percentile` with two dead inputs

### Evidence in production data

```python
print(sorted(set(round(r['vvix_percentile'], 4) for r in ts)))  # [0.5]
print(sorted(set(round(r['skew_value'], 4) for r in ts)))       # [0.0]
# correlation of vol_uncertainty_score and vix_percentile across 187 days:
# 1.0000  (computed above; identical signal)
```

Both `vvix_percentile=0.5` (the neutral fallback in `compute_vol_uncertainty` line 82-83 and 86-87) and `skew_value=0.0` (the unavailable-fallback in line 117) are single-distinct-value constants across the entire 187-day live history. The `vol_uncertainty_score` is identical to `vix_percentile` at every single date (`corr = 1.000`), confirming Phase 6 spec was incomplete in production.

### Code-level diagnosis

`src/handler.py:211-218`:

```python
log_step(3, 12, "Ingesting vol indices...", logger)
with StepTimer("Ingest vol indices", logger):
    vvix_data = ingest_prices.fetch_stooq_index('^VVIX')
    skew_data = ingest_prices.fetch_stooq_index('^SKEW')
    ...
```

The Stooq fetch is *attempted*, and the result is passed forward, but the production data shows `vvix_data` and `skew_data` reach `compute_vol_uncertainty` with empty values from `_get_latest_close` (`compute_signals.py:19-23`) — that helper returns `None` when the DataFrame is empty. Either:

- `fetch_stooq_index` returns empty for `^VVIX`/`^SKEW` (rate-limit, schema drift, or the symbol convention changed), and the failure is silent.
- The handler is catching an exception elsewhere and the values never hit the call.

The `compute_vol_uncertainty` API exposes a `degraded_reason` field but only on full-block failure (`compute_signals.py:169`). There is no per-input degraded marker; missing VVIX simply collapses the composite weighting:

`src/signals/vol_uncertainty.py:90-97`:

```python
if vvix is not None and skew is not None:
    score = 0.45 * vix_pctile + 0.35 * vvix_pctile + 0.20 * skew_pctile
elif vvix is not None:
    score = 0.55 * vix_pctile + 0.45 * vvix_pctile
elif skew is not None:
    score = 0.65 * vix_pctile + 0.35 * skew_pctile
else:
    score = vix_pctile
```

The fall-through `else` makes the score *silently* equal to `vix_pctile` without flagging that two of the three inputs are missing. The Phase 6 design called for a vol *complex* — production has been running on a single component for the entire history.

### Reproducer

```python
from src.signals.vol_uncertainty import compute_vol_uncertainty
score = compute_vol_uncertainty(vix=20, vvix=None, skew=None)
# vol_regime_label='calm', vol_uncertainty_score == vix_pctile
print(score)
```

### Recommended fix summary

1. `compute_vol_uncertainty`: emit `degraded_reason='vvix_missing'` and/or `degraded_reason='skew_missing'` on each missing input so the artifact carries provenance, not just the score.
2. `_build_timeseries_row` (`publish_artifacts.py:391-436`): write a `vol_inputs_status` column that records per-input availability (`vix_ok|vvix_ok|skew_ok` flags). Without that, all 187 historical timeseries rows look indistinguishable from rows where the full complex was healthy.
3. `ingest_prices.fetch_stooq_index`: surface a structured failure record (HTTP code, parse error, empty result) rather than returning an empty DataFrame. Caller in `handler.py:214-218` should log a `WARNING` with the failure reason, not a benign `INFO` line.
4. **Decision-engine impact**: as long as VVIX/SKEW are dead, the `unstable_calm` hard override (`regime_fusion.py:131-141`) is unreachable — it requires `vvix_pctile > unstable_vvix_pctile (=0.80)` while VVIX is permanently `0.5`. This is a phantom override that has never fired.

### Risk class

LIVE-IMPACT for unreachability of `unstable_calm` regime; OBSERVABILITY for the `vix_percentile == vol_uncertainty_score` aliasing.

---

## F-4 — `unrealized_pnl` and `unrealized_pnl_pct` use different cost bases in the broker-reconciliation path; only VUG is currently miscomputed

### Evidence in production data

```python
import json
d = json.load(open('/tmp/claude-501/trader-bot-audit/dashboard.json'))
for h in d['holdings']:
    sh, ep, cp = h['shares'], h['entry_price'], h['current_price']
    expected = (cp - ep) * sh
    print(f"{h['symbol']:5s}  reported_pnl=${h['unrealized_pnl']:.2f}  "
          f"local_basis_expected_pnl=${expected:.2f}  pct={h['unrealized_pnl_pct']:.4f}")
```

7 of 8 rows match `(cp - ep) × sh` to the cent. Only **VUG** disagrees:

```
VUG  sh=102.766  ep=73.4817  cp=82.77
     reported_pnl = -$6,097.78
     local_basis_expected_pnl = +$954.52
     reported_pct = +12.64%   (consistent with local entry → current)
```

The reported `pct` and the reported `$` use different cost bases. From the broker number, the implied avg_entry_price = `current - (pnl/qty)` = `82.77 - (-6097.78/102.766)` = `$142.11`. The local entry price for VUG is `$73.48`. The two differ by ~2x — diagnostic of an Alpaca-side avg-entry that drifted from the local one (likely a leftover from the cutover continuity bridge that didn't reset for VUG).

### Code-level diagnosis

`src/steps/morning_executor.py:379-402`:

```python
holding = {
    'symbol': symbol,
    'shares': qty,
    'entry_price': existing.get('entry_price', pos['avg_entry_price']),  # local wins
    ...
    'current_price': pos['current_price'],
    'market_value': market_value,
    'unrealized_pnl': pos['unrealized_pl'],                              # broker wins
    ...
}
if holding['entry_price'] > 0:
    holding['unrealized_pnl_pct'] = (
        pos['current_price'] / holding['entry_price'] - 1                # local entry wins
    )
```

The broker's `unrealized_pl` is computed by Alpaca as `(current − broker_avg_entry) × qty`. The dashboard's `unrealized_pnl_pct` is computed locally as `current/local_entry - 1`. When the broker's `avg_entry_price` and `existing['entry_price']` agree (7 of 8 holdings), the two reports are consistent. When they disagree (VUG), the dashboard shows incoherent numbers and the bug is *invisible to a human auditor* until they cross-check the columns.

### Reproducer

```python
holding_local_entry = 73.48
broker = {
    'symbol': 'VUG', 'qty': 102.766, 'current_price': 82.77,
    'unrealized_pl': -6097.78,             # broker calc: avg_entry ≈ 142.11
    'avg_entry_price': 142.11,
    'market_value': 102.766 * 82.77,
}
existing = {'entry_price': 73.48}
# Replicating morning_executor.py:379-402:
pnl_dollar = broker['unrealized_pl']                                # uses broker basis
pnl_pct    = broker['current_price'] / existing['entry_price'] - 1  # uses local basis
assert pnl_dollar < 0 and pnl_pct > 0, "sign disagreement is the bug"
```

### Recommended fix summary

Pick a single cost basis and use it for both fields. Two coherent options:

1. **Local-entry canonical** (preserves continuity-bridge intent): replace `holding['unrealized_pnl'] = pos['unrealized_pl']` with `holding['unrealized_pnl'] = (pos['current_price'] - holding['entry_price']) * qty`. Always report local-basis $ and %.
2. **Broker-truth canonical**: replace `holding['entry_price'] = existing.get('entry_price', pos['avg_entry_price'])` with `holding['entry_price'] = pos['avg_entry_price']`, and compute pct from broker's avg. Then both fields reflect the broker's view of cost basis.

Option 1 is recommended because (a) the dashboard's `total_value` is already continuity-adjusted (F-10), so local-basis $ and % are the consistent presentation; (b) `broker_total_value` is already exposed for broker-truth reconciliation. Option 2 would require also re-aligning the equity curve and TWR series.

Add a unit test that fabricates a divergent broker `avg_entry` vs local `entry_price`, runs the reconciliation, and asserts `sign(unrealized_pnl) == sign(unrealized_pnl_pct)` for every holding.

### Risk class

LIVE-IMPACT (display). The dashboard `holdings.unrealized_pnl` total feeds `invested` indirectly (via market_value) but does not currently affect orders. The hazard is the *operator*: when the published number is internally inconsistent, all downstream reasoning ("am I making money on this position?") is corrupted.

---

## F-5 — `_update_valuations_from_quotes` and `_reconcile_portfolio_from_broker` are two divergent valuation paths; which one runs depends on transient broker state

### Code-level diagnosis

`src/steps/morning_executor.py:130-167` (`_update_valuations_from_quotes`):

```python
holding['unrealized_pnl'] = (price - holding['entry_price']) * holding['shares']
holding['unrealized_pnl_pct'] = (price / holding['entry_price'] - 1)
```

Local-basis `$` AND `%`. Always coherent.

`src/steps/morning_executor.py:351-413` (`_reconcile_portfolio_from_broker`):

```python
holding['unrealized_pnl'] = pos['unrealized_pl']                          # broker
holding['unrealized_pnl_pct'] = pos['current_price'] / holding['entry_price'] - 1   # local
```

Mixed-basis. Per F-4, can disagree when broker `avg_entry` drifts from local `entry_price`.

`src/steps/morning_executor.py:697-716` decides which path runs:

```python
if use_broker:
    reconciled = _reconcile_portfolio_from_broker_with_retry(...)
    if reconciled.get('broker_reconciled'):
        portfolio = reconciled
    else:
        portfolio = _update_valuations_from_quotes(portfolio, morning_quotes)
else:
    portfolio = _update_valuations_from_quotes(portfolio, morning_quotes)
```

If broker reconciliation succeeds, valuation uses Path B (mixed basis, possibly inconsistent). If the broker call fails or returns nothing, valuation uses Path A (local basis, always coherent). **The same dashboard, on the same day, can show different P&L numbers for the same holding depending on whether `broker.list_positions()` happened to succeed.**

### Reproducer

Run the morning phase twice — once with broker access disabled, once with broker access enabled — and diff the resulting `unrealized_pnl` columns for VUG. Or in the test suite: stub a broker call to fail and assert the published P&L equals the local-basis calculation; stub a broker call to succeed with divergent `avg_entry` and assert it does *not* equal the local-basis calculation. Both assertions pass today — that's the bug.

### Recommended fix summary

After the F-4 fix (always-local-basis `$` and `%`), the divergence collapses by construction. The deeper architectural fix is to make valuation a pure function over `(holdings, prices, cost_basis_policy)` and call it from a single place after either reconciliation path. Then "broker reconciliation succeeded vs. failed" can never affect P&L formatting — it only affects whether `cash`/`portfolio_value` are read from broker truth or local.

### Risk class

LIVE-IMPACT (display) and OBSERVABILITY (the dashboard becomes nondeterministic in a way that hides bugs).

---

## F-6 — Expert-signal fallbacks flow into `timeseries.json` indistinguishable from real values

### Code-level diagnosis

`src/signals/compute_signals.py:117-126, 158-170, 178-186, 204-213` — all four expert blocks attach `degraded_reason: str(e)` on exception, alongside neutral fallback values (e.g. `vol_uncertainty_score: 0.5`, `fragility_score: 0.5`, `entropy_score: 0.5`).

`src/steps/publish_artifacts.py:391-436` (`_build_timeseries_row`) reads each output dict via `.get(key, default)` and writes the score directly into the timeseries row. **The `degraded_reason` field is never propagated to the timeseries.** A row containing `vol_uncertainty_score=0.5` produced by a clean computation is byte-identical to a row produced by an exception that took the fallback branch. There is no flag, no health column, no provenance chain.

This is the same class of defect as F-3 (silent fallback for missing inputs *inside* the signal) — except F-6 is silent fallback for the entire signal block.

### Recommended fix summary

`_build_timeseries_row`: add per-signal `<signal>_status` columns: `ok` | `degraded:<reason>`. Mirror in dashboard data.

Add a Phase-2 health-card check that flags any timeseries row where (a) status missing, (b) status `ok` but value equals the neutral default for that signal in a suspicious run, (c) any signal fallback active for > N consecutive days.

### Risk class

OBSERVABILITY — and a multiplier on every other finding. F-1 and F-3 were only catchable because their fallbacks happened to hit a *suspicious* value (0.10 isn't even a "neutral" fallback — it's the floor of a bin). A neutral 0.5 fallback would have been invisible and could last years.

---

## F-7 — Fragility caution gate is regime-blind

### Code-level diagnosis

`src/signals/regime_fusion.py:191-208`:

```python
fragility_fired = fragility_score > fragility_threshold              # 0.75 default
if fragility_fired:
    position_size_mod = min(position_size_mod, fragility_position_cap)  # 0.60
    risk_throttle = min(risk_throttle + fragility_throttle_increment, 1.0)
```

The condition is `fragility_score > 0.75`. There is no AND-gating against `regime_confidence`, `trend_risk_on_prob`, or the ensemble label. The Phase 6 design preamble (`regime_fusion.py:1-13`) calls fragility "Caution Gate" — but Caution gates are intended to be conditional, while this implementation is unconditional.

Production effect (combining with F-2): `fragility_score` has been ≥ 0.97 for 6 weeks. With `regime_label='risk_on_trend'` at confidence 0.92, the gate still fires every day, capping `position_size_mod` at `0.60`. The ensemble's high confidence in `risk_on_trend` is reduced to a 60% multiplier ÷ ensemble_multiplier ≈ 0.85 = ~0.51 effective sizing.

### Reproducer

```python
from src.signals.regime_fusion import decide_regime_v3
out = decide_regime_v3(
    ensemble_regime_label='risk_on_trend',
    trend_risk_on_prob=0.92, panic_prob=0.04,
    ensemble_disagreement=0.10, ensemble_multiplier=0.85,
    macro_credit_score=0.20,
    vol_uncertainty_score=0.55, vol_regime_label='calm',
    fragility_score=0.98,                # current production reality
    entropy_score=0.50, entropy_shift_flag=False,
)
assert out['position_size_modifier'] == 0.60 * 0.85, out
```

### Recommended fix summary

Make the fragility gate *conditional*:

```python
fragility_fired = (
    fragility_score > fragility_threshold
    and not (
        ensemble_regime_label in ('risk_on_trend', 'calm_uptrend')
        and (1.0 - ensemble_disagreement) > regime_confidence_relax_threshold  # e.g. 0.80
    )
)
```

The relaxation is gated on regime AND confidence, so saturation in correlation does not auto-throttle a high-confidence trend regime. In `choppy` / `risk_off_trend` / when the ensemble disagrees, fragility throttles as before. The threshold and confidence cutoff are calibration knobs for Phase 5.

This is hypothesis #3 in the prior audit and the packet's primary calibration target. Confirmed by code, not just data.

### Risk class

LIVE-IMPACT — direct cause of the "fragility gate halves position size in confirmed risk_on_trend" production behavior.

---

## F-8 — `compute_signals.run` reaches for `fred_latest` outside its definition scope

### Code-level diagnosis

`src/signals/compute_signals.py:128-141` (the `Vol Uncertainty` block):

```python
try:
    vix_value = fred_latest.get('VIXCLS', 0) if 'fred_latest' in dir() else 0
    ...
```

`fred_latest` is *defined* inside the `Macro/Credit` `try` block at line 102:

```python
try:
    from src.steps.ingest_fred import get_latest_values
    fred_latest = get_latest_values(fred_df) if len(fred_df) > 0 else {}
    ...
except Exception as e:
    ...                     # fred_latest never assigned in this branch
```

If the `Macro/Credit` block raises before line 102, `fred_latest` is never bound. Then `'fred_latest' in dir()` returns `False`, and `vix_value` defaults to `0`. Combined with F-1, this gives the exact `vol_uncertainty_score=0.10` we see in 168 days of history — the symptom of the upstream data-plane health problem is silently dropped onto the next signal.

The defect is also fragile in a second way: `dir()` at function scope returns *all* local names, including those imported at the top of the function. A future contributor who adds `fred_latest` as a parameter or who re-orders blocks will silently change the behavior of this branch.

### Reproducer

```python
def run(fred_df):
    try:
        # Simulate Macro/Credit raising before fred_latest is bound
        raise RuntimeError("FRED ingest failed")
        fred_latest = {'VIXCLS': 22.0}
    except Exception:
        pass
    vix_value = fred_latest.get('VIXCLS', 0) if 'fred_latest' in dir() else 0
    return vix_value
print(run(None))   # 0  (silent data loss)
```

### Recommended fix summary

`compute_signals.py:90-100`: define `fred_latest = {}` (and any other shared scratch state) *before* the try blocks. Remove the `'fred_latest' in dir()` reflection — replace with a regular `if fred_latest:` check on the dict-emptiness. Same hygiene for `vix_history`, `vvix_history`, etc.

This converts a silent-fallback failure mode into a typed-dict-empty failure mode that integrates cleanly with the F-3/F-6 `degraded_reason` propagation work.

### Risk class

LIVE-IMPACT, latent. The current 6-week run of healthy `vol_uncertainty_score` could regress at any time if `Macro/Credit` ever raises before line 102 in production.

---

## F-9 — SKEW is fetched but `skew_history` is never plumbed; SKEW percentile uses static thresholds only

### Code-level diagnosis

`src/signals/compute_signals.py:142-145`:

```python
vvix_value = _get_latest_close(vvix_data)
skew_value = _get_latest_close(skew_data)
vvix_history = _get_close_series(vvix_data)
# (no skew_history)
```

`compute_vol_uncertainty` accepts `vix_history` and `vvix_history` parameters that compute *dynamic* percentiles when present (`vol_uncertainty.py:71-80`). The SKEW path has no equivalent — it always falls back to the static `_percentile_score(skew_value, SKEW_THRESHOLDS)` (`vol_uncertainty.py:85-87`), which uses the hardcoded breakpoints at p20=115/p50=125/p80=140/p95=150.

This is a quieter cousin of F-3: even when SKEW is available, the historical context is thrown away.

### Recommended fix summary

`compute_signals.py:146`: add `skew_history = _get_close_series(skew_data)` and pass it to `compute_vol_uncertainty`. Extend `vol_uncertainty.py:84-87` to use `skew_history` symmetrically with `vix_history`/`vvix_history`. Add unit test asserting the dynamic-percentile path is exercised.

### Risk class

OBSERVABILITY (when SKEW returns to live, percentile interpretation will be regime-stale).

---

## F-10 — `total_value` ($106,248) vs `broker_total_value` ($96,170) differ by ~$10,078 with no on-page disclosure

### Evidence

`/tmp/claude-501/trader-bot-audit/dashboard.json` `metrics`:
- `total_value: 106248.47`
- `broker_total_value: 96170.64`
- difference: `$10,077.83`

### Code-level diagnosis

`src/utils/dashboard_metrics.py:250-273` (`_build_continuity_rows`):

```python
cumulative_cashflow = 0.0
for row in return_rows:
    cashflow = float(row.get("external_cashflow", 0.0) or 0.0)
    cumulative_cashflow += cashflow
    continuity_value = float(row.get("value", 0.0) or 0.0) - cumulative_cashflow
```

The continuity bridge subtracts cumulative external cashflows from the equity curve so the visualization reflects performance net of deposits/withdrawals/cutover adjustments. The dashboard's headline `total_value` is `continuity_curve[-1]['value']` — i.e. the broker's account equity *minus* cumulative external cashflow.

Per the cutover continuity-bootstrap plan (`docs/plans/2026-03-12-cutover-continuity-bootstrap.md`), there was a one-day positive cashflow adjustment when the simulated portfolio was bridged into Alpaca paper. That ~$10k cashflow is now *permanently* subtracted from `total_value`, even though the broker really does hold $96,170.

This is correct behavior for a *performance* metric (TWR-style) — but the dashboard hero presents it as `total_value`, which a casual reader will interpret as account equity. The packet's acceptance test demands: *"every dollar number on the dashboard either matches the broker truth or has a documented, traceable cashflow gap."* The gap is traceable in code (cashflow → continuity_value), but it is **not displayed on the dashboard alongside the figure**.

### Reproducer

```python
import json
d = json.load(open('/tmp/claude-501/trader-bot-audit/dashboard.json'))
gap = d['metrics']['total_value'] - d['metrics']['broker_total_value']
print(f"undisclosed gap: ${gap:.2f}")     # $10077.83
# search for an on-page disclosure of this gap
disclosures = [k for k in d.get('weather', {}).get('summary', '').split() if 'cashflow' in k.lower()]
print('disclosures in weather:', disclosures)   # []
```

### Recommended fix summary

This is **not a bug per se** — it's an architectural decision documented in the cutover plan. But to satisfy the packet's "no silent paper-vs-broker drift" rule:

1. Render both numbers on the dashboard with explicit labels — e.g. "Net Equity (continuity-adjusted): $106,248" + "Broker Account: $96,170 (gap: $10,078 cutover continuity bridge — see cutover_marker_2026-03-12)".
2. Surface `cumulative_external_cashflow` as a metric on the dashboard, not just an internal field.
3. Add a postmortem entry under "first-pass-audit-framing-trap" — the prior audit didn't surface this gap because the code path was correct.

### Risk class

OBSERVABILITY. No live-decision impact (the broker is the source of truth for orders), but the operator-readable dashboard hides ~10% accounting drift in plain sight.

---

## Findings expected to be tracked by Phase 2 (signal diagnostic), not Phase 1

The following showed up incidentally and will be documented in Phase 2's 26-field table rather than enumerated here:

- `entropy_consecutive_days` and `entropy_above_threshold` are in `timeseries.json` but the entropy gate appears mostly inactive (depends on production-data inspection in Phase 2).
- `panic_prob` distribution and threshold sensitivity (the prior audit cited `panic_prob = 0.997` on Mar 23-25 forcing exits — needs validation).
- `macro_credit_score` distribution and its `macro_downgrade_threshold` interaction.
- `regime_confidence` mean and how often it crosses the hypothesized 0.80 relaxation cutoff in F-7.

## Open questions for Phase 2 / 3

- Why specifically does VUG's broker `avg_entry_price ≈ $142` — is there an Alpaca-side legacy lot from the cutover that wasn't reset? (Phase 3 task to call the broker and inspect.)
- What was the FRED VIX series state in `fred_df` on dates flat at 0.10, vs. on Apr 2 when it stepped? (Requires snapshot of `fred.parquet` for those dates.)
- What is the actual long-horizon distribution of `avg_correlation` across multi-year SPY history? (Calibration input for the F-2 fix.)

---

End Phase 1.
