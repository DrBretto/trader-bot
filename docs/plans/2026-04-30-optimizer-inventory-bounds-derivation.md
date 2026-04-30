# Phase 1 — Optimizer inventory bounds derivation

Date: 2026-04-30
Packet: `docs/plans/2026-04-30-optimizer-empirical-mutation-packet.md`
Output: updated `optimizer/discovery/param_inventory.json`.

## §A — Method

For each constant in `optimizer/discovery/param_inventory.json` whose `where_defined` source file uses it as a normalization input (z-score mean/std, or percentile bin baseline), tag with `parameter_class: "normalization_constant"`. Compute empirical bounds from the optimizer's loaded dataset (`optimizer/data_access.load_optimizer_dataset` with `max_days=900`, returning 177 aligned snapshots from `s3://investment-system-data/daily/` 2025-08-04 → 2026-04-30).

Bounds derivation rules:
- **Mean constant** (e.g. `AVG_CORR_MEAN`): `min = p1(input)`, `max = p99(input)`. Input filtered to non-zero rows (zero is the documented neutral fallback per F-1 / F-3).
- **Std constant** (e.g. `AVG_CORR_STD`): `min = max(1e-3, empirical_std * 0.25)`, `max = empirical_std * 4.0`. 16x search range centered geometrically on the empirical std.
- **Percentile-bin constant** (e.g. `vol_uncertainty.vix_thresholds.p20`): bounds = ±N around the historical realized percentile (informed by the prior audit's 1272-day yfinance VIX/VVIX/SKEW history).

Two flavors of tag are produced:

1. `parameter_class: "normalization_constant"` + `empirical_statistic: "<expr>"` — input field is in the optimizer's `signal_row` and the empirical-mutation operator (Phase 2) can compute the statistic at runtime.
2. `parameter_class: "normalization_constant"` + `empirical_statistic: null` — bounds tightened from historical knowledge but the input is not in `signal_row` (e.g., raw VIX/VVIX/SKEW values are not stored in the daily signal artifact). The empirical-mutation operator skips these gracefully; they get tighter bounds at least, so random mutation has a chance.

## §B — In-scope normalization constants (8 with empirical_statistic, 12 tag-only)

### Fragility (`src/signals/fragility.py`)

| name                       | current | new bounds         | empirical mean | empirical_statistic                          |
|----------------------------|--------:|--------------------|---------------:|-----------------------------------------------|
| `fragility.AVG_CORR_MEAN`  |   0.30  | [0.3881, 0.5999]   | **0.4847**     | `mean(avg_correlation, exclude_zero=True)`    |
| `fragility.AVG_CORR_STD`   |   0.15  | [0.0181, 0.2893]   | 0.0723         | `std(avg_correlation, exclude_zero=True)`     |
| `fragility.PC1_MEAN`       |   0.45  | [0.5206, 0.6870]   | **0.5929**     | `mean(pc1_explained, exclude_zero=True)`      |
| `fragility.PC1_STD`        |   0.12  | [0.0150, 0.2408]   | 0.0602         | `std(pc1_explained, exclude_zero=True)`       |

The current values are mostly **outside the new bounds** — the search space was so wide that the GA never converged on the empirical band. With these bounds in place, random mutation can only produce values within the empirical realistic range; the empirical-mutation operator (Phase 2) injects the exact mean as one candidate per generation.

Verification gate (per packet §Verification gate): `AVG_CORR_MEAN ≈ 0.477 ± 0.02`. **Empirical computed = 0.4847**, +0.0077 from the audit's 900-day yfinance reference (0.4774). Within tolerance. ✓

### Macro / credit (`src/signals/macro_credit.py`, defaults at `optimizer/replay.py:79-82`)

| name                            | current | new bounds         | empirical mean | empirical_statistic                              |
|---------------------------------|--------:|--------------------|---------------:|---------------------------------------------------|
| `macro_credit.SLOPE_MEAN`       |   1.50  | [0.5500, 0.6655]   | **0.6044**     | `mean(yield_slope_10y_3m, exclude_zero=True)`     |
| `macro_credit.SLOPE_STD`        |   1.00  | [0.0082, 0.1306]   | 0.0327         | `std(yield_slope_10y_3m, exclude_zero=True)`      |
| `macro_credit.HY_SPREAD_MEAN`   |   0.00  | [-0.0235, 0.0167]  | -0.0005        | `mean(hy_spread_proxy, exclude_zero=False)`       |
| `macro_credit.HY_SPREAD_STD`    |   0.02  | [0.0032, 0.0505]   | 0.0126         | `std(hy_spread_proxy, exclude_zero=False)`        |

`SLOPE_MEAN=1.50` is dramatically out of the empirical band — same class of staleness as the fragility constants. `SLOPE_STD=1.00` is similarly miles too wide. The macro-credit normalization has the same disease the fragility audit surfaced; the empirical-mutation operator will propose corrected values on the first run after promotion.

**Caveat**: only 16 non-zero `yield_slope_10y_3m` samples (post-2026-04-02 DGS3MO availability). Bounds are honest given the data we have. Re-derive these specific bounds in 2026-Q3 once 6+ months of post-fix data has accumulated.

### Volatility complex (`src/signals/vol_uncertainty.py`)

These are bin boundaries in a piecewise-linear percentile mapping. Tagged but with `empirical_statistic: null` because the optimizer's `signal_row` does not store raw `vix_value` / `vvix_value` (only the percentiles after binning). The empirical-mutation operator skips these; tighter bounds restrict random mutation to historically-plausible regions.

| name                                        | current | new bounds       | source rationale                                                |
|---------------------------------------------|--------:|------------------|------------------------------------------------------------------|
| `vol_uncertainty.vix_thresholds.p20`        |    13.0 | [9.0, 16.0]      | VIX p20 historically near 13. Bounds ±3.                         |
| `vol_uncertainty.vix_thresholds.p50`        |    17.0 | [14.0, 21.0]     | VIX median historically 17.93 (1272d yfinance). Bounds ±3.       |
| `vol_uncertainty.vix_thresholds.p80`        |    25.0 | [21.0, 30.0]     | VIX p80 historically near 25. Bounds ±4.                         |
| `vol_uncertainty.vix_thresholds.p95`        |    30.0 | [25.0, 40.0]     | VIX p95 historically near 30. Bounds spans normal-to-stress.     |
| `vol_uncertainty.vvix_thresholds.p20`       |    75.0 | [65.0, 85.0]     | VVIX p20 historically near 75. ±10.                              |
| `vol_uncertainty.vvix_thresholds.p50`       |    85.0 | [75.0, 95.0]     | VVIX p50 historically near 85. ±10.                              |
| `vol_uncertainty.vvix_thresholds.p80`       |   105.0 | [95.0, 115.0]    | VVIX p80 historically near 105. ±10.                             |
| `vol_uncertainty.vvix_thresholds.p95`       |   120.0 | [110.0, 140.0]   | VVIX p95 historically near 120.                                  |
| `vol_uncertainty.skew_thresholds.p20`       |   115.0 | [105.0, 125.0]   | SKEW p20 historically near 115. ±10.                             |
| `vol_uncertainty.skew_thresholds.p50`       |   125.0 | [115.0, 135.0]   | SKEW p50 historically near 125. ±10.                             |
| `vol_uncertainty.skew_thresholds.p80`       |   140.0 | [130.0, 150.0]   | SKEW p80 historically near 140. ±10.                             |
| `vol_uncertainty.skew_thresholds.p95`       |   150.0 | [140.0, 160.0]   | SKEW p95 historically near 150. ±10.                             |

These don't have `empirical_statistic` because `vix_value` etc. aren't in the optimizer's `signal_row` (only `vix_percentile`). To upgrade to full empirical-mutation in the future, plumb raw VIX/VVIX/SKEW values through to `signal_row` (out of this packet's scope; logged as a follow-up).

## §C — Out of scope (left as `decision_threshold`)

Constants in fragility / macro / vol that are NOT normalization constants:

- `entropy_shift.z_threshold` (a decision cutoff, not a normalization mean/std)
- `vol_uncertainty.regime_thresholds.{panic,unstable}.*` (percentile-on-percentile, not normalization of an input)
- `vol_uncertainty.dynamic_history_min_obs` (sample-count minimum, not a normalization)
- `macro_credit.{slope_weight, hy_spread_weight}` (weighting fractions with sum-to-1 constraint)
- `*.window_days`, `*.MIN_DAYS`, `*.MIN_SYMBOLS`, etc. (configuration thresholds)

These remain tagged `decision_threshold` and use the existing bounds-and-mutation logic.

## §D — Empirical-bounds JSON snapshot

The full empirical computation is preserved at `$TMPDIR/optimizer_empirical_bounds.json`:

```json
{
  "computed_at": "2026-04-30",
  "dataset_size": 177,
  "date_range": "2025-08-04 to 2026-04-29",
  "inputs": {
    "avg_correlation":     {"n_nonzero": 52, "mean": 0.4847, "std": 0.0723, "p1": 0.3881, "p99": 0.5999},
    "pc1_explained":       {"n_nonzero": 52, "mean": 0.5929, "std": 0.0602, "p1": 0.5206, "p99": 0.6870},
    "yield_slope_10y_3m":  {"n_nonzero": 16, "mean": 0.6044, "std": 0.0327, "p1": 0.5500, "p99": 0.6655},
    "hy_spread_proxy":     {"n_nonzero": 52, "mean": -0.0005, "std": 0.0126, "p1": -0.0235, "p99": 0.0167}
  }
}
```

## §E — Exit criterion

Every fragility / macro / vol normalization constant has a `parameter_class` tag, derived bounds, and a named statistic (or null with rationale). The verification target `AVG_CORR_MEAN ≈ 0.477 ± 0.02` is computable from the dataset (filtered mean = 0.4847, in tolerance). Phase 2 (empirical re-derivation operator) can now build on this schema.
