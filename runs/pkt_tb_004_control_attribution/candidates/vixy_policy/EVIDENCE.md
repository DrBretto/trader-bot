# VIXY policy candidate — evidence base (Risk Architect, PKT-TB-004)

Assembled 2026-06-10, before variant tuning. All strategy-performance reads are
pre-holdout (< 2026-03-11) or non-performance market/production facts.

## 1. Structural decay is real and large

Source: S3 daily prices (`runs/pkt_tb_004_control_attribution/cache/prices_20260610.parquet`,
range 2025-06-11..2026-06-09, 250 sessions).

- Buy-and-hold VIXY: **-50.6% total / -50.9% annualized**.
- Pre-holdout segment (2025-06-11..2026-03-10): **-33.5%**.
- Rolling 21-day buy-and-hold: **median -8.6%**, mean -5.0%, only **26%** of
  windows positive. The bleed is the asset's normal state (futures roll /
  contango), not an episode.
- Spikes are real but short: best 10-day window **+25.7%** (2026-03-06 panic
  cluster); month-level +21.7% (2026-03) immediately followed by -18.9% and
  -15.1% (2026-04/05). Any hold past the spike-capture window is negative-sum.

## 2. The strategy's existing controls do NOT model the decay

- `universe.csv`: `VIXY,vol,volatility,0,1` — **leverage_flag=0**, so the
  leveraged hold cap (`leveraged_constraints.max_hold_days=10`) does not apply.
  VIXY is the only `sector=volatility` symbol.
- `regime_compatibility` vol multipliers: 0.85 (calm_uptrend) .. 1.30
  (high_vol_panic) — a scoring tilt, never an exclusion, and it *boosts* VIXY
  in panic while the panic force-sell (asset_class `vol` not in
  [bond, commodity]) simultaneously force-SELLS it and the panic buy filter
  blocks buying it. The stack treats VIXY as a risk asset in exactly the
  regime where it is the hedge — it can only be bought in benign regimes,
  i.e. during the bleed.
- Health score is structurally blind to roll decay. Production night run
  2026-06-04 (S3 `daily/2026-06-04/decisions.json` watchlist):
  `health_score=0.954`, `vol_bucket="low"`, while `return_21d=-15.3%`,
  `return_63d=-17.6%`. A steady bleed reads as "low-vol, healthy,
  mean-reversion" to the scorer — score 0.68 cleared the buy bar and the sim
  book bought 219 sh @ ~23.47 (~$5.1k).
- LLM risk layer saw VIXY on 2026-06-03/04/05 and flagged only generic
  "liquidity"/"regulatory" (severity 2, no veto, confidence_adjustment
  0.1-0.2). Contango bleed was never named. The Haiku layer does not catch it.
- The trailing stop (10% from peak) does eventually fire on a decaying
  holding, but only after surrendering up to 10% from each local peak —
  it bounds episodes, not the policy gap.

## 3. Exposure in the books

- **Canon line (dashboard.json, 2026-06-09):** VIXY held in two lots,
  ~$10.6k = **~9.3% of a $114k book** (entries ~23.57 / ~22.955, currently
  +$464 unrealized on the early-June vol uptick).
- **Sim/intent book (S3 portfolio_state):** bought 2026-06-04, no VIXY held
  as of 2026-06-09 (book diverges from canon line).
- **Optimizer-harness C00 baseline (5 seeds, full period):** VIXY is
  **never bought** — 0 fills in 26 traded symbols. The replay path from
  $100k/2025-08-04 never has VIXY clear the buy bar.

## 4. What this implies for the candidate design

- **Universe exclusion (eligible=0)** is over-broad given the data (it would
  also be a `config/universe.csv` change, outside this packet's write
  surface): the spike capture is real (+25%/10d) and exclusion forecloses it
  permanently. Not chosen.
- **Decay budget (sell at cumulative -X% vs entry)** duplicates the trailing
  stop with a worse trigger (entry-anchored instead of peak-anchored).
  Not chosen.
- **Hold cap** (chosen): `decision_params['vol_decay_constraints'] =
  {'max_hold_days': N, 'sectors': ['volatility']}` — calendar cap analogous
  to the leveraged hold cap, measured against the price data's as-of date so
  it is replay-consistent (the leveraged cap uses wall-clock `now()` and is
  therefore distorted in replays — flagged to the chair separately). It
  preserves spike capture (spikes resolve inside 2-3 weeks), bounds the
  bleed (median 21d hold = -8.6%), and needs no new scoring machinery.
  N=10 trading-equivalent calendar days ~ 2 weeks; grid {5, 10, 15, 21}.

## 5. Honest limitation (stated up front)

The optimizer replay never buys VIXY, so the E1/E2 counterfactual for this
candidate is expected to be **exactly zero delta** — the replay can prove
harmlessness, not benefit. The benefit case lives on the canon line (9.3% of
the book today) and in any future path where the scorer again rates a
bleeding VIXY at health 0.95. The verdict this evidence supports is
"cheap structural insurance, proven harmless in replay," not
"proven-improvement."
