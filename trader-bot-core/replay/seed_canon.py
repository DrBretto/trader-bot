"""P6 — seed the clean canon ledger by REPLAY (PKT-TRADER-BOT-SEED-CANON-BY-REPLAY).

The payoff of the clean rebuild: the corrected line. This orchestrates P4's replay
keystone + P5's ledger into the operator's LOCKED SCOPE:

  1. Reconstruct ONLY the post-split window (2026-06-12 .. latest settled) with the
     corrected spine + the RECORDED regime the algorithm consumed (NOT the
     deterministic picker — that is forward-only). Picks are RECOMPUTED, never
     re-marked.
  2. The pre-split line (<= 2026-06-11 frozen champion, terminal 114772.39) is KEPT
     continuous and BYTE-UNCHANGED — it is copied verbatim (same content_sha) and
     never reconstructed (a different, non-reproducible model).
  3. The reconstruction ANCHORS CONTINUOUSLY at the pre-split terminal at the split
     point — canon+comparison at 114772.39, benchmark at its own SPY terminal — so
     there is NO seam at the join.
  4. Fixes apply FORWARD, not retroactively — the reconstruction uses the substrate
     as it was (GDELT frozen at its 06-10 seed, CBOE null post-06-09), never the new
     forward-fetch capabilities.
  5. Ship the accurate line + REPORT the delta vs the old contaminated line as
     information (no sign-off gate).

WHY A SEPARATE CLEAN LEDGER (not an in-place supersede of ``canon/equity_ledger/``):
the production ledger's post-split grid is itself contaminated — it carries PHANTOM
weekend/holiday leaves (2026-06-13/-19/-20/-27, none of which have a settled SPY
bar) and is MISSING real trading days (2026-06-26, -29). The append-only + supersede
ledger has NO delete, so an in-place transform to the real trading-day grid is
impossible; and "production repoint is P9" means production must stay untouched
here. So P6 seeds a SEPARATE clean ledger (a distinct S3 prefix) on the REAL
trading-day grid: pre-split byte-copied, post-split reconstructed. Production's
ledger is left fully intact (the most non-destructive form of "supersede the
contaminated ones" — the corrected artifact supersedes the contaminated one as the
go-forward source of truth, formalized at the P9 cutover). The DELTA_REPORT records
the old-vs-corrected difference.

FORWARD-NIGHTLY (the ONE path): ``forward_nightly`` appends exactly ONE settled leaf
per day onto the corrected frontier, ``forward_confirmed=True``, using the
DETERMINISTIC ``forecast.regime.regime(D)`` picker (forward-only). No prior settled
leaf moves. Reconstruction and forward share the SAME replay marking loop
(``replay.driver.replay``) and the SAME ``run_engine`` selection, so replaying a
recent day reproduces the forward path's leaf (replay==forward).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from replay.driver import _OHLCVStore, replay, ReplayResult, CORE_ROOT
from forecast import recorded_regime as RR

# The split point (pre-split terminal / continuous-anchor date). The frozen
# champion owns <= this date (byte-immutable, non-reproducible model); the
# reconstruction owns forward of it. Matches decide.cutover.DEFAULT_FORWARD_BOUNDARY.
SPLIT_DATE = "2026-06-11"
FIRST_POST_SPLIT = "2026-06-12"

ISSUED_BY = "seed-canon-by-replay@trader-bot"

# The production (contaminated) ledger prefix and the NEW clean ledger prefix.
PROD_PREFIX = "canon/equity_ledger/"
CLEAN_PREFIX = "canon/equity_ledger_clean/"

# The incumbent brain's ACTIVE decision params + regime_compatibility (what the
# legacy production incumbent trades) — the SAME source the live shadow read
# (shadow_nightly.build_production_ctx). The INDEPENDENT challenger's incumbent
# ranker consumes these verbatim.
INCUMBENT_CONFIG_KEY = "config/decision_params.active.json"


def load_incumbent_config(s3=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(decision_params, regime_compatibility) for the ported incumbent, read from
    the live S3 config the production incumbent uses. Fail-loud (the packet STOP
    condition): a missing/garbled config is surfaced, never stubbed."""
    import boto3
    s3 = s3 or boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    obj = json.loads(s3.get_object(Bucket=RR.S3_BUCKET, Key=INCUMBENT_CONFIG_KEY)["Body"].read())
    dparams = obj.get("decision_params")
    rcompat = obj.get("regime_compatibility") or {}
    if not dparams:
        raise RuntimeError(
            f"incumbent config {INCUMBENT_CONFIG_KEY} has no decision_params — "
            f"STOP; the challenger incumbent cannot run without its frozen params")
    return dparams, rcompat


def build_independent_challenger(a_comp: float, s3=None, log_dir=None):
    """Construct the INDEPENDENT challenger (ported incumbent + M1 tilt) anchored at
    the pre-split comparison terminal ``a_comp``. Bound to the EXTENDED core OHLCV
    store + the organ_inputs dir ``forecast.inference`` writes fresh as-of-D."""
    from forecast import shadow_lib as SL
    from challenger.independent import IndependentChallenger
    dparams, rcompat = load_incumbent_config(s3=s3)
    return IndependentChallenger(
        cash_anchor=a_comp, organ_dir=SL.ORGAN_DIR, ohlcv_dir=SL.CACHE_OHLCV,
        decision_params=dparams, regime_compat=rcompat, log_dir=log_dir)


# --------------------------------------------------------------------------- #
# substrate — extend the OHLCV store from S3 so the post-split window can be
# marked + forecast (the seed store is frozen at 2026-06-10; the settled bars
# come from S3 daily/<D>/prices.parquet, exactly like the forward path).
# --------------------------------------------------------------------------- #
def prepare_substrate(state_dir: Optional[str] = None) -> _OHLCVStore:
    """Seed the writable STATE caches and splice the post-split settled bars from
    S3 into the OHLCV store. Returns an ``_OHLCVStore`` bound to the EXTENDED store
    (so marks + the trading-day grid reach the settled window). GDELT stays frozen
    at its 06-10 seed (substrate as-it-was — fixes are forward-only)."""
    if state_dir:
        os.environ["BRAIN_STATE_DIR"] = str(state_dir)
    from forecast import shadow_lib as SL
    from forecast import inference as FI
    from forecast import features_gdelt as FG
    from store.ohlcv_store import _extend_ohlcv_from_s3

    SL.STATE.mkdir(parents=True, exist_ok=True)
    FI.ensure_seed_caches()
    _extend_ohlcv_from_s3(SL, FI)                       # splice settled post-split bars
    out = SL.STORE / "gdelt_features.parquet"
    if not out.exists():
        FG.build_gdelt_features(daily_dir=SL.GDELT_DAILY_DIR, out_path=out)
    return _OHLCVStore(root=SL.CACHE_OHLCV)


def latest_settled(ohlcv: _OHLCVStore, on_or_before: Optional[str] = None) -> str:
    """The most recent settled trading day present in the (extended) SPY grid."""
    days = [d for d in ohlcv.trading_days() if (on_or_before is None or d <= on_or_before)]
    if not days:
        raise RuntimeError("no settled trading days in the OHLCV grid")
    return days[-1]


def post_split_window(ohlcv: _OHLCVStore, d1: Optional[str] = None) -> List[str]:
    """The REAL settled trading-day grid of the post-split window (strictly after
    the split date, through ``d1`` or the latest settled day). Derived from the
    settled SPY bars — so phantom weekend/holiday folders never enter the grid."""
    hi = d1 or latest_settled(ohlcv)
    return [d for d in ohlcv.trading_days() if SPLIT_DATE < d <= hi]


# --------------------------------------------------------------------------- #
# pre-split — byte-identical copy of the frozen-champion leaves
# --------------------------------------------------------------------------- #
def copy_presplit(prod_ledger, clean_ledger, boundary: str = SPLIT_DATE) -> List[dict]:
    """Copy every prod leaf dated <= ``boundary`` into the clean ledger
    BYTE-IDENTICAL (same content_sha, same body), then rebuild the clean manifest +
    cache from those leaves. The pre-split line is preserved verbatim — never
    reconstructed. Returns the copied leaves."""
    copied = []
    for leaf in prod_ledger.list_leaves():
        if leaf.get("date", "") <= boundary:
            clean_ledger._put_leaf_write_once(leaf)     # write-once, content-addressed → byte-identical
            copied.append(leaf)
    clean_ledger.rebuild(write=True)                    # supersede-aware manifest+cache fold
    return copied


def anchor_from_frontier(ledger, boundary: str = SPLIT_DATE) -> Tuple[float, float, float]:
    """Return (canon, benchmark, comparison) anchors = the boundary leaf's displayed
    values (the pre-split terminal the reconstruction joins continuously)."""
    for leaf in ledger.list_leaves():
        if leaf.get("date") == boundary:
            comp = leaf.get("comparison")
            return (float(leaf["value"]), float(leaf["benchmark"]),
                    float(comp if comp is not None else leaf["value"]))
    raise RuntimeError(f"no boundary leaf at {boundary} to anchor on")


# --------------------------------------------------------------------------- #
# regime functions
# --------------------------------------------------------------------------- #
def recorded_regime_fn(dates: List[str], s3=None) -> Callable[[str], str]:
    """A regime function that returns the RECORDED regime the algorithm consumed for
    each reconstruction day (fail-loud on any unresolved day). Verified against
    ``recorded_regime.VERIFIED_WINDOW``."""
    rmap = RR.recorded_regime_map(dates, s3=s3, verify=True)

    def _fn(D: str) -> str:
        if D not in rmap:
            raise RR.RecordedRegimeUnavailable(
                f"recorded regime requested for {D} outside the resolved window")
        return rmap[D]
    _fn.regime_map = rmap                                # expose for the receipt
    return _fn


def mixed_regime_fn(recorded_map: Mapping[str, str]) -> Callable[[str], str]:
    """History days use the RECORDED regime; any day NOT in the recorded map (a
    forward day) uses the DETERMINISTIC as-of-D picker on the extended seed store.
    This is the ONE go-forward path: recorded for the reconstructed window,
    deterministic forward of it."""
    from forecast import shadow_lib as SL
    from forecast.regime import regime as _regime

    ext_seed = SL.STATE / "cache"                        # extended {ohlcv,fred,cboe}

    def _fn(D: str) -> str:
        if D in recorded_map:
            return recorded_map[D]
        return _regime(D, seed_root=ext_seed)
    return _fn


# --------------------------------------------------------------------------- #
# the reconstruction (recorded regime, continuous anchor, correct grid)
# --------------------------------------------------------------------------- #
def seed_canon_by_replay(
    clean_ledger,
    ohlcv: _OHLCVStore,
    *,
    d1: Optional[str] = None,
    regime_fn: Optional[Callable[[str], str]] = None,
    state_dir: Optional[str] = None,
    selected_universe_sink: Optional[str] = None,
    independent_challenger: bool = True,
    s3=None,
) -> ReplayResult:
    """Reconstruct the post-split window onto the clean ledger (which already holds
    the byte-unchanged pre-split leaves through ``SPLIT_DATE``). Recorded regime,
    recomputed picks, continuous anchor at the pre-split terminal, one leaf per real
    settled trading day. Non-destructive: append-only at the frontier.

    ``independent_challenger`` (default True, the V2 correction): the blue-dotted
    line is the ported incumbent (decision_engine) selection + the ported M1 tilt,
    run INDEPENDENTLY (NOT the coupled ``publish.challenger`` M1-tilt-of-two-stage).
    The challenger anchors at the pre-split comparison terminal and is marked by the
    SAME settled-close machinery as canon/SPY."""
    window = post_split_window(ohlcv, d1)
    if not window:
        raise RuntimeError("empty post-split window — nothing to reconstruct")
    a_canon, a_bench, a_comp = anchor_from_frontier(clean_ledger, SPLIT_DATE)
    rfn = regime_fn or recorded_regime_fn(window)
    chal = build_independent_challenger(a_comp, s3=s3) if independent_challenger else None
    return replay(
        window[0], window[-1], ledger=clean_ledger, state_dir=state_dir,
        issued_by=ISSUED_BY, regime_fn=rfn, ohlcv=ohlcv,
        write_genesis=False, genesis_date=SPLIT_DATE,
        genesis_anchor=a_canon, bench_anchor=a_bench, comparison_anchor=a_comp,
        segment="new_brain", source="native_two_stage", model_id_prefix="replay@",
        selected_universe_sink=selected_universe_sink, challenger=chal,
    )


# --------------------------------------------------------------------------- #
# forward-nightly (the ONE path) — append exactly one settled leaf/day
# --------------------------------------------------------------------------- #
def forward_nightly(
    clean_ledger,
    ohlcv: _OHLCVStore,
    D: str,
    recorded_map: Mapping[str, str],
    *,
    state_dir: Optional[str] = None,
    independent_challenger: bool = True,
    s3=None,
) -> dict:
    """Append the next settled day ``D`` onto the corrected frontier as a single
    forward leaf (``forward_confirmed=True``), using the DETERMINISTIC regime picker
    for ``D``. No prior settled leaf moves (append-only at the frontier).

    The value is produced by replaying the FULL window [first-post-split .. D] to a
    scratch ledger (so the book chains correctly across the reconstructed days,
    recorded regime for history + deterministic for D), then appending ONLY the
    terminal (D) leaf to the real clean ledger. Because the marking loop and
    run_engine are the SAME as the reconstruction, replaying D reproduces this
    forward leaf (replay==forward)."""
    from replay._fake_s3 import FakeS3
    from lines.ledger import EquityLedger

    front = clean_ledger.frontier()
    if front is not None and D <= front["date"]:
        raise RuntimeError(f"forward date {D} is not past the frontier {front['date']}")

    a_canon, a_bench, a_comp = anchor_from_frontier(clean_ledger, SPLIT_DATE)
    rfn = mixed_regime_fn(recorded_map)

    # scratch full-window replay to compute D's chained displayed values
    scratch = EquityLedger(FakeS3(), prefix="scratch/fwd/")
    chal = build_independent_challenger(a_comp, s3=s3) if independent_challenger else None
    res = replay(
        FIRST_POST_SPLIT, D, ledger=scratch, state_dir=state_dir,
        issued_by=ISSUED_BY, regime_fn=rfn, ohlcv=ohlcv,
        write_genesis=True, genesis_date=SPLIT_DATE,
        genesis_anchor=a_canon, bench_anchor=a_bench, comparison_anchor=a_comp,
        segment="new_brain", source="native_two_stage", model_id_prefix="forward@",
        challenger=chal,
    )
    term = res.terminal_leaf
    if term.get("date") != D:
        raise RuntimeError(f"scratch replay terminal is {term.get('date')}, expected {D}")

    # append ONLY day D onto the real corrected frontier, tagged forward_confirmed
    return clean_ledger.append(
        date=D, value=term["value"], benchmark=term["benchmark"],
        comparison=term["comparison"], segment="new_brain",
        model_id=term["model_id"], source="native_two_stage", issued_by=ISSUED_BY,
        extra={"forward_confirmed": True},
    )
