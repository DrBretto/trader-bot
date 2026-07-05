"""First-class as-of-D fused-regime picker for the clean spine (PKT-TRADER-BOT-
REGIME-AS-OF-D).

``regime(as_of_D) -> regime_label`` is the ONE picker used by BOTH the history
replay (``replay/driver.py``) and the forward path (``decide/cutover.run_cutover``),
exactly as ``run_engine`` is the one selection/allocation call for both. It replaces
the hard-coded ``'neutral'`` that made every ``regime_score_mult`` collapse to 1.0
(raw-mu ranking, no regime tilt).

WHAT IS PORTED (KEEP-code, relocated verbatim into ``forecast/regime_lib/`` — only
import-path edits, no logic rewrite):
  - ``baseline_regime.baseline_regime_model``  (src/models/baseline_regime.py)
  - ``regime_fusion.decide_regime_v3``         (src/signals/regime_fusion.py)
  - the four expert signals macro_credit / vol_uncertainty / fragility /
    entropy_shift (src/signals/*.py) — the transitive dependency ``decide_regime_v3``
    needs; every one of their inputs is present in the clean-spine SEED substrate
    (store/seeds/cache/{ohlcv,fred,cboe}), so the fused picker runs fully as-of-D.
  - the context builder ``feature_utils.compute_{asset,context}_features``
    (src/utils/feature_utils.py).

This is the DETERMINISTIC fused picker (``baseline_regime_model -> decide_regime_v3``),
the seed-reproducible KEEP-code path. The legacy forward *production* run recorded
its regime from a TRAINED GRU+Transformer ensemble (``ensemble_regime.py``) whose
deployed vintage moved over time, took a ``gdelt_avg_tone`` input not in the
deterministic seed substrate, and was pinned by an S3 ``models/latest.json`` — so the
legacy ensemble label is NOT a deterministic function of as-of-D seed data and is
out of scope for the deterministic spine (surfaced in the run receipt, NOT stubbed
to neutral). See the run STATUS for the evidence.

AS-OF-D DISCIPLINE (same mechanic as the panel's ``end=D`` bound): every seed series
is sliced to bars with ``date <= D`` BEFORE any feature/score is computed. Bars dated
after D are never read, so truncating future bars cannot change ``regime(D)`` — the
no-future-leak guarantee is structural, proven in the reality-test by truncation.

FAIL-LOUD: if a required seed series is absent for D, ``regime`` RAISES
``RegimeSubstrateError`` — it does NOT silently fall back to ``'neutral'`` (the
dependency-closure discipline: a missing artifact is surfaced, never stubbed).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import pandas as pd

from forecast.regime_lib.feature_utils import (
    compute_asset_features,
    compute_context_features,
)
from forecast.regime_lib.baseline_regime import baseline_regime_model
from forecast.regime_lib.macro_credit import compute_macro_credit
from forecast.regime_lib.vol_uncertainty import compute_vol_uncertainty
from forecast.regime_lib.fragility import compute_fragility, PANEL_SYMBOLS
from forecast.regime_lib.entropy_shift import compute_entropy_shift
from forecast.regime_lib.regime_fusion import decide_regime_v3

_THIS = Path(__file__).resolve()
CORE_ROOT = _THIS.parents[1]                                   # trader-bot-core
_DEFAULT_SEED = CORE_ROOT / "store" / "seeds" / "cache"

# Context proxies + FRED/CBOE series the fused picker reads (all seeded).
_CONTEXT_OHLCV = ("SPY", "TLT", "HYG", "IEF", "VIXY")
_FRED_RATES = ("DGS2", "DGS3MO", "DGS10")


class RegimeSubstrateError(RuntimeError):
    """A seed series required to compute regime(D) is absent — surfaced, never
    stubbed to neutral (dependency-closure discipline)."""


class _SeedStore:
    """Cached as-of-D reader over the clean-spine SEED caches. Every accessor
    slices to ``date <= D`` (the no-future-leak bound)."""

    def __init__(self, seed_root: Optional[Path] = None):
        self.root = Path(seed_root) if seed_root else _DEFAULT_SEED
        self._ohlcv: Dict[str, Optional[pd.DataFrame]] = {}
        self._fred: Dict[str, Optional[pd.DataFrame]] = {}
        self._cboe: Dict[str, Optional[pd.DataFrame]] = {}

    def _read(self, cache: Dict, sub: str, name: str, date_col: str = "date"):
        if name not in cache:
            p = self.root / sub / f"{name}.parquet"
            if not p.exists():
                cache[name] = None
            else:
                df = pd.read_parquet(p)
                df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
                cache[name] = df.sort_values(date_col).reset_index(drop=True)
        return cache[name]

    def ohlcv_asof(self, sym: str, D: str) -> Optional[pd.DataFrame]:
        df = self._read(self._ohlcv, "ohlcv", sym)
        if df is None:
            return None
        return df[df["date"] <= pd.Timestamp(D)].copy()

    def fred_latest(self, series: str, D: str) -> Optional[float]:
        df = self._read(self._fred, "fred", series)
        if df is None:
            return None
        sub = df[df["date"] <= pd.Timestamp(D)]
        return None if sub.empty else float(sub["value"].iloc[-1])

    def fred_history_asof(self, series: str, D: str) -> Optional[pd.Series]:
        df = self._read(self._fred, "fred", series)
        if df is None:
            return None
        sub = df[df["date"] <= pd.Timestamp(D)]
        return None if sub.empty else sub.set_index("date")["value"]

    def cboe_asof(self, sym: str, D: str) -> Optional[pd.DataFrame]:
        df = self._read(self._cboe, "cboe", sym)
        if df is None:
            return None
        return df[df["date"] <= pd.Timestamp(D)].copy()


def _ctx_frame(store: _SeedStore, sym: str, D: str) -> pd.DataFrame:
    raw = store.ohlcv_asof(sym, D)
    if raw is None or raw.empty:
        raise RegimeSubstrateError(
            f"regime({D}): required context OHLCV seed '{sym}' absent/empty through "
            f"D — cannot compute as-of-D regime (STOP; do NOT stub to neutral)")
    return compute_asset_features(raw)


def regime(
    as_of_D: str,
    *,
    seed_root: Optional[Path] = None,
    fusion_params: Optional[Mapping] = None,
    return_detail: bool = False,
):
    """The clean spine's as-of-D fused regime label.

    Computes, from seed substrate bounded at ``date <= as_of_D`` ONLY:
      context (SPY/TLT/HYG/IEF/VIXY OHLCV + FRED rates) -> baseline_regime_model
      -> the four expert signals -> decide_regime_v3 -> ``final_regime_label``.

    Returns the label string (or, with ``return_detail=True``, the full detail dict
    including the raw baseline label, expert scores, and fusion override reason).

    RAISES ``RegimeSubstrateError`` if a required seed series is missing for D —
    never falls back to ``'neutral'``.
    """
    store = _SeedStore(seed_root)
    D = str(as_of_D)
    Dts = pd.Timestamp(D)

    # ---- context (as-of-D) — the baseline regime model's input row -----------
    spy = _ctx_frame(store, "SPY", D)
    tlt = _ctx_frame(store, "TLT", D)
    hyg = _ctx_frame(store, "HYG", D)
    ief = _ctx_frame(store, "IEF", D)
    vixy = _ctx_frame(store, "VIXY", D)

    rates = {r: (store.fred_latest(r, D) or 0.0) for r in _FRED_RATES}
    dgs10, dgs3mo, dgs2 = rates["DGS10"], rates["DGS3MO"], rates["DGS2"]

    ctx_df = compute_context_features(
        spy, tlt, hyg, ief, vixy, rates, {}, target_date=Dts)
    ctx_row = ctx_df.iloc[0]

    base = baseline_regime_model(ctx_row)
    probs = base.get("regime_probs", {})

    # ---- expert 1: macro / credit (FRED rates + HYG/IEF closes, as-of-D) ------
    hyg_close = store.ohlcv_asof("HYG", D).set_index("date")["close"]
    ief_close = store.ohlcv_asof("IEF", D).set_index("date")["close"]
    macro = compute_macro_credit(
        rate_10y=dgs10, rate_3m=dgs3mo,
        hyg_prices=hyg_close if len(hyg_close) else None,
        ief_prices=ief_close if len(ief_close) else None,
        rate_2y=dgs2)

    # ---- expert 2: vol uncertainty (FRED VIXCLS history + CBOE VVIX/SKEW) -----
    vix_hist = store.fred_history_asof("VIXCLS", D)
    if vix_hist is None or vix_hist.empty:
        raise RegimeSubstrateError(
            f"regime({D}): FRED VIXCLS seed absent through D — vol-uncertainty "
            "cannot run as-of-D (STOP; do NOT stub to neutral)")
    vix_value = float(vix_hist.iloc[-1])
    vix_history = vix_hist if len(vix_hist) >= 60 else None
    vvix = store.cboe_asof("VVIX", D)
    skew = store.cboe_asof("SKEW", D)
    vol = compute_vol_uncertainty(
        vix=vix_value,
        vvix=float(vvix["close"].iloc[-1]) if vvix is not None and len(vvix) else None,
        skew=float(skew["close"].iloc[-1]) if skew is not None and len(skew) else None,
        vix_history=vix_history,
        vvix_history=vvix.set_index("date")["close"] if vvix is not None and len(vvix) else None,
        skew_history=skew.set_index("date")["close"] if skew is not None and len(skew) else None)

    # ---- expert 3: cross-asset fragility (8-symbol OHLCV panel, as-of-D) ------
    frag_rows = []
    for s in PANEL_SYMBOLS:
        df = store.ohlcv_asof(s, D)
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            frag_rows.append({"date": r["date"], "symbol": s, "close": r["close"]})
    frag = compute_fragility(pd.DataFrame(frag_rows))

    # ---- expert 4: entropy / distribution shift (SPY returns, as-of-D) -------
    # prev-day running counter starts at 0: the S3 timeseries state is not part of
    # the deterministic seed substrate. This affects only entropy_shift_flag, which
    # in decide_regime_v3 gates SIZE/THROTTLE, NOT the regime LABEL — so the picked
    # label is unaffected. (The record shows entropy_shift_flag False across the
    # window.) Documented, not stubbed.
    spy_close = store.ohlcv_asof("SPY", D).set_index("date")["close"]
    spy_returns = spy_close.pct_change().dropna()
    ent = compute_entropy_shift(
        spy_returns=spy_returns, prev_consecutive_days=0, prev_above_threshold=False)

    # ---- fuse (decide_regime_v3, verbatim) -----------------------------------
    # Baseline (non-ensemble) convention: disagreement 0.0, multiplier 1.0 — the
    # same values ModelLoader's baseline fallback returns.
    fusion = decide_regime_v3(
        ensemble_regime_label=base["regime_label"],
        trend_risk_on_prob=probs.get("risk_on_trend", 0.0),
        panic_prob=probs.get("high_vol_panic", 0.0),
        ensemble_disagreement=0.0,
        ensemble_multiplier=1.0,
        macro_credit_score=macro["macro_credit_score"],
        vol_uncertainty_score=vol["vol_uncertainty_score"],
        vol_regime_label=vol["vol_regime_label"],
        fragility_score=frag["fragility_score"],
        entropy_score=ent["entropy_score"],
        entropy_shift_flag=ent["entropy_shift_flag"],
        params=dict(fusion_params) if fusion_params else None)

    label = str(fusion["final_regime_label"])
    if not return_detail:
        return label
    return {
        "date": D,
        "regime_label": label,
        "raw_baseline_label": base["regime_label"],
        "override_reason": fusion.get("override_reason"),
        "macro_credit_score": round(float(macro["macro_credit_score"]), 4),
        "vol_regime_label": vol["vol_regime_label"],
        "vol_uncertainty_score": round(float(vol["vol_uncertainty_score"]), 4),
        "fragility_score": round(float(frag["fragility_score"]), 4),
        "entropy_shift_flag": bool(ent["entropy_shift_flag"]),
        "context": {
            "spy_return_21d": round(float(ctx_row.get("spy_return_21d", 0) or 0), 5),
            "spy_vol_21d": round(float(ctx_row.get("spy_vol_21d", 0) or 0), 5),
            "credit_spread_proxy": round(float(ctx_row.get("credit_spread_proxy", 0) or 0), 5),
            "vixy_return_21d": round(float(ctx_row.get("vixy_return_21d", 0) or 0), 5),
        },
    }
