"""feeds/contracts.py — the OHLCVBar boundary contract.

This is the shared boundary the clean rebuild is built on: every feed emits rows
that pass ``coerce_to_ohlcv`` before anything downstream (store.extend, features,
P2's gdelt join) touches them. Getting the dtypes right HERE is the fix for the
June-10 freeze class: that freeze was a pandas dtype-mismatch inside a
``pd.concat`` that a bare ``except`` swallowed, so a silently-empty splice looked
like a healthy no-op. The rule this module enforces:

  * dtypes are pinned and coerced explicitly, never inferred at concat time;
  * a coercion that cannot succeed RAISES ``OHLCVContractError`` — it is never
    swallowed, so an upstream feed regression trips loudly at the boundary
    instead of leaking a wrong-dtype / empty frame into the store.

OHLCVBar schema
---------------
    date   : datetime64[ns]   (tz-naive — an exchange-local trading date)
    symbol : str  (pandas object dtype holding python str)
    open   : float64
    high   : float64
    low    : float64
    close  : float64
    volume : int64
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

# Canonical column order. Anything downstream may rely on this exact ordering.
OHLCV_COLUMNS: tuple[str, ...] = (
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

# The pinned dtypes, by column. `symbol` is python-str inside an object column;
# `date` is a tz-naive datetime64[ns]. The rest are fixed-width numerics.
OHLCV_FLOAT_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")
OHLCV_DTYPES: dict[str, str] = {
    "date": "datetime64[ns]",
    "symbol": "object",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "int64",
}


class OHLCVContractError(ValueError):
    """Raised when a frame cannot be coerced to the OHLCVBar contract.

    This is deliberately a hard failure. The whole point of the boundary is that
    a bad/empty/wrong-dtype feed result RAISES here rather than being swallowed
    into a silent freeze downstream (the ISSUE that took the night phase down on
    2026-06-10).
    """


def _require_columns(df: pd.DataFrame) -> None:
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise OHLCVContractError(
            f"OHLCV frame missing required columns {missing}; "
            f"got columns={list(df.columns)}"
        )


def coerce_to_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce ``df`` to the OHLCVBar contract, or RAISE ``OHLCVContractError``.

    Coerce-or-raise (never a bare-except swallow):

      * required columns must all be present;
      * ``date`` -> tz-naive datetime64[ns] (a tz-aware input is localized to
        naive; an unparseable date raises);
      * ``symbol`` -> python str;
      * ``open/high/low/close`` -> float64 (a non-numeric value raises);
      * ``volume`` -> int64 (NaN / non-finite / non-numeric raises — an int64
        column cannot hold NaN, so a missing volume is a genuine contract
        violation, not something to silently fill).

    Returns a NEW frame with exactly ``OHLCV_COLUMNS`` in order and a fresh index.
    An empty-but-well-formed frame (0 rows, right columns) coerces fine — the
    caller decides whether emptiness is acceptable; the contract only guarantees
    dtypes.
    """
    if not isinstance(df, pd.DataFrame):
        raise OHLCVContractError(f"expected a DataFrame, got {type(df).__name__}")

    _require_columns(df)
    out = df.loc[:, list(OHLCV_COLUMNS)].copy()

    # date -> tz-naive datetime64[ns]
    try:
        dt = pd.to_datetime(out["date"], errors="raise")
    except (ValueError, TypeError) as exc:
        raise OHLCVContractError(f"date column not parseable to datetime: {exc}") from exc
    if isinstance(dt.dtype, pd.DatetimeTZDtype) or getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)
    out["date"] = dt.astype("datetime64[ns]")

    # symbol -> str
    if out["symbol"].isna().any():
        raise OHLCVContractError("symbol column contains null values")
    out["symbol"] = out["symbol"].astype(str)

    # open/high/low/close -> float64
    for col in OHLCV_FLOAT_COLUMNS:
        try:
            num = pd.to_numeric(out[col], errors="raise")
        except (ValueError, TypeError) as exc:
            raise OHLCVContractError(
                f"{col} column not coercible to float64: {exc}"
            ) from exc
        out[col] = num.astype("float64")

    # volume -> int64 (NaN / non-finite / non-numeric all raise here)
    try:
        vol = pd.to_numeric(out["volume"], errors="raise")
    except (ValueError, TypeError) as exc:
        raise OHLCVContractError(
            f"volume column not coercible to numeric: {exc}"
        ) from exc
    if len(vol) and not np.isfinite(vol.to_numpy(dtype="float64", na_value=np.nan)).all():
        raise OHLCVContractError(
            "volume column contains NaN/non-finite values; int64 cannot hold them "
            "(a missing volume is a contract violation, not a fill-with-zero)"
        )
    out["volume"] = vol.astype("int64")

    return out.reset_index(drop=True)


def empty_ohlcv() -> pd.DataFrame:
    """A well-formed, zero-row OHLCVBar frame with the pinned dtypes."""
    cols = {name: pd.Series([], dtype=OHLCV_DTYPES[name]) for name in OHLCV_COLUMNS}
    return pd.DataFrame(cols)


def concat_ohlcv(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate already-coerced OHLCV frames AFTER re-asserting the contract.

    Never ``pd.concat`` raw feed output — that is the exact shape of the freeze.
    Each frame is coerced first (so dtypes are pinned identically), then
    concatenated. An empty input yields a well-formed empty frame.
    """
    coerced = [coerce_to_ohlcv(f) for f in frames if f is not None and len(f) > 0]
    if not coerced:
        return empty_ohlcv()
    return coerce_to_ohlcv(pd.concat(coerced, ignore_index=True))
