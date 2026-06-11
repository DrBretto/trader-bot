"""PKT-TB-007 — point-in-time beta-hat / sigma-hat from the TB-006 OHLCV cache
(BUILD_SPEC_007 §2.5).

beta_hat_i  = trailing 126d OLS beta of symbol i's daily returns to SPY
sigma_hat_i = trailing 21d realized vol (daily return sd, NOT annualized — it
              only ever appears in ratios where the unit cancels)

Both computed strictly through D-1's close: only ohlcv rows with date < D are
used. Returns from adj_close (split/dividend safe; e.g. the VUG 6:1 split).
Cache source: runs/pkt_tb_006_clean_sheet_brain/prototype/cache/ohlcv/ —
read-only TB-006 import (ANALYST_AUDIT §C).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

PROTO = Path(__file__).resolve().parent
REPO = PROTO.parents[2]
TB006_OHLCV = REPO / "runs" / "pkt_tb_006_clean_sheet_brain" / "prototype" / "cache" / "ohlcv"

BETA_WINDOW = 126
SIGMA_WINDOW = 21
MIN_BETA_OBS = 60   # below this the symbol is excluded from the projection
MIN_SIGMA_OBS = 10


class RiskStats:
    def __init__(self, ohlcv_dir: Path = TB006_OHLCV):
        self.ohlcv_dir = Path(ohlcv_dir)
        self._ret: Dict[str, pd.Series] = {}      # symbol -> daily returns indexed by date str

    def _returns(self, symbol: str) -> Optional[pd.Series]:
        if symbol in self._ret:
            return self._ret[symbol]
        p = self.ohlcv_dir / f"{symbol}.parquet"
        if not p.exists():
            self._ret[symbol] = None
            return None
        df = pd.read_parquet(p, columns=["date", "adj_close"]).dropna()
        df = df.sort_values("date")
        idx = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        px = pd.Series(df["adj_close"].to_numpy(dtype=np.float64), index=idx.to_numpy())
        ret = px.pct_change().dropna()
        self._ret[symbol] = ret
        return ret

    def beta_sigma(self, symbol: str, date: str) -> Tuple[Optional[float], Optional[float]]:
        """(beta_hat, sigma_hat) using returns strictly before `date` (YYYY-MM-DD).
        None where the history is insufficient (caller excludes the name)."""
        r = self._returns(symbol)
        spy = self._returns("SPY")
        if r is None or spy is None:
            return None, None
        r = r[r.index < date]
        spy = spy[spy.index < date]
        # sigma: trailing 21 of the symbol's own returns
        sigma = None
        if len(r) >= MIN_SIGMA_OBS:
            tail = r.to_numpy()[-SIGMA_WINDOW:]
            sigma = float(np.std(tail, ddof=1))
        # beta: trailing 126 on common dates
        beta = None
        common = r.index.intersection(spy.index)
        if len(common) >= MIN_BETA_OBS:
            common = common[-BETA_WINDOW:]
            x = spy.loc[common].to_numpy()
            y = r.loc[common].to_numpy()
            vx = np.var(x, ddof=1)
            if vx > 0:
                beta = float(np.cov(y, x, ddof=1)[0, 1] / vx)
        return beta, sigma

    def table(self, symbols, date: str) -> Dict[str, Dict[str, float]]:
        out = {}
        for s in sorted(set(symbols)):
            b, sg = self.beta_sigma(s, date)
            if b is not None and sg is not None:
                out[s] = {"beta": b, "sigma": sg}
        return out
