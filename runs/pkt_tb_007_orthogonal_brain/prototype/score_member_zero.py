"""PKT-TB-007 — member zero: the deployed RankingMLP scored over the panel.

C2 gate needs the deployed ranker's per-day per-symbol signal (BUILD_SPEC_007
§4.4 / TOURNAMENT §4.4.2 member-zero rows; M1's acceptance IC-delta conjunct).
Loads models/ranking_expanded_unconditioned/{ranking_mlp.pt,
ranking_normalization.json} (read-only), recomputes the production
RANKING_FEATURES from the TB-006 OHLCV cache with the production formulas
(src/utils/feature_utils.compute_asset_features), valued through D-1 close
(panel input convention), normalizes with the deployed normalization, and
writes store/member_zero_scores.npz (dates, symbols, score [N,S]).

Forced choice (ledgered): adj_close is used as the price series (production's
data source is dividend/split-adjusted at fetch time); the deployed
normalization JSON carries no regime/structural one-hots, so the model is the
pure 10-feature variant (verified at load).
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROTO = Path(__file__).resolve().parent
REPO = PROTO.parents[2]
TB6 = REPO / "runs" / "pkt_tb_006_clean_sheet_brain" / "prototype"
MODEL_DIR = TB6 / "cache" / "s3" / "models" / "ranking_expanded_unconditioned"
sys.path.insert(0, str(REPO))

from training.models.ranking_mlp import RankingMLP, RANKING_FEATURES  # noqa: E402


def main():
    t0 = time.time()
    import torch
    z = np.load(TB6 / "store" / "panel.npz", allow_pickle=False)
    dates = np.asarray(z["dates"]).astype(str)
    symbols = [str(s) for s in z["symbols"]]
    dts = pd.DatetimeIndex(pd.to_datetime(dates))
    N, S = len(dates), len(symbols)

    norm = json.loads((MODEL_DIR / "ranking_normalization.json").read_text())
    assert "regime_calm_uptrend" not in norm and "ac_equity" not in norm, \
        "deployed normalization carries one-hots — scorer must be extended"
    model = RankingMLP(input_dim=len(RANKING_FEATURES))
    model.load_state_dict(torch.load(MODEL_DIR / "ranking_mlp.pt",
                                     weights_only=True))
    model.eval()

    cache = TB6 / "cache" / "ohlcv"
    spy = pd.read_parquet(cache / "SPY.parquet")
    cal = pd.DatetimeIndex(pd.to_datetime(spy["date"]).dt.normalize()) \
        .unique().sort_values()
    pos = cal.searchsorted(dts)
    n_cal = len(cal)
    A = np.full((n_cal, S), np.nan)
    for j, sym in enumerate(symbols):
        df = pd.read_parquet(cache / f"{sym}.parquet")
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.set_index("date").reindex(cal)
        A[:, j] = df["adj_close"].to_numpy(dtype=np.float64)
    Ad = pd.DataFrame(A)
    r1 = Ad.pct_change(1)
    feats = {
        "return_1d": r1,
        "return_5d": Ad.pct_change(5),
        "return_21d": Ad.pct_change(21),
        "return_63d": Ad.pct_change(63),
        "vol_21d": r1.rolling(21).std() * np.sqrt(252),
        "vol_63d": r1.rolling(63).std() * np.sqrt(252),
    }
    pk63 = Ad.rolling(63, min_periods=1).max()
    feats["drawdown_63d"] = (Ad - pk63) / pk63
    feats["trend_63d"] = np.log(Ad.clip(lower=0.01)).diff(63) / 63
    spy_j = symbols.index("SPY")
    feats["rel_strength_21d"] = feats["return_21d"].sub(
        feats["return_21d"].iloc[:, spy_j], axis=0)
    feats["rel_strength_63d"] = feats["return_63d"].sub(
        feats["return_63d"].iloc[:, spy_j], axis=0)

    X = np.zeros((n_cal, S, len(RANKING_FEATURES)), dtype=np.float32)
    for fi, f in enumerate(RANKING_FEATURES):
        raw = np.nan_to_num(feats[f].to_numpy(), nan=0.0, posinf=0.0,
                            neginf=0.0)
        mu = norm.get(f, {}).get("mean", 0.0)
        sd = norm.get(f, {}).get("std", 1.0) or 1.0
        if sd < 1e-8:
            sd = 1.0
        X[:, :, fi] = (raw - mu) / sd
    Xp = X[pos - 1]                       # features through D-1 close
    with torch.no_grad():
        score = model.net(torch.from_numpy(Xp.reshape(N * S, -1))) \
            .numpy().reshape(N, S)
    score = np.where(z["symbol_mask"].astype(bool), score, np.nan)
    np.savez_compressed(PROTO / "store" / "member_zero_scores.npz",
                        dates=z["dates"], symbols=z["symbols"],
                        score=score.astype(np.float32))
    man = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
           "model_dir": str(MODEL_DIR), "features": RANKING_FEATURES,
           "convention": "features through D-1 close; adj_close basis "
                         "(forced choice, ledgered)",
           "wall_clock_s": round(time.time() - t0, 1)}
    (PROTO / "store" / "member_zero_manifest.json").write_text(
        json.dumps(man, indent=1))
    print("member zero scored:", score.shape, "finite frac",
          float(np.isfinite(score).mean()), f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
