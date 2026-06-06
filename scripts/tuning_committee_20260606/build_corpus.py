"""
Phase 1 (committee execution 2026-06-06): rebuild the per-asset feature corpus.

Pulls daily OHLCV for the full universe via yfinance back to 2015, computes the
10 ASSET_FEATURES the health autoencoder expects -- using the repo's exact
definitions from src/utils/feature_utils.py PLUS the missing drawdown_21d
(committee finding: feature_utils computes drawdown_63d but NOT drawdown_21d,
which the health model lists in ASSET_FEATURES; the np.random at
train_health.py:418 is only the dummy fallback). Writes a single tidy parquet.

HOLDOUT discipline: this just builds features for ALL dates; the split that
holds out 2026-03-11+ happens at train/eval time. Building features over all
dates is not leakage (features are point-in-time, computed only from past prices).
"""
import sys, os
import numpy as np
import pandas as pd

REPO = "/Users/drbretto/Desktop/Projects/trader-bot"
OUT_REPO = os.path.join(REPO, "training/data/asset_features_history.parquet")
RUN_DIR = "/Users/drbretto/Desktop/Projects/Infotropy Book/book-factory/runs/20260606_trader-bot-tuning-execution"
START = "2014-06-01"   # extra lead so 63d windows are valid from ~2015
END = "2026-06-06"
HOLDOUT_START = "2026-03-11"

ASSET_FEATURES = ['return_1d','return_5d','return_21d','return_63d',
                  'vol_21d','vol_63d','drawdown_21d','drawdown_63d',
                  'rel_strength_21d','rel_strength_63d']

def load_universe():
    df = pd.read_csv(os.path.join(REPO, "config/universe.csv"))
    return df['symbol'].tolist()

def compute_features_for_symbol(close: pd.Series) -> pd.DataFrame:
    """Mirror src/utils/feature_utils.compute_asset_features, plus drawdown_21d."""
    out = pd.DataFrame(index=close.index)
    out['close'] = close
    out['return_1d']  = close.pct_change(1)
    out['return_5d']  = close.pct_change(5)
    out['return_21d'] = close.pct_change(21)
    out['return_63d'] = close.pct_change(63)
    r1 = out['return_1d']
    out['vol_21d'] = r1.rolling(21).std() * np.sqrt(252)
    out['vol_63d'] = r1.rolling(63).std() * np.sqrt(252)
    peak21 = close.rolling(21, min_periods=1).max()
    peak63 = close.rolling(63, min_periods=1).max()
    out['drawdown_21d'] = (close - peak21) / peak21   # THE FIX (was missing)
    out['drawdown_63d'] = (close - peak63) / peak63
    return out

def main():
    syms = load_universe()
    print(f"Universe: {len(syms)} symbols")
    import yfinance as yf
    raw = yf.download(syms, start=START, end=END, auto_adjust=True, progress=False)
    close = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw[['Close']]
    close = close.dropna(how='all')
    print(f"Price matrix: {close.shape[0]} dates {close.index.min().date()} -> {close.index.max().date()}, {close.shape[1]} symbols")

    # SPY returns for relative strength
    if 'SPY' not in close.columns:
        raise SystemExit("SPY missing from download")
    spy_feat = compute_features_for_symbol(close['SPY'])

    frames = []
    dropped = []
    for sym in close.columns:
        s = close[sym].dropna()
        if len(s) < 120:
            dropped.append((sym, len(s)))
            continue
        f = compute_features_for_symbol(s)
        f['rel_strength_21d'] = f['return_21d'] - spy_feat['return_21d'].reindex(f.index)
        f['rel_strength_63d'] = f['return_63d'] - spy_feat['return_63d'].reindex(f.index)
        f = f.reset_index().rename(columns={'index':'date','Date':'date'})
        f['symbol'] = sym
        frames.append(f)
    corpus = pd.concat(frames, ignore_index=True)
    corpus['date'] = pd.to_datetime(corpus['date'])
    # keep rows where all 10 features are present
    before = len(corpus)
    corpus = corpus.dropna(subset=ASSET_FEATURES)
    print(f"Rows: {before} -> {len(corpus)} after dropping warmup NaNs")
    print(f"Dropped short-history symbols: {dropped}")

    cols = ['date','symbol','close'] + ASSET_FEATURES
    corpus = corpus[cols].sort_values(['date','symbol']).reset_index(drop=True)
    corpus.to_parquet(OUT_REPO, index=False)
    print(f"WROTE {OUT_REPO}  ({len(corpus)} rows)")

    # summary for the run record
    span = f"{corpus['date'].min().date()} -> {corpus['date'].max().date()}"
    n_pre = (corpus['date'] < HOLDOUT_START).sum()
    n_hold = (corpus['date'] >= HOLDOUT_START).sum()
    per_sym = corpus.groupby('symbol')['date'].agg(['min','max','count'])
    summary = {
        'rows': len(corpus), 'symbols': corpus['symbol'].nunique(), 'span': span,
        'pre_holdout_rows': int(n_pre), 'holdout_rows': int(n_hold),
        'years': round((corpus['date'].max()-corpus['date'].min()).days/365.25, 2),
        'drawdown_21d_is_random': False,
        'drawdown_21d_stats': corpus['drawdown_21d'].describe().to_dict(),
        'dropped_short_history': dropped,
    }
    os.makedirs(RUN_DIR, exist_ok=True)
    import json
    with open(os.path.join(RUN_DIR, "corpus_build_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    per_sym.to_csv(os.path.join(RUN_DIR, "corpus_per_symbol_coverage.csv"))
    print("SUMMARY:", json.dumps({k:summary[k] for k in ['rows','symbols','span','years','pre_holdout_rows','holdout_rows']}, default=str))

if __name__ == "__main__":
    main()
