"""Smoke-test inspector for PKT-TB-006 GDELT cache (records parquet)."""
import os

import pandas as pd

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gdelt_cache", "records")

print("MARKER inspect2")
df = pd.read_parquet(os.path.join(BASE, "20260609.parquet"))
print("shape:", df.shape)
print("ts dtype:", df["ts"].dtype, "| ts range:", df["ts"].min(), "->", df["ts"].max())
print("nonempty quotations:", int((df["quotations"].str.len() > 0).sum()))
print("distinct domains:", df["source_domain"].nunique())
sub = df[df["quotations"].str.len() > 0]
r = sub.iloc[0] if len(sub) else df.iloc[0]
for c in df.columns:
    print(f"{c}: {str(r[c])[:180]}")
for f in sorted(os.listdir(BASE)):
    print(f, os.path.getsize(os.path.join(BASE, f)) // 1024, "KB")
