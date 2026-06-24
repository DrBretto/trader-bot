"""Canonical clean-core artifacts (FP-08).

The canon package holds the from-scratch clean-core primitives the equity-line
rebuild stands on. Today it contains the append-only, content-addressed equity
ledger (FP-08-1). It is deliberately dependency-light (JSON + an S3 client) — no
torch/pandas/requests — so it can live on the thin, non-inference publish branch.
"""
