"""PKT-TB-007 — R-LLM falsifier conditioner input (TOURNAMENT §4.4.9 / BUILD_SPEC §2.6).

`llm_disag` = std over the 27 `llm_sent_*` LEVEL buckets (unweighted), z-scored
on its trailing 252-artifact window, joined to decision date D by
visible_from <= D (merge_asof backward, 4-calendar-day staleness tolerance —
the Phase-A probe's join, llm_role_probe.py). Width multiplier
g = clip(1 + 0.25*tanh(z), 0.75, 1.25); missing/stale artifact => z=None, g=1.

Writes store/rllm_disag_007.json: {date: {z, g, llm_disag, art_visible_from}}.
$0 — reads the TB-006 cached parquet only; no Bedrock.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

PROTO = Path(__file__).resolve().parent
REPO = PROTO.parents[2]
TB6 = REPO / "runs" / "pkt_tb_006_clean_sheet_brain" / "prototype"
Z_WINDOW = 252
Z_MINP = 64
OUT = PROTO / "store" / "rllm_disag_007.json"


def g_of_z(z: float) -> float:
    return float(np.clip(1.0 + 0.25 * math.tanh(z), 0.75, 1.25))


def main():
    panel = np.load(TB6 / "store" / "panel.npz", allow_pickle=True)
    buckets = [str(b) for b in panel["buckets"]]
    assert len(buckets) == 27, f"expected 27 buckets, got {len(buckets)}"
    cols = [f"llm_sent_{b}" for b in buckets]

    llm = pd.read_parquet(TB6 / "store" / "llm_features.parquet")
    llm = llm[llm["llm_available"] == 1.0].copy()
    llm.index = pd.to_datetime(llm.index)
    llm["visible_from"] = pd.to_datetime(llm["visible_from"])
    llm = llm[llm.index >= "2024-08-15"]            # contiguous run only (probe)
    llm = llm.sort_values("visible_from")

    sent = llm[cols].to_numpy(dtype=float)
    disag = np.array([np.nanstd(row) if np.isfinite(row).sum() >= 5 else np.nan
                      for row in sent])
    s = pd.Series(disag, index=llm["visible_from"].to_numpy())
    mu = s.rolling(Z_WINDOW, min_periods=Z_MINP).mean()
    sd = s.rolling(Z_WINDOW, min_periods=Z_MINP).std(ddof=1)
    z = (s - mu) / sd

    art = pd.DataFrame({"visible_from": llm["visible_from"].to_numpy(),
                        "llm_disag": disag, "z": z.to_numpy()}
                       ).sort_values("visible_from")

    # decision dates = live nightly panel dates (2026-01-31 -> 2026-06-10 dirs)
    dates = sorted(p.stem for p in (PROTO / "store" / "nightly_007").glob("*.json"))
    dd = pd.DataFrame({"D": pd.to_datetime(dates)})
    j = pd.merge_asof(dd, art, left_on="D", right_on="visible_from",
                      tolerance=pd.Timedelta(days=4), direction="backward")

    out = {}
    for _, row in j.iterrows():
        d = row["D"].strftime("%Y-%m-%d")
        if pd.isna(row["z"]):
            out[d] = {"z": None, "g": 1.0, "llm_disag": None,
                      "art_visible_from": None}
        else:
            out[d] = {"z": round(float(row["z"]), 6),
                      "g": round(g_of_z(float(row["z"])), 6),
                      "llm_disag": round(float(row["llm_disag"]), 6),
                      "art_visible_from": row["visible_from"].strftime("%Y-%m-%d")}
    OUT.write_text(json.dumps(
        {"spec": "TOURNAMENT_007 §4.4.9 / BUILD_SPEC_007 §2.6",
         "signal": "llm_disag = unweighted std over 27 llm_sent_* level buckets",
         "z": f"trailing {Z_WINDOW}-artifact rolling z (minp {Z_MINP})",
         "join": "visible_from <= D, backward, 4d tolerance",
         "g": "clip(1 + 0.25*tanh(z), 0.75, 1.25); missing => 1.0",
         "dates": out}, indent=1))
    zs = [v["z"] for v in out.values() if v["z"] is not None]
    print(f"wrote {OUT} — {len(out)} dates, {len(zs)} with z "
          f"(z range [{min(zs):+.2f}, {max(zs):+.2f}])")


if __name__ == "__main__":
    main()
