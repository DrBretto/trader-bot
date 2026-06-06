"""
Constrained tuning sweep (committee execution 2026-06-06).

Unlike the genetic search (which overfits a degenerate config across 200+ params
on 3 short folds), this varies ONLY the misfire levers the timeline audit
implicated, freezes everything else, and replays each variant on the REAL engine
(optimizer.replay.run_replay_for_dates) over:
  - pre-holdout (2025-08-04 .. 2026-03-10)  = selection window
  - holdout      (2026-03-11 .. latest)     = validation, never tuned on
Same ranking model + seed across all variants, so differences = the params only.
"""
import json, copy, numpy as np
from optimizer.config import load_optimizer_config
from optimizer.data_access import load_optimizer_dataset
from optimizer.promote import load_active_bundle
from optimizer.replay import run_replay_for_dates

RUN_DIR="/Users/drbretto/Desktop/Projects/Infotropy Book/book-factory/runs/20260606_trader-bot-tuning-execution"
HOLDOUT="2026-03-11"
cfg=load_optimizer_config("config/optimizer.committee_20260606.json")
cfg.holdout_start=""          # load ALL dates; we split manually
ds=load_optimizer_dataset(cfg)
dates=sorted(str(s.date) for s in ds.snapshots)
pre=[d for d in dates if d<HOLDOUT]; hold=[d for d in dates if d>=HOLDOUT]
print(f"pre-holdout {len(pre)} ({pre[0]}..{pre[-1]})  holdout {len(hold)} ({hold[0]}..{hold[-1]})")

base=load_active_bundle(cfg.config_dir_path)
ens=base.get("ensemble",{})
print("current ensemble params:", json.dumps(ens))

# optional ranking model (same for all variants -> fair)
ranking_model=ranking_norm=None; blend=0.0
try:
    from training.models.ranking_mlp import RankingMLP
    import os, torch
    mdir="models/ranking_expanded_unconditioned"
    if os.path.exists(mdir+"/ranking_mlp.pt"):
        ranking_norm=json.load(open(mdir+"/ranking_normalization.json"))
        ranking_model=RankingMLP.load(mdir+"/ranking_mlp.pt") if hasattr(RankingMLP,'load') else None
        blend=0.35 if ranking_model else 0.0
except Exception as e:
    print("ranking model not loaded (replay still valid, blend=0):", str(e)[:80])

def variant(name, **changes):
    b=copy.deepcopy(base)
    b.setdefault("ensemble",{}); b.setdefault("decision_params",{})
    for k,v in changes.items():
        if k.startswith("ens."): b["ensemble"][k[4:]]=v
        elif k.startswith("dp."): b["decision_params"][k[3:]]=v
        elif k.startswith("cash."):
            b["decision_params"].setdefault("min_cash_reserve_by_regime",{})[k[5:]]=v
    return name,b

# current ensemble defaults (fill if missing) to tune from
cur=lambda k,d: ens.get(k,d)
variants=[
  variant("V0_champion"),
  # soften disagreement penalty (the 60%-every-day cut)
  variant("V1_soft_disagreement", **{"ens.disagreement_penalty_scale":min(0.5*cur('disagreement_penalty_scale',1.0),cur('disagreement_penalty_scale',1.0)), "ens.clip_min":max(0.6,cur('clip_min',0.0))}),
  # down-weight the broken 9%-acc GRU
  variant("V2_downweight_gru", **{"ens.gru_weight":0.2, "ens.transformer_weight":0.8}),
  # deploy more in benign regimes + slightly looser gates
  variant("V3_deploy_benign", **{"cash.risk_on_trend":0.05,"cash.calm_uptrend":0.05,"dp.buy_score_threshold":0.58,"dp.min_health_buy":0.53}),
  # all three, sane (no max_positions/sell-rule degeneracy)
  variant("V4_combined", **{"ens.disagreement_penalty_scale":min(0.5*cur('disagreement_penalty_scale',1.0),cur('disagreement_penalty_scale',1.0)),"ens.clip_min":max(0.6,cur('clip_min',0.0)),"ens.gru_weight":0.2,"ens.transformer_weight":0.8,"cash.risk_on_trend":0.05,"cash.calm_uptrend":0.05,"dp.buy_score_threshold":0.58,"dp.min_health_buy":0.53}),
]

import pandas as pd
_corpus=pd.read_parquet("training/data/asset_features_history.parquet")
_corpus["date"]=pd.to_datetime(_corpus["date"])
_spy=_corpus[_corpus["symbol"]=="SPY"].set_index("date")["close"].sort_index()
def spy_ret(dl):
    s=_spy.reindex(pd.to_datetime(sorted(dl))).ffill().dropna()
    return float(s.iloc[-1]/s.iloc[0]-1) if len(s)>=2 else None

def metrics(res, label, dl):
    steps=res.steps
    vals=[s.start_value for s in steps]+[steps[-1].end_value] if steps else []
    fills=res.fills
    buys=sum(1 for f in fills if str(f.get('action','')).upper().startswith('BUY')) if fills else 0
    if len(vals)<2: return {'config':label,'note':'no path'}
    ret=vals[-1]/vals[0]-1
    arr=np.array(vals); dd=float((arr/np.maximum.accumulate(arr)-1).min())
    sr=spy_ret(dl)
    # exposure proxy: fraction of steps holding (end_value moved by market, not just cash)
    traded=sum(getattr(s,'traded_notional',0) or 0 for s in steps)
    return {'config':label,'return':round(ret,4),'spy':None if sr is None else round(sr,4),
            'excess':None if sr is None else round(ret-sr,4),'maxDD':round(dd,4),
            'n_fills':len(fills),'buys':buys,'traded_notional':round(traded,0)}

out={"pre_holdout":[], "holdout":[]}
for name,b in variants:
    for seg,dl in [("pre_holdout",pre),("holdout",hold)]:
        res=run_replay_for_dates(ds, dl, b, cfg.random_seed, 100000.0, ranking_model, ranking_norm, blend)
        out[seg].append(metrics(res,name,dl))
for seg in ["pre_holdout","holdout"]:
    print(f"\n=== {seg} ===")
    for r in out[seg]:
        print(f"  {r.get('config'):22} ret={r.get('return')} spy={r.get('spy')} excess={r.get('excess')} maxDD={r.get('maxDD')} buys={r.get('buys')} fills={r.get('n_fills')}")
json.dump(out, open(RUN_DIR+"/constrained_tune_results.json","w"), indent=2, default=str)
print("\nWROTE constrained_tune_results.json")
