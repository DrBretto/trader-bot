"""
AUDIT the real live algorithm timeline (committee execution 2026-06-06, v2 — correct target).

Pulls the system's OWN daily decision artifacts from S3 (decisions.json,
portfolio_state.json, inference.json) over the full live period and finds where
the tuning misfired: benign regimes where the book was throttled into cash,
disagreement/risk-throttle sizing cuts, and rallies missed while under-deployed.
No reconstruction -- the system's actual outputs.
"""
import boto3, json, numpy as np, pandas as pd
b3=boto3.client("s3","us-east-1"); BK="investment-system-data"
RUN_DIR="/Users/drbretto/Desktop/Projects/Infotropy Book/book-factory/runs/20260606_trader-bot-tuning-execution"

dates=sorted([k["Prefix"].split("/")[1] for k in
              b3.list_objects_v2(Bucket=BK,Prefix="daily/",Delimiter="/").get("CommonPrefixes",[])])
def get(dd,f):
    try: return json.loads(b3.get_object(Bucket=BK,Key=f"daily/{dd}/{f}")["Body"].read())
    except Exception: return None

rows=[]
for dd in dates:
    dec=get(dd,"decisions.json"); ps=get(dd,"portfolio_state.json"); inf=get(dd,"inference.json")
    if not dec: continue
    em=dec.get("expert_metrics",{}) or {}; ens=dec.get("ensemble_metrics",{}) or {}
    reg=dec.get("regime")
    acts=dec.get("actions",[]) or []
    bc=dec.get("buy_candidates",[]) or []
    rows.append({
        "date":dd,"regime":reg,
        "regime_conf":em.get("regime_confidence"),
        "disagreement":ens.get("disagreement"),
        "pos_size_mult":ens.get("position_size_multiplier"),
        "risk_throttle":em.get("risk_throttle_factor"),
        "pos_size_mod":em.get("position_size_modifier"),
        "eff_exposure":em.get("effective_exposure"),
        "override":em.get("override_reason"),
        "n_buy_cand":len(bc),
        "n_buy_act":sum(1 for a in acts if a.get("action")=="BUY"),
        "cash_pct":(ps or {}).get("cash_pct"),
        "gross_exp":(ps or {}).get("gross_exposure"),
        "pval":(ps or {}).get("portfolio_value"),
        "bench":(ps or {}).get("benchmark_value"),
    })
df=pd.DataFrame(rows)
df["date"]=pd.to_datetime(df["date"])
df=df.sort_values("date").reset_index(drop=True)
df.to_csv(RUN_DIR+"/LIVE_TIMELINE_AUDIT.csv",index=False)

benign=df["regime"].isin(["risk_on_trend","calm_uptrend"])
print(f"Live days: {len(df)} ({df.date.min().date()} -> {df.date.max().date()})")
print(f"\nregime distribution:\n{df['regime'].value_counts().to_string()}")
print(f"\nbenign-regime days (risk_on/calm): {benign.sum()} ({benign.mean():.0%})")
print(f"avg gross_exposure overall: {df['gross_exp'].mean():.2f} | benign: {df.loc[benign,'gross_exp'].mean():.2f}")
print(f"avg cash_pct overall: {df['cash_pct'].mean():.2f} | benign: {df.loc[benign,'cash_pct'].mean():.2f}")
print(f"avg risk_throttle overall: {df['risk_throttle'].mean():.3f} | benign: {df.loc[benign,'risk_throttle'].mean():.3f}")
print(f"avg pos_size_mult overall: {df['pos_size_mult'].mean():.3f} | benign: {df.loc[benign,'pos_size_mult'].mean():.3f}")
print(f"avg disagreement: {df['disagreement'].mean():.3f}")

# MISFIRE 1: benign regime, had buy candidates, but heavily throttled & in cash
mis=df[benign & (df["n_buy_cand"]>0) & (df["gross_exp"]<0.4)]
print(f"\nMISFIRE — benign regime + buy candidates available + gross_exposure<0.40: "
      f"{len(mis)} days ({len(mis)/max(1,benign.sum()):.0%} of benign days)")
print("  avg buy_candidates on those days:", round(mis['n_buy_cand'].mean(),1),
      "| avg buys taken:", round(mis['n_buy_act'].mean(),1),
      "| avg risk_throttle:", round(mis['risk_throttle'].mean(),3),
      "| avg pos_size_mult:", round(mis['pos_size_mult'].mean(),3))

# MISFIRE 2: how often risk_throttle pinned low (<=0.3) in benign regimes
low_throttle=df[benign & (df["risk_throttle"]<=0.3)]
print(f"\nMISFIRE — risk_throttle<=0.30 while in a benign regime: {len(low_throttle)} days "
      f"({len(low_throttle)/max(1,benign.sum()):.0%} of benign days)")

# holdout slice
ho=df[df["date"]>=pd.Timestamp("2026-03-11")]
print(f"\n=== holdout {ho.date.min().date()}->{ho.date.max().date()} ({len(ho)}d) ===")
print(f"  avg gross_exposure {ho['gross_exp'].mean():.2f}, avg cash {ho['cash_pct'].mean():.2f}, "
      f"avg risk_throttle {ho['risk_throttle'].mean():.3f}")
print(f"  benign days in holdout: {ho['regime'].isin(['risk_on_trend','calm_uptrend']).sum()}/{len(ho)}")

import json as J
summ={"live_days":len(df),"benign_frac":float(benign.mean()),
      "avg_gross_exp_benign":float(df.loc[benign,'gross_exp'].mean()),
      "avg_cash_benign":float(df.loc[benign,'cash_pct'].mean()),
      "avg_risk_throttle_benign":float(df.loc[benign,'risk_throttle'].mean()),
      "misfire_benign_underdeployed_days":int(len(mis)),
      "misfire_low_throttle_benign_days":int(len(low_throttle))}
J.dump(summ, open(RUN_DIR+"/LIVE_TIMELINE_AUDIT_summary.json","w"), indent=2)
print("\nWROTE LIVE_TIMELINE_AUDIT.csv + summary.json")
