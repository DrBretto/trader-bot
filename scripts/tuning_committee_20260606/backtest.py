"""
Phase 3-4 (committee execution 2026-06-06): what would working vs broken health
models actually have done?

Faithful long-only portfolio sim on the rebuilt feature corpus. The ONLY thing
that varies across the core configs is the SOURCE of health_score; the decision
gates, sizing, caps, regime cash floors, and sell rules are held identical and
mirror src/steps/decision_engine.py. This isolates the model-fix effect.

Health sources:
  broken_v501   : production health_v20260501.pkl  (the collapsed artifact)
  current_v601  : production health_v20260601.pkl  (current latest.json, anti-predictive)
  working_base  : deterministic rank composite = repo baseline_health logic (the
                  thing the NN was supposed to approximate; positive, full-range)
  repaired_nn   : a LayerNorm NN trained on pre-holdout to match working_base (proof
                  a trained net CAN be non-broken)

Configs:
  C0 broken_v501  + flat thresholds
  C1 current_v601 + flat thresholds
  C2 working_base + flat thresholds
  C3 working_base + regime-conditional thresholds + relaxed benign brakes
  C4 repaired_nn  + regime-conditional thresholds + relaxed benign brakes

final_score (buy ranking, stand-in for the ranking-MLP blend): transparent
cross-sectional momentum-quality rank, identical across configs. Regime label:
the repo's own SPY-percentile rule (metrics.label_regimes), identical across configs.

HOLDOUT: 2026-03-11+ is evaluated, never used to fit anything. repaired_nn trained
only on date<2026-03-11. Down-windows (2020,2022) are in-sample for repaired_nn and
deterministic for working_base -- used as protection checks, not OOS-alpha claims.
"""
import os, json, pickle
import numpy as np, pandas as pd, torch
from training.models.health_autoencoder import create_health_model

REPO="/Users/drbretto/Desktop/Projects/trader-bot"
RUN_DIR="/Users/drbretto/Desktop/Projects/Infotropy Book/book-factory/runs/20260606_trader-bot-tuning-execution"
HOLDOUT_START=pd.Timestamp("2026-03-11")
FEAT10=['return_1d','return_5d','return_21d','return_63d','vol_21d','vol_63d',
        'drawdown_21d','drawdown_63d','rel_strength_21d','rel_strength_63d']
torch.manual_seed(7); np.random.seed(7)

# ---------- health sources ----------
def working_base_health(df):
    """repo baseline_health composite (deterministic, per-date)."""
    g=df.groupby('date')
    mom=0.6*g['return_63d'].rank(pct=True)+0.4*g['return_21d'].rank(pct=True)
    rs=g['rel_strength_63d'].rank(pct=True)
    dd=(-df['drawdown_63d']).groupby(df['date']).rank(pct=True)
    risk=0.6*g['vol_63d'].rank(pct=True)+0.4*(1-dd)
    return (0.45*mom+0.35*rs+0.20*(1-risk)).clip(0,1)

def final_score_rank(df):
    """buy ranking stand-in (momentum-quality), per-date in [0,1]."""
    g=df.groupby('date')
    return (0.5*g['return_63d'].rank(pct=True)+0.3*g['rel_strength_63d'].rank(pct=True)
            +0.2*(1-g['vol_63d'].rank(pct=True)))

def score_production(df, pkl_path):
    art=pickle.load(open(pkl_path,"rb"))
    cfg=dict(art["model_config"]); mt=cfg.pop("model_type","autoencoder")
    m=create_health_model(model_type=mt,**cfg); m.load_state_dict(art["model_state"]); m.eval()
    fc=art["feature_cols"]; mu=np.array(art["normalization"]["mean"]); sd=np.array(art["normalization"]["std"])
    X=(df[fc].values-mu)/np.where(sd==0,1,sd)
    with torch.no_grad():
        h=m(torch.tensor(X,dtype=torch.float32))['health_score'].squeeze(-1).numpy()
    return pd.Series(h, index=df.index)

def train_repaired_nn(df):
    import torch.nn as nn
    class AE(nn.Module):
        def __init__(s,d=10,latent=16,hid=(64,32),dr=0.2):
            super().__init__()
            def blk(i,o):return [nn.Linear(i,o),nn.LayerNorm(o),nn.ReLU(),nn.Dropout(dr)]
            e=[];p=d
            for hh in hid:e+=blk(p,hh);p=hh
            e.append(nn.Linear(p,latent));s.enc=nn.Sequential(*e)
            de=[];p=latent
            for hh in reversed(hid):de+=blk(p,hh);p=hh
            de.append(nn.Linear(p,d));s.dec=nn.Sequential(*de)
            s.h=nn.Sequential(nn.Linear(latent,latent//2),nn.ReLU(),nn.Linear(latent//2,1),nn.Sigmoid())
        def forward(s,x):z=s.enc(x);return s.dec(z),s.h(z)
    tr=df[df['date']<HOLDOUT_START]
    y=working_base_health(tr).values
    mu=tr[FEAT10].mean().values; sd=tr[FEAT10].std().replace(0,1).values
    Xtr=(tr[FEAT10].values-mu)/sd
    m=AE(); opt=torch.optim.Adam(m.parameters(),1e-3); mse=torch.nn.MSELoss()
    Xt=torch.tensor(Xtr,dtype=torch.float32); yt=torch.tensor(y,dtype=torch.float32).unsqueeze(1); n=len(Xt)
    for _ in range(30):
        m.train(); perm=torch.randperm(n)
        for i in range(0,n,256):
            idx=perm[i:i+256]
            if len(idx)<8: continue
            recon,h=m(Xt[idx]); loss=mse(recon,Xt[idx])+mse(h,yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    X=(df[FEAT10].values-mu)/sd
    with torch.no_grad(): h=m(torch.tensor(X,dtype=torch.float32))[1].squeeze(-1).numpy()
    # report correlation on holdout
    ho=df['date']>=HOLDOUT_START
    corr=np.corrcoef(h[ho.values], working_base_health(df).values[ho.values])[0,1]
    return pd.Series(h,index=df.index), float(corr)

# ---------- regime (repo SPY-percentile rule) ----------
def regime_series(df):
    spy=df[df['symbol']=='SPY'].set_index('date').sort_index()
    r21=spy['return_21d']; v21=spy['vol_21d']
    vp=v21.expanding(min_periods=60).quantile  # point-in-time percentiles
    out={}
    v40=v21.expanding(min_periods=60).quantile(0.40)
    v50=v21.expanding(min_periods=60).quantile(0.50)
    v85=v21.expanding(min_periods=60).quantile(0.85)
    for d in spy.index:
        rr=r21.get(d,0); vv=v21.get(d,0)
        if pd.isna(rr) or pd.isna(vv): out[d]='risk_on_trend'; continue
        if vv>(v85.get(d) or 1e9) and rr<-0.02: out[d]='high_vol_panic'
        elif rr<-0.03: out[d]='risk_off_trend'
        elif rr>0.03 and vv<(v40.get(d) or 0): out[d]='calm_uptrend'
        elif abs(rr)<0.01 and vv>(v50.get(d) or 1e9): out[d]='choppy'
        else: out[d]='risk_on_trend'
    return out

# ---------- portfolio sim ----------
CASH_FLOOR={'calm_uptrend':0.10,'risk_on_trend':0.10,'choppy':0.20,'risk_off_trend':0.40,'high_vol_panic':0.40}
CASH_FLOOR_RELAXED={'calm_uptrend':0.04,'risk_on_trend':0.04,'choppy':0.20,'risk_off_trend':0.40,'high_vol_panic':0.40}
BUY_BY_REGIME={'calm_uptrend':0.55,'risk_on_trend':0.55,'choppy':0.65,'risk_off_trend':0.70,'high_vol_panic':0.72}
HEALTH_BY_REGIME={'calm_uptrend':0.46,'risk_on_trend':0.46,'choppy':0.55,'risk_off_trend':0.62,'high_vol_panic':0.65}

def simulate(df, hcol, dates, regime, *, regime_cond, relaxed, start=100000.0,
             buy_flat=0.65, mh_flat=0.60, max_pos=8, max_pos_relaxed=12,
             max_w=0.2, trail=0.10, sell_h=0.35, sell_days=3):
    px=df.pivot_table(index='date',columns='symbol',values='close')
    feats=df.set_index(['date','symbol'])
    cash=start; pos={}  # sym -> {shares,entry,peak,low_days}
    eq=[]; mp = max_pos_relaxed if relaxed else max_pos
    for d in dates:
        if d not in px.index: continue
        reg=regime.get(d,'risk_on_trend')
        day=df[df['date']==d]
        prices=px.loc[d]
        # mark to market
        val=cash+sum(p['shares']*prices.get(s,np.nan) for s,p in pos.items() if not pd.isna(prices.get(s,np.nan)))
        # sells: trailing stop or health<sell_h for sell_days
        for s in list(pos.keys()):
            pr=prices.get(s,np.nan)
            if pd.isna(pr): continue
            p=pos[s]; p['peak']=max(p['peak'],pr)
            hrow=day[day['symbol']==s]
            hh=float(hrow[hcol].iloc[0]) if len(hrow) else 0.0
            p['low_days']=p['low_days']+1 if hh<sell_h else 0
            if pr<=p['peak']*(1-trail) or p['low_days']>=sell_days:
                cash+=p['shares']*pr; del pos[s]
        # buy gates
        buy_t = (BUY_BY_REGIME[reg] if regime_cond else buy_flat)
        mh_t  = (HEALTH_BY_REGIME[reg] if regime_cond else mh_flat)
        floor = (CASH_FLOOR_RELAXED if relaxed else CASH_FLOOR)[reg]
        cand=day.copy()
        cand=cand[(cand[hcol]>=mh_t)&(cand['fscore']>=buy_t)]
        cand=cand[~cand['symbol'].isin(pos.keys())]
        if reg=='high_vol_panic':
            cand=cand[cand['symbol'].isin(['TLT','IEF','GLD','SHY','BIL','UUP'])]
        cand=cand.sort_values('fscore',ascending=False)
        # invest up to (1-floor) of value, equal-ish weight by health, cap max_w
        investable=val*(1-floor)
        cur_invested=val-cash
        room=max(0.0, investable-cur_invested)
        slots=mp-len(pos)
        for _,row in cand.iterrows():
            if slots<=0 or room<val*0.01: break
            s=row['symbol']; pr=prices.get(s,np.nan)
            if pd.isna(pr) or pr<=0: continue
            alloc=min(val*max_w, room, val/ mp + val*0.0)
            alloc=min(alloc,room)
            sh=alloc/pr
            if sh*pr<250: continue
            pos[s]={'shares':sh,'entry':pr,'peak':pr,'low_days':0}; cash-=sh*pr; room-=sh*pr; slots-=1
        val=cash+sum(p['shares']*prices.get(s,np.nan) for s,p in pos.items() if not pd.isna(prices.get(s,np.nan)))
        eq.append((d,val,1-(val-cash)/val if val>0 else 1.0))
    e=pd.DataFrame(eq,columns=['date','value','cash_frac']).set_index('date')
    return e

def metrics(e, px_spy, label):
    if len(e)<2: return {'config':label,'note':'no data'}
    ret=e['value'].iloc[-1]/e['value'].iloc[0]-1
    spy=px_spy.reindex(e.index).ffill()
    spy_ret=spy.iloc[-1]/spy.iloc[0]-1
    roll=e['value'].cummax(); dd=(e['value']/roll-1).min()
    # rally capture: algo return / spy return when spy up
    rc=ret/spy_ret if spy_ret>0 else np.nan
    daily=e['value'].pct_change().dropna()
    sharpe=(daily.mean()/daily.std()*np.sqrt(252)) if daily.std()>0 else 0
    return {'config':label,'total_return':round(ret,4),'spy_return':round(float(spy_ret),4),
            'excess_vs_spy':round(ret-float(spy_ret),4),'rally_capture':None if pd.isna(rc) else round(rc,3),
            'max_drawdown':round(float(dd),4),'avg_cash_frac':round(float(e['cash_frac'].mean()),3),
            'sharpe':round(float(sharpe),2)}

def main():
    df=pd.read_parquet(os.path.join(REPO,"training/data/asset_features_history.parquet"))
    df['date']=pd.to_datetime(df['date'])
    uni=pd.read_csv(os.path.join(REPO,"config/universe.csv"))[['symbol','asset_class']]
    df=df.merge(uni,on='symbol',how='left')
    df['fscore']=final_score_rank(df)
    df['h_working']=working_base_health(df)
    df['h_v501']=score_production(df,"/tmp/claude/tbmodels/health_v20260501.pkl")
    df['h_v601']=score_production(df,"/tmp/claude/tbmodels/health_v20260601.pkl")
    df['h_repaired'],corr=train_repaired_nn(df)
    print(f"repaired_nn holdout corr with working target: {corr:.3f}")
    reg=regime_series(df)
    px_spy=df[df['symbol']=='SPY'].set_index('date')['close'].sort_index()

    windows={'holdout_2026':(HOLDOUT_START,pd.Timestamp("2026-06-05")),
             'bear_2022':(pd.Timestamp("2022-01-03"),pd.Timestamp("2022-12-30")),
             'covid_2020':(pd.Timestamp("2020-02-03"),pd.Timestamp("2020-06-30"))}
    configs=[
      ('C0_broken_v501','h_v501',False,False),
      ('C1_current_v601','h_v601',False,False),
      ('C2_working_flat','h_working',False,False),
      ('C3_working_regimecond','h_working',True,True),
      ('C4_repaired_nn_regimecond','h_repaired',True,True),
    ]
    results={}
    for wname,(a,b) in windows.items():
        dates=sorted(df[(df['date']>=a)&(df['date']<=b)]['date'].unique())
        rows=[]
        for cname,hcol,rc,rel in configs:
            e=simulate(df,hcol,dates,reg,regime_cond=rc,relaxed=rel)
            rows.append(metrics(e,px_spy,cname))
        results[wname]=rows
        print(f"\n=== {wname} ({pd.Timestamp(a).date()}->{pd.Timestamp(b).date()}) ===")
        spyr=rows[0]['spy_return']
        print(f"  SPY: {spyr:+.2%}")
        for r in rows:
            print(f"  {r['config']:26} ret={r['total_return']:+.2%} excessSPY={r['excess_vs_spy']:+.2%} "
                  f"maxDD={r['max_drawdown']:+.2%} cash={r['avg_cash_frac']:.0%} rallycap={r['rally_capture']}")
    json.dump(results, open(RUN_DIR+"/backtest_results.json","w"), indent=2, default=str)
    print("\nWROTE backtest_results.json")

if __name__=="__main__":
    main()
