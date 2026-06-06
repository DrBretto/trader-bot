"""
Root-cause isolation for the health-score collapse (committee execution 2026-06-06).

The first demo showed BatchNorm does NOT collapse when trained on the rebuilt
11.77yr corpus with drawdown_21d present. So the committee's "BatchNorm is the
prime suspect" is likely WRONG. This script isolates the true cause by training
under controlled conditions and measuring the holdout whole-universe-per-date
score distribution for each:

  1 full corpus  + BatchNorm + all features        (control)
  2 THIN corpus  + BatchNorm + all features        (thin-data hypothesis)
  3 full corpus  + BatchNorm + drawdown_21d zeroed  (missing-feature hypothesis)
  4 THIN corpus  + BatchNorm + drawdown_21d zeroed  (both bugs together)
  5 full corpus  + LayerNorm + all features         (proposed repair)

"Collapse" per operator note #1 = whole universe <= ~0.48, narrow band.
"""
import os, json
import numpy as np, pandas as pd, torch, torch.nn as nn

REPO="/Users/drbretto/Desktop/Projects/trader-bot"
RUN_DIR="/Users/drbretto/Desktop/Projects/Infotropy Book/book-factory/runs/20260606_trader-bot-tuning-execution"
HOLDOUT_START=pd.Timestamp("2026-03-11")
THIN_START=pd.Timestamp("2025-08-04")   # the stored-artifact start the operator cited
FEATURES=['return_1d','return_5d','return_21d','return_63d','vol_21d','vol_63d',
          'drawdown_21d','drawdown_63d','rel_strength_21d','rel_strength_63d']
torch.manual_seed(7); np.random.seed(7)

def health_target(df):
    g=df.groupby('date')
    mom=g['return_63d'].rank(pct=True)*0.6+g['return_21d'].rank(pct=True)*0.4
    rs=g['rel_strength_63d'].rank(pct=True)
    risk=g['vol_63d'].rank(pct=True)*0.6+(1-(-df['drawdown_63d']).groupby(df['date']).rank(pct=True))*0.4
    return (0.45*mom+0.35*rs+0.20*(1-risk)).clip(0,1).values

class AE(nn.Module):
    def __init__(self,norm='bn',d=10,latent=16,hid=(64,32),dropout=0.2):
        super().__init__()
        def blk(i,o):
            n=nn.BatchNorm1d(o) if norm=='bn' else nn.LayerNorm(o)
            return [nn.Linear(i,o),n,nn.ReLU(),nn.Dropout(dropout)]
        e=[];p=d
        for h in hid:e+=blk(p,h);p=h
        e.append(nn.Linear(p,latent));self.encoder=nn.Sequential(*e)
        dec=[];p=latent
        for h in reversed(hid):dec+=blk(p,h);p=h
        dec.append(nn.Linear(p,d));self.decoder=nn.Sequential(*dec)
        self.health=nn.Sequential(nn.Linear(latent,latent//2),nn.ReLU(),nn.Linear(latent//2,1),nn.Sigmoid())
    def forward(self,x):z=self.encoder(x);return self.decoder(z),self.health(z)

def train(model,X,y,epochs=25,bs=256):
    opt=torch.optim.Adam(model.parameters(),lr=1e-3);mse=nn.MSELoss()
    Xt=torch.tensor(X,dtype=torch.float32);yt=torch.tensor(y,dtype=torch.float32).unsqueeze(1);n=len(Xt)
    for _ in range(epochs):
        model.train();perm=torch.randperm(n)
        for i in range(0,n,bs):
            idx=perm[i:i+bs]
            if len(idx)<8:continue
            recon,h=model(Xt[idx]);loss=mse(recon,Xt[idx])+mse(h,yt[idx])
            opt.zero_grad();loss.backward();opt.step()
    return model

def score_per_date(model,df,Xz):
    model.eval();out=[];df=df.reset_index(drop=True)
    for d,idx in df.groupby('date').groups.items():
        idx=list(idx)
        with torch.no_grad():_,h=model(torch.tensor(Xz[idx],dtype=torch.float32))
        out.append(pd.DataFrame({'date':d,'health':h.squeeze(1).numpy()}))
    return pd.concat(out,ignore_index=True)

def summ(sc):
    s=sc['health'];pd_=sc.groupby('date')['health'].agg(['min','max'])
    return {'min':round(float(s.min()),3),'max':round(float(s.max()),3),
            'mean':round(float(s.mean()),3),'std':round(float(s.std()),3),
            'frac_above_0.60':round(float((s>0.60).mean()),3),
            'avg_per_date_max':round(float(pd_['max'].mean()),3),
            'avg_per_date_band':round(float((pd_['max']-pd_['min']).mean()),3),
            'COLLAPSED': bool(pd_['max'].mean()<0.52)}

def run_condition(df, train_mask, norm, zero_dd):
    feats=list(FEATURES)
    work=df.copy()
    if zero_dd: work['drawdown_21d']=0.0
    tr=work[train_mask].copy()
    mu=tr[feats].mean().values; sd=tr[feats].std().replace(0,1).values
    Xtr=(tr[feats].values-mu)/sd
    ho=work[work['date']>=HOLDOUT_START].copy()
    Xho=(ho[feats].values-mu)/sd
    m=AE(norm=norm); m=train(m,Xtr,tr['y'].values)
    return summ(score_per_date(m,ho,Xho)), len(tr)

def main():
    df=pd.read_parquet(os.path.join(REPO,"training/data/asset_features_history.parquet"))
    df['date']=pd.to_datetime(df['date']); df['y']=health_target(df)
    pre=df['date']<HOLDOUT_START
    thin=(df['date']>=THIN_START)&(df['date']<HOLDOUT_START)
    conds={
      '1_full_BN_allfeat':(pre,'bn',False),
      '2_thin_BN_allfeat':(thin,'bn',False),
      '3_full_BN_dd21zero':(pre,'bn',True),
      '4_thin_BN_dd21zero':(thin,'bn',True),
      '5_full_LN_allfeat':(pre,'ln',False),
    }
    res={}
    for name,(mask,norm,zdd) in conds.items():
        torch.manual_seed(7); np.random.seed(7)
        d,ntr=run_condition(df,mask,norm,zdd)
        d['train_rows']=int(ntr)
        res[name]=d
        print(f"{name:22} train_rows={ntr:6d}  max~{d['avg_per_date_max']:.3f} "
              f"band~{d['avg_per_date_band']:.3f} pass0.60={d['frac_above_0.60']:.2f} "
              f"COLLAPSED={d['COLLAPSED']}")
    with open(os.path.join(RUN_DIR,"health_rootcause.json"),"w") as fh:
        json.dump(res,fh,indent=2)
    print("\nWROTE health_rootcause.json")

if __name__=="__main__":
    main()
