"""
Phase 2 (committee execution 2026-06-06): empirically test the health-score
collapse and the LayerNorm repair.

Trains TWO health autoencoders on the SAME pre-holdout data with the SAME target,
differing ONLY in the normalization layer:
  A = BatchNorm1d  (current repo architecture, training/models/health_autoencoder.py)
  B = LayerNorm    (committee-proposed repair)
Then scores the WHOLE UNIVERSE per-date on holdout dates in eval() mode -- exactly
how src/models/loader.py does inference -- and compares the score distributions.

Hypothesis (operator note #1 + model panelist): BatchNorm's running stats are the
global pooled feature distribution; a single date's cross-section is not that
distribution, so eval-mode re-centering squashes pre-sigmoid activations -> narrow
low band. LayerNorm has no running stats -> train/eval parity -> full-range scores.

HOLDOUT: trained ONLY on date < 2026-03-11. Holdout used for scoring/diagnosis only.
"""
import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO = "/Users/drbretto/Desktop/Projects/trader-bot"
RUN_DIR = "/Users/drbretto/Desktop/Projects/Infotropy Book/book-factory/runs/20260606_trader-bot-tuning-execution"
HOLDOUT_START = pd.Timestamp("2026-03-11")
FEATURES = ['return_1d','return_5d','return_21d','return_63d','vol_21d','vol_63d',
            'drawdown_21d','drawdown_63d','rel_strength_21d','rel_strength_63d']
torch.manual_seed(7); np.random.seed(7)

def health_target(df):
    """Within-date rank composite (mirrors repo rank-only target: relative, no
    absolute direction). Same target for A and B so the ONLY difference is the norm."""
    g = df.groupby('date')
    mom = g['return_63d'].rank(pct=True)*0.6 + g['return_21d'].rank(pct=True)*0.4
    rs  = g['rel_strength_63d'].rank(pct=True)
    risk= g['vol_63d'].rank(pct=True)*0.6 + (1-(-df['drawdown_63d']).groupby(df['date']).rank(pct=True))*0.4
    return (0.45*mom + 0.35*rs + 0.20*(1-risk)).clip(0,1).values

class AE(nn.Module):
    def __init__(self, norm='bn', d=10, latent=16, hid=(64,32), dropout=0.2):
        super().__init__()
        def block(i,o):
            n = nn.BatchNorm1d(o) if norm=='bn' else nn.LayerNorm(o)
            return [nn.Linear(i,o), n, nn.ReLU(), nn.Dropout(dropout)]
        enc=[]; prev=d
        for h in hid: enc+=block(prev,h); prev=h
        enc.append(nn.Linear(prev,latent)); self.encoder=nn.Sequential(*enc)
        dec=[]; prev=latent
        for h in reversed(hid): dec+=block(prev,h); prev=h
        dec.append(nn.Linear(prev,d)); self.decoder=nn.Sequential(*dec)
        self.health=nn.Sequential(nn.Linear(latent,latent//2),nn.ReLU(),nn.Linear(latent//2,1),nn.Sigmoid())
    def forward(self,x):
        z=self.encoder(x); return self.decoder(z), self.health(z)

def train(model, X, y, epochs=25, bs=256):
    opt=torch.optim.Adam(model.parameters(), lr=1e-3)
    mse=nn.MSELoss()
    Xt=torch.tensor(X,dtype=torch.float32); yt=torch.tensor(y,dtype=torch.float32).unsqueeze(1)
    n=len(Xt)
    for ep in range(epochs):
        model.train(); perm=torch.randperm(n)
        for i in range(0,n,bs):
            idx=perm[i:i+bs]
            if len(idx)<8: continue
            xb,yb=Xt[idx],yt[idx]
            recon,h=model(xb)
            loss=mse(recon,xb)+mse(h,yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return model

def score_per_date(model, df, Xz):
    """Score whole universe per date in eval() mode (as loader.py does)."""
    model.eval()
    out=[]
    df=df.reset_index(drop=True)
    for d, idx in df.groupby('date').groups.items():
        idx=list(idx)
        with torch.no_grad():
            _,h=model(torch.tensor(Xz[idx],dtype=torch.float32))
        out.append(pd.DataFrame({'date':d,'symbol':df.loc[idx,'symbol'].values,
                                 'health':h.squeeze(1).numpy()}))
    return pd.concat(out, ignore_index=True)

def dist(s):
    return {'min':float(s.min()),'max':float(s.max()),'mean':float(s.mean()),
            'std':float(s.std()),'p05':float(s.quantile(.05)),'p95':float(s.quantile(.95)),
            'range':float(s.max()-s.min()),'frac_below_0.48':float((s<0.48).mean()),
            'frac_above_0.60':float((s>0.60).mean())}

def main():
    df=pd.read_parquet(os.path.join(REPO,"training/data/asset_features_history.parquet"))
    df['date']=pd.to_datetime(df['date'])
    df['y']=health_target(df)
    train_df=df[df['date']<HOLDOUT_START].copy()
    hold_df =df[df['date']>=HOLDOUT_START].copy()
    print(f"train rows {len(train_df)} (<{HOLDOUT_START.date()}), holdout rows {len(hold_df)}")

    # standardize on TRAIN stats only (no leakage)
    mu=train_df[FEATURES].mean().values; sd=train_df[FEATURES].std().replace(0,1).values
    Xtr=((train_df[FEATURES].values-mu)/sd)
    Xho=((hold_df[FEATURES].values-mu)/sd)

    res={}
    for norm,label in [('bn','A_BatchNorm_current'),('ln','B_LayerNorm_repair')]:
        m=AE(norm=norm)
        m=train(m, Xtr, train_df['y'].values)
        sc=score_per_date(m, hold_df, Xho)
        res[label]=dist(sc['health'])
        # per-date band width (the operator's "whole universe <= 0.48" claim)
        per_date=sc.groupby('date')['health'].agg(['min','max','mean'])
        res[label]['avg_per_date_band_width']=float((per_date['max']-per_date['min']).mean())
        res[label]['avg_per_date_max']=float(per_date['max'].mean())
        torch.save(m.state_dict(), os.path.join(REPO,f"models/health_demo_{norm}.pt"))
        print(f"\n{label}: {json.dumps(res[label],indent=0)}")

    with open(os.path.join(RUN_DIR,"health_collapse_diagnostic.json"),"w") as fh:
        json.dump(res, fh, indent=2)
    print("\nWROTE health_collapse_diagnostic.json")
    # verdict
    a=res['A_BatchNorm_current']; b=res['B_LayerNorm_repair']
    print(f"\nVERDICT: BatchNorm holdout score range={a['range']:.3f} (max {a['max']:.3f}, "
          f"{a['frac_above_0.60']*100:.0f}% would pass 0.60 gate) "
          f"vs LayerNorm range={b['range']:.3f} (max {b['max']:.3f}, "
          f"{b['frac_above_0.60']*100:.0f}% would pass 0.60 gate)")

if __name__=="__main__":
    main()
