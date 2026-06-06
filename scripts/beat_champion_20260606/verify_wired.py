"""End-to-end wiring check: build the variant from the REPO config
config/decision_params.active.json (the deploy source) and use the REAL
extender._build_champion_strategy() overlay (now topup 1.1). Confirms the wired
artifacts reproduce the beat-the-champion number through the real harness.
"""
import json, os, contextlib
from src.utils.three_line_replay import replay_engine as RE
from src.utils.three_line_replay.replay_engine import run_variant
from src.utils.three_line_replay.extender import _build_champion_strategy
from scripts.beat_champion_20260606.runner import make_cache, _start_value, _maxdd, seed_at

c = make_cache(); uni = c.get_csv('config/universe.csv')

def valid(dates):
    out=[]
    for d in dates:
        try:
            inf=c.get_json(f'daily/{d}/inference.json')
            if isinstance(inf,dict) and 'date' in inf and 'regime' in inf: out.append(d)
        except Exception: pass
    return out

dates=c.list_daily_dates()
hold=valid([d for d in dates if d>='2026-03-11'])
ins =valid([d for d in dates if d<='2026-03-10'])

active=json.load(open('config/decision_params.active.json'))
def variant_from_active(a):
    return RE.VariantConfig(
        name='wired_champion',
        decision_params=a['decision_params'],
        regime_compatibility=a['regime_compatibility'],
        signal_overrides=a.get('signals',{}),
        regime_fusion_overrides=a.get('regime_fusion',{}),
        decision_engine_overrides=a.get('decision_engine',{}),
        ensemble_overrides=a.get('ensemble',{}),
        transaction_cost_overrides=a.get('transaction_costs',{}),
    )

def run(window, seed_date):
    v=variant_from_active(active); strat=_build_champion_strategy()
    orig=RE.START_PORTFOLIO_DATE; RE.START_PORTFOLIO_DATE=seed_date
    try:
        seed=seed_at(c,seed_date); start=_start_value(c,seed,window[1])
        with open(os.devnull,'w') as dn, contextlib.redirect_stdout(dn):
            res=run_variant(c,v,strat,window,uni)
    finally:
        RE.START_PORTFOLIO_DATE=orig
    eq=[start]+[r['ending_value'] for r in res['timeline']]
    return eq[-1]/start-1.0, _maxdd(eq)

print('topup overlay name:', _build_champion_strategy().name)
print('repo active max_position_weight:', active['decision_params']['max_position_weight'])
print('repo active buy_score_threshold_by_regime:', active['decision_params'].get('buy_score_threshold_by_regime'))
hr,hd=run(hold,'2026-03-11'); ir,idd=run(ins,'2026-01-02')
print(f"WIRED (repo config + real overlay): holdout {hr*100:+.2f}%  hdd {hd*100:.2f}%   in-sample {ir*100:+.2f}%")
print(f"vs incumbent holdout +9.39% -> beats_champion={hr>0.0939}")
json.dump({'wired_holdout':hr,'wired_hold_dd':hd,'wired_insample':ir,'overlay':_build_champion_strategy().name,
           'mpw':active['decision_params']['max_position_weight']},
          open('runs/beat_champion_20260606/verify_wired.json','w'),indent=1)
print('saved verify_wired.json')
