"""Stage-4: refine around the stage-3 robust winner and enforce the mandatory
down-window protection check. Final candidate must:
  - beat incumbent on BOTH in-sample and holdout (robust, not overfit)
  - not worsen holdout maxdd materially
  - pass down-window: dw_dd >= incumbent_dw_dd - 0.02 (DOWN-1 rule)
"""
import json
from scripts.beat_champion_20260606.runner import make_cache, champion_strategy, run_real

c = make_cache(); uni = c.get_csv('config/universe.csv')

def valid(dates):
    out = []
    for d in dates:
        try:
            inf = c.get_json(f'daily/{d}/inference.json')
            if isinstance(inf, dict) and 'date' in inf and 'regime' in inf:
                out.append(d)
        except Exception:
            pass
    return out

dates = c.list_daily_dates()
hold = valid([d for d in dates if d >= '2026-03-11'])
ins  = valid([d for d in dates if d <= '2026-03-10'])
dw   = [d for d in hold if '2026-03-25' <= d <= '2026-04-10']
CASH = {"calm_uptrend":0.1,"risk_on_trend":0.1,"choppy":0.2,"risk_off_trend":0.4,"high_vol_panic":0.4}

champ = champion_strategy()
inc_h = run_real(c, uni, hold, '2026-03-11', strategy=champ)
inc_i = run_real(c, uni, ins,  '2026-01-02', strategy=champ)
inc_dw= run_real(c, uni, dw,   '2026-03-11', strategy=champ)
INC_H, INC_I, INC_DWD, INC_HDD = inc_h['return'], inc_i['return'], inc_dw['maxdd'], inc_h['maxdd']
print(f"INCUMBENT ins {INC_I*100:+.2f}% hold {INC_H*100:+.2f}% hold_dd {INC_HDD*100:.2f}% dw_dd {INC_DWD*100:.2f}%")

def cfg(mpw, topup, bst):
    dp = {'max_positions': 8, 'max_position_weight': mpw,
          'min_cash_reserve_by_regime': dict(CASH),
          'buy_score_threshold_by_regime': {'calm_uptrend':bst,'risk_on_trend':bst,
                                            'choppy':0.65,'risk_off_trend':0.68,'high_vol_panic':0.72}}
    return dp, champion_strategy(topup_trigger=topup)

rows = []
for mpw in (0.25, 0.28, 0.30):
  for topup in (1.15, 1.10, 1.05):
    for bst in (0.62, 0.60, 0.58):
        dp, strat = cfg(mpw, topup, bst)
        ri = run_real(c, uni, ins,  '2026-01-02', dp_overrides=dp, strategy=strat)
        rh = run_real(c, uni, hold, '2026-03-11', dp_overrides=dp, strategy=strat)
        rd = run_real(c, uni, dw,   '2026-03-11', dp_overrides=dp, strategy=strat)
        ok_robust = rh['return'] > INC_H and ri['return'] >= INC_I
        ok_dd = rh['maxdd'] >= INC_HDD - 0.02
        ok_dw = rd['maxdd'] >= INC_DWD - 0.02
        rows.append({'mpw':mpw,'topup':topup,'bst':bst,'ins':ri['return'],'hold':rh['return'],
                     'hold_dd':rh['maxdd'],'dw_dd':rd['maxdd'],
                     'robust':ok_robust,'ok_dd':ok_dd,'ok_dw':ok_dw,
                     'pass':ok_robust and ok_dd and ok_dw})

print("\nALL (sorted by holdout); PASS = robust + dd-ok + downwindow-ok:")
for r in sorted(rows,key=lambda x:-x['hold']):
    print(f"  mpw{r['mpw']} top{r['topup']} bst{r['bst']} | ins {r['ins']*100:+6.2f}% hold {r['hold']*100:+6.2f}% "
          f"hdd {r['hold_dd']*100:6.2f}% dwdd {r['dw_dd']*100:6.2f}% | {'PASS' if r['pass'] else 'fail'}")

passing = [r for r in rows if r['pass']]
best = sorted(passing, key=lambda r:(-r['hold'], -r['ins']))[0] if passing else None
print("\nFINAL:", json.dumps(best) if best else "NONE PASSED")
json.dump({'incumbent':{'ins':INC_I,'hold':INC_H,'hold_dd':INC_HDD,'dw_dd':INC_DWD},
           'rows':rows,'final':best}, open('runs/beat_champion_20260606/search_stage4.json','w'),indent=1)
print("saved search_stage4.json")
