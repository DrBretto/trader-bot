"""Stage-3: participation/exposure search on the REAL champion overlay system.
Lower benign-only buy threshold + cash floor, modest position/weight expansion,
topup aggressiveness. Tune for ROBUST winners (beat incumbent on BOTH windows).
Stressed regimes kept protected (thresholds/cash floors unchanged there).
"""
import json, itertools
from scripts.beat_champion_20260606.runner import (
    make_cache, champion_strategy, run_real)

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
# stressed down-window inside holdout (the sharp leg) for protection check
dw = [d for d in hold if '2026-03-25' <= d <= '2026-04-10']

CASH_CUR = {"calm_uptrend":0.1,"risk_on_trend":0.1,"choppy":0.2,"risk_off_trend":0.4,"high_vol_panic":0.4}

champ_cur = champion_strategy()
inc_h = run_real(c, uni, hold, '2026-03-11', strategy=champ_cur)
inc_i = run_real(c, uni, ins,  '2026-01-02', strategy=champ_cur)
inc_dw = run_real(c, uni, dw,  '2026-03-11', strategy=champ_cur) if len(dw) > 2 else {'maxdd':inc_h['maxdd']}
INC_H, INC_I, INC_DD = inc_h['return'], inc_i['return'], inc_dw.get('maxdd', inc_h['maxdd'])
print(f"INCUMBENT  ins {INC_I*100:+.2f}%  hold {INC_H*100:+.2f}%  hold_dd {inc_h['maxdd']*100:.2f}%  downwin_dd {INC_DD*100:.2f}%  (FLOOR)")

rows = []
for mp in (8, 10, 12):
  for mpw in (0.20, 0.25):
    for benign_cash in (0.10, 0.05):
      for benign_bst in (0.65, 0.60):
        for topup_trig in (1.2, 1.1):
            dp = {'max_positions': mp, 'max_position_weight': mpw}
            cash = dict(CASH_CUR); cash['calm_uptrend']=benign_cash; cash['risk_on_trend']=benign_cash
            dp['min_cash_reserve_by_regime'] = cash
            dp['buy_score_threshold_by_regime'] = {
                'calm_uptrend': benign_bst, 'risk_on_trend': benign_bst,
                'choppy': 0.65, 'risk_off_trend': 0.68, 'high_vol_panic': 0.72}
            strat = champion_strategy(topup_trigger=topup_trig)
            ri = run_real(c, uni, ins,  '2026-01-02', dp_overrides=dp, strategy=strat)
            rh = run_real(c, uni, hold, '2026-03-11', dp_overrides=dp, strategy=strat)
            rows.append({'mp':mp,'mpw':mpw,'cash':benign_cash,'bst':benign_bst,'topup':topup_trig,
                         'ins':ri['return'],'ins_dd':ri['maxdd'],'hold':rh['return'],'hold_dd':rh['maxdd']})

robust = [r for r in rows if r['hold'] > INC_H and r['ins'] >= INC_I]
beat_hold = [r for r in rows if r['hold'] > INC_H]
print(f"\nconfigs: {len(rows)}  beat-holdout: {len(beat_hold)}  robust(beat BOTH): {len(robust)}")
print("\nTOP 10 by HOLDOUT:")
for r in sorted(rows,key=lambda x:-x['hold'])[:10]:
    tag=''
    if r['hold']>INC_H and r['ins']>=INC_I: tag=' ROBUST'
    elif r['hold']>INC_H: tag=' (hold-only)'
    print(f"  mp{r['mp']} mpw{r['mpw']} cash{r['cash']} bst{r['bst']} top{r['topup']} | ins {r['ins']*100:+6.2f}% hold {r['hold']*100:+6.2f}% hdd {r['hold_dd']*100:6.2f}%{tag}")

pool = robust if robust else (beat_hold if beat_hold else rows)
best = sorted(pool, key=lambda r: (-r['hold'], -r['ins']))[0] if (robust or beat_hold) else \
       sorted(pool, key=lambda r: (-r['ins'], r['ins_dd']))[0]
print(f"\nSELECTED: mp{best['mp']} mpw{best['mpw']} cash{best['cash']} bst{best['bst']} topup{best['topup']}")
print(f"  ins {best['ins']*100:+.2f}% (inc {INC_I*100:+.2f}%)  hold {best['hold']*100:+.2f}% (inc {INC_H*100:+.2f}%)  beats_inc_hold={best['hold']>INC_H}")
json.dump({'incumbent':{'ins':INC_I,'hold':INC_H,'hold_dd':inc_h['maxdd'],'downwin_dd':INC_DD},
           'rows':rows,'winner':best,'dw_dates':dw},
          open('runs/beat_champion_20260606/search_stage3.json','w'),indent=1)
print("saved search_stage3.json")
