"""Confirm the exact winning config on the real system and emit the wired
decision_params + overlay knobs. No leakage: params chosen on in-sample, here we
just reproduce the locked numbers on all windows.
"""
import json, copy
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

# ---- WINNING CONFIG (chosen on in-sample, validated out-of-sample) ----
WIN_DP = {
    'max_position_weight': 0.30,             # was 0.20
    'buy_score_threshold_by_regime': {        # was flat 0.65; benign ~unchanged, stressed PROTECTED
        'calm_uptrend': 0.62, 'risk_on_trend': 0.62,
        'choppy': 0.65, 'risk_off_trend': 0.68, 'high_vol_panic': 0.72},
}
WIN_TOPUP = 1.10                              # was 1.20
strat = champion_strategy(topup_trigger=WIN_TOPUP)

champ0 = champion_strategy()
inc = {'ins': run_real(c,uni,ins,'2026-01-02',strategy=champ0)['return'],
       'hold': run_real(c,uni,hold,'2026-03-11',strategy=champ0)['return']}
ri = run_real(c,uni,ins, '2026-01-02', dp_overrides=WIN_DP, strategy=strat)
rh = run_real(c,uni,hold,'2026-03-11', dp_overrides=WIN_DP, strategy=strat)
rd = run_real(c,uni,dw,  '2026-03-11', dp_overrides=WIN_DP, strategy=strat)

print("=== WIRED CONFIG CONFIRMATION (real three_line_replay, overlays on) ===")
print(f"incumbent : ins {inc['ins']*100:+.2f}%   hold {inc['hold']*100:+.2f}%")
print(f"new config: ins {ri['return']*100:+.2f}%   hold {rh['return']*100:+.2f}%   "
      f"hold_dd {rh['maxdd']*100:.2f}%   dw_dd {rd['maxdd']*100:.2f}%")
print(f"HOLDOUT BEAT: {rh['return']*100:+.2f}% vs {inc['hold']*100:+.2f}%  "
      f"(+{(rh['return']-inc['hold'])*100:.2f} pp)  -> beats_champion={rh['return']>inc['hold']}")

out = {'winning_decision_params': WIN_DP, 'winning_topup_trigger': WIN_TOPUP,
       'incumbent': inc,
       'new': {'ins': ri['return'], 'hold': rh['return'], 'hold_dd': rh['maxdd'], 'dw_dd': rd['maxdd']},
       'beats_champion_holdout': bool(rh['return'] > inc['hold'])}
json.dump(out, open('runs/beat_champion_20260606/FINAL_RESULT.json','w'), indent=1)
print("saved FINAL_RESULT.json")
