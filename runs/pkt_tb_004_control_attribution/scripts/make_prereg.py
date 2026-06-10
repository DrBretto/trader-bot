"""Generate PREREG_BATTERY.json from battery_cells.py. Run BEFORE the battery;
commit the output before any holdout replay executes."""
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(__file__))

from battery_cells import HOLDOUT_START, LLM_ERA_START, SEEDS, build_cells  # noqa: E402

from optimizer.config import load_optimizer_config  # noqa: E402
from optimizer.promote import load_active_bundle  # noqa: E402

cfg = load_optimizer_config(os.path.join(REPO, 'config/optimizer.committee_20260606.json'))
base = load_active_bundle(cfg.config_dir_path)
cells = build_cells(base)

prereg = {
    'packet': 'PKT-TB-004-CONTROL-STACK-ATTRIBUTION-V1-20260609',
    'frozen_from': 'Methodologist prescription (committee panel role 3), adopted by chair',
    'holdout_start': HOLDOUT_START,
    'llm_era_start': LLM_ERA_START,
    'seeds': SEEDS,
    'base_bundle_version': base.get('version_id'),
    'decision_rules': {
        'meaningful_delta': [
            'paired daily-delta |t| >= 2.0 on identical valuation dates (median seed)',
            'Newey-West lag-5 |t| >= 1.8',
            'sign of delta total return identical across all seeds',
            '|delta total return| > B = 3 * 1.15bps * sqrt(sum_ON wf^2 + sum_OFF wf^2)',
        ],
        'keep_cut': ('full-period AND holdout deltas agree in sign with full-period '
                     'clearing the gate; holdout-only marginal effects are noise per '
                     'the multiple-candidates rule'),
        'cut_list_extra': ('delta~0 cut-list entries must also cite T2-ALLOFF or a '
                           'Tier-3 cell showing the layer inert with its '
                           'mechanism-group partners off'),
    },
    'tier1b_trigger': ('run T1b-C/N/R iff T1-LLM paired daily |t| >= 2.0 on the '
                       'LLM-era segment (either period read); else SKIPPED-PARENT-NULL'),
    'tier3_triggers': {
        'S1_interaction_mass': ('|R| > max(0.25 * sum_i |delta_i|, 2*SE(R)) on '
                                'pre-holdout -> all 6 pairs among top-4 layers by |delta|'),
        'S2_wrong_sign': ('pre-holdout |t| >= 2.0 with sign opposite design claim -> '
                          'pair with 3 largest-|delta| neighbors in mechanism group'),
        'S3_instability': ('sign flip pre-holdout vs holdout, both |t| >= 1.5 -> pair '
                           'with T1-J; DIAGNOSTIC only, never changes a verdict'),
        'cap': 10,
        'mechanism_groups': {
            'SIZE': ['T1-A', 'T1-B', 'T1-LLM', 'T1-D', 'T1-E', 'T1-F'],
            'LABEL': ['T1-G', 'T1-H', 'T1-I', 'T1-J'],
            'SELL': ['T1-K', 'T1-L', 'T1-M', 'T1-O', 'T1-P', 'T1-Q'],
            'BUYFILTER': ['T1-S', 'T1-T'],
        },
    },
    'holdout_look_budget': ('<= 33 attribution reads + 3 named candidates + <= 5 '
                            'panel-added candidates = K <= 41'),
    'cells': [
        {'cell_id': c['cell_id'], 'tier': c['tier'], 'layers': c['layers'],
         'description': c['description'], 'bundle_hash': c['bundle_hash']}
        for c in cells.values()
    ],
}

out = os.path.join(REPO, 'runs/pkt_tb_004_control_attribution/prereg/PREREG_BATTERY.json')
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w') as f:
    json.dump(prereg, f, indent=1)
print('wrote', out, f'({len(prereg["cells"])} cells)')
