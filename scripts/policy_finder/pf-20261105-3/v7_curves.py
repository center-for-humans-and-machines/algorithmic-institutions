"""v7: floored shock dose curves (c = 0..19) and timing of the best v4-v6 finals."""
import json, sys, numpy as np
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261105-3')
B = '/opt/algorithmic-institutions/plots/simulation/policy_finder/'
from v7_pool_feats import curve
for v in ['v4', 'v5', 'v6']:
    f = json.load(open(B + f'pf-20261105-3-{v}_sweep/final/sweep.json'))
    for q in f['points']:
        p = q['params']
        print(f"{q['name'][14:]:8s} {q['pool']:6.1f}", ' '.join(f'{int(x):2d}' for x in curve(v, p, np.arange(20))),
              '| h', p['horizon'], 'lthr', p['lthr'], 'lstop', p['lstop'])
