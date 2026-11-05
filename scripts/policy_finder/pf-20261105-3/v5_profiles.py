"""v5: one-at-a-time profiles of the thr-free v4 form (dose pmax (1 - c/20)^gamma,
shock < horizon, free riders c < lthr until lstop) on the surrogate, 3 seeds x 2000."""
import sys, warnings, numpy as np, torch as th
warnings.filterwarnings('ignore')
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261105-3')
from v5_levers import reparam, simulate
th.set_num_threads(2)
B = dict(pmax=30, gamma=2.2, cfree=20, horizon=7, lthr=4, lstop=15)
def sc(**kw):
    q = dict(B); q.update(kw)
    return np.mean([simulate(reparam(**q), n=2000, seed=s)[0] for s in (601, 602, 603)])
b = sc(); print('base', round(b, 2), flush=True)
for k, vals in [('gamma', [1.4, 1.7, 2.0, 2.5, 2.9, 3.4]), ('pmax', [22, 26, 34, 40, 50]),
                ('horizon', [4, 5, 6, 8, 9, 11]), ('lthr', [0, 2, 3, 5, 6, 8]), ('lstop', [10, 12, 14, 16, 18, 21])]:
    print(k, ' '.join(f'{x}:{sc(**{k: x}) - b:+.1f}' for x in vals), flush=True)
