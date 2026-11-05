"""v4: merged late phase (free riders keep the shock dose until lstop) vs flat ldose."""
import sys, json, warnings, numpy as np, torch as th
warnings.filterwarnings('ignore')
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261105-3')
from v4_sim import simulate
from v4_levers import f_late2, f_late, f_base
th.set_num_threads(2)
def sc(f):
    return np.mean([simulate(f, n=3000, seed=s)[0] for s in (401, 402, 403)])
print('flat ldose 23', sc(lambda c,v,p,t: f_late2(c,v,p,t,21,31,2.3,6,3.5,23,16)))
for lthr in (2.5, 3.5, 4.5, 6):
    for lstop in (14, 16, 18):
        for h in (5, 6, 7):
            print('merged lthr', lthr, 'lstop', lstop, 'h', h, round(sc(lambda c,v,p,t: f_late(c,v,p,t,21,31,2.3,h,lthr,lstop)), 2), flush=True)
