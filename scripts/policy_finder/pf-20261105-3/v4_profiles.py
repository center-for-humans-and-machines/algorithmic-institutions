"""v4: one-at-a-time profiles of the late2 family around its surrogate optimum."""
import sys, json, warnings, numpy as np, torch as th
warnings.filterwarnings('ignore')
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261105-3')
from v4_sim import simulate
from v4_levers import f_late2
th.set_num_threads(2)
best = json.loads(sys.argv[1]) if len(sys.argv) > 1 else dict(thr=21, pmax=31, gamma=2.3, horizon=6, lthr=3.5, ldose=23, lstop=16)
grid = dict(thr=[17, 19, 21, 23], pmax=[24, 28, 32, 38], gamma=[1.5, 2.0, 2.5, 3.0], horizon=[3, 4, 5, 6, 7, 8, 10],
            lthr=[1.5, 2.5, 3.5, 5, 7, 9], ldose=[8, 14, 20, 26, 30], lstop=[12, 14, 16, 18, 20, 23])
if len(sys.argv) > 2: grid = {k: grid[k] for k in sys.argv[2].split(',')}
def sc(prm):
    return np.mean([simulate(lambda c, v, p, t: f_late2(c, v, p, t, **prm), n=3000, seed=s)[0] for s in (401, 402, 403)])
b = sc(best); print('best', round(b, 2), best, flush=True)
for k, xs in grid.items():
    out = []
    for x in xs:
        prm = dict(best); prm[k] = x
        out.append(f'{x}:{sc(prm) - b:+.1f}')
    print(k, ' '.join(out), flush=True)
