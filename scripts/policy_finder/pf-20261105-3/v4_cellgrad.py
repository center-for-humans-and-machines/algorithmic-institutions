"""v4: finite-difference value of each (c band, round band) cell of a dose table around
the base optimum, with common random numbers. python v4_cellgrad.py [delta] [n]"""
import sys, warnings, numpy as np, torch as th
warnings.filterwarnings('ignore')
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261105-3')
from v4_sim import simulate
from v4_table import make, init_from_base, NC, NT, CB, TB
th.set_num_threads(2)
d = float(sys.argv[1]) if len(sys.argv) > 1 else 4
n = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
base = np.clip(init_from_base(), 0, 30)
print(np.round(base.reshape(NC, NT), 1))
seeds = [21, 22]
b = np.mean([simulate(make(base), n=n, seed=s)[0] for s in seeds])
print('base', b, flush=True)
up = np.zeros((NC, NT)); dn = np.zeros((NC, NT))
for i in range(NC):
    for j in range(NT):
        k = i * NT + j
        t1 = base.copy(); t1[k] = min(30, t1[k] + d)
        up[i, j] = np.mean([simulate(make(t1), n=n, seed=s)[0] for s in seeds]) - b
        if base[k] > 0:
            t2 = base.copy(); t2[k] = max(0, t2[k] - d)
            dn[i, j] = np.mean([simulate(make(t2), n=n, seed=s)[0] for s in seeds]) - b
        print(f'c[{CB[i]},{CB[i+1]}) t[{TB[j]},{TB[j+1]}) dose {base[k]:.1f} up {up[i,j]:+.2f} down {dn[i,j]:+.2f}', flush=True)
np.save('/tmp/v4_cellgrad.npy', np.stack([up, dn]))
print('rows c bands', CB, 'cols t bands', TB)
print('up'); print(np.round(up, 1)); print('down'); print(np.round(dn, 1))
