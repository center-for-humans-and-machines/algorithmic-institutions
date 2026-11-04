"""v6: natural experiment on the randomised stop round. For window start s (4, 8, 12): sweep points
whose stop round equals s (A: no punishment from s on) vs >= s+4 (B: punishes through the window),
identical policies before s. Conditional on my group's size during rounds s..s+3, compare my pool
in rounds s..23 (mean per round, empty = 0), size at s+4 and my mean punishment per member in the
window. Uses v4 and v5 sweeps (t_end) and v1 (t_stop)."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, 'scripts/policy_finder/pf-20261104-7')
from v5_panel import load
pd.set_option('display.width', 250)
rows = []
EXTRA = int(sys.argv[1]) if len(sys.argv) > 1 else -1  # B: stop round in s+4..s+4+EXTRA (-1: any)
NB = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8] if len(sys.argv) > 2 else [-1, 0, 2, 5, 8]
for V, key in [('pf-20261104-7-v5', 't_end'), ('pf-20261104-7-v4', 't_end'), ('pf-20261104-7', 't_stop')]:
    try:
        r = load(V)
    except Exception as e:
        print('skip', V, e); continue
    sw = json.load(open(f'plots/simulation/policy_finder/{V}_sweep/sweep.json'))
    te = {int(q['name'].split('_s')[-1]): q['params'][key] for q in sw['points']}
    tend = np.array([te[i] for i in r['point']])
    c, p, v, g = r['c'].astype(float), r['p'].astype(float), r['v'].astype(float), r['g'].astype(float)
    pool = (1.6 * c * v * g - p * g).sum(1)
    size = g.sum(1)
    pm = (p * g).sum(1) / np.maximum(size, 1)
    for s in [4, 8, 12]:
        fixed = (size[:, s:s+4] == size[:, [s]]).all(1)
        hi = s + 4 + EXTRA if EXTRA >= 0 else 99
        arm = np.where(tend == s, 'A_stop', np.where((tend >= s + 4) & (tend <= hi), 'B_go', ''))
        m = fixed & (arm != '')
        rows.append(pd.DataFrame({'V': V, 's': s, 'n': size[m, s], 'arm': arm[m],
                                  'pool_rest': pool[m, s:].mean(1), 'n_next': size[m, s + 4],
                                  'n_end': size[m, 20:].mean(1), 'pwin': pm[m, s:s+4].mean(1)}))
x = pd.concat(rows)
x['nb'] = pd.cut(x.n, NB)
for s in [4, 8, 12]:
    y = x[x.s == s]
    t = y.groupby(['nb', 'arm'], observed=True).agg(N=('pool_rest', 'size'), pool=('pool_rest', 'mean'),
        se=('pool_rest', lambda a: a.std() / np.sqrt(len(a))), n_next=('n_next', 'mean'), n_end=('n_end', 'mean'), pwin=('pwin', 'mean')).round(2)
    print(f'=== window from round {s}'); print(t.to_string())
    d = t.pool.unstack('arm'); print('B - A pool:', (d['B_go'] - d['A_stop']).round(2).to_dict())
