# v7: does the best opening slope depend on how much shortfall the game has at round 0?
# D = total shortfall below the round-0 mean over our members (c0 is set before any punishment,
# so D is exogenous to the design). Per D tercile: pool 0-23 by k0 quartile and by k1 quartile
# (v6 sweep design params, joined via the point index), and a quadratic fit in log slope.
import numpy as np, json
d = np.load('/tmp/pf-20261106-9-v6_sweep.npz')
c = d['c'].astype(float); p = d['p'].astype(float); g = d['g']; ours = d['ours']; opp = d['opp']; pt = d['pt']
s = json.load(open('plots/simulation/policy_finder/pf-20261106-9-v6_sweep/sweep.json'))
names = [q['name'] for q in s['points']]
idx = {int(n.split('_s')[-1]): i for i, n in enumerate(names)}
par = {k: np.array([s['points'][idx[j]]['params'][k] for j in pt]) for k in ['k0', 'k1', 'kn', 's0', 'nb']}
mine = (g == ours[:, None, None])
pool = (1.6 * (c * mine).sum(1) - (p * mine).sum(1)).mean(1)
c0 = c[:, :, 0]; m0 = (c0 * mine[:, :, 0]).sum(1, keepdims=True) / 4
D = (np.clip(m0 - c0, 0, None) * mine[:, :, 0]).sum(1)
q = np.quantile(D, [1 / 3, 2 / 3]); b = np.digitize(D, q)
print('D tercile bounds', q)
for k in ['k0', 'k1', 's0']:
    x = np.log(par[k]) if k != 's0' else par[k]
    qq = np.quantile(x, [.25, .5, .75]); kb = np.digitize(x, qq)
    for t in range(3):
        sel = b == t
        row = [pool[sel & (kb == j)].mean() for j in range(4)]
        A = np.column_stack([np.ones(sel.sum()), x[sel], x[sel] ** 2]); bb = np.linalg.lstsq(A, pool[sel], rcond=None)[0]
        pk = -bb[1] / (2 * bb[2]) if bb[2] < 0 else np.nan
        print(f'{k} D-tercile {t} (D {D[sel].mean():5.1f}): by quartile ' + ' '.join(f'{r:5.1f}' for r in row)
              + f'  quad peak {np.exp(pk) if k != "s0" else pk:6.2f}')
