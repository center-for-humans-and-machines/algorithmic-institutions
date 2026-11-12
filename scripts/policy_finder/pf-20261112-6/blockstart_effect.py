"""v3: natural experiment at each block start b = 4k (k = 1..4) in v2's sweep. Points with n_blocks = k and
n_blocks = k+1 play identically up to round b-1; at b only the latter punish. Game-level regression of outcomes
(rest-of-game lead, members at later rounds, own/other contributions) on punishment dealt at round b, controlling
for the state at b."""
import numpy as np, json
D = np.load('/tmp/v2sweep.npz')
c, p, g, mg, name = D['c'], D['p'], D['g'], D['mine_g'], D['name']
pts = {q['name']: q['params'] for q in json.load(open('plots/simulation/policy_finder/pf-20261112-6-v2_sweep/sweep.json'))['points']}
nb = np.array([pts[n]['n_blocks'] if pts[n]['amt'] >= 1 else 0 for n in name])
mine = g == mg[:, None, None]
own_pool = (1.6 * c * mine - p * mine).sum(1); oth_pool = (1.6 * c * ~mine - p * ~mine).sum(1)
size = mine.sum(1).astype(float)
own_c = (c * mine).sum(1); oth_c = (c * ~mine).sum(1)
for k in [0, 1, 2, 3, 4]:
    b = 4 * k
    sel = (nb == k) | (nb == k + 1)
    Pb = (p[:, :, b] * mine[:, :, b]).sum(1)
    lowb = ((c[:, :, b] <= 3) * mine[:, :, b]).sum(1)
    ctrl = [own_c[:, b], oth_c[:, b], lowb, mg] + ([size[:, b]] if b > 0 else []) + ([own_c[:, b - 1], (p[:, :, :b] * mine[:, :, :b]).sum((1, 2))] if b > 0 else [])
    X = np.column_stack([np.ones(sel.sum()), Pb[sel]] + [x[sel] for x in ctrl])
    XtXi = np.linalg.inv(X.T @ X)
    def reg(y):
        bb = XtXi @ X.T @ y[sel]
        r = y[sel] - X @ bb
        return bb[1], np.sqrt(XtXi[1, 1] * (r ** 2).mean())
    out = {'lead_rest(x n_rounds/24)': reg((own_pool[:, b:] - oth_pool[:, b:]).sum(1) / 24),
           'size+4': reg(size[:, min(b + 4, 23)]), 'size+8': reg(size[:, min(b + 8, 23)]),
           'own_c b+1..b+3': reg(own_c[:, b + 1:b + 4].mean(1)), 'oth_c b+1..b+3': reg(oth_c[:, b + 1:b + 4].mean(1)),
           'own_c b+4..b+7': reg(own_c[:, b + 4:b + 8].mean(1))}
    print(f'b={b}: games {sel.sum()}, punished games {(Pb[sel] > 0).sum()}, mean P|>0 {Pb[sel][Pb[sel] > 0].mean():.1f}')
    print('   ' + '  '.join(f'{k_} {v[0]:+.4f}({v[1]:.4f})' for k_, v in out.items()))
