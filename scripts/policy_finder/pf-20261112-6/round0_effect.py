"""v3: the effect of punishment in round 0 alone. v2 points with n_blocks = 1 punish only in round 0 (members with
c0 <= thr, a flat amt); points with n_blocks = 0 or amt < 1 never punish. Round-0 contributions are drawn before any
manager acts, so given them the punishment is as good as random (it depends only on the point's thr and amt).
Game-level regressions of outcomes on round-0 punishment dealt, controlling for round-0 contributions."""
import numpy as np, json
D = np.load('/tmp/v2sweep.npz')
c, p, g, mg, name, v = D['c'], D['p'], D['g'], D['mine_g'], D['name'], D['v']
pts = {q['name']: q['params'] for q in json.load(open('plots/simulation/policy_finder/pf-20261112-6-v2_sweep/sweep.json'))['points']}
nb = np.array([pts[n]['n_blocks'] if pts[n]['amt'] >= 1 else 0 for n in name])
sel = (nb <= 1)
mine = g == mg[:, None, None]
pool = lambda m: (1.6 * c * m - p * m).sum(1)  # [N, 24]
own_pool, oth_pool = pool(mine), pool(~mine)
lead = (own_pool - oth_pool).mean(1)
P0 = (p[:, :, 0] * mine[:, :, 0]).sum(1)
n_pun0 = ((p[:, :, 0] * mine[:, :, 0]) > 0).sum(1)
c0own = (c[:, :, 0] * mine[:, :, 0]).sum(1); c0oth = (c[:, :, 0] * ~mine[:, :, 0]).sum(1)
z0own = ((c[:, :, 0] <= 2) * mine[:, :, 0]).sum(1)
size = mine.sum(1)  # [N,24]
own_c = (c * mine).sum(1); oth_c = (c * ~mine).sum(1)
def reg(y, lab):
    X = np.column_stack([np.ones(sel.sum()), P0[sel], c0own[sel], c0oth[sel], z0own[sel], mg[sel]])
    b, *_ = np.linalg.lstsq(X, y[sel], rcond=None)
    r = y[sel] - X @ b
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (r ** 2).mean()))
    print(f'{lab:28s} P0 {b[1]:+.4f} ({se[1]:.4f})')
print('games', sel.sum(), 'with P0>0', (P0[sel] > 0).sum(), 'mean P0 | >0', P0[sel][P0[sel] > 0].mean())
reg(lead, 'game lead')
reg(own_pool.mean(1), 'own pool')
reg(oth_pool.mean(1), 'other pool')
for r in [1, 2, 3, 4, 8, 12, 20]:
    reg(own_c[:, r], f'own sum c round {r}')
    reg(oth_c[:, r], f'other sum c round {r}')
    reg(size[:, r].astype(float), f'own size round {r}')
# per-capita contribution, own group, rounds 1-3
reg((own_c[:, 1:4] / np.maximum(size[:, 1:4], 1)).mean(1), 'own c per member r1-3')
reg((oth_c[:, 1:4] / np.maximum(8 - size[:, 1:4], 1)).mean(1), 'other c per member r1-3')
