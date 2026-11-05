# v7: finer: by sextile of the round-0 group mean m0, the effect on the game's mean pool of
# (a) total round-0 punishment P0 given the sorted round-0 contributions, (b) the design's
# log k1 and log kn and s0 (exogenous), same controls. Over v6's sweep.
import numpy as np, json
d = np.load('/tmp/pf-20261106-9-v6_sweep.npz')
c = d['c'].astype(float); p = d['p'].astype(float); g = d['g']; ours = d['ours']; opp = d['opp']; pt = d['pt']
s = json.load(open('plots/simulation/policy_finder/pf-20261106-9-v6_sweep/sweep.json'))
idx = {int(q['name'].split('_s')[-1]): i for i, q in enumerate(s['points'])}
par = {k: np.array([s['points'][idx[j]]['params'][k] for j in pt]) for k in ['k0', 'k1', 'kn', 's0', 'nb']}
mine = (g == ours[:, None, None])
pool = (1.6 * (c * mine).sum(1) - (p * mine).sum(1)).mean(1)
c0 = c[:, :, 0]; m0 = (c0 * mine[:, :, 0]).sum(1) / 4
c0s = np.sort(np.where(mine[:, :, 0], c0, np.inf), 1)[:, :4]
P0 = (p[:, :, 0] * mine[:, :, 0]).sum(1)
q = np.quantile(m0, np.arange(1, 6) / 6); b = np.digitize(m0, q)
def ols(y, X):
    X = np.column_stack([np.ones(len(y))] + X)
    bb = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ bb; s2 = r @ r / (len(y) - X.shape[1]); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return bb, se
for t in range(6):
    sl = b == t
    ctrl = [c0s[sl][:, j] for j in range(4)] + [c0s[sl][:, j] ** 2 for j in range(4)] + [(opp[sl] == o).astype(float) for o in (2, 3, 4)]
    b1, s1 = ols(pool[sl], [P0[sl] / 24] + ctrl)
    b2, s2 = ols(pool[sl], [np.log(par['k0'][sl]), np.log(par['k1'][sl]), np.log(par['kn'][sl]), par['s0'][sl], par['nb'][sl]] + ctrl)
    print(f'm0 sextile {t} ({m0[sl].mean():5.2f}) pool {pool[sl].mean():5.1f}: dpool/dP0(per-round units) {b1[1]:+.2f}({s1[1]:.2f}) | '
          + ' '.join(f'{n} {bb:+.2f}({ss:.2f})' for n, bb, ss in zip(['lk0', 'lk1', 'lkn', 's0', 'nb'], b2[1:6], s2[1:6])))
