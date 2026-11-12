"""v2: point-level regression of the v1 sweep's lead on punishment summed per game by round group
(round 0, rounds 1-3, block starts 4/8/12/16, other non-decision rounds 5-18, decision rounds 7..19, 20-22),
split by target (contribution 0-4 vs 5+). Coefficients are pool points of lead per punishment point per game;
the direct cost is 1/24 = 0.042."""
import numpy as np, json
D = np.load('/tmp/v1sweep.npz')
c, p, g, mg, name = D['c'], D['p'], D['g'], D['mine_g'], D['name']
pts = {q['name']: q for q in json.load(open('plots/simulation/policy_finder/pf-20261112-6_sweep/sweep.json'))['points']}
mine = g == mg[:, None, None]
pm = p * mine
lo = c < 5
r = np.arange(24)
groups = {'r0': r == 0, 'r1_3': (r >= 1) & (r <= 3), 'bstart': np.isin(r, [4, 8, 12, 16]),
          'mid': (r >= 5) & (r <= 18) & ~np.isin(r, [7, 8, 11, 12, 15, 16]), 'dec': np.isin(r, [7, 11, 15, 19]) | (r == 3) * 0 > 0,
          'late': (r >= 20)}
groups['dec'] = np.isin(r, [7, 11, 15, 19])
groups['mid'] = (r >= 5) & (r <= 18) & ~np.isin(r, [7, 8, 11, 12, 15, 16])
names = sorted(set(name))
F = []
for n in names:
    s = name == n
    row = [ (pm[s][:, :, gm] * lo[s][:, :, gm]).sum((1, 2)).mean() for gm in groups.values()]
    row.append((pm[s] * ~lo[s]).sum((1, 2)).mean())
    F.append(row)
F = np.array(F); y = np.array([pts[n]['pool'] for n in names])
X = np.column_stack([np.ones(len(F)), F])
b, res, *_ = np.linalg.lstsq(X, y, rcond=None)
sig2 = ((y - X @ b) ** 2).sum() / (len(y) - X.shape[1])
se = np.sqrt(np.diag(sig2 * np.linalg.inv(X.T @ X)))
for lab, bb, ss in zip(['const'] + [k + '_low' for k in groups] + ['any_5plus'], b, se):
    print(f'{lab:12s} {bb:+.4f} ({ss:.4f})')
print('mean sums', dict(zip(list(groups) + ['5plus'], F.mean(0).round(1))))
