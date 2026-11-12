"""v3: point-level regression of the v2 sweep's lead on punishment summed per game, by block-start round
(0, 4, 8, 12, 16) and by target contribution (0-4 vs 5+). Also members/own/other. Direct cost 1/24 = 0.042."""
import sys, numpy as np, json
sys.path.insert(0, 'scripts/policy_finder/pf-20261112-6')
from load_sweep import load
base = 'plots/simulation/policy_finder/pf-20261112-6-v2_sweep/'
D = load(base)
np.savez('/tmp/v2sweep.npz', **D)
c, p, g, mg, name = D['c'], D['p'], D['g'], D['mine_g'], D['name']
pts = {q['name']: q for q in json.load(open(base + 'sweep.json'))['points']}
mine = g == mg[:, None, None]
pm = p * mine
lo = c < 5
names = sorted(set(name))
F = []
for n in names:
    s = name == n
    row = [(pm[s][:, :, r] * lo[s][:, :, r]).sum(1).mean() for r in [0, 4, 8, 12, 16]]
    row.append((pm[s] * ~lo[s]).sum((1, 2)).mean())
    F.append(row)
F = np.array(F)
labs = ['const', 'r0_low', 'r4_low', 'r8_low', 'r12_low', 'r16_low', 'any_5plus']
for key in ['pool', 'members', 'own', 'other']:
    y = np.array([pts[n][key] for n in names])
    X = np.column_stack([np.ones(len(F)), F])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    sig2 = ((y - X @ b) ** 2).sum() / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(sig2 * np.linalg.inv(X.T @ X)))
    print(key, '  '.join(f'{l} {bb:+.4f}({ss:.4f})' for l, bb, ss in zip(labs, b, se)))
    # pooled block starts
    Xp = np.column_stack([np.ones(len(F)), F[:, :5].sum(1), F[:, 5]])
    b, *_ = np.linalg.lstsq(Xp, y, rcond=None)
    sig2 = ((y - Xp @ b) ** 2).sum() / (len(y) - 3)
    se = np.sqrt(np.diag(sig2 * np.linalg.inv(Xp.T @ Xp)))
    print('   pooled: const %.3f bstart_low %+.4f(%.4f) 5plus %+.4f(%.4f)' % (b[0], b[1], se[1], b[2], se[2]))
print('mean sums', F.mean(0).round(2), 'max', F.max(0).round(1))
