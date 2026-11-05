# v7: per-opponent quadratic surface of the pool over v6's params; where is each opponent's optimum?
import json, numpy as np, itertools
s = json.load(open('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-7-v6_sweep/sweep.json'))
pts = s['points']
names = ['c0', 'w', 'k', 'gamma', 'decay', 'follow']
X = np.array([[p['params']['c0'], p['params']['w'], np.log(p['params']['k']), p['params']['gamma'], np.log(p['params']['decay']), p['params']['follow']] for p in pts])
mu, sd = X.mean(0), X.std(0); Z = (X - mu) / sd
nP = len(names)
def design(Z):
    cols = [np.ones(len(Z))] + [Z[:, i] for i in range(nP)] + [Z[:, i] * Z[:, j] for i, j in itertools.combinations_with_replacement(range(nP), 2)]
    return np.stack(cols, 1)
D = design(Z)
grid = np.array(list(itertools.product(np.linspace(-1.7, 1.7, 7), repeat=nP)))
Dg = design(grid)
print('se per opponent per point', np.mean([p['pools']['opponent_1']['pool_se'] for p in pts]))
for o in s['opponents'] + ['all']:
    y = np.array([p['pool'] if o == 'all' else p['pools'][o]['pool'] for p in pts])
    m = np.array([p['members'] if o == 'all' else p['pools'][o]['members'] for p in pts])
    b, *_ = np.linalg.lstsq(D, y, rcond=1e-6)
    pred = Dg @ b; i = pred.argmax(); g = grid[i] * sd + mu
    r2 = 1 - ((y - D @ b) ** 2).mean() / y.var()
    print(o, 'r2 %.2f' % r2, 'peak %.1f' % pred[i], dict(zip(names, [round(v, 2) for v in [g[0], g[1], np.exp(g[2]), g[3], np.exp(g[4]), g[5]]])), 'mean pool %.1f members %.2f' % (y.mean(), m.mean()))
    print('   lin', dict(zip(names, np.round(b[1:nP + 1], 2))))
