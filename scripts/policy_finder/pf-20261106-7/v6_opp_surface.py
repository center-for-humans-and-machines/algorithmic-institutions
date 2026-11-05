# v6: per-opponent quadratic surface of the pool over v5's params; where is each opponent's optimum?
import json, numpy as np, itertools
s = json.load(open('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-7-v5_sweep/sweep.json'))
pts = s['points']
names = ['c0', 'w', 'k', 'decay', 'follow']
X = np.array([[p['params']['c0'], p['params']['w'], np.log(p['params']['k']), np.log(p['params']['decay']), p['params']['follow']] for p in pts])
mu, sd = X.mean(0), X.std(0); Z = (X - mu) / sd
def design(Z):
    cols = [np.ones(len(Z))] + [Z[:, i] for i in range(5)] + [Z[:, i] * Z[:, j] for i, j in itertools.combinations_with_replacement(range(5), 2)]
    return np.stack(cols, 1)
D = design(Z)
grid = np.array(list(itertools.product(np.linspace(-1.7, 1.7, 9), repeat=5)))
Dg = design(grid)
for o in s['opponents'] + ['all']:
    y = np.array([p['pool'] if o == 'all' else p['pools'][o]['pool'] for p in pts])
    m = np.array([p['members'] if o == 'all' else p['pools'][o]['members'] for p in pts])
    b, *_ = np.linalg.lstsq(D, y, rcond=None)
    pred = Dg @ b; i = pred.argmax(); g = grid[i] * sd + mu
    r2 = 1 - ((y - D @ b) ** 2).mean() / y.var()
    print(o, 'r2 %.2f' % r2, 'peak %.1f' % pred[i], dict(zip(names, [round(v, 2) for v in [g[0], g[1], np.exp(g[2]), np.exp(g[3]), g[4]]])))
    # linear slopes on standardized
    print('   lin', dict(zip(names, np.round(b[1:6], 2))))
