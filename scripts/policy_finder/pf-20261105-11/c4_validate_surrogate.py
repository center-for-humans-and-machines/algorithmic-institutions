"""v3: does the surrogate reproduce the real sweep scores? Held-out and in-sample v1/v2 points,
the finals, and zero."""
import json, numpy as np, torch as th, sys
sys.path.insert(0, 'scripts/policy_finder/pf-20261105-11')
from surrogate import load
sim = load()
held = th.load('/tmp/pf3/surrogate.pt', weights_only=False)['held']

def r_v1(T, P, g, E):
    def f(c, v, p, t):
        s = th.relu(T - c[..., -1]) / T
        x = th.where(s > 0, P * s ** g, th.zeros_like(s))
        return x if t <= 23 - E else th.zeros_like(x)
    return f

def r_v2(T, A, B, S):
    def f(c, v, p, t):
        s = th.relu(T - c[..., -1])
        x = th.where(s > 0, A + B * s, th.zeros_like(s))
        return x if t <= S else th.zeros_like(x)
    return f

out = []
for tag, fn, name in [('v1', r_v1, 'pf-20261105-11'), ('v2', r_v2, 'pf-20261105-11-v2')]:
    sw = json.load(open(f'plots/simulation/policy_finder/{name}_sweep/sweep.json'))
    hs = set(h.split()[-1] for h in held[tag])
    for i, pt in enumerate(sw['points']):
        if pt['name'] in hs or i % 8 == 0:
            r = sim.run(fn(**pt['params']), B=1000, seed=i)
            out.append((tag, pt['name'] in hs, pt['pool'], r['score']))
    fin = json.load(open(f'plots/simulation/policy_finder/{name}_sweep/final/sweep.json'))
    for pt in fin['points'][:3]:
        r = sim.run(fn(**pt['params']), B=4000, seed=7)
        print('final', tag, pt['params'], 'real', round(pt['pool'], 1), 'sur', round(r['score'], 1))
r = sim.run(lambda c, v, p, t: th.zeros_like(c[..., -1]), B=4000)
print('zero sur', round(r['score'], 1), 'c by round', r['c_round'].round(1))
o = np.array([(a[2], a[3], a[1]) for a in out])
for h in [0, 1]:
    m = o[:, 2] == h
    print('held' if h else 'in', m.sum(), 'corr', np.corrcoef(o[m, 0], o[m, 1])[0, 1].round(3),
          'bias', (o[m, 1] - o[m, 0]).mean().round(1), 'rmse', np.sqrt(((o[m, 1] - o[m, 0]) ** 2).mean()).round(1))
top = o[o[:, 0] > 630]
print('top pts', len(top), 'corr', np.corrcoef(top[:, 0], top[:, 1])[0, 1].round(3), 'bias', (top[:, 1] - top[:, 0]).mean().round(1))
