"""v6: every version's final uses seed 1000, so the round-0 lineups (and most early draws) are identical across
final points: common random numbers. Paired differences of game scores against v4's GP peak, with paired se."""
import numpy as np, json
B = 'plots/simulation/policy_finder/'
sc = {}; par = {}
for tag, ver in [('v4f', 'pf-20261105-11-v4'), ('v5f', 'pf-20261105-11-v5'), ('v3f', 'pf-20261105-11-v3')]:
    z = np.load(f'/tmp/pf6/{tag}.npz', allow_pickle=True)
    c = z['c'].astype(float); p = z['p'].astype(float); v = z['v'].astype(float); c = c * v
    n = v.sum(1)
    pay = 20 - c - p + ((1.6 * c.sum(1) - p.sum(1)) / np.maximum(n, 1))[:, None, :]
    s = ((pay * v).sum(1) / np.maximum(n, 1) * (n > 0)).sum(1)
    runs = np.array([r.split('managed by ')[1] for r in z['run']])
    sw = json.load(open(f'{B}{ver}_sweep/final/sweep.json'))
    for pt in sw['points']:
        m = runs == pt['name']
        sc[pt['name']] = s[m]; par[pt['name']] = pt['params']
        assert abs(s[m].mean() - pt['pool']) < 0.05, (pt['name'], s[m].mean(), pt['pool'])
ref = sc['pf-20261105-11-v4_gp']
for k in sorted(sc, key=lambda k: -sc[k].mean()):
    d = sc[k] - ref
    print(f'{k:24s} {sc[k].mean():6.1f} diff {d.mean():+5.2f} paired se {d.std()/np.sqrt(len(d)):.2f}  ',
          {a: round(b, 2) for a, b in par[k].items()})
