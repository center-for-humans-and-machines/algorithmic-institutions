"""v6: per-episode outcomes in v5's sweep (both parts), group 0 = ours. For every game:
score (our pool, mean over rounds), members by block, contribution per member by block,
punishment by block, and early states (round-0 group mean, n zero givers, other group's mean).
Saves v6_episodes.parquet for later scripts."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, 'scripts/policy_finder/pf-20261104-5')
from v3_load import load
B = 'plots/simulation/policy_finder/pf-20261104-5-v5_sweep/'
sw = {p['name']: p for p in json.load(open(B + 'sweep.json'))['points']}
out = []
for part in ['per_round_p1of2.parquet', 'per_round_p2of2.parquet']:
    A = load(B + part)
    c, p, g, v = A['c'], A['p'], A['g'] == 0, A['v']
    n = g.sum(2)
    pool = np.where(n > 0, (1.6 * c * g).sum(2) - (p * g).sum(2), 0)
    pool1 = np.where(n < 8, (1.6 * c * ~g).sum(2) - (p * ~g).sum(2), 0)
    D = pd.DataFrame(dict(run=[r.split('managed by ')[1].split('_vs_')[0] for r in A['run']],
                          episode=A['episode'], score=pool.mean(1), other=pool1.mean(1)))
    for b in range(6):
        sl = slice(4 * b, 4 * b + 4)
        D[f'n{b}'] = n[:, sl].mean(1)
        D[f'cm{b}'] = (c * g)[:, sl].sum((1, 2)) / np.maximum(g[:, sl].sum((1, 2)), 1)
        D[f'p{b}'] = (p * g)[:, sl].sum((1, 2))
        D[f'pool{b}'] = pool[:, sl].mean(1)
    D['c0mean'] = (c[:, 0] * g[:, 0]).sum(1) / np.maximum(g[:, 0].sum(1), 1)
    D['c0other'] = (c[:, 0] * ~g[:, 0]).sum(1) / np.maximum((~g[:, 0]).sum(1), 1)
    D['n_low0'] = ((c[:, 0] <= 4) & g[:, 0]).sum(1)
    D['n_hi0'] = ((c[:, 0] >= 12) & g[:, 0]).sum(1)
    D['n_start'] = n[:, 0]
    out.append(D)
    del A
D = pd.concat(out, ignore_index=True)
for k in ['k', 'thr', 'w', 'd', 't_stop']:
    D[k] = [sw[x]['params'][k] for x in D.run]
D.to_parquet('scripts/policy_finder/pf-20261104-5/v6_episodes.parquet')
print(D.describe().T.round(2).to_string())
