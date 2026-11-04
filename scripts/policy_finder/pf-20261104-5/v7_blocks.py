"""v7: per game and block table of v6's sweep (both parts), group 0 = ours.
Per block b: our size n_b (mean over block), our punishment total P_b and per member,
our contribution per member cm_b, other group's cm and punishment, share gap, and the
design params. Saves v7_blocks.parquet."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, 'scripts/policy_finder/pf-20261104-5')
from v3_load import load
B = 'plots/simulation/policy_finder/pf-20261104-5-v6_sweep/'
sw = {p['name']: p for p in json.load(open(B + 'sweep.json'))['points']}
out = []
for part in ['per_round_p1of2.parquet', 'per_round_p2of2.parquet']:
    A = load(B + part)
    c, p, g = A['c'], A['p'], A['g'] == 0
    n = g.sum(2)
    pool = np.where(n > 0, (1.6 * c * g).sum(2) - (p * g).sum(2), 0)
    pool1 = np.where(n < 8, (1.6 * c * ~g).sum(2) - (p * ~g).sum(2), 0)
    run = np.array([r.split('managed by ')[1].split('_vs_')[0] for r in A['run']])
    for b in range(6):
        sl = slice(4 * b, 4 * b + 4)
        gb, cb, pb = g[:, sl], c[:, sl], p[:, sl]
        nb = n[:, 4 * b]
        D = pd.DataFrame(dict(run=run, episode=A['episode'], block=b, n=nb,
                              P=(pb * gb).sum((1, 2)), Po=(pb * ~gb).sum((1, 2)),
                              cm=(cb * gb).sum((1, 2)) / np.maximum(gb.sum((1, 2)), 1),
                              cmo=(cb * ~gb).sum((1, 2)) / np.maximum((~gb).sum((1, 2)), 1),
                              pool=pool[:, sl].mean(1), poolo=pool1[:, sl].mean(1),
                              score=pool.mean(1)))
        D['n_next'] = n[:, 4 * b + 4] if b < 5 else np.nan
        # contribution per member in the rest of the game after this block
        rest = slice(4 * b + 4, 24)
        if b < 5:
            D['cm_rest'] = (c[:, rest] * g[:, rest]).sum((1, 2)) / np.maximum(g[:, rest].sum((1, 2)), 1)
            D['pool_rest'] = pool[:, rest].mean(1)
        out.append(D)
    del A
D = pd.concat(out, ignore_index=True)
for k in ['k', 'thr', 'w', 't_stop', 'm']:
    D[k] = D.run.map(lambda x: sw[x]['params'][k])
D.to_parquet('scripts/policy_finder/pf-20261104-5/v7_blocks.parquet')
print(D.groupby('block')[['n', 'P', 'Po', 'cm', 'cmo', 'pool', 'poolo']].mean().round(2))
