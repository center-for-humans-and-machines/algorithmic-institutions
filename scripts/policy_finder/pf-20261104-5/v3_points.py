"""v3: per v2 sweep point (our group = group 0), summaries of how much punishment went where
(decision rounds vs others; by the punished member's contribution class) and the outcomes
(score, late members, late member contribution). Writes v3_points.csv."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, 'scripts/policy_finder/pf-20261104-5')
from v3_load import load
B = 'plots/simulation/policy_finder/pf-20261104-5-v2_sweep/'
sw = {p['name'] if 'name' in p else None: p for p in json.load(open(B + 'sweep.json'))['points']}
rows = []
for part in ['per_round_p1of2.parquet', 'per_round_p2of2.parquet']:
    A = load(B + part)
    c, p, g, v = A['c'], A['p'], A['g'], A['v']
    ing = (g == 0)
    for run in np.unique(A['run']):
        m = A['run'] == run
        cc, pp, gg, vv = c[m], p[m], ing[m], v[m]
        n = gg.sum(2)  # games x rounds
        pool = (1.6 * cc * gg).sum(2) - (pp * gg * vv).sum(2)
        pool = np.where(n > 0, pool, 0)
        pm = (pp * gg).sum(2)  # total punishment per round
        r = np.arange(24)
        dec = np.isin(r, [3, 7, 11, 15, 19])
        early = r <= 11
        row = dict(run=run.split('managed by ')[1].split('_vs_')[0], score=pool.mean(),
                   P_dec=pm[:, dec & early].sum(1).mean(), P_non=pm[:, ~dec & early].sum(1).mean(),
                   P_late=pm[:, r >= 12].sum(1).mean(),
                   n_late=n[:, 12:].mean(), cm_late=((cc * gg).sum(2)[:, 12:].sum() / max(gg[:, 12:].sum(), 1)),
                   n_4=n[:, 4:8].mean(), pool_03=pool[:, :4].mean())
        for lo, hi, lab in [(0, 2, 'P_c0_2'), (3, 6, 'P_c3_6'), (7, 10, 'P_c7_10'), (11, 20, 'P_c11_')]:
            cl = (cc >= lo) & (cc <= hi) & gg
            row[lab] = (pp * cl)[:, :12].sum((1, 2)).mean()
        rows.append(row)
    del A, c, p, g, v
D = pd.DataFrame(rows)
for k in ['k', 'thr', 'cap', 'decay', 't_stop']:
    D[k] = [sw.get(f'pf-20261104-5-v2_s{x.split("_s")[-1]}', {}).get('params', {}).get(k, np.nan) if '_s' in x else np.nan for x in D.run]
D.to_csv('scripts/policy_finder/pf-20261104-5/v3_points.csv', index=False)
print(D.describe().T.round(2))
