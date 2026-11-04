"""v4: per v3 sweep point (our group = group 0): punishment by round window and by the
punished member's contribution class, plus outcomes (score, members per block, member
contribution per block, other group's score). Writes v4_points.csv."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, 'scripts/policy_finder/pf-20261104-5')
from v3_load import load
B = 'plots/simulation/policy_finder/pf-20261104-5-v3_sweep/'
sw = {p['name']: p for p in json.load(open(B + 'sweep.json'))['points']}
rows = []
for part in ['per_round_p1of2.parquet', 'per_round_p2of2.parquet']:
    A = load(B + part)
    for run in np.unique(A['run']):
        m = A['run'] == run
        cc, pp, vv = A['c'][m], A['p'][m], A['v'][m]
        gg = A['g'][m] == 0
        oo = A['g'][m] == 1
        n = gg.sum(2)
        pool = np.where(n > 0, (1.6 * cc * gg).sum(2) - (pp * gg).sum(2), 0)
        no = oo.sum(2)
        opool = np.where(no > 0, (1.6 * cc * oo).sum(2) - (pp * oo).sum(2), 0)
        name = run.split('managed by ')[1].split('_vs_')[0]
        row = dict(run=name, score=pool.mean(), oscore=opool.mean())
        for b in range(6):
            s = slice(4 * b, 4 * b + 4)
            row[f'n{b}'] = n[:, s].mean()
            row[f'cm{b}'] = (cc * gg)[:, s].sum() / max(gg[:, s].sum(), 1)
            row[f'pool{b}'] = pool[:, s].mean()
        pm = (pp * gg).sum(2)
        for r in range(6):
            row[f'P_r{r}'] = pm[:, r].mean()
        row['P_r6_11'] = pm[:, 6:12].sum(1).mean()
        row['P_r12_'] = pm[:, 12:].sum(1).mean()
        for lo, hi, lab in [(0, 2, 'c0_2'), (3, 6, 'c3_6'), (7, 10, 'c7_10'), (11, 20, 'c11_')]:
            cl = (cc >= lo) & (cc <= hi) & gg
            row['P0_' + lab] = (pp * cl)[:, :3].sum((1, 2)).mean()
            row['P3_' + lab] = (pp * cl)[:, 3:].sum((1, 2)).mean()
        rows.append(row)
    del A
D = pd.DataFrame(rows)
for k in ['k', 'thr', 'cap', 'budget', 'quiet', 't_stop']:
    D[k] = [sw.get(x, {}).get('params', {}).get(k, np.nan) for x in D.run]
D['pool_sw'] = [sw.get(x, {}).get('pool', np.nan) for x in D.run]
D.to_csv('scripts/policy_finder/pf-20261104-5/v4_points.csv', index=False)
print(D.describe().T.round(2))
