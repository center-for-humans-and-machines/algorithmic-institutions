"""v5: per v4 sweep point (group 0 = ours): punishment sums by block (0,1,2,3+), block
position (start vs rest) and the punished member's contribution class, plus outcomes.
Writes v5_points.csv, then regresses the score on the punishment sums."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, 'scripts/policy_finder/pf-20261104-5')
from v3_load import load
B = 'plots/simulation/policy_finder/pf-20261104-5-v4_sweep/'
sw = {p['name']: p for p in json.load(open(B + 'sweep.json'))['points']}
CL = [(0, 4, 'lo'), (5, 8, 'mid'), (9, 11, 'near'), (12, 20, 'hi')]
rows = []
for part in ['per_round_p1of2.parquet', 'per_round_p2of2.parquet']:
    A = load(B + part)
    for run in np.unique(A['run']):
        m = A['run'] == run
        cc, pp = A['c'][m], A['p'][m]
        gg = A['g'][m] == 0
        n = gg.sum(2)
        pool = np.where(n > 0, (1.6 * cc * gg).sum(2) - (pp * gg).sum(2), 0)
        name = run.split('managed by ')[1].split('_vs_')[0]
        row = dict(run=name, score=pool.mean(), n_late=n[:, 12:].mean(),
                   cm_late=(cc * gg)[:, 12:].sum() / max(gg[:, 12:].sum(), 1),
                   ptot=(pp * gg).sum((1, 2)).mean())
        for b, sl in [('b0', slice(0, 4)), ('b1', slice(4, 8)), ('b2', slice(8, 12)), ('b3', slice(12, 24))]:
            for lo, hi, lab in CL:
                cl = (cc >= lo) & (cc <= hi) & gg
                x = (pp * cl)[:, sl]
                row[f'{b}s_{lab}'] = x[:, 0].sum(1).mean()
                row[f'{b}r_{lab}'] = x[:, 1:].sum((1, 2)).mean()
        rows.append(row)
    del A
D = pd.DataFrame(rows)
for k in ['k', 'thr', 'w', 'budget', 't_stop']:
    D[k] = [sw[x]['params'][k] for x in D.run]
D.to_csv('scripts/policy_finder/pf-20261104-5/v5_points.csv', index=False)
import numpy.linalg as la
def reg(y, cols):
    X = np.column_stack([np.ones(len(D))] + [D[c] for c in cols])
    b, *_ = la.lstsq(X, D[y], rcond=None)
    r = D[y] - X @ b
    s2 = (r ** 2).sum() / (len(D) - X.shape[1])
    se = np.sqrt(np.diag(s2 * la.inv(X.T @ X)))
    print(y, 'R2 %.2f' % (1 - r.var() / D[y].var()))
    for c, bb, ss in zip(['const'] + cols, b, se):
        print('  %-12s %8.4f (%.4f)' % (c, bb, ss))
cols = [c for c in D.columns if c[:2] in ('b0', 'b1', 'b2', 'b3') and c[2] in 'sr']
# collapse b3 to one total to limit collinearity
D['b3'] = D[[c for c in cols if c.startswith('b3')]].sum(1)
cols = [c for c in cols if not c.startswith('b3')] + ['b3']
for y in ['score', 'n_late', 'cm_late']:
    reg(y, cols)
print(D[['score'] + cols].describe().T.round(2))
