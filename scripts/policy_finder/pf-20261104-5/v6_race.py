"""v6: is the block-0 punishment's value state dependent? Joins v6_migr (round-3 rows, part 1)
with v6_episodes, defines G0 = block-0 share gap with our block-0 punishment added back
(the gap we would have had without punishing, ignoring its lift), and within G0 bins
regresses the score, dn and late cm on our block-0 punishment."""
import numpy as np, pandas as pd
from numpy.linalg import lstsq
M = pd.read_parquet('scripts/policy_finder/pf-20261104-5/v6_migr.parquet')
E = pd.read_parquet('scripts/policy_finder/pf-20261104-5/v6_episodes.parquet')
E = E.iloc[:64000].reset_index(drop=True)   # part 1, same game order as v3_load
s = M[M.r == 3].reset_index(drop=True)
assert (s.run.values == E.run.values).all()
s['score'] = E.score; s['cm_late'] = E[['cm3', 'cm4', 'cm5']].mean(1); s['n_late'] = E[['n3', 'n4', 'n5']].mean(1)
s['dn'] = s.n_after - 4
s['gap'] = s.s0b - s.s1b
s['G0'] = s.gap + s.pblk / 16
bins = [-60, -12, -6, -3, 0, 3, 6, 12, 60]
s['Gb'] = pd.cut(s.G0, bins)
print('G0 distribution', s.G0.describe().round(2).to_dict())
for gb, q in s.groupby('Gb', observed=True):
    X = np.column_stack([np.ones(len(q)), q.pblk / 10])
    out = []
    for y in ['score', 'dn', 'cm_late', 'n_late']:
        b, *_ = lstsq(X, q[y].values, rcond=None)
        out.append(f'{y} {b[0]:.1f} {b[1]:+.2f}')
    print(gb, len(q), 'per 10 pts of block-0 punishment:', ' | '.join(out))
s.drop(columns="Gb").to_parquet("scripts/policy_finder/pf-20261104-5/v6_race.parquet")
# design-based version: the state is the round-0 contribution gap (set before any punishment),
# the treatment the rule's harshness h = its mean block-0 punishment over all its games
s['h'] = s.groupby('run').pblk.transform('mean') / 10
s['c0gap'] = E.c0mean - E.c0other
s['c0mean'] = E.c0mean
print('c0gap sd', s.c0gap.std().round(2))
for gb, q in s.groupby(pd.cut(s.c0gap, [-20, -4, -2, -1, 0, 1, 2, 4, 20]), observed=True):
    X = np.column_stack([np.ones(len(q)), q.h, q.c0mean])
    out = []
    for y in ['score', 'dn', 'cm_late', 'n_late']:
        b, *_ = lstsq(X, q[y].values, rcond=None)
        out.append(f'{y} {q[y].mean():.1f} {b[1]:+.2f}')
    print(gb, len(q), 'per 10 pts of design harshness:', ' | '.join(out))
