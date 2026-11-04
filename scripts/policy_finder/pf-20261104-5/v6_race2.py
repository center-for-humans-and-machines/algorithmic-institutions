"""v6: marginal value of block-0 harshness at v5's peak level (~77 points), by race state.
State: round-0 contribution gap (ours - theirs, before any punishment). Per state bin, fit
score/dn/cm_late ~ h + h^2 + c0mean over all v5 designs (h = design's mean block-0
punishment / 10) and read the slope at h = 7.7. Also by v5 design t_stop>=4 only.
Also the same with the round-0 *share* gap after round-0 punishments (what the guard sees)."""
import numpy as np, pandas as pd
from numpy.linalg import lstsq
s = pd.read_parquet('scripts/policy_finder/pf-20261104-5/v6_race.parquet')
E = pd.read_parquet('scripts/policy_finder/pf-20261104-5/v6_episodes.parquet').iloc[:64000].reset_index(drop=True)
s['h'] = s.groupby('run').pblk.transform('mean') / 10
s['c0gap'] = E.c0mean - E.c0other; s['c0mean'] = E.c0mean
h0 = 7.7
rows = []
for gb, q in s.groupby(pd.cut(s.c0gap, [-20, -4, -2, -0.5, 0.5, 2, 4, 20]), observed=True):
    X = np.column_stack([np.ones(len(q)), q.h - h0, (q.h - h0) ** 2, q.c0mean])
    r = {'bin': str(gb), 'n': len(q)}
    for y in ['score', 'dn', 'cm_late', 'n_late']:
        b, *_ = lstsq(X, q[y].values, rcond=None)
        res = q[y].values - X @ b
        se = np.sqrt((res ** 2).sum() / (len(q) - 4) * np.linalg.inv(X.T @ X)[1, 1])
        r[y] = f'{b[1]:+.2f}({se:.2f})'
    rows.append(r)
print(pd.DataFrame(rows).to_string())
