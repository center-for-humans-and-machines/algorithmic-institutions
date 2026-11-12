"""v6: slot fixed-effects over all five sweeps (1280 points). Features: points dealt per game (/10) at rounds 0, 4, 8,
12 split newcomer/stayer x c<=5/>5, plus other phases' totals; linear + squared. Prints coefficients and saves
slot-adjusted leads (/tmp/pf/adj6.csv). Then v5's slot-adjusted surface by params."""
import numpy as np, pandas as pd
z = np.load('/tmp/pf/points.npz')
V = ['v1', 'v2', 'v3', 'v4', 'v5']
cls = ['nlo', 'nhi', 'slo', 'shi']
other = {'r1-3': [1, 2, 3], 'r5-7': [5, 6, 7], 'r9-11': [9, 10, 11], 'r13-19': list(range(13, 20)), 'r20-23': [20, 21, 22, 23]}
rows = []
for v in V:
    F = z[f'{v}_F'] / 10  # [pts, 24, 4]
    d = {}
    d['r0:lo'] = F[:, 0, [0, 2]].sum(1); d['r0:hi'] = F[:, 0, [1, 3]].sum(1)
    for r in [4, 8, 12]:
        for j, cname in enumerate(cls): d[f'r{r}:{cname}'] = F[:, r, j]
    for ph, rr in other.items(): d[ph] = F[:, rr, :].sum((1, 2))
    df = pd.DataFrame(d)
    for k in ['lead', 'own', 'oth', 'mem']: df[k] = z[f'{v}_{k}']
    df['slot'] = z[f'{v}_slot']; df['v'] = v
    rows.append(df)
df = pd.concat(rows, ignore_index=True)
feat = [c for c in df.columns if c.startswith('r')]
X = df[feat].values; X2 = np.concatenate([X, X ** 2], 1)
names = feat + [f + '^2' for f in feat]
S = df.slot.values
def demean(a):
    a = np.asarray(a, float).copy()
    for s in np.unique(S):
        i = S == s; a[i] -= a[i].mean(0)
    return a
Xd = demean(X2)
print('mean pts/game:', ' '.join(f'{n}:{x*10:.1f}' for n, x in zip(feat, X.mean(0))))
for k in ['lead', 'own', 'oth', 'mem']:
    y = demean(df[k].values)
    b, *_ = np.linalg.lstsq(Xd, y, rcond=None); r = y - Xd @ b
    s2 = (r ** 2).sum() / (len(y) - 256 - len(names)); se = np.sqrt(np.diag(s2 * np.linalg.pinv(Xd.T @ Xd)))
    print(f'== {k} resid sd {np.sqrt(s2):.2f}')
    print('  ' + ' '.join(f'{n}:{bb:+.2f}({ss:.2f})' for n, bb, ss in zip(names, b, se)))
    if k == 'lead':
        # slot effects
        alpha = {}
        for s in np.unique(S):
            i = S == s; alpha[s] = (df.lead.values[i] - X2[i] @ b).mean()
        a = np.array([alpha[s] for s in S]); a -= np.mean(list(alpha.values()))
        df['adj'] = df.lead - a
        print('slot effect sd %.2f' % np.std(list(alpha.values())))
print(df.groupby('v')[['lead', 'adj', 'mem']].agg(['mean', 'std']).round(2))
df.to_csv('/tmp/pf/adj6.csv', index=False)
