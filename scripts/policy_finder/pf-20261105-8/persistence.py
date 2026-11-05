"""Event study in the humanlike sim: a low contributor punished (or not) at round t,
its contribution and the others' over t+1..t+8. Also the same in the human pilots."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
r = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
r['player'] = r.participant_code.str.split('_').str[0].astype(int)
r = r[r.run.str.contains('humanlike')]
C = r.pivot_table(index=['episode', 'player'], columns='round_number', values='contribution').values.reshape(1000, 4, 24)
V = r.pivot_table(index=['episode', 'player'], columns='round_number', values='contribution_valid').values.reshape(1000, 4, 24).astype(bool)
P = r.pivot_table(index=['episode', 'player'], columns='round_number', values='punishment').values.reshape(1000, 4, 24)
C = np.where(V, C, np.nan)
rows = []
for t in range(2, 14):
    for i in range(4):
        c = C[:, i, t]; p = P[:, i, t]
        others = np.nanmean(np.delete(C, i, axis=1), axis=1)  # [ep, round]
        sel = (c <= 6) & V[:, i, t]
        for e in np.where(sel)[0]:
            rows.append(dict(t=t, p=p[e], c=c[e], cprev=C[e, i, t-1], oth=others[e, t],
                             **{f'd{k}': C[e, i, t+k] - c[e] for k in range(1, 9) if t+k < 24},
                             **{f'o{k}': others[e, t+k] - others[e, t] for k in range(1, 9) if t+k < 24}))
d = pd.DataFrame(rows)
d['pbin'] = pd.cut(d.p, [-1, 0, 3, 7, 12, 30])
cols = [f'd{k}' for k in range(1, 9)] + [f'o{k}' for k in (1, 3, 5, 8)]
print(d.groupby('pbin')[cols].mean().round(2))
print(d.groupby('pbin').size())
# regression to control for c, cprev, oth
import numpy.linalg as la
for k in (1, 3, 5, 8):
    dd = d.dropna(subset=[f'd{k}', 'cprev'])
    X = np.column_stack([np.ones(len(dd)), dd.p, dd.c, dd.cprev, dd.oth, dd.t])
    b = la.lstsq(X, dd[f'd{k}'], rcond=None)[0]
    bo = la.lstsq(X, dd[f'o{k}'], rcond=None)[0] if f'o{k}' in dd else [np.nan]*6
    print(k, 'own per p-point', round(b[1], 3), 'others per p-point', round(bo[1], 3))
print('--- bin dummies with controls (c, cprev, oth, t); own and others(mean of 3) change')
bins = [(0.5, 1.5), (1.5, 3.5), (3.5, 7.5), (7.5, 12.5), (12.5, 30)]
for k in (1, 3, 5, 8):
    dd = d.dropna(subset=[f'd{k}', 'cprev'])
    D = np.column_stack([((dd.p > lo) & (dd.p <= hi)).astype(float) for lo, hi in bins])
    X = np.column_stack([np.ones(len(dd)), D, dd.c, dd.cprev, dd.oth, dd.t])
    b = la.lstsq(X, dd[f'd{k}'], rcond=None)[0]
    bo = la.lstsq(X, dd[f'o{k}'], rcond=None)[0]
    print(k, 'own', np.round(b[1:6], 2), 'others', np.round(bo[1:6], 2))
print(d.groupby(pd.cut(d.p, [-1, 0, 1, 3, 7, 12, 30]))[['c', 'cprev', 'oth']].mean().round(2))
