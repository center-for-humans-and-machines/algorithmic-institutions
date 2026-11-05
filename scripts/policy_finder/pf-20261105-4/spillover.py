"""Spillover and dose response in the humanlike reference sim (clones)."""
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings('ignore')
sim = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
d = sim[sim.run.str.endswith('humanlike')].copy()
d['pl'] = d.participant_code.str.split('_').str[0].astype(int)
d.loc[~d.contribution_valid, 'contribution'] = np.nan
C = d.pivot_table(index=['episode', 'round_number'], columns='pl', values='contribution', dropna=False)
P = d.pivot_table(index=['episode', 'round_number'], columns='pl', values='punishment')
C = C.unstack('round_number'); P = P.unstack('round_number')
c = C.values.reshape(len(C), 4, 24) if False else np.stack([C[i].values for i in range(4)], 1)
p = np.stack([P[i].values for i in range(4)], 1)
rows = []
for t in range(23):
    for i in range(4):
        o = [j for j in range(4) if j != i]
        rows.append(pd.DataFrame(dict(
            t=t, c=c[:, i, t], c1=c[:, i, t + 1], p=p[:, i, t],
            po=np.nansum(p[:, o, t], 1), co=np.nanmean(c[:, o, t], 1),
            minc=np.nanmin(c[:, :, t], 1))))
x = pd.concat(rows).dropna()
x['dc'] = x.c1 - x.c
# spillover: unpunished high contributors, by punishment dealt to others
y = x[(x.p == 0) & (x.c >= 10)]
print('unpunished c>=10 players: dc by total punishment dealt to others')
print(y.groupby(pd.cut(y.po, [-1, 0, 5, 10, 20, 90])).dc.agg(['mean', 'count']).round(2))
y = x[(x.p == 0) & (x.c >= 10)]
print('same, by others mean contribution')
print(y.groupby(pd.cut(y.co, [-1, 5, 10, 15, 20])).dc.agg(['mean', 'count']).round(2))
# dose response for low contributors
for lo, hi in ((0, 5), (6, 10), (11, 15)):
    z = x[(x.c >= lo) & (x.c <= hi)]
    print(f'c {lo}-{hi}: dc by punishment')
    print(z.groupby(pd.cut(z.p, [-1, 0, 2, 4, 6, 8, 10, 14, 20, 30])).dc.agg(['mean', 'count']).round(2))
# by round: are clones responding differently late?
z = x[x.c <= 10]
print('c<=10 dc by round band and punished or not')
print(z.pivot_table(index=pd.cut(z.t, [-1, 3, 7, 11, 15, 22]), columns=z.p > 3, values='dc').round(2))
