# v4: per-player effect of the first and the second hit (v3 sweep, my group only):
# contribution in rounds 2-3 and 9-11 vs punishment in round 0 and round 1, by round-0 contribution
import pandas as pd, numpy as np, pyarrow.parquet as pq, re, statsmodels.api as sm
base = '/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-5-v3_sweep/'
cols = ['episode', 'participant_code', 'round_number', 'punishment', 'contribution', 'group_id', 'run']
out = []
for f in ['per_round_p1of2.parquet', 'per_round_p2of2.parquet']:
    import pyarrow.compute as pc
    tb = pq.read_table(base + f, columns=cols, read_dictionary=['run'], filters=[('round_number', '<', 12)])
    d = tb.to_pandas(); del tb
    for k in ['punishment', 'contribution', 'group_id', 'round_number']: d[k] = d[k].astype('int8')
    cats = d.run.cat.categories
    seat = np.array([0 if re.search(r'managed by pf-', r) else 1 for r in cats])
    d['seat'] = seat[d.run.cat.codes.values]
    d['run'] = d.run.cat.codes
    d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
    n = len(d) // 96
    P = d.punishment.values.reshape(n, 8, 12).astype(float)
    C = d.contribution.values.reshape(n, 8, 12).astype(float)
    G = d.group_id.values.reshape(n, 8, 12)
    S = d.seat.values.reshape(n, 8, 12)[:, :, 0]
    del d
    mine0 = G[:, :, 0] == S
    x = pd.DataFrame({'c0': C[:, :, 0].ravel(), 'c1': C[:, :, 1].ravel(), 'p0': P[:, :, 0].ravel(), 'p1': P[:, :, 1].ravel(),
                      'c23': C[:, :, 2:4].mean(-1).ravel(), 'c911': C[:, :, 9:12].mean(-1).ravel(),
                      'stay4': (G[:, :, 4] == G[:, :, 0]).ravel(), 'stay12': (G[:, :, 9:12] == G[:, :, :1]).all(-1).ravel(),
                      'mine': mine0.ravel()})
    out.append(x[x.mine])
X = pd.concat(out, ignore_index=True)
X['cb'] = pd.cut(X.c0, [-1, 2, 5, 8, 11, 20])
X['p0b'] = pd.cut(X.p0, [-1, 0, 5, 10, 20, 29, 30])
X['p1b'] = pd.cut(X.p1, [-1, 0, 10, 29, 30])
pd.set_option('display.width', 250)
for v in ['c1', 'c23', 'c911', 'stay4']:
    print(v, 'by c0 band x p0 band')
    print(X.pivot_table(index='cb', columns='p0b', values=v, aggfunc='mean', observed=True).round(2))
print('counts'); print(X.pivot_table(index='cb', columns='p0b', values='c1', aggfunc='size', observed=True))
# second hit: among c0<=5 hit 30 in round 0, by c1 band and p1
Y = X[(X.c0 <= 5) & (X.p0 >= 25)].copy()
Y['c1b'] = pd.cut(Y.c1, [-1, 3, 6, 9, 20])
for v in ['c23', 'c911', 'stay4']:
    print(v, 'hit 25+ at round 0 (c0<=5): by c1 band x p1 band')
    print(Y.pivot_table(index='c1b', columns='p1b', values=v, aggfunc='mean', observed=True).round(2))
print(Y.pivot_table(index='c1b', columns='p1b', values='c23', aggfunc='size', observed=True))
# OLS on c911 for stayers, low contributors
Z = X[X.stay12 & (X.c0 <= 8)]
m = sm.OLS(Z.c911, sm.add_constant(Z[['c0', 'c1', 'p0', 'p1']])).fit(); print('c911 stayers c0<=8', m.params.round(3).to_dict())
m = sm.OLS(Z.c911, sm.add_constant(Z[['c0', 'p0']])).fit(); print('c911 ~ c0 + p0', m.params.round(3).to_dict())
