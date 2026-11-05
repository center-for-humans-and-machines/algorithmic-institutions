# v3: contribution response to punishment by block position and contribution level (v2 final data, all managers)
import pandas as pd, numpy as np, pyarrow.parquet as pq, statsmodels.api as sm
base = '/opt/algorithmic-institutions/plots/simulation/policy_finder/'
cols = ['episode', 'participant_code', 'round_number', 'punishment', 'contribution', 'group_id', 'run']
d = pq.read_table(base + 'pf-20261106-5-v2_sweep/final/per_round.parquet', columns=cols, read_dictionary=['run']).to_pandas()
d['run'] = d.run.cat.codes
d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
n = len(d) // 192
P = d.punishment.values.reshape(n, 8, 24).astype(float)
C = d.contribution.values.reshape(n, 8, 24).astype(float)
G = d.group_id.values.reshape(n, 8, 24)
del d
rows = []
for t in range(0, 20):
    for k in [1, 4]:
        if t + k > 23: continue
        x = pd.DataFrame({'c': C[:, :, t].ravel(), 'p': P[:, :, t].ravel(), 'dc': (C[:, :, t + k] - C[:, :, t]).ravel(),
                          'stay': (G[:, :, t + k] == G[:, :, t]).ravel()})
        x = x.sample(frac=0.15, random_state=t)
        x['t'] = t; x['k'] = k; x['pos'] = t % 4
        rows.append(x)
X = pd.concat(rows, ignore_index=True)
X['cb'] = pd.cut(X.c, [-1, 3, 7, 11, 15, 20])
X['pb'] = pd.cut(X.p, [-1, 0, 4, 9, 15, 30])
pd.set_option('display.width', 250)
for k in [1, 4]:
    print('k', k, '(stayers within k rounds)')
    print(X[(X.k == k)].pivot_table(index='cb', columns='pb', values='dc', aggfunc='mean', observed=True).round(2))
# slope of dc on p for c<=11, by pos and k
for k in [1, 4]:
    for pos in range(4):
        y = X[(X.k == k) & (X.pos == pos) & (X.c <= 11)]
        m = sm.OLS(y.dc, sm.add_constant(y[['p', 'c']])).fit()
        print('k', k, 'pos', pos, 'slope p', round(m.params.p, 3), 'n', len(y))
# by phase of game
for k in [4]:
    for lo, hi in [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20)]:
        y = X[(X.k == k) & (X.t >= lo) & (X.t < hi) & (X.c <= 11)]
        m = sm.OLS(y.dc, sm.add_constant(y[['p', 'c']])).fit()
        print('k', k, 'rounds', lo, hi, 'slope p', round(m.params.p, 3))
