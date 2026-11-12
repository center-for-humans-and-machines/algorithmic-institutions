"""v5: slot-adjusted v2 and v3 scores (from v5_adj.py's /tmp/pf/adj.csv) by v2's lo and hi and v3's hi (ex-ante params)."""
import numpy as np, pandas as pd
z = np.load('/tmp/pf/feats.npz'); df = pd.read_csv('/tmp/pf/adj.csv')
d2 = df[df.v == 'v2'].copy(); P = z['v2_params']
for i, n in enumerate(['D', 'lo', 'hi', 'w', 'blocks']): d2[n] = P[:, i]
print('v2 by lo'); print(d2.groupby('lo').adj.agg(['mean', 'sem', 'count']).round(2).T.to_string())
print('v2 by hi'); print(d2.groupby('hi').adj.agg(['mean', 'sem', 'count']).round(2).T.to_string())
X = pd.get_dummies(d2[['lo']].astype(int).astype(str)).astype(float)
for n in ['hi', 'w', 'blocks']: X = X.join(pd.get_dummies(d2[n].astype(int), prefix=n, drop_first=True).astype(float))
X['logD'] = np.log(d2.D)
b, *_ = np.linalg.lstsq(X.values, d2.adj.values, rcond=None); r = d2.adj.values - X.values @ b
se = np.sqrt(np.diag(((r ** 2).sum() / (len(r) - X.shape[1])) * np.linalg.pinv(X.values.T @ X.values)))
print(' '.join(f'{c}:{x:+.2f}({s:.2f})' for c, x, s in zip(X.columns, b, se)))
d3 = df[df.v == 'v3'].copy(); P = z['v3_params']
for i, n in enumerate(['D', 'hi', 'first', 'nblocks', 'w']): d3[n] = P[:, i]
print('v3 by hi'); print(d3.groupby('hi').adj.agg(['mean', 'sem', 'count']).round(2).T.to_string())
print('v3 by first'); print(d3.groupby('first').adj.agg(['mean', 'sem', 'count']).round(2).T.to_string())
