"""v5: slot-adjusted v4 scores. Slot effects from a fixed-effects model over all four sweeps (features: points dealt
per phase, split by target c<=5 / >5, linear and squared); then v4's lead minus its slot effect, against v4's params."""
import numpy as np, pandas as pd
z = np.load('/tmp/pf/feats.npz')
phases = {'r0': [0], 'r1-3': [1, 2, 3], 'r4': [4], 'r5-7': [5, 6, 7], 'r8': [8], 'r9-11': [9, 10, 11], 'r12': [12],
          'r13-19': list(range(13, 20)), 'r20-23': [20, 21, 22, 23]}
V = ['v1', 'v2', 'v3', 'v4']
X, Y, S, VV = [], [], [], []
for v in V:
    F = z[f'{v}_F']; cols = []
    for ph, rr in phases.items():
        a, b = F[:, 0, rr].sum(1) / 10, F[:, 1:, rr].sum((1, 2)) / 10
        cols += [a, b, a ** 2, b ** 2]
    X.append(np.stack(cols, 1)); Y.append(z[f'{v}_lead']); S.append(z[f'{v}_slot']); VV += [v] * len(F)
X = np.concatenate(X); Y = np.concatenate(Y); S = np.concatenate(S); VV = np.array(VV)
D = pd.get_dummies(S).values.astype(float)
A = np.concatenate([X, D], 1)
b, *_ = np.linalg.lstsq(A, Y, rcond=None)
alpha = b[X.shape[1]:]; alpha -= alpha.mean()
res = Y - A @ b
print('resid sd %.2f' % res.std(), 'slot effect sd %.2f' % alpha.std())
adj = Y - alpha[np.searchsorted(np.unique(S), S)]
df = pd.DataFrame(dict(v=VV, slot=S, lead=Y, adj=adj))
for v in V:
    P = z[f'{v}_params']; df.loc[df.v == v, 'pidx'] = np.arange((df.v == v).sum())
print(df.groupby('v')[['lead', 'adj']].agg(['mean', 'std']).round(2))
d4 = df[df.v == 'v4'].copy(); P = z['v4_params']
for i, n in enumerate(['D0', 'D4', 'D8', 'D12', 'hi']): d4[n] = P[:, i]
pd.set_option('display.width', 200)
for n in ['D0', 'D4', 'D8', 'D12']:
    print(n, d4.groupby(pd.cut(d4[n], [0, 5, 10, 15, 20, 25, 30]), observed=True).adj.agg(['mean', 'count']).round(2).T.to_string())
print('hi', d4.groupby('hi').adj.agg(['mean', 'count']).round(2).T.to_string())
# linear+quadratic model in params for v4 (adj)
cols = {}
for n in ['D0', 'D4', 'D8', 'D12']:
    cols[n] = d4[n] / 10; cols[n + '^2'] = (d4[n] / 10) ** 2
for h in range(3, 10): cols[f'hi{h}'] = (d4.hi == h).astype(float)
Xq = pd.DataFrame(cols); Xq['1'] = 1.0
bq, *_ = np.linalg.lstsq(Xq.values, d4.adj.values, rcond=None)
r = d4.adj.values - Xq.values @ bq; s2 = (r ** 2).sum() / (len(r) - Xq.shape[1])
se = np.sqrt(np.diag(s2 * np.linalg.inv(Xq.values.T @ Xq.values)))
print('v4 quad model resid sd %.2f' % np.sqrt(s2)); print(' '.join(f'{c}:{x:+.2f}({s:.2f})' for c, x, s in zip(Xq.columns, bq, se)))
d4.sort_values('adj', ascending=False).head(15).round(2).to_csv('/dev/stdout')
# best of other versions
for v in ['v2', 'v3']:
    dd = df[df.v == v].copy(); P = z[f'{v}_params']
    dd['params'] = [np.round(p, 1).tolist() for p in P]
    print(v, dd.sort_values('adj', ascending=False).head(8)[['slot', 'lead', 'adj', 'params']].round(2).to_string())
df.to_csv('/tmp/pf/adj.csv', index=False)
