"""v3: per point of fine on a member at round t, by phase t%4 (0 = first round of a block,
3 = decision round), the change in the member's own future input to my pool (S_k), in
the round-t co-members' input (O_k) and in my group's size (N_k), k=1..8. OLS with
contribution dummies, c_{t-1}, others' mean, group size, previous fine, round dummies."""
import pandas as pd, numpy as np, numpy.linalg as la
x = pd.read_parquet('/tmp/v3_value.parquet')
x['ph'] = x.t % 4
H = 8
def fit(y, outs):
    X = [np.ones(len(y)), y.p, (y.p > 0), y.op, y.cp, y.om, y.n, y.pprev] + [(y.c == v).values for v in range(1, 21)] + [(y.t == v).values for v in range(2, 23)]
    X = np.column_stack(X).astype(float)
    res = {}
    for o in outs:
        m = y[o].notna().values
        b = la.lstsq(X[m], y[o].values[m], rcond=None)[0]
        res[o] = (b[1], b[2], b[3])
    return res
outs = [f'{a}{k}' for k in range(1, H + 1) for a in 'SON']
for lo, hi in [(0, 3), (4, 10), (11, 20)]:
    for ph in range(4):
        y = x[(x.ph == ph) & (x.c >= lo) & (x.c <= hi) & (x.t <= 15)]
        r = fit(y, outs)
        S = np.array([r[f'S{k}'][0] for k in range(1, H + 1)]); O = np.array([r[f'O{k}'][0] for k in range(1, H + 1)])
        N = np.array([r[f'N{k}'][0] for k in range(1, H + 1)])
        Sa = np.array([r[f'S{k}'][1] for k in range(1, H + 1)])
        Oa = np.array([r[f'O{k}'][1] for k in range(1, H + 1)])
        print(f'c {lo}-{hi} phase {ph} n={len(y)} fined={(y.p>0).mean():.2f}  per point: S {np.round(S,3)} O {np.round(O,3)} N {np.round(N,4)}')
        print(f'   any-fine dummy: S {np.round(Sa,2)} O {np.round(Oa,2)};  8-round value per point 1.6*(S+O) sum = {1.6*(S+O).sum():.2f}  (cost 1)')
