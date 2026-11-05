"""v7: value of one extra late-phase round at round t, for a free rider, first vs repeat hit.
Uses v5 and v6 sweeps (both: free riders c<2 punished in rounds horizon<=t<end).
Compares episodes of rules with end == t+1 (hit at t only) vs end == t (no hit from t on).
Outcome: sum over rounds t..23 of the group's mean valid payoff; also mean contribution of others/own t+1..t+6."""
import numpy as np
def payoff(C, P, V):
    C = C.astype(float) * V; P = P.astype(float) * V
    n = V.sum(1, keepdims=True).clip(1)
    pay = 20 - C - P + (1.6 * C.sum(1, keepdims=True) - P.sum(1, keepdims=True)) / n
    return (pay * V).sum(1) / n[:, 0]          # [ep, 24]
rows = []
for f, hcol, ecol in (('/tmp/v6sweep.npz', 2, 3), ('/tmp/v5sweep.npz', 3, 4)):
    z = np.load(f); C, P, V, R, prm = z['C'].astype(float), z['P'].astype(float), z['V'], z['R'], z['prm']
    H = prm[R, hcol].astype(int); E = prm[R, ecol].astype(int)
    G = payoff(C, P, V); Gc = np.cumsum(G[:, ::-1], 1)[:, ::-1]   # sum from t on
    for t in range(9, 20):
        sel_e = ((E == t + 1) | (E == t)) & (H <= t - 1)
        for i in range(4):
            fr = sel_e & V[:, i, t] & (C[:, i, t] < 2)
            idx = np.where(fr)[0]
            if len(idx) == 0: continue
            prior = (P[idx, i, :t] > 0) & (np.arange(t)[None] >= H[idx, None])   # late-phase hits before t
            nprior = prior.sum(1)
            open_hits = ((P[idx, i, :t] > 0) & (np.arange(t)[None] < H[idx, None])).sum(1)
            prevc = C[idx, i, t - 1]
            oth = [j for j in range(4) if j != i]
            othc = (C[idx][:, oth, t]).mean(1)
            fut = min(t + 6, 23)
            own_f = C[idx, i, t + 1:fut + 1].mean(1); oth_f = C[idx][:, oth, t + 1:fut + 1].mean((1, 2))
            for k, e in enumerate(idx):
                rows.append((t, int(E[e] == t + 1), nprior[k], open_hits[k], prevc[k], othc[k], Gc[e, t], own_f[k], oth_f[k], P[e, i, t]))
A = np.array(rows)
t, tr, npri, oh, prevc, othc, g, own, oth, pt = A.T
print('n', len(A), 'treated', tr.sum(), 'mean dose when treated', pt[tr == 1].mean())
def reg(mask, name):
    X = np.column_stack([np.ones(mask.sum()), tr[mask], othc[mask], prevc[mask]] + [(t[mask] == k).astype(float) for k in range(10, 20)])
    for yn, y in (('payoff t..23', g), ('own c +1..6', own), ('others c +1..6', oth)):
        b, res, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
        r = y[mask] - X @ b; s2 = r @ r / (len(r) - X.shape[1])
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
        print(f'{name:28s} n={mask.sum():5d} {yn:15s} effect {b[1]:+6.2f} (se {se[1]:.2f})')
reg(np.ones(len(A), bool), 'all')
reg(npri == 0, 'first late hit')
reg(npri > 0, 'repeat late hit')
reg((npri == 0) & (prevc >= 5), 'first, prev c>=5 (fresh drop)')
reg((npri == 0) & (prevc < 5), 'first, prev c<5')
fresh = (npri == 0) & (prevc >= 5)
def reg2(mask, name):
    X = np.column_stack([np.ones(mask.sum()), tr[mask], othc[mask], prevc[mask], t[mask]])
    b, *_ = np.linalg.lstsq(X, g[mask], rcond=None); r = g[mask] - X @ b; s2 = r @ r / (len(r) - 5)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X))); print(f'{name:34s} n={mask.sum():5d} payoff effect {b[1]:+6.2f} (se {se[1]:.2f})')
for lo, hi in ((9, 11), (12, 13), (14, 15), (16, 17), (18, 19)):
    reg2(fresh & (t >= lo) & (t <= hi), f'fresh drop, t {lo}-{hi}')
    reg2(~fresh & (t >= lo) & (t <= hi), f'not fresh, t {lo}-{hi}')
for lo, hi in ((5, 9), (10, 14), (15, 21)):
    reg2(fresh & (prevc >= lo) & (prevc < hi), f'fresh drop, prev c {lo}-{hi-1}')
for a, b_ in ((0, 1), (1, 3), (3, 30)):
    reg2((npri == 0) & (prevc >= 5) & (oh >= a) & (oh < b_), f'fresh, opening hits {a}-{b_-1}')

