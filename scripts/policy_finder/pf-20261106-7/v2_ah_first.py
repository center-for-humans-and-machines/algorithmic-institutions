# v2: is the artificial humans' response to punishment about round 0, or about the first punishment a player gets?
import numpy as np
d = np.load('/tmp/v1sweep.npz'); C, P, G, V = [d[k].astype(np.float32) for k in 'CPGV']
rng = np.random.default_rng(2); idx = rng.choice(len(C), 40000, replace=False)
C, P, G, V = C[idx], P[idx], G[idx], V[idx]
cum = np.concatenate([np.zeros_like(P[:, :, :1]), np.cumsum(P, -1)[:, :, :-1]], -1)  # punishment before round t
K = 4
X, Y = [], []
for t in range(0, 24 - K):
    m = (G[:, :, t+K] == G[:, :, t]) & (V[:, :, t] > 0) & (V[:, :, t+K] > 0) & (C[:, :, t] <= 12)
    grp = np.stack([((G[:, :, t] == g) * C[:, :, t]).sum(1) / np.maximum((G[:, :, t] == g).sum(1), 1) for g in [0, 1]], 1)
    gm = np.take_along_axis(grp, G[:, :, t].astype(int), 1)
    p = P[:, :, t]; first = (cum[:, :, t] == 0).astype(float); r0 = float(t == 0)
    blockstart = float(t % 4 == 0 and t > 0)
    X.append(np.stack([np.ones(m.sum()), C[:, :, t][m], gm[m], p[m], (p * first)[m], (p * r0)[m], (p * blockstart)[m], np.sqrt(p)[m], np.full(m.sum(), t), np.log1p(cum[:, :, t])[m]], 1)); Y.append(C[:, :, t+K][m])
X, Y = np.concatenate(X), np.concatenate(Y)
names = ['const', 'own c', 'group mean c', 'p', 'p*first-ever', 'p*round0', 'p*blockstart', 'sqrt p', 't', 'log1p cum p before']
b = np.linalg.lstsq(X, Y, rcond=None)[0]
print(f'c(t+{K}) for c(t)<=12 stayers:', dict(zip(names, np.round(b, 3))))
# nonparametric: punished players in round t at c<=9, by whether first ever punishment; extra c(t+4) over unpunished peers in same round and c band
for t in [0, 1, 2, 4, 5, 8, 12]:
    out = []
    for first in [1, 0]:
        for a, bb in [(3, 6), (7, 12), (13, 30)]:
            m = (G[:, :, t+4] == G[:, :, t]) & (C[:, :, t] <= 9) & (P[:, :, t] >= a) & (P[:, :, t] <= bb) & ((cum[:, :, t] == 0) == bool(first))
            m0 = (G[:, :, t+4] == G[:, :, t]) & (C[:, :, t] <= 9) & (P[:, :, t] == 0) & ((cum[:, :, t] == 0) == bool(first))
            if m.sum() > 200 and m0.sum() > 200:
                out.append(f'first={first} p{a}-{bb}: {(C[:, :, t+4]-C[:, :, t])[m].mean() - (C[:, :, t+4]-C[:, :, t])[m0].mean():+.2f} (n{m.sum()})')
    print('round', t, ' | '.join(out))
