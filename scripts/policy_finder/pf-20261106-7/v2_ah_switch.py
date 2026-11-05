# v2: what makes the artificial humans switch groups (from v1's simulated games)
import numpy as np
d = np.load('/tmp/v1sweep.npz'); C, P, G, V = [d[k].astype(np.float32) for k in 'CPGV']
rng = np.random.default_rng(1); idx = rng.choice(len(C), 60000, replace=False)
C, P, G, V = C[idx], P[idx], G[idx], V[idx]
rows = []
for dec in [3, 7, 11, 15, 19]:
    t0 = dec - 3
    sl = slice(t0, dec + 1)
    g = G[:, :, dec]
    sw = (G[:, :, dec + 1] != g).astype(float)
    own_p = P[:, :, sl].mean(-1); own_c = C[:, :, sl].mean(-1)
    own_p_last = P[:, :, dec]
    gstat = {}
    for name, X in [('p', P), ('c', C)]:
        per = []
        for gg in [0, 1]:
            m = (G[:, :, sl] == gg)
            per.append((X[:, :, sl] * m).sum((1, 2)) / np.maximum(m.sum((1, 2)), 1))
        per = np.stack(per, 1)
        gstat[name + '_own'] = np.take_along_axis(per, g.astype(int), 1)
        gstat[name + '_oth'] = np.take_along_axis(per, 1 - g.astype(int), 1)
    n = np.stack([(G[:, :, dec] == gg).sum(1) for gg in [0, 1]], 1).astype(float)
    n_own = np.take_along_axis(n, g.astype(int), 1); n_oth = 8 - n_own
    rows.append(np.stack([sw, own_p, own_c, gstat['p_own'], gstat['p_oth'], gstat['c_own'], gstat['c_oth'], n_own, own_p_last, np.full_like(sw, dec)], -1).reshape(-1, 10))
R = np.concatenate(rows)
names = ['sw','own_p','own_c','p_own','p_oth','c_own','c_oth','n_own','own_p_last','dec']
col = {k: R[:, i] for i, k in enumerate(names)}
print('switch rate', col['sw'].mean(), 'by decision', [round(col['sw'][col['dec']==k].mean(),3) for k in [3,7,11,15,19]])
# logistic regression
import torch as th
X = np.stack([col['own_p'], col['p_own'] - col['p_oth'], col['p_oth'], col['own_c'], col['c_own'] - col['c_oth'], col['n_own'] - col['n_oth'] if False else col['n_own'] - 4, col['own_p_last']], 1)
xn = ['own_p', 'dp(own-oth grp)', 'p_oth', 'own_c', 'dc(own-oth grp)', 'n_own-4', 'own_p_last']
Xt = th.tensor(np.c_[np.ones(len(X)), X], dtype=th.float64); yt = th.tensor(col['sw'], dtype=th.float64)
w = th.zeros(Xt.shape[1], dtype=th.float64, requires_grad=True)
opt = th.optim.LBFGS([w], max_iter=200)
def cl():
    opt.zero_grad(); l = th.nn.functional.binary_cross_entropy_with_logits(Xt @ w, yt); l.backward(); return l
opt.step(cl)
print(dict(zip(['const'] + xn, np.round(w.detach().numpy(), 3))))
# simple tables
print('switch rate by own block-mean punishment band')
for a, b in [(0,0),(0.01,1),(1,2.5),(2.5,5),(5,10),(10,30)]:
    m = (col['own_p'] >= a) & (col['own_p'] <= b); print(a, b, round(col['sw'][m].mean(), 3), int(m.sum()))
print('switch rate by group punishment diff band')
for a, b in [(-30,-3),(-3,-1),(-1,-0.2),(-0.2,0.2),(0.2,1),(1,3),(3,30)]:
    m = (col['p_own'] - col['p_oth'] >= a) & (col['p_own'] - col['p_oth'] < b); print(a, b, round(col['sw'][m].mean(), 3), int(m.sum()))
print('switch rate by group contribution diff band')
for a, b in [(-20,-3),(-3,-1),(-1,1),(1,3),(3,20)]:
    m = (col['c_own'] - col['c_oth'] >= a) & (col['c_own'] - col['c_oth'] < b); print(a, b, round(col['sw'][m].mean(), 3), int(m.sum()))
