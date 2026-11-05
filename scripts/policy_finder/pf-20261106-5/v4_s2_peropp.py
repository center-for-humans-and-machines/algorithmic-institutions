# v4: per-opponent GP surfaces on v3's sweep: does each opponent want a different peak?
import numpy as np, pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
d = pd.read_csv('/tmp/v3sweep.csv')
P = ['target_open', 'n_open', 'target_late', 'window', 'slope', 'cap', 't_end']
X = d[P].copy(); X['slope'] = np.log(X.slope)
lo, hi = X.min(), X.max(); Xn = ((X - lo) / (hi - lo)).values
rng = np.random.default_rng(0)
cand = rng.random((60000, len(P)))
for c in ['n_open', 'window', 't_end']:
    i = P.index(c); cand[:, i] = np.round(cand[:, i] * (hi[c] - lo[c])) / (hi[c] - lo[c])
out = {}
for o in ['pool', 'opponent_1', 'opponent_2', 'opponent_3', 'opponent_4']:
    y = d[o].values; m = y.mean()
    gp = GaussianProcessRegressor(ConstantKernel(4) * Matern([0.5] * len(P), nu=2.5) + WhiteKernel(4), normalize_y=False, n_restarts_optimizer=2, random_state=0)
    gp.fit(Xn, y - m)
    mu = gp.predict(cand) + m
    out[o] = mu
    b = cand[mu.argmax()] * (hi - lo).values + lo.values
    print(o, 'peak', round(mu.max(), 2), dict(zip(P, np.round(b, 2))))
# cross-evaluate: each opponent's peak under the others
idx = {o: out[o].argmax() for o in out}
print('rows: peak of; cols: value under')
print(pd.DataFrame({o2: {o1: round(out[o2][idx[o1]], 2) for o1 in out} for o2 in out}))
