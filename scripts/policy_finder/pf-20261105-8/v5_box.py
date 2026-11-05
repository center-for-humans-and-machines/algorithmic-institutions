"""v5: score candidate reduced boxes on the pooled GP of v5_pool (refit here)."""
import json, numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from scipy.stats import qmc
A = np.load('/tmp/x/pool.npy')
def tf(M):  # open, std, rate, dose, hor, floor, end -> features
    X = M[:, :7].copy(); X[:, 2] = np.log(X[:, 2]); X[:, 5] = np.log(X[:, 5]); X[:, 3] = np.minimum(X[:, 3], 45)
    X[:, 6] = np.maximum(X[:, 6] - X[:, 4], 0); return X
X = tf(A); y = A[:, 7]; lo, hi = X.min(0), X.max(0); keep = y > 600
gp = GaussianProcessRegressor(C(100) * RBF(np.ones(7) * .3, (1e-2, 1e2)) + WhiteKernel(10), normalize_y=True, random_state=0)
gp.fit(((X - lo) / (hi - lo))[keep], y[keep])
pred = lambda M: gp.predict((tf(M) - lo) / (hi - lo))
def box(name, f, n=4096):
    U = qmc.Sobol(f.dim, seed=1).random(n); M = f(U); m = pred(M); k = np.argmax(m)
    print(name, 'mean', round(m.mean(), 1), 'p10', round(np.percentile(m, 10), 1), 'max', round(m.max(), 1), 'at', np.round(M[k], 3))
class B:
    def __init__(s, rng, rel_open=None, floor=2.0): s.r = rng; s.dim = len(rng); s.ro = rel_open; s.fl = floor
    def __call__(s, U):
        V = {}
        for i, (k, (a, b, lg)) in enumerate(s.r.items()):
            V[k] = np.exp(np.log(a) + U[:, i] * (np.log(b) - np.log(a))) if lg else a + U[:, i] * (b - a)
            if k in ('hor', 'end'): V[k] = np.floor(a + U[:, i] * (b - a + 1))
        op = V.get('open', V['std'] * (s.ro or 1))
        fl = V.get('floor', np.full(len(U), s.fl))
        return np.c_[op, V['std'], V['rate'], V['dose'], V['hor'], fl, V['end']]
base = dict(std=(19, 30, 0), rate=(0.09, 0.16, 1), dose=(24, 36, 0), hor=(4, 9, 0), end=(11, 19, 0))
box('5d open=std', B(base))
box('5d open=.85std', B(base, 0.85))
box('6d +open', B(dict(base, open=(14, 28, 0))))
box('6d +open floor1', B(dict(base, open=(14, 28, 0)), floor=1.0))
box('6d +open floor4', B(dict(base, open=(14, 28, 0)), floor=4.0))
box('7d v4 box', B(dict(std=(17, 32, 0), rate=(0.07, 0.18, 1), dose=(22, 42, 0), hor=(2, 10, 0), end=(9, 24, 0), open=(6, 30, 0), floor=(0.5, 8, 1))))
pk = np.array([[22.7, 22.7, 0.112, 28.7, 7, 2.0, 14]])
for j, nm, grid in [(1, 'std', np.linspace(17, 32, 7)), (2, 'rate', np.geomspace(0.08, 0.18, 7)), (3, 'dose', np.linspace(20, 40, 6)),
                    (4, 'hor', np.arange(3, 11)), (6, 'end', np.arange(9, 22, 2)), (5, 'floor', np.array([0.6, 1, 1.5, 2, 3, 5, 8]))]:
    M = np.repeat(pk, len(grid), 0); M[:, j] = grid
    if j == 1: M[:, 0] = grid
    print(nm, ' '.join(f'{a:.3g}:{b:.1f}' for a, b in zip(grid, pred(M))))
