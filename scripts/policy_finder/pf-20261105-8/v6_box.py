"""v6: score candidate boxes of the (free, mid, horizon, end) parametrisation on the pooled GP of v6_pool."""
import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from v6_pool import CS
from scipy.stats import qmc
gp, lo, hi = pickle.load(open('/tmp/x/gp6.pkl', 'rb'))
ZERO = float(sys.argv[1]) if len(sys.argv) > 1 else 22.
ANCH = float(sys.argv[2]) if len(sys.argv) > 2 else 8.
BOX = [float(a) for a in sys.argv[3:11]] if len(sys.argv) > 10 else [18, 30, 4, 16, 4, 10, 10, 20]

def feats(free, mid, hor, end, floor=2.):
    k = np.log(mid / free) / np.log((ZERO - ANCH) / ZERO)
    P = np.floor(np.clip(free[:, None] * (np.clip(ZERO - CS, 0, None) / ZERO) ** k[:, None], 0, 30))
    return np.column_stack([P, hor, end, np.full(len(free), floor)])

def pred(*a):
    X = feats(*a); return gp.predict((X - lo) / (hi - lo + 1e-9), return_std=True)

box = dict(free=tuple(BOX[0:2]), mid=tuple(BOX[2:4]), hor=tuple(BOX[4:6]), end=tuple(BOX[6:8]))
q = qmc.Sobol(4, seed=0).random(4096)
v = [box[k][0] + q[:, i] * (box[k][1] - box[k][0]) for i, k in enumerate(box)]
v[2] = np.floor(box['hor'][0] + q[:, 2] * (box['hor'][1] - box['hor'][0] + 1)); v[3] = np.floor(box['end'][0] + q[:, 3] * (box['end'][1] - box['end'][0] + 1))
m, s = pred(*v)
print('box mean', m.mean().round(1), 'p10', np.percentile(m, 10).round(1))
j = np.argmax(m); best = [x[j] for x in v]; print('peak', np.round(best, 2), m[j].round(1), s[j].round(1))
for i, k in enumerate(box):
    g = np.linspace(box[k][0] - (2 if k != 'mid' else 2), box[k][1] + 2, 9)
    if k in ('hor', 'end'): g = np.round(g)
    a = [np.full(9, b) for b in best]; a[i] = g
    if k == 'mid': a[i] = np.maximum(g, 1)
    mm, _ = pred(*a)
    print(k, ' '.join(f'{x:.3g}:{y:.1f}' for x, y in zip(a[i], mm)))
