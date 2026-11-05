"""v6: the pooled v4+v5 GP (v6_pool.py) read in the v6 coordinates:
dose = pmax (1 - c/20)^g, g = log2(pmax/d10), for c <= 10;
dose = d10 ((20 - c)/10)^k, k = log(d10/d16)/log(2.5), for 10 < c < 20.
Prints the fitted peak, profiles and the integer dose tables."""
import sys, numpy as np
sys.argv = ['x']
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import v6_pool as P
gp, lo, hi = P.gp, P.lo, P.hi


def dose_new(c, pm, d10, d16):
    c = np.asarray(c, float)
    g = np.log2(pm / d10); k = np.log(d10 / d16) / np.log(2.5)
    a = pm * np.clip(1 - c / 20, 0, 1) ** g
    b = d10 * np.clip((20 - c) / 10, 0, 1) ** k
    return np.minimum(30, np.where(c <= 10, a, b)) * (c < 20)


def F(pm, d10, d16, hz, lt, ls):
    return np.stack([dose_new(0, pm, d10, d16), d10 * np.ones_like(pm), dose_new(16, pm, d10, d16), hz, lt, ls], -1)


def pred(*a):
    X = (F(*a) - lo) / (hi - lo)
    return gp.predict(X)


rng = np.random.default_rng(1); n = 200000
pm = rng.uniform(24, 40, n); d10 = rng.uniform(4, 10, n); d16 = rng.uniform(0.3, 4, n)
hz = rng.integers(3, 11, n).astype(float); lt = rng.integers(1, 8, n).astype(float); ls = rng.integers(10, 21, n).astype(float)
ok = d16 < d10 * 0.8
m = pred(pm, d10, d16, hz, lt, ls); m[~ok] = -1e9
i = m.argmax(); pk = np.array([pm[i], d10[i], d16[i], hz[i], lt[i], ls[i]])
print('peak', pk.round(2), m[i].round(2))
names = ['pmax', 'd10', 'd16', 'horizon', 'lthr', 'lstop']
grids = [np.linspace(24, 40, 9), np.linspace(4, 10, 13), np.linspace(0.3, 4, 10), np.arange(3, 11.), np.arange(1, 8.), np.arange(10, 21.)]
for j, nm in enumerate(names):
    A = np.tile(pk, (len(grids[j]), 1)); A[:, j] = grids[j]
    print(nm, ' '.join(f'{g:.1f}:{v:.1f}' for g, v in zip(grids[j], pred(*A.T))))
cs = np.arange(20)
for lab, args in [('peak', pk[:3]), ('v5 base', (30, 7, 0.93)), ('d16=2', (30, 7, 2)), ('d16=3', (30, 7, 3))]:
    print(lab, np.floor(dose_new(cs, *args)).astype(int))
print('v4 s187', np.floor(P.dose(cs, 29.38, 23, 2.59)).astype(int))
print('v4 gp  ', np.floor(P.dose(cs, 30.08, 24, 2.93)).astype(int))
