"""v6: does the best (T, A, B) depend on the round-0 lineup? The lineup is set before any fine, and the sweeps draw
params independently of it, so E[score | lineup bin, params] is a clean conditional policy value.
Per bin of the lineup feature: quadratic in (T, A, log B) (S kept 5-12 and as a linear term) fitted to games."""
import numpy as np, itertools, sys
z = np.load('/tmp/pf6/groups.npz'); R = z['R']; cols = list(z['cols'])
g = {k: R[:, i] for i, k in enumerate(cols)}
feat = sys.argv[1] if len(sys.argv) > 1 else 'mean0'
reg = (g['T'] >= 11) & (g['T'] <= 16.5) & (g['B'] >= 1.0) & (g['B'] <= 3.2) & (g['A'] <= 5.5) & (g['S'] >= 5) & (g['S'] <= 12) & (g['n0'] == 4)
print('games in region', reg.sum())
def design(T, A, lB, Sv):
    X = [np.ones_like(T), T, A, lB, Sv, T*T, A*A, lB*lB, T*A, T*lB, A*lB, Sv*Sv]
    return np.column_stack(X)
TT, AA, BB = np.meshgrid(np.linspace(11, 16.5, 23), np.linspace(0.5, 5, 10), np.log(np.linspace(1.0, 3.2, 23)), indexing='ij')
grid = design(TT.ravel(), AA.ravel(), BB.ravel(), np.full(TT.size, 7.5))
f = g[feat]
qs = np.nanquantile(f[reg], [0, .2, .4, .6, .8, 1.0])
glob_fit = None
fits = []
for lo, hi in zip(qs[:-1], qs[1:]):
    m = reg & (f >= lo) & (f <= hi)
    X = design(g['T'][m], g['A'][m], np.log(g['B'][m]), g['S'][m]); y = g['score'][m]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pr = grid @ beta
    k = pr.argmax()
    fits.append((m.mean() / reg.mean(), beta))
    print(f'{feat} [{lo:5.2f},{hi:5.2f}] n{m.sum():6d} best T {TT.ravel()[k]:5.2f} A {AA.ravel()[k]:4.2f} B {np.exp(BB.ravel()[k]):4.2f} -> {pr[k]:6.1f}')
# fixed policy: maximise share-weighted mean over bins
tot = sum(w * (grid @ b) for w, b in fits)
k = tot.argmax()
print(f'fixed best T {TT.ravel()[k]:5.2f} A {AA.ravel()[k]:4.2f} B {np.exp(BB.ravel()[k]):4.2f} -> {tot[k]:6.1f}')
adapt = sum(w * (grid @ b).max() for w, b in fits)
print(f'adaptive (per-bin best) -> {adapt:6.1f}; per-bin value at fixed best:', [round(float((grid @ b)[k]), 1) for w, b in fits])
