"""v2: candidate rule shapes on the v2 surrogate, coarse grids. Prints best few per family."""
import numpy as np, itertools, sys
from v2_surrogate import Surrogate
S = Surrogate()
def cur(Ch): return Ch[:, :, -1]
fams = {}
# A: v1
def A(T, s, H):
    return lambda Ch, Ph, t: (t < H) * s * np.clip(T - cur(Ch), 0, None)
fams['A_v1'] = (A, dict(T=[12, 13.75, 15.5], s=[1.6, 2.2, 3], H=[8, 12, 16]))
# B: standard declines linearly to 0 at round H, dose s*(T(t)-c)
def B(T, s, H):
    return lambda Ch, Ph, t: s * np.clip(T * max(1 - t / H, 0) - cur(Ch), 0, None)
fams['B_decl'] = (B, dict(T=[13, 15, 17, 19], s=[1.6, 2.2, 3], H=[14, 20, 28, 40]))
# C: bar declines, dose from the full standard
def Cr(T, s, H):
    return lambda Ch, Ph, t: s * np.clip(T - cur(Ch), 0, None) * (cur(Ch) < T * max(1 - t / H, 0))
fams['C_bar'] = (Cr, dict(T=[12, 13.75, 15.5], s=[1.6, 2.2, 3], H=[14, 20, 28, 40]))
# D: v1 until H1, afterwards only free riders below L (same dose) until 23
def D(T, s, H, L):
    return lambda Ch, Ph, t: s * np.clip(T - cur(Ch), 0, None) * ((t < H) | (cur(Ch) < L))
fams['D_late_fr'] = (D, dict(T=[13.75, 15.5], s=[2.2, 3], H=[8, 12], L=[1, 3, 6]))
# E: v1 with standard relative to group: T_eff = min(T, others' mean + d)
def E(T, s, H, d):
    def r(Ch, Ph, t):
        c = cur(Ch); o = (c.sum(1, keepdims=True) - c) / 3
        return (t < H) * s * np.clip(np.minimum(T, o + d) - c, 0, None)
    return r
fams['E_rel'] = (E, dict(T=[13.75, 15.5], s=[2.2, 3], H=[12], d=[-2, 0, 3, 6]))
# F: power of the shortfall: p = s*T*(short/T)^g  (g>1 concentrates on free riders)
def F(T, s, H, g):
    return lambda Ch, Ph, t: (t < H) * s * T * (np.clip(T - cur(Ch), 0, None) / T) ** g
fams['F_pow'] = (F, dict(T=[12, 13.75, 15.5, 17], s=[2.2, 3], H=[10], g=[0.5, 0.75, 1.5, 2]))
# J: escalation (e>0) or forgiveness (e<0) with the number of earlier punishments
def J(T, s, H, e):
    def r(Ch, Ph, t):
        n = (Ph > 0).sum(2) if t > 0 else np.zeros(Ch.shape[:2])
        return (t < H) * s * np.clip(T - cur(Ch), 0, None) * np.clip(1 + e * n, 0, None)
    return r
fams['J_esc'] = (J, dict(T=[13.75], s=[2.2], H=[10], e=[-0.3, -0.15, 0.15, 0.4]))
# K: credit: standard for a player is min(T, previous c + d) after round 0
def K(T, s, H, d):
    def r(Ch, Ph, t):
        c = cur(Ch); std = np.minimum(T, Ch[:, :, -2] + d) if t > 0 else T
        return (t < H) * s * np.clip(std - c, 0, None)
    return r
fams['K_credit'] = (K, dict(T=[13.75], s=[2.2], H=[10], d=[0, 3, 6]))
fams['F_pow2'] = (F, dict(T=[15.5, 17, 19, 21], s=[1.6, 2.2, 3], H=[10, 13], g=[1, 1.5, 2, 3]))
# D2: v1 until H, then free riders below L (same dose) until H2
def D2(T, s, H, L, H2):
    return lambda Ch, Ph, t: s * np.clip(T - cur(Ch), 0, None) * ((t < H) | ((cur(Ch) < L) & (t < H2)))
fams['D2_tail'] = (D2, dict(T=[13.75], s=[2.2], H=[10], L=[1, 4], H2=[14, 18]))
# G: p = M * (shortfall/T)^g : M the dose of a zero contributor, T the standard, g the curvature
def Gr(T, M, H, g):
    return lambda Ch, Ph, t: (t < H) * M * (np.clip(T - cur(Ch), 0, None) / T) ** g
fams['G_pow'] = (Gr, dict(T=[17, 19, 21, 23], M=[24, 30, 38], H=[10], g=[1.5, 2, 2.5, 3]))
fams['G_powH'] = (Gr, dict(T=[19], M=[30], H=[6, 8, 10, 12, 14, 16], g=[2]))
fams['G_fine'] = (Gr, dict(T=[15, 17, 19], M=[30, 45], H=[10], g=[1, 1.25, 1.5, 1.75, 2.25]))
which = sys.argv[1:] or list(fams)
for name in which:
    f, grid = fams[name]
    res = []
    for vals in itertools.product(*grid.values()):
        kw = dict(zip(grid, vals))
        sc = S.play(f(**kw), n=4000, seed=7)[0]
        res.append((sc, kw))
    res.sort(key=lambda x: -x[0])
    print(name, [(round(s, 1), kw) for s, kw in res[:12]], flush=True)
