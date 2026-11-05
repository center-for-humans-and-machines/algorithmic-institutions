"""v7: value per punishment point in the opening's last rounds, by the punished player's type.
Groups at round t whose rule has horizon t+1 (curve applied at t) vs horizon t (only c<2 hit), end > t+1.
Would-be dose of every valid c>=2 player computed from the rule; types:
 stuck  = hit at t-1 and c_t <= c_{t-1};  rising = hit at t-1 and c_t > c_{t-1};  fresh = not hit at t-1.
Outcome: group payoff summed over t..23. Effect per would-be point = coefficient on treat x dose."""
import numpy as np, sys
exec(open('scripts/policy_finder/pf-20261105-8/v7_late.py').read().split('rows = []')[0])
def dose_v6(prm, c):
    free, mid = prm[:, 0:1], prm[:, 1:2]
    k = np.log2(free / mid); return np.floor(np.clip(free * (np.clip(20 - c, 0, None) / 20) ** k, 0, 30))
def dose_v5(prm, c):
    s, r, d = prm[:, 0:1], prm[:, 1:2], prm[:, 2:3]
    return np.floor(np.clip(d * (np.clip(s - c, 0, None) / s) ** (r * s), 0, 30))
out = []
for f, hcol, ecol, df in (('/tmp/v6sweep.npz', 2, 3, dose_v6), ('/tmp/v5sweep.npz', 3, 4, dose_v5)):
    z = np.load(f); C, P, V, R, prm = z['C'].astype(float), z['P'].astype(float), z['V'], z['R'], z['prm']
    H = prm[R, hcol].astype(int); E = prm[R, ecol].astype(int)
    G = payoff(C, P, V); Gc = np.cumsum(G[:, ::-1], 1)[:, ::-1]
    for t in range(3, 10):
        idx = np.where(((H == t) | (H == t + 1)) & (E > t + 1))[0]
        c = C[idx, :, t]; v = V[idx, :, t]
        b = df(prm[R[idx]], c) * v * (c >= 2)
        hit = P[idx, :, t - 1] > 0; up = c > C[idx, :, t - 1]
        types = {'stuck': hit & ~up, 'rising': hit & up, 'fresh': ~hit}
        D = np.column_stack([(b * m).sum(1) for m in types.values()])
        tr = (H[idx] == t + 1).astype(float)
        out.append(np.column_stack([np.full(len(idx), t), tr, D, (c * v).sum(1) / v.sum(1).clip(1), Gc[idx, t]]))
A = np.concatenate(out); t, tr, D, mc, g = A[:, 0], A[:, 1], A[:, 2:5], A[:, 5], A[:, 6]
for lo, hi in ((3, 4), (5, 6), (7, 9)):
    m = (t >= lo) & (t <= hi)
    X = np.column_stack([np.ones(m.sum()), D[m] * tr[m, None], D[m], mc[m], tr[m]] + [(t[m] == k).astype(float) for k in range(lo + 1, hi + 1)])
    bb, *_ = np.linalg.lstsq(X, g[m], rcond=None); r = g[m] - X @ bb; s2 = r @ r / (len(r) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    print(f't {lo}-{hi} n={m.sum()}: per point  stuck {bb[1]:+.2f} ({se[1]:.2f})  rising {bb[2]:+.2f} ({se[2]:.2f})  fresh {bb[3]:+.2f} ({se[3]:.2f});  mean pts/group stuck {D[m,0].mean():.1f} rising {D[m,1].mean():.1f} fresh {D[m,2].mean():.1f}')
# split the fresh points at t 5-9 by the current contribution: mild drop (c >= 8) vs deep drop (2 <= c < 8)
out = []
for f, hcol, ecol, df in (('/tmp/v6sweep.npz', 2, 3, dose_v6), ('/tmp/v5sweep.npz', 3, 4, dose_v5)):
    z = np.load(f); C, P, V, R, prm = z['C'].astype(float), z['P'].astype(float), z['V'], z['R'], z['prm']
    H = prm[R, hcol].astype(int); E = prm[R, ecol].astype(int)
    G = payoff(C, P, V); Gc = np.cumsum(G[:, ::-1], 1)[:, ::-1]
    for t in range(5, 10):
        idx = np.where(((H == t) | (H == t + 1)) & (E > t + 1))[0]
        c = C[idx, :, t]; v = V[idx, :, t]
        b = df(prm[R[idx]], c) * v * (c >= 2)
        hit = P[idx, :, t - 1] > 0
        types = [hit, ~hit & (c >= 8), ~hit & (c < 8)]
        D = np.column_stack([(b * m).sum(1) for m in types])
        tr = (H[idx] == t + 1).astype(float)
        out.append(np.column_stack([np.full(len(idx), t), tr, D, (c * v).sum(1) / v.sum(1).clip(1), Gc[idx, t]]))
A = np.concatenate(out); t, tr, D, mc, g = A[:, 0], A[:, 1], A[:, 2:5], A[:, 5], A[:, 6]
X = np.column_stack([np.ones(len(t)), D * tr[:, None], D, mc, tr] + [(t == k).astype(float) for k in range(6, 10)])
bb, *_ = np.linalg.lstsq(X, g, rcond=None); r = g - X @ bb; s2 = r @ r / (len(r) - X.shape[1])
se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
print(f't 5-9 per point: repeat {bb[1]:+.2f} ({se[1]:.2f}) fresh mild c>=8 {bb[2]:+.2f} ({se[2]:.2f}) fresh deep 2-7 {bb[3]:+.2f} ({se[3]:.2f}); pts/group {D.mean(0).round(2)}')
