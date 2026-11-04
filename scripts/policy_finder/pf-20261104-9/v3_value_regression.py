"""Design-level value of punishment points in v3's sweep: regress each point's pool (sweep.json) on
the punishment it dealt per episode in classes (round class x c/others'-mean class)."""
import numpy as np, json
z = np.load('/tmp/pf/v3.npz'); C = z['C'].astype(float); V = z['V']; P = z['P'].astype(float); G = z['G']
NP, NE = C.shape[:2]
sw = json.load(open('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261104-9-v3_sweep/sweep.json'))
pool = np.zeros(NP)
for q in sw['points']:
    pool[int(q['name'].split('_s')[-1])] = q['pool']
m = (G == 0) * V
k = m.sum(2, keepdims=True)
o = ((C * m).sum(2, keepdims=True) - C * m) / np.maximum(k - m, 1)
ratio = C / np.maximum(o, 0.5)
Pm = P * (G == 0)
rcls = {'r0': [0], 'r4,8': [4, 8], 'mid<=10': [1, 2, 5, 6, 9, 10], 'late': list(range(11, 24))}
ccls = {'<0.5': (0, 0.5), '0.5-0.8': (0.5, 0.8), '>=0.8': (0.8, 99)}
feats, names = [], []
for rn, rs in rcls.items():
    for cn, (lo, hi) in ccls.items():
        mask = np.zeros_like(P, bool); mask[..., rs] = True
        mask &= (ratio >= lo) & (ratio < hi)
        feats.append((Pm * mask).sum((2, 3)).mean(1)); names.append(f'{rn} {cn}')
F = np.column_stack(feats)
keep = F.std(0) > 0.5
F = F[:, keep]; names = [n for n, kk in zip(names, keep) if kk]
X = np.column_stack([np.ones(NP), F])
b, *_ = np.linalg.lstsq(X, pool, rcond=None)
res = pool - X @ b; se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * res.var() * NP / (NP - X.shape[1]))
print('R2', 1 - res.var() / pool.var())
for n, bb, s, f in zip(['const'] + names, b, se, [0] + list(F.mean(0))):
    print(f'{n:18s} {bb:+.3f} ({s:.3f})  mean points/episode {f:.1f}')
print('pool per episode-round; coefficient = pool per round per point per episode -> x24 = pool total per point')
