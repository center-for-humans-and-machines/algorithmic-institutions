"""v3: who absorbs the punishment in the maintenance phase (rounds 3+) of v2's good runs,
and does it buy anything? Groups punished player-rounds by own c and by past punishment."""
import sys; sys.path.insert(0, 'scripts/policy_finder/pf-20261105-3')
import numpy as np
from v3_load import load
a = load(); c, p, v, pool = a['c'], a['p'], a['v'], a['pool']
# sanity: recompute score of run 0
def score(c, p, v):
    n = v.sum(1)
    pay = 20 - c - p + ((1.6 * (c * v).sum(1) - (p * v).sum(1)) / np.maximum(n, 1))[:, None]
    m = (pay * v).sum(1) / np.maximum(n, 1)
    return m.sum(-1).mean()
print('check', score(c[0], p[0], v[0]), pool[0])
good = pool > 640
print('good runs', good.sum())
C, P, V = c[good], p[good], v[good]
cum = np.cumsum(P, -1)  # punishment through round t
for lo, hi in [(3, 6), (6, 12)]:
    print(f'--- rounds {lo}-{hi-1}: punishment share by own c and past punishment (sum over rounds 0..t-1)')
    tot = 0; rows = []
    for t in range(lo, hi):
        pt = P[..., t]; ct = C[..., t]; past = cum[..., t-1]; ok = V[..., t] & V[..., t+1]
        nxt = C[..., t+1] - ct
        rows.append((pt[ok], ct[ok], past[ok], nxt[ok]))
    pt, ct, past, nxt = [np.concatenate(x) for x in zip(*rows)]
    T = pt.sum()
    for cl, ch in [(0, 3), (3, 8), (8, 13), (13, 21)]:
        for pl, ph in [(0, 1), (1, 30), (30, 60), (60, 1000)]:
            m = (ct >= cl) & (ct < ch) & (past >= pl) & (past < ph)
            if m.sum() < 200: continue
            pun = m & (pt > 0); un = m & (pt == 0)
            print(f'c {cl:2d}-{ch-1:2d} past {pl:3d}-{ph:4d}: share of pts {pt[m].sum()/T:5.2f} freq {m.mean():.3f} '
                  f'mean p {pt[m].mean():5.1f}  dc next | p>0 {nxt[pun].mean() if pun.sum()>50 else np.nan:5.2f} (n {pun.sum()}) | p=0 {nxt[un].mean() if un.sum()>50 else np.nan:5.2f} (n {un.sum()})')
