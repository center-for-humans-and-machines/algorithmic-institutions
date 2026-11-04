"""v7: the last_round cut-off as a natural experiment in v2's sweep (last_round 8..23).
For round r, points with last_round = r ('one more round', B) vs last_round = r-1 (A) differ only in
whether they punish at round r. Compare per-episode my-group pool over rounds r..23 (sum), members,
and c per member at round 23. Also within-point regression: episode pool r..23 on own round-r spend,
controlling for state at r (sum c at r, members at r, pool r-1)."""
import numpy as np, json
d = json.load(open('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261104-9-v2_sweep/sweep.json'))
LR = np.zeros(256, int); PAR = {}
for p in d['points']:
    i = int(p['name'].split('_s')[-1]); LR[i] = p['params']['last_round']; PAR[i] = p['params']
z = np.load('/tmp/pf/v2.npz'); C = z['C'].astype(np.float32); V = z['V']; P = z['P'].astype(np.float32); G = z['G']
mine = (G == 0)
pool = 1.6 * (C * mine).sum(2) - (P * mine).sum(2)   # [pt, ep, round]
mem = mine.sum(2)
print('points per last_round', np.bincount(LR))
for r in [10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23]:
    A = LR == r - 1; B = LR == r
    if A.sum() == 0 or B.sum() == 0: continue
    pa = pool[A][:, :, r:].sum(-1); pb = pool[B][:, :, r:].sum(-1)
    spend = (P * mine)[B][:, :, :, r].sum(-1).mean()
    # point-level means to get a fair SE
    ma, mb = pa.mean(1), pb.mean(1)
    se = np.sqrt(ma.var(ddof=1) / len(ma) + mb.var(ddof=1) / len(mb))
    # state at r-1 to check comparability
    sa = pool[A][:, :, :r].sum(-1).mean(); sb = pool[B][:, :, :r].sum(-1).mean()
    print(f'r={r}: nA={A.sum()} nB={B.sum()} spend_B@r={spend:.1f}  pool r..23: A={ma.mean():.1f} B={mb.mean():.1f} diff={mb.mean()-ma.mean():+.1f} (se {se:.1f}); pool 0..r-1 A={sa:.0f} B={sb:.0f}')
# within-point regression for points that punish at r (LR >= r), episode-level
print('within-point: pool r+1..23 on round-r spend, state controls, point fixed effects')
for r in [12, 16, 20, 21, 22]:
    sel = np.where(LR >= r)[0]
    ys, Xs = [], []
    for i in sel:
        s = (P[i] * mine[i])[:, :, r].sum(-1); y = pool[i][:, r + 1:].sum(-1)
        cs = (C[i] * mine[i])[:, :, r].sum(-1); n = mem[i][:, r]
        lo = ((C[i][:, :, r] <= 5) & mine[i][:, :, r]).sum(-1)
        X = np.column_stack([s, cs, n, lo, pool[i][:, r - 1]])
        X = X - X.mean(0); ys.append(y - y.mean()); Xs.append(X)
    X = np.concatenate(Xs); y = np.concatenate(ys)
    b = np.linalg.lstsq(X, y, rcond=None)[0]; res = y - X @ b; se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * res.var())
    print(f'r={r} N={len(y)} spend coef {b[0]:+.3f} ({se[0]:.3f}) per point  [sum c {b[1]:+.2f}, members {b[2]:+.1f}, n_le5 {b[3]:+.1f}]  (cost of the point itself excluded: pool from r+1)')
