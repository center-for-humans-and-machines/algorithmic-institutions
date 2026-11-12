"""v2: in the v1 sweep (rule-run groups), how a member's later contributions and leaving respond to the
punishment it got, by its contribution and round. Across design points the same contribution draws different
punishments, which is close to an experiment."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, 'scripts/policy_finder/pf-20261112-6')
import os
from load_sweep import load
if not os.path.exists('/tmp/v1sweep.npz'):
    D = load('plots/simulation/policy_finder/pf-20261112-6_sweep/')
    np.savez('/tmp/v1sweep.npz', **D)
    del D
D = np.load('/tmp/v1sweep.npz')
c, p, g, v, mg = D['c'], D['p'], D['g'], D['v'], D['mine_g']
N = c.shape[0]
mine = (g == mg[:, None, None])
rows = []
for t in range(23):
    m = mine[:, :, t] & (v[:, :, t] == 1)
    stay1 = g[:, :, t + 1] == g[:, :, t]
    for h in (1, 2, 3, 4):
        if t + h > 23: continue
        pass
    rec = dict(t=np.full(m.sum(), t), c=c[:, :, t][m], p=p[:, :, t][m],
               cn1=c[:, :, t + 1][m], leave=(~stay1)[m].astype(float),
               cn3=c[:, :, t + 1:min(t + 4, 24)].mean(-1)[m],
               gc=(c[:, :, t] * mine[:, :, t]).sum(1, keepdims=True).repeat(8, 1)[m] / mine[:, :, t].sum(1, keepdims=True).repeat(8, 1)[m])
    rows.append(pd.DataFrame({k: np.asarray(x, dtype=np.float32) for k, x in rec.items()}))
R = pd.concat(rows)
R['pb'] = pd.cut(R.p, [-1, 0, 1, 2, 4, 7, 12, 30])
R['cb'] = pd.cut(R.c, [-1, 0, 2, 5, 9, 20])
R['pos'] = np.where(R.t >= 20, 'late', 'pos' + (R.t % 4).astype(str))
print('mean next-3-round contribution (cn3) minus c, by c bin x punishment bin (rounds 0-19 staying not required)')
R['d3'] = R.cn3 - R.c
print(R.pivot_table(index='cb', columns='pb', values='d3', aggfunc='mean', observed=False).round(2))
print(R.pivot_table(index='cb', columns='pb', values='d3', aggfunc='size', observed=False))
dec = R[R.t.isin([3, 7, 11, 15, 19])]
print('leave prob at decision rounds by c bin x punishment bin (punished in the decision round)')
print(dec.pivot_table(index='cb', columns='pb', values='leave', aggfunc='mean', observed=False).round(3))
R.drop(columns=['pb','cb']).to_parquet('/tmp/v1resp.parquet')
