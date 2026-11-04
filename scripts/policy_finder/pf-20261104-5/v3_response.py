"""v3: player-level response in v2's sweep (part 1, our group 0, rounds 0-10): change from c(t) to
mean c over t+1..t+4 for players who stay in group 0 through t+4, by contribution class at t and
punishment at t; residualised on the group's mean contribution at t."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'scripts/policy_finder/pf-20261104-5')
from v3_load import load
A = load('plots/simulation/policy_finder/pf-20261104-5-v2_sweep/per_round_p1of2.parquet')
c, p, g, v = A['c'], A['p'], A['g'] == 0, A['v']
rows = []
for t in range(0, 11):
    stay = g[:, t:t + 5].all(1) & v[:, t]
    fut = c[:, t + 1:t + 5].mean(1)
    nxt = c[:, t + 1]
    gm = (c[:, t] * g[:, t]).sum(1, keepdims=True) / np.maximum(g[:, t].sum(1, keepdims=True), 1) * np.ones((1, 8))
    rows.append(pd.DataFrame(dict(c=c[:, t][stay], p=p[:, t][stay], d4=(fut - c[:, t])[stay], d1=(nxt - c[:, t])[stay], gm=gm[stay], t=t)))
D = pd.concat(rows)
D['cl'] = pd.cut(D.c, [-1, 2, 6, 10, 20], labels=['0-2', '3-6', '7-10', '11+'])
D['pb'] = pd.cut(D.p, [-1, 0, 3, 7, 12, 20, 30], labels=['0', '1-3', '4-7', '8-12', '13-20', '21-30'])
print(D.pivot_table(index='cl', columns='pb', values='d4', aggfunc='mean', observed=True).round(2))
print(D.pivot_table(index='cl', columns='pb', values='d1', aggfunc='mean', observed=True).round(2))
print(D.pivot_table(index='cl', columns='pb', values='d4', aggfunc='size', observed=True))
for cl in ['0-2', '3-6', '7-10']:
    s = D[D.cl == cl]
    X = np.c_[np.ones(len(s)), s.c, s.p, s.gm, s.t]
    b = np.linalg.lstsq(X, s.d4.values, rcond=None)[0]
    print(cl, 'd4 on [1,c,p,gm,t]', b.round(3))
