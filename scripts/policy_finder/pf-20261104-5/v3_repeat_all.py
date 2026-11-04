"""v3: as v3_repeat.py, but the past punishment counts every manager's (ours and ah's), and the
sample is players in group 0 at t whose past includes time in ah's group (newcomers)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'scripts/policy_finder/pf-20261104-5')
from v3_load import load
A = load('plots/simulation/policy_finder/pf-20261104-5-v2_sweep/per_round_p1of2.parquet')
c, p, g, v = A['c'], A['p'], A['g'] == 0, A['v']
cum_all = np.concatenate([np.zeros_like(p[:, :1]), np.cumsum(p, 1)[:, :-1]], 1)
cum_oth = np.concatenate([np.zeros_like(p[:, :1]), np.cumsum(p * ~g, 1)[:, :-1]], 1)
rows = []
for t in range(4, 16):
    stay = g[:, t:t + 3].all(1) & v[:, t] & (c[:, t] <= 6)
    rows.append(pd.DataFrame(dict(c=c[:, t][stay], p=p[:, t][stay], ca=cum_all[:, t][stay], co=cum_oth[:, t][stay],
                                  d2=(c[:, t + 1:t + 3].mean(1) - c[:, t])[stay])))
D = pd.concat(rows)
D['pb'] = pd.cut(D.p, [-1, 0, 7, 15, 30], labels=['0', '1-7', '8-15', '16-30'])
for col in ['ca', 'co']:
    D['b'] = pd.cut(D[col], [-1, 0, 10, 30, 60, 999])
    print('0-6 givers, rounds 4-15, by', col)
    print(D.pivot_table(index='b', columns='pb', values='d2', aggfunc='mean', observed=True).round(2))
    print(D.pivot_table(index='b', columns='pb', values='d2', aggfunc='size', observed=True))
