"""v3: in v2's sweep part 1 (group 0, rounds 1-10): does the response to a punishment depend on
how much the player was punished before (cumulative, own, rounds < t), and how often a punished
0-2 giver is still 0-2 next round? Also: leaving rate of the punished by class."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'scripts/policy_finder/pf-20261104-5')
from v3_load import load
A = load('plots/simulation/policy_finder/pf-20261104-5-v2_sweep/per_round_p1of2.parquet')
c, p, g, v = A['c'], A['p'], A['g'] == 0, A['v']
cum = np.concatenate([np.zeros_like(p[:, :1]), np.cumsum(p * g, 1)[:, :-1]], 1)  # punishment from us before t
rows = []
for t in range(1, 11):
    stay = g[:, t:t + 3].all(1) & v[:, t]
    rows.append(pd.DataFrame(dict(c=c[:, t][stay], p=p[:, t][stay], cum=cum[:, t][stay], d2=(c[:, t + 1:t + 3].mean(1) - c[:, t])[stay],
                                  low_next=(c[:, t + 1] <= 2)[stay], t=t)))
D = pd.concat(rows)
D['cl'] = pd.cut(D.c, [-1, 2, 6, 10, 20], labels=['0-2', '3-6', '7-10', '11+'])
D['cb'] = pd.cut(D.cum, [-1, 0, 10, 30, 60, 999], labels=['0', '1-10', '11-30', '31-60', '61+'])
D['pb'] = pd.cut(D.p, [-1, 0, 7, 15, 30], labels=['0', '1-7', '8-15', '16-30'])
for cl in ['0-2', '3-6']:
    s = D[D.cl == cl]
    print(cl, 'mean change over next 2 rounds by past punishment (rows) and current (cols)')
    print(s.pivot_table(index='cb', columns='pb', values='d2', aggfunc='mean', observed=True).round(2))
    print(s.pivot_table(index='cb', columns='pb', values='d2', aggfunc='size', observed=True))
s = D[D.cl == '0-2']
print('0-2 givers: P(still 0-2 next round) by past punishment, current')
print(s.pivot_table(index='cb', columns='pb', values='low_next', aggfunc='mean', observed=True).round(2))
