"""How does next-round contribution change with this round's punishment, by contribution level?"""
import sys; sys.path.insert(0, 'scripts/policy_finder/pf-20261104-2')
from human_load import load
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
d = load()
print('games', d.game.nunique(), 'rows', len(d))
print(d[['contribution','punishment','common_good']].describe().round(2))
print('manager timeout rate', d.manager_no_input.mean().round(3), 'player timeout', d.player_no_input.mean().round(3))
v = d[(d.player_no_input == 0) & (d.next_player_no_input == 0) & d.next_contribution.notna()].copy()
v['dc'] = v.next_contribution - v.contribution
v['pbin'] = pd.cut(v.punishment, [-1, 0, 2, 5, 10, 30])
v['cbin'] = pd.cut(v.contribution, [-1, 4, 9, 14, 19, 20])
print(v.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='mean').round(2))
print(v.pivot_table(index='cbin', columns='pbin', values='dc', aggfunc='size'))
# punishment relative to others: deficit
v['deficit'] = v.others_mean_c - v.contribution
v['dbin'] = pd.cut(v.deficit, [-21, -3, 0, 3, 6, 21])
print(v.pivot_table(index='dbin', columns='pbin', values='dc', aggfunc='mean').round(2))
print(v.pivot_table(index='dbin', columns='pbin', values='dc', aggfunc='size'))
# OLS of dc on punishment, controlling contribution and deficit
import numpy.linalg as la
v = v[v.deficit.notna()]
X = np.column_stack([np.ones(len(v)), v.contribution, v.deficit, v.punishment, v.punishment * (v.contribution >= 15)])
b = la.lstsq(X, v.dc.values, rcond=None)[0]
print('OLS dc ~ 1 + c + deficit + p + p*(c>=15):', b.round(3))

# per-point marginal response for under-contributors only (deficit > 0), split by deficit size
for lo, hi in [(0, 3), (3, 8), (8, 21)]:
    w = v[(v.deficit > lo) & (v.deficit <= hi)]
    X = np.column_stack([np.ones(len(w)), w.contribution, w.deficit, w.punishment])
    b = la.lstsq(X, w.dc.values, rcond=None)[0]
    print(f'deficit ({lo},{hi}] n={len(w)}: dc per punishment point {b[3]:.3f}')
# cumulative: does punishment in rounds t raise contribution 2-3 rounds later?
